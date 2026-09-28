"""LSTM next-step forecaster (Telemanom-style prediction error detector).

Trained on a selected observation history to predict the next report from a
trailing window. Scores are derived, causal prediction errors, not observed
fault labels. The fault status of the context is not assumed to be known.

Reference: Hundman et al., KDD 2018 (nonparametric dynamic thresholding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .lstm_autoencoder import TrainConfig, _loss_function


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class ForecasterDetector:
    def __init__(self, window: int = 24, hidden: int = 96, layers: int = 2,
                 dropout: float = 0.1, seed: int = 42):
        self.cfg_model = dict(window=window, hidden=hidden, layers=layers,
                              dropout=dropout)
        self.seed = seed
        self.model: LSTMForecaster | None = None
        self.feature_cols: list[str] = []
        self.err_scale: np.ndarray | None = None   # per-channel train err scale
        self.err_norm: float = 1.0
        self._device = "cpu"
        self.train_cfg: dict = {}

    def _make_pairs(self, blocks: list[np.ndarray], rng: np.random.Generator,
                    n: int, *, balance_blocks: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
        wb = [b for b in blocks if len(b) >= self.cfg_model["window"] + 1]
        w = self.cfg_model["window"]
        counts = np.array([len(b) - w for b in wb], dtype=np.int64)
        if balance_blocks:
            bi = rng.integers(0, len(wb), size=n)
        else:
            probs = counts / counts.sum()
            bi = rng.choice(len(wb), size=n, p=probs)
        X = np.empty((n, w, wb[0].shape[1]), dtype=np.float32)
        Y = np.empty((n, wb[0].shape[1]), dtype=np.float32)
        for bidx in np.unique(bi):
            rows = np.flatnonzero(bi == bidx)
            blk = wb[int(bidx)]
            # Next-step targets: the last valid window start is len(blk)-w-1 so
            # that blk[start+w] is the in-bounds step after the window.  high is
            # exclusive, so it must be len(blk)-w (NOT +1, which indexes past the
            # end).  ``counts`` above uses the same len(blk)-w pair count.
            starts = rng.integers(0, len(blk) - w, size=len(rows))
            view = np.lib.stride_tricks.sliding_window_view(
                blk, w, axis=0).transpose(0, 2, 1)
            X[rows] = np.nan_to_num(view[starts], nan=0.0)
            Y[rows] = np.nan_to_num(blk[starts + w], nan=0.0)
        return X, Y

    def fit(self, train_blocks: list[np.ndarray], feature_cols: list[str],
            val_blocks: list[np.ndarray] | None = None,
            cfg: TrainConfig | None = None, device: str | None = None,
            log=print) -> "ForecasterDetector":
        cfg = cfg or TrainConfig(window=self.cfg_model["window"])
        self.train_cfg = dict(cfg.__dict__)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log(f"[forecaster] device={device} blocks={len(train_blocks)} window={cfg.window}")

        self.feature_cols = list(feature_cols)
        self.model = LSTMForecaster(len(feature_cols), self.cfg_model["hidden"],
                                    self.cfg_model["layers"],
                                    self.cfg_model["dropout"]).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        loss_fn = _loss_function(cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

        best_val, best_state, bad = math.inf, None, 0
        Xv = Yv = None
        if val_blocks:
            Xv, Yv = self._make_pairs(
                val_blocks, np.random.default_rng(1),
                min(8192, 4 * cfg.batch_size),
                balance_blocks=cfg.balance_groups)
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            n_batches = max(1, cfg.windows_per_epoch // cfg.batch_size)
            tot = 0.0
            for _ in range(n_batches):
                X, Y = self._make_pairs(train_blocks, rng, cfg.batch_size,
                                         balance_blocks=cfg.balance_groups)
                xb = torch.from_numpy(X).to(device)
                yb = torch.from_numpy(Y).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    loss = loss_fn(self.model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            sched.step()
            if Xv is not None:
                v = self._val_err(Xv, Yv, cfg.batch_size)
                log(f"[forecaster] epoch {epoch:02d} train={tot/n_batches:.5f} val={v:.5f}")
                if v < best_val - 1e-4:
                    best_val, bad = v, 0
                    best_state = {k: t.detach().cpu().clone()
                                  for k, t in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        log(f"[forecaster] early stop @ {epoch}")
                        break
            else:
                log(f"[forecaster] epoch {epoch:02d} train={tot/n_batches:.5f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # per-channel error scale from clean training predictions
        Xc, Yc = self._make_pairs(train_blocks, np.random.default_rng(2), 8192,
                                  balance_blocks=cfg.balance_groups)
        errs = np.abs(self._predict(Xc) - Yc)
        self.err_scale = np.quantile(errs, 0.99, axis=0) + 1e-6
        s = (errs / self.err_scale).mean(axis=1)
        self.err_norm = float(np.quantile(s, 0.99)) or 1.0
        return self

    @torch.no_grad()
    def _predict(self, X: np.ndarray, bs: int = 512) -> np.ndarray:
        self.model.eval()
        outs = []
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s:s + bs]).to(self._device)
            with torch.amp.autocast("cuda", enabled=self._device == "cuda"):
                outs.append(self.model(xb).float().cpu().numpy())
        return np.concatenate(outs)

    def _val_err(self, Xv: np.ndarray, Yv: np.ndarray, bs: int) -> float:
        return float(np.mean((self._predict(Xv, bs) - Yv) ** 2))

    def score_series(self, block: np.ndarray, ewma_halflife: float = 3.0) -> np.ndarray:
        w = self.cfg_model["window"]
        n = len(block)
        if n <= w:
            return np.zeros(n)
        clean = np.nan_to_num(block, nan=0.0)
        wins = np.lib.stride_tricks.sliding_window_view(clean, w, axis=0)
        wins = wins.transpose(0, 2, 1)                   # (n-w+1, w, F)
        n_pairs = len(wins) - 1
        # Keep peak host RAM bounded while scoring multi-year station blocks.
        err = np.empty((n_pairs, clean.shape[1]), dtype=np.float32)
        chunk = max(512 * 8, 512)
        for base in range(0, n_pairs, chunk):
            # Cap the window slice at n_pairs: the final sliding window (index
            # n_pairs) has no next-step target, so scoring it would make
            # _predict(X) one row longer than Y and break the broadcast.
            end = min(base + chunk, n_pairs)
            X = np.ascontiguousarray(wins[base:end])
            Y = clean[w + base:w + end]
            err[base:end] = np.abs(self._predict(X) - Y)
        e = (err / self.err_scale).mean(axis=1)
        # Causal EWMA with a raw-score floor.  A centered convolution would
        # use future observations and is invalid for an online AWS detector.
        # Failed/nonfinite predictions are unavailable score evidence. A
        # complete-series maximum here would leak later errors backward.
        e = np.where(np.isfinite(e), e, 0.0)
        alpha = 1.0 - np.exp(np.log(0.5) / max(float(ewma_halflife), 1e-6))
        smooth = np.empty_like(e)
        if len(e):
            smooth[0] = e[0]
            for i in range(1, len(e)):
                smooth[i] = alpha * e[i] + (1.0 - alpha) * smooth[i - 1]
            e = np.maximum(smooth, e)
        # No forecast exists for the first window. Copying the first future
        # error backward would issue alerts before the evidence arrived.
        e = np.concatenate([np.zeros(w), e])
        return e / self.err_norm

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "cfg_model": self.cfg_model,
                    "feature_cols": self.feature_cols, "err_scale": self.err_scale,
                    "err_norm": self.err_norm, "train_cfg": self.train_cfg,
                    "scoring_mode": "causal_next_step"}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ForecasterDetector":
        blob = torch.load(path, map_location=device, weights_only=False)
        obj = cls(**blob["cfg_model"])
        obj.model = LSTMForecaster(len(blob["feature_cols"]), blob["cfg_model"]["hidden"],
                                   blob["cfg_model"]["layers"], blob["cfg_model"]["dropout"])
        obj.model.load_state_dict(blob["state_dict"])
        obj.model.to(device).eval()
        obj.feature_cols = blob["feature_cols"]
        obj.err_scale = np.asarray(blob["err_scale"])
        obj.err_norm = blob["err_norm"]
        obj.train_cfg = dict(blob.get("train_cfg", {}))
        obj._device = device
        return obj

"""LSTM sequence autoencoder (PyTorch) — the deep branch of the ensemble.

Trained on a selected observation history, whose fault status may be unknown.
The causal anomaly score uses the last timestep's reconstruction error from a
trailing window. Earlier points receive no score until a full window exists.
Scores/reconstructions are model outputs, never observed values or labels.

Runs offline on Kaggle (torch is preinstalled). Deterministic seeding.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_features)
        self.n_features, self.hidden, self.layers = n_features, hidden, layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, (h, c) = self.encoder(x)
        # repeat last hidden state across the window as decoder input
        dec_in = enc[:, -1:, :].repeat(1, x.size(1), 1)
        out, _ = self.decoder(dec_in, (h, c))
        return self.head(out)


@dataclass
class TrainConfig:
    window: int = 168
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 5
    windows_per_epoch: int = 65536   # sampled windows per epoch
    val_windows: int = 8192
    # Every station|source group is a deployment domain.  Sampling windows in
    # proportion to history length lets the longest stations dominate the
    # representation and quietly hurts leave-station-out generalisation.
    # Equal block sampling is the safer default; callers can disable it for
    # the legacy length-weighted behaviour.
    balance_groups: bool = True
    # Weather archives can contain a small number of unlabelled natural faults.
    # Smooth-L1 keeps those points from dominating an otherwise clean batch,
    # while ``mse`` remains available for exact legacy reproduction.
    loss: str = "huber"
    huber_beta: float = 1.0


class WindowSampler:
    """Memory-efficient on-the-fly window sampling from many station blocks."""
    def __init__(self, blocks: list[np.ndarray], window: int,
                 rng: np.random.Generator, *, balance_blocks: bool = True):
        self.blocks = [b for b in blocks if len(b) >= window + 1]
        self.window = window
        self.rng = rng
        self.balance_blocks = bool(balance_blocks)
        self.counts = np.array([len(b) - window for b in self.blocks], dtype=np.int64)

    def sample(self, n: int) -> np.ndarray:
        if not len(self.blocks):
            raise ValueError(f"no blocks contain at least window+1={self.window + 1} rows")
        # Equal block probabilities make each station|source contribute the
        # same number of training examples, independent of archive length.
        # This matters for the explicit unseen-station holdout protocol.
        if self.balance_blocks:
            bi = self.rng.integers(0, len(self.blocks), size=n)
        else:
            probs = self.counts / self.counts.sum()
            bi = self.rng.choice(len(self.blocks), size=n, p=probs)
        out = np.empty((n, self.window, self.blocks[0].shape[1]), dtype=np.float32)
        # Grouped indexed reads avoid a Python loop over every item in a large
        # batch.  The strided views are zero-copy; only the selected windows
        # are materialised into the batch tensor.
        for bidx in np.unique(bi):
            rows = np.flatnonzero(bi == bidx)
            blk = self.blocks[int(bidx)]
            starts = self.rng.integers(0, len(blk) - self.window + 1,
                                       size=len(rows))
            view = np.lib.stride_tricks.sliding_window_view(
                blk, self.window, axis=0).transpose(0, 2, 1)
            out[rows] = np.nan_to_num(view[starts], nan=0.0)
        return out

    def all_windows(self, max_windows: int = 20000) -> np.ndarray:
        """Dense (strided) window extraction for validation scoring."""
        if not len(self.blocks) or max_windows <= 0:
            return np.empty((0, self.window, 0), dtype=np.float32)
        wins = []
        for blk, cnt in zip(self.blocks, self.counts):
            stride = max(1, (len(blk) - self.window) // max(1, max_windows // len(self.blocks)))
            for s in range(0, len(blk) - self.window + 1, stride):
                wins.append(np.nan_to_num(blk[s:s + self.window], nan=0.0))
        return np.stack(wins) if wins else np.empty((0,))


def _loss_function(cfg: TrainConfig) -> nn.Module:
    """Return the configured reconstruction loss.

    ``SmoothL1Loss`` is materially less sensitive to a few naturally faulty
    observations in a supposedly clean archive.  Keeping the choice explicit
    makes old MSE-trained artifacts reproducible when needed.
    """
    if str(cfg.loss).lower() in {"huber", "smooth_l1", "smoothl1"}:
        return nn.SmoothL1Loss(beta=float(cfg.huber_beta))
    return nn.MSELoss()


class LSTMAEDetector:
    def __init__(self, window: int = 168, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, seed: int = 42):
        self.cfg_model = dict(window=window, hidden=hidden, layers=layers, dropout=dropout)
        self.seed = seed
        self.model: LSTMAutoencoder | None = None
        self.feature_cols: list[str] = []
        self.err_norm: float = 1.0
        self.train_cfg: dict = {}

    # ------------------------------------------------------------------ train
    def fit(self, train_blocks: list[np.ndarray], feature_cols: list[str],
            val_blocks: list[np.ndarray] | None = None, cfg: TrainConfig | None = None,
            device: str | None = None, log=print) -> "LSTMAEDetector":
        cfg = cfg or TrainConfig(window=self.cfg_model["window"])
        self.train_cfg = dict(cfg.__dict__)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[lstm-ae] device={device}  blocks={len(train_blocks)}")

        self.feature_cols = list(feature_cols)
        self.model = LSTMAutoencoder(len(feature_cols), **{k: v for k, v in
                                     self.cfg_model.items() if k != "window"}).to(device)
        sampler = WindowSampler(train_blocks, cfg.window, rng,
                                balance_blocks=cfg.balance_groups)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        loss_fn = _loss_function(cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

        best_val, best_state, bad = math.inf, None, 0
        val_win = None
        if val_blocks:
            val_win = WindowSampler(val_blocks, cfg.window, rng,
                                    balance_blocks=cfg.balance_groups).all_windows(
                                        cfg.val_windows)

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            n_batches = max(1, cfg.windows_per_epoch // cfg.batch_size)
            tot = 0.0
            for _ in range(n_batches):
                wb = torch.from_numpy(sampler.sample(cfg.batch_size)).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    rec = self.model(wb)
                    loss = loss_fn(rec, wb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            sched.step()

            if val_win is not None and len(val_win):
                v = self._batch_errors(val_win, device, cfg.batch_size).mean()
                log(f"[lstm-ae] epoch {epoch:02d} train={tot/n_batches:.5f} val={v:.5f}")
                if v < best_val - 1e-4:
                    best_val, bad = v, 0
                    best_state = {k: t.detach().cpu().clone()
                                  for k, t in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        log(f"[lstm-ae] early stop @ epoch {epoch}")
                        break
            else:
                log(f"[lstm-ae] epoch {epoch:02d} train={tot/n_batches:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        # calibration: error scale from training distribution
        cal = sampler.sample(min(4096, cfg.windows_per_epoch))
        errs = self._batch_errors(cal, device, cfg.batch_size)
        self.err_norm = float(np.quantile(errs, 0.99)) or 1.0
        self._device = device
        return self

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def _batch_errors(self, arr: np.ndarray, device: str, bs: int) -> np.ndarray:
        """Per-timestep reconstruction error, mean over features: (n, window)."""
        self.model.eval()
        errs = []
        for s in range(0, len(arr), bs):
            w = torch.from_numpy(arr[s:s + bs]).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                rec = self.model(w)
            e = (rec - w).pow(2).mean(dim=2)          # (b, window)
            errs.append(e.float().cpu().numpy())
        return np.concatenate(errs) if errs else np.empty((0, self.cfg_model["window"]))

    def score_series(self, block: np.ndarray, device: str | None = None,
                     batch_size: int = 512) -> np.ndarray:
        """Score each report using only its trailing window.

Warm-up scores are zero (no available evidence), not a normal-status label.
Retrospective overlap averaging belongs only in ``reconstruct``.
"""
        device = device or getattr(self, "_device", "cpu")
        window = self.cfg_model["window"]
        n = len(block)
        scores = np.zeros(n, dtype=np.float64)
        if n < window:
            return scores
        clean = np.nan_to_num(np.asarray(block), nan=0.0)
        n_w = n - window + 1
        # Score in bounded chunks.  A dense (N-window, window, F) allocation
        # is needlessly large for the full multi-year corpus.
        view = np.lib.stride_tricks.sliding_window_view(clean, window, axis=0)
        view = view.transpose(0, 2, 1)
        chunk_windows = max(batch_size * 8, batch_size)
        for base in range(0, n_w, chunk_windows):
            arr = np.ascontiguousarray(view[base:base + chunk_windows])
            errs = self._batch_errors(arr, device, batch_size)
            start = base + window - 1
            scores[start:start + len(errs)] = errs[:, -1]
        return scores / self.err_norm

    @torch.no_grad()
    def reconstruct(self, block: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Retrospective predicted reconstruction; never observed values.

Uses future context, so it must not supply an online detection metric.
"""
        device = getattr(self, "_device", "cpu")
        window = self.cfg_model["window"]
        n = len(block)
        if n == 0:
            return np.empty((0, block.shape[1]), dtype=np.float32)
        clean = np.nan_to_num(block, nan=0.0)
        if n < window:
            pad = np.repeat(clean[:1], window - n, axis=0)
            clean = np.concatenate([pad, clean])
        wins = np.lib.stride_tricks.sliding_window_view(clean, window, axis=0)
        wins = wins.transpose(0, 2, 1).copy()                    # (n', w, F)
        recs = []
        for s in range(0, len(wins), batch_size):
            w = torch.from_numpy(wins[s:s + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                rec = self.model(w).float().cpu().numpy()
            recs.append(rec)
        recs = np.concatenate(recs)
        acc = np.zeros((n, clean.shape[1])); cnt = np.zeros(n)
        for i in range(len(wins)):
            j = n - len(wins) + i
            seg = recs[i][-min(window, n - j):]
            acc[j:j + len(seg)] += seg
            cnt[j:j + len(seg)] += 1
        cnt[cnt == 0] = 1
        return acc / cnt[:, None]

    # ------------------------------------------------------------------ persist
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "cfg_model": self.cfg_model, "feature_cols": self.feature_cols,
                    "err_norm": self.err_norm, "train_cfg": self.train_cfg,
                    "scoring_mode": "causal_last_timestep"}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LSTMAEDetector":
        blob = torch.load(path, map_location=device, weights_only=False)
        obj = cls(**blob["cfg_model"])
        obj.model = LSTMAutoencoder(len(blob["feature_cols"]), **
                                    {k: v for k, v in blob["cfg_model"].items()
                                     if k != "window"})
        obj.model.load_state_dict(blob["state_dict"])
        obj.model.to(device).eval()
        obj.feature_cols = blob["feature_cols"]
        obj.err_norm = blob["err_norm"]
        obj.train_cfg = dict(blob.get("train_cfg", {}))
        obj._device = device
        return obj


def save_arch_meta(path: str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

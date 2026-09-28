"""Transformer autoencoder for AWS anomaly detection (PyTorch).

Self-attention encoder over the multivariate observation window, information
bottleneck at the final token, self-attention decoder reconstructing every
timestep. Causal anomaly scores use only the last reconstructed timestep of
each trailing window (same protocol as the LSTM-AE branch). Scores and
reconstructions are model outputs, not observations or ground truth.

Design follows the spirit of Anomaly-Transformer (Xu et al., ICLR 2022) and
TranAD (Tuli et al., VLDB 2022) while staying dependency-free/offline: we use
standard attention blocks without the minimax association loss (which proved
finicky to tune reliably on heterogeneous station data).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .lstm_autoencoder import TrainConfig, WindowSampler, _loss_function


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerAE(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, nhead: int = 8,
                 layers: int = 3, ff: int = 256, bottleneck: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.to_latent = nn.Linear(d_model, bottleneck)
        self.from_latent = nn.Linear(bottleneck, d_model)
        dec_layer = nn.TransformerEncoderLayer(      # self-attentive decoder
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, max(1, layers - 1))
        self.head = nn.Linear(d_model, n_features)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.posenc(self.inp(x))
        z = self.encoder(h)
        b = self.from_latent(self.to_latent(z[:, -1]))          # last-token bottleneck
        dec_in = b.unsqueeze(1) + self.posenc(
            torch.zeros_like(z[:, : x.size(1)]))
        out = self.decoder(dec_in)
        return self.head(out)


@dataclass
class TransformerTrainConfig(TrainConfig):
    window: int = 48    # attention pays per-token; shorter windows are effective


class TransformerAEDetector:
    def __init__(self, window: int = 48, d_model: int = 128, nhead: int = 8,
                 layers: int = 3, ff: int = 256, bottleneck: int = 64,
                 dropout: float = 0.1, seed: int = 42):
        self.cfg_model = dict(window=window, d_model=d_model, nhead=nhead,
                              layers=layers, ff=ff, bottleneck=bottleneck,
                              dropout=dropout)
        self.seed = seed
        self.model: TransformerAE | None = None
        self.feature_cols: list[str] = []
        self.err_norm: float = 1.0
        self._device = "cpu"
        self.train_cfg: dict = {}

    def fit(self, train_blocks: list[np.ndarray], feature_cols: list[str],
            val_blocks: list[np.ndarray] | None = None,
            cfg: TransformerTrainConfig | None = None,
            device: str | None = None, log=print) -> "TransformerAEDetector":
        cfg = cfg or TransformerTrainConfig(window=self.cfg_model["window"])
        self.train_cfg = dict(cfg.__dict__)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        log(f"[tx-ae] device={device} blocks={len(train_blocks)} window={cfg.window}")

        self.feature_cols = list(feature_cols)
        self.model = TransformerAE(len(feature_cols),
                                   **{k: v for k, v in self.cfg_model.items()
                                      if k != "window"}).to(device)
        sampler = WindowSampler(train_blocks, cfg.window, rng,
                                balance_blocks=cfg.balance_groups)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        loss_fn = _loss_function(cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

        val_win = None
        if val_blocks:
            val_win = WindowSampler(val_blocks, cfg.window, rng,
                                    balance_blocks=cfg.balance_groups).all_windows(
                                        cfg.val_windows)

        best_val, best_state, bad = math.inf, None, 0
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            n_batches = max(1, cfg.windows_per_epoch // cfg.batch_size)
            tot = 0.0
            for _ in range(n_batches):
                wb = torch.from_numpy(sampler.sample(cfg.batch_size)).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    loss = loss_fn(self.model(wb), wb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            sched.step()
            if val_win is not None and len(val_win):
                v = self._batch_errors(val_win, device, cfg.batch_size).mean()
                log(f"[tx-ae] epoch {epoch:02d} train={tot/n_batches:.5f} val={v:.5f}")
                if v < best_val - 1e-4:
                    best_val, bad = v, 0
                    best_state = {k: t.detach().cpu().clone()
                                  for k, t in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        log(f"[tx-ae] early stop @ {epoch}")
                        break
            else:
                log(f"[tx-ae] epoch {epoch:02d} train={tot/n_batches:.5f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        cal = sampler.sample(min(4096, cfg.windows_per_epoch))
        self.err_norm = float(np.quantile(self._batch_errors(cal, device, cfg.batch_size), 0.99)) or 1.0
        return self

    @torch.no_grad()
    def _batch_errors(self, arr: np.ndarray, device: str, bs: int) -> np.ndarray:
        self.model.eval()
        errs = []
        for s in range(0, len(arr), bs):
            w = torch.from_numpy(arr[s:s + bs]).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                rec = self.model(w)
            errs.append((rec - w).pow(2).mean(dim=2).float().cpu().numpy())
        return np.concatenate(errs) if errs else np.empty((0, self.cfg_model["window"]))

    def score_series(self, block: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Causal trailing-window score; warm-up zero means no evidence."""
        window = self.cfg_model["window"]
        n = len(block)
        scores = np.zeros(n, dtype=np.float64)
        if n < window:
            return scores
        clean = np.nan_to_num(np.asarray(block), nan=0.0)
        n_w = n - window + 1
        view = np.lib.stride_tricks.sliding_window_view(clean, window, axis=0)
        view = view.transpose(0, 2, 1)
        chunk_windows = max(batch_size * 8, batch_size)
        for base in range(0, n_w, chunk_windows):
            arr = np.ascontiguousarray(view[base:base + chunk_windows])
            errs = self._batch_errors(arr, self._device, batch_size)
            start = base + window - 1
            scores[start:start + len(errs)] = errs[:, -1]
        return scores / self.err_norm

    @torch.no_grad()
    def reconstruct(self, block: np.ndarray) -> np.ndarray:
        """Retrospective predicted values using future context, not observations."""
        window = self.cfg_model["window"]
        n = len(block)
        if n == 0:
            return np.empty((0, block.shape[1]), dtype=np.float32)
        clean = np.nan_to_num(block, nan=0.0)
        n_w = max(1, n - window + 1)
        if n_w == 1 and n < window:
            return self.model(torch.from_numpy(clean[None]).to(self._device))[0].cpu().numpy()[:n]
        wins = np.stack([clean[i:i + window] for i in range(n_w)])
        acc = np.zeros((n, block.shape[1])); cnt = np.zeros(n)
        for s in range(0, len(wins), 512):
            w = torch.from_numpy(wins[s:s + 512]).to(self._device)
            with torch.amp.autocast("cuda", enabled=self._device == "cuda"):
                rec = self.model(w).float().cpu().numpy()
            for i, s0 in enumerate(range(s, min(s + 512, n_w))):
                acc[s0:s0 + window] += rec[i]
                cnt[s0:s0 + window] += 1
        cnt[cnt == 0] = 1
        return acc / cnt[:, None]

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "cfg_model": self.cfg_model,
                    "feature_cols": self.feature_cols, "err_norm": self.err_norm,
                    "train_cfg": self.train_cfg,
                    "scoring_mode": "causal_last_timestep"}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TransformerAEDetector":
        blob = torch.load(path, map_location=device, weights_only=False)
        obj = cls(**blob["cfg_model"])
        obj.model = TransformerAE(len(blob["feature_cols"]),
                                  **{k: v for k, v in blob["cfg_model"].items()
                                     if k != "window"})
        obj.model.load_state_dict(blob["state_dict"])
        obj.model.to(device).eval()
        obj.feature_cols = blob["feature_cols"]
        obj.err_norm = blob["err_norm"]
        obj.train_cfg = dict(blob.get("train_cfg", {}))
        obj._device = device
        return obj

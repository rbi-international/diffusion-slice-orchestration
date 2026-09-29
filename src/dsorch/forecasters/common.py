"""Shared forecaster utilities: scaling, conditioning windows, LWMA anchor, base class."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from ..repro import derive_seed


class MinMaxScaler:
    """Per-resource min-max scaling fitted on the fitting segment only."""

    def __init__(self) -> None:
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None

    def fit(self, traces: np.ndarray) -> "MinMaxScaler":
        flat = traces.reshape(-1, 2)
        self.lo = flat.min(axis=0)
        self.hi = flat.max(axis=0)
        return self

    @property
    def span(self) -> np.ndarray:
        return np.maximum(self.hi - self.lo, 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / self.span

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.span + self.lo


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def conditioning(norm_hist: np.ndarray, slice_idx: int, num_slices: int) -> np.ndarray:
    """13-dimensional condition: five-slot bandwidth-CPU history (oldest first) + slice one-hot."""
    return np.concatenate([norm_hist.reshape(-1), one_hot(slice_idx, num_slices)]).astype(np.float32)


def build_windows(norm: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Training pairs (condition, next-slot target) from a normalized (T, S, 2) segment."""
    t_len, s_num, _ = norm.shape
    conds, targets = [], []
    for t in range(k, t_len):
        for s in range(s_num):
            conds.append(conditioning(norm[t - k : t, s, :], s, s_num))
            targets.append(norm[t, s, :])
    return np.asarray(conds, np.float32), np.asarray(targets, np.float32)


def lwma(history: np.ndarray, a: float = 0.35, b: float = 0.65) -> np.ndarray:
    """Linearly weighted moving average of Eq. (17); weights rise from oldest to newest."""
    k = len(history)
    i = np.arange(1, k + 1, dtype=float)
    w = a + b * (i - 1) / max(k - 1, 1)
    return (history * w[:, None]).sum(axis=0) / w.sum()


def count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class Forecaster:
    """Base class. Distributional forecasters implement ``_sample_norm``."""

    name = "base"
    distributional = True
    model_id = 0

    def __init__(self, cfg: Dict, num_slices: int, history: int) -> None:
        self.cfg = cfg
        self.num_slices = num_slices
        self.k = history
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.loss_history: list[dict] = []

    # -- training -----------------------------------------------------------
    def fit(self, traces: np.ndarray, seed: int) -> "Forecaster":
        self.scaler.fit(traces)
        norm = self.scaler.transform(traces)
        conds, targets = build_windows(norm, self.k)
        torch.manual_seed(derive_seed(seed, self.model_id))
        self._fit(conds, targets, derive_seed(seed, self.model_id, 1))
        self.fitted = True
        return self

    def _fit(self, conds: np.ndarray, targets: np.ndarray, seed: int) -> None:
        raise NotImplementedError

    # -- inference ----------------------------------------------------------
    def conditions_for(self, traces: np.ndarray, t_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stack conditions for every (t, s) with t in t_idx, using history t-k..t-1."""
        norm = self.scaler.transform(traces)
        conds, sl = [], []
        for t in t_idx:
            for s in range(self.num_slices):
                conds.append(conditioning(norm[t - self.k : t, s, :], s, self.num_slices))
                sl.append(s)
        return np.asarray(conds, np.float32), np.asarray(sl)

    def sample(self, traces: np.ndarray, t_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
        """Predictive samples in original units, shape (len(t_idx), S, n, 2).

        Each (t, s) item draws its noise from its own generator seeded by
        (seed, model, t, s), so results do not depend on batching or order.
        """
        conds, _ = self.conditions_for(traces, t_idx)
        item_seeds = [
            derive_seed(seed, self.model_id, int(t), s)
            for t in t_idx
            for s in range(self.num_slices)
        ]
        with torch.no_grad():
            out = self._sample_norm(conds, n, item_seeds)          # (B, n, 2) normalized
        out = self.scaler.inverse(out)
        return np.maximum(out, 0.0).reshape(len(t_idx), self.num_slices, n, 2)

    def _sample_norm(self, conds: np.ndarray, n: int, item_seeds: list[int]) -> np.ndarray:
        raise NotImplementedError

    def num_parameters(self) -> Dict[str, int]:
        return {}


def item_noise(item_seeds: list[int], shape: Tuple[int, ...]) -> torch.Tensor:
    """Independent standard normal noise of ``shape`` for every item, stacked on dim 0."""
    chunks = []
    for s in item_seeds:
        g = torch.Generator().manual_seed(int(s))
        chunks.append(torch.randn(shape, generator=g))
    return torch.stack(chunks, dim=0)

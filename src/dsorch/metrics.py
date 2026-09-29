"""Orchestration and forecast metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def orchestration_metrics(df: pd.DataFrame, test_start: int) -> Dict[str, float]:
    """Submitted-request metrics over the held-out slots (Section "Evaluation metrics")."""
    d = df[df["slot"] >= test_start]
    slots = d.drop_duplicates("slot")
    u = d[["u_bw", "u_cpu"]].to_numpy()
    tgt = d[["tgt_bw", "tgt_cpu"]].to_numpy()
    alloc = d[["a_bw", "a_cpu"]].to_numpy()
    out = {
        "miss_rate": 100.0 * d["miss"].mean(),
        "acceptance": 100.0 * d["admitted"].mean(),
        "mean_latency_ms": d["latency_ms"].mean(),
        "p95_latency_ms": d["latency_ms"].quantile(0.95),
        "p99_latency_ms": d["latency_ms"].quantile(0.99),
        "admitted_miss_rate": 100.0 * d.loc[d["admitted"], "latency_ms"].gt(d.loc[d["admitted"], "deadline_ms"]).mean()
        if d["admitted"].any() else np.nan,
        "isolation": slots["isolation"].mean(),
        "utilization": 100.0 * slots["utilization"].mean(),
        "fragmentation": slots["fragmentation"].mean(),
        "over_reservation": 100.0 * np.maximum(tgt - u, 0).sum() / u.sum(),
        "realized_reservation_ratio": alloc.sum() / u.sum(),
        "under_reservation_share": 100.0 * (tgt < u).any(axis=1).mean(),
    }
    for name, g in d.groupby("slice"):
        out[f"p95_{name}_ms"] = g["latency_ms"].quantile(0.95)
        out[f"miss_{name}"] = 100.0 * g["miss"].mean()
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Forecast quality
# ---------------------------------------------------------------------------

def pinball(y: np.ndarray, q_pred: np.ndarray, tau: float) -> np.ndarray:
    e = y - q_pred
    return np.maximum(tau * e, (tau - 1.0) * e)


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per item and dimension. samples (..., n, 2), y (..., 2)."""
    term1 = np.abs(samples - y[..., None, :]).mean(axis=-2)
    s = np.sort(samples, axis=-2)
    n = samples.shape[-2]
    i = np.arange(1, n + 1)[:, None]
    # E|X - X'| = 2/n^2 * sum_i (2i - n - 1) x_(i)
    term2 = (2.0 / n**2) * ((2 * i - n - 1) * s).sum(axis=-2)
    return term1 - 0.5 * term2


def crps_quantiles(qs: np.ndarray, levels: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quantile-grid approximation CRPS ~ 2 * mean_tau pinball_tau. qs (..., L, 2)."""
    e = y[..., None, :] - qs
    lv = levels[:, None]
    return 2.0 * np.maximum(lv * e, (lv - 1.0) * e).mean(axis=-2)


def energy_score(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-dimensional energy score per item. samples (B, n, 2), y (B, 2)."""
    t1 = np.linalg.norm(samples - y[:, None, :], axis=-1).mean(axis=-1)
    diff = samples[:, :, None, :] - samples[:, None, :, :]
    t2 = np.linalg.norm(diff, axis=-1).mean(axis=(-1, -2))
    return t1 - 0.5 * t2

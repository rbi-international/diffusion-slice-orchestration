"""Demand traces: controlled burst workloads and the public B5G trace replay."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .system import SliceSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
B5G_DEFAULT = REPO_ROOT / "data" / "external" / "public_b5g_trace.csv"


def controlled_trace(
    slices: Sequence[SliceSpec], slots: int, burst_factor: float, seed: int
) -> np.ndarray:
    """Generate a (slots, S, 2) bandwidth-CPU demand trace.

    Periodic component, log-normal noise (sigma 0.08, 0.11, 0.14 by slice),
    4 + int(2 * burst_factor) burst windows of 8-23 slots with amplitudes
    0.25-0.75 times the burst factor, five-slot URLLC alarm bursts, and
    periodic mMTC device-density pulses. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    traces = np.zeros((slots, len(slices), 2), dtype=float)
    t = np.arange(slots)
    for s, spec in enumerate(slices):
        phase = rng.uniform(0, 2 * np.pi)
        periodic = 1.0 + 0.12 * np.sin(t / (18.0 + 3 * s) + phase)
        noise = rng.lognormal(mean=0.0, sigma=0.08 + 0.03 * s, size=slots)
        burst = np.ones(slots)
        for _ in range(4 + int(2 * burst_factor)):
            start = rng.integers(15, max(20, slots - 35))
            length = rng.integers(8, 24)
            burst[start : start + length] += rng.uniform(0.25, 0.75) * burst_factor
        if spec.name == "URLLC":
            for _ in range(6):
                start = rng.integers(10, slots - 10)
                burst[start : start + 5] += 0.45 * burst_factor
        if spec.name == "mMTC":
            burst += 0.12 * burst_factor * (np.sin(t / 7.0) > 0.85)
        traces[:, s, 0] = spec.base_bw * periodic * noise * burst
        traces[:, s, 1] = spec.base_cpu * (0.96 + 0.08 * periodic) * noise * (0.85 + 0.25 * burst)
    return traces


def b5g_available(path: Path = B5G_DEFAULT) -> bool:
    return Path(path).exists()


def b5g_segments(path: Path = B5G_DEFAULT) -> List[int]:
    df = pd.read_csv(path, usecols=["seed"])
    return sorted(int(s) for s in df["seed"].unique())


def b5g_trace(
    slices: Sequence[SliceSpec], segment: int, slots: int = 260, path: Path = B5G_DEFAULT
) -> np.ndarray:
    """Return the (slots, S, 2) replay tensor for one seed-selected trace segment.

    The processed trace uses the FlowPathQoS CPU proxy and maps the public
    label mIoT to mMTC. Its fitting-window statistics were frozen before the
    held-out slots were produced (see data/README.md).
    """
    df = pd.read_csv(path)
    df["slice"] = df["slice"].replace({"mIoT": "mMTC"})
    seg = df[df["seed"] == segment]
    if seg.empty:
        raise ValueError(f"segment {segment} not found in {path}")
    out = np.zeros((slots, len(slices), 2), dtype=float)
    for s, spec in enumerate(slices):
        sub = seg[seg["slice"] == spec.name].sort_values("slot")
        if len(sub) < slots:
            raise ValueError(f"segment {segment} slice {spec.name} has {len(sub)} < {slots} slots")
        out[:, s, 0] = sub["bw_demand"].to_numpy()[:slots]
        out[:, s, 1] = sub["cpu_demand"].to_numpy()[:slots]
    return out

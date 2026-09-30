"""Risk-budgeted joint reservation (rule RB), study B.

Implements docs/PREREGISTRATION_B.md version 3.1, section 2. Section numbers
in comments refer to that document.

Terminology
-----------
bank      Predictive distributions of one fitted forecaster for a set of
          slots: samples (generators) or a quantile grid (QuantileMLP).
reserve   The reservation vector r for a risk level eps (section 2.2).
ladder    Risk levels eps_s * 1.5^k used by capacity-aware relaxation (2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .policies import Placement, place
from .system import SystemModel

EPS_MIN, EPS_MAX = 0.01, 0.60            # section 2.3
LAMBDA_STEP, LAMBDA_MAX = 0.01, 6.0      # section 2.2
G_GRID = np.logspace(-1.0, 1.0, 41)      # section 2.4, contains 1.0 exactly
LADDER_FACTOR, LADDER_LEVELS = 1.5, 12   # section 2.5
MAX_RELAX_STEPS = 12                     # section 2.5
U_GRID = (np.arange(400) + 0.5) / 400.0  # integration grid for QuantileMLP shortfall


def clip_eps(eps):
    return np.clip(eps, EPS_MIN, EPS_MAX)


# ---------------------------------------------------------------------------
# Predictive-distribution banks
# ---------------------------------------------------------------------------

class SampleBank:
    """Generator samples x (n_t, S, M, 2) in demand units."""

    kind = "samples"

    def __init__(self, samples: np.ndarray) -> None:
        self.x = samples
        self.M = samples.shape[2]
        self.med = np.median(samples, axis=2)                               # (n, S, 2)
        self.h = np.maximum(np.quantile(samples, 0.95, axis=2) - self.med, 1e-6)
        lam = (samples - self.med[:, :, None, :]) / self.h[:, :, None, :]   # (n, S, M, 2)
        self.lam_sorted = np.sort(np.maximum(lam.max(axis=-1), 0.0), axis=2)  # minimal lambda per sample

    def _coverage_count(self, r: np.ndarray) -> np.ndarray:
        return (self.x <= r[:, :, None, :]).all(axis=-1).sum(axis=2)

    def reserve(self, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Smallest grid lambda with joint coverage >= 1 - eps. Returns (r, lambda_capped)."""
        eps = np.broadcast_to(eps, self.med.shape[:2])
        need = np.clip(np.ceil((1.0 - eps) * self.M - 1e-9).astype(int), 1, self.M)   # samples to cover
        lam_need = np.take_along_axis(self.lam_sorted, (need - 1)[:, :, None], axis=2)[:, :, 0]
        lam = np.ceil(lam_need / LAMBDA_STEP - 1e-9) * LAMBDA_STEP
        capped = lam > LAMBDA_MAX + 1e-12
        lam = np.minimum(lam, LAMBDA_MAX)
        r = self.med + lam[:, :, None] * self.h
        # guard against floating-point edge cases: step up until the grid value really covers
        for _ in range(3):
            short = (self._coverage_count(r) < need) & ~capped & (lam < LAMBDA_MAX)
            if not short.any():
                break
            lam = np.where(short, np.minimum(lam + LAMBDA_STEP, LAMBDA_MAX), lam)
            r = self.med + lam[:, :, None] * self.h
        return r, capped

    def shortfall(self, r: np.ndarray) -> np.ndarray:
        """sum_d E[(x_d - r_d)+] / med_d, unit-free (section 2.5)."""
        es = np.maximum(self.x - r[:, :, None, :], 0.0).mean(axis=2)          # (n, S, 2)
        return (es / np.maximum(self.med, 1e-6)).sum(axis=-1)


class QuantileBank:
    """QuantileMLP quantile grid q (n_t, S, L, 2) at sorted levels (L,)."""

    kind = "quantiles"

    def __init__(self, q: np.ndarray, levels: Sequence[float]) -> None:
        self.q = q
        self.levels = np.asarray(levels, dtype=float)
        self.med = self._interp(np.full(q.shape[:2], 0.5))
        self._u_values = self._interp_many(U_GRID)                          # (n, S, U, 2)

    def _interp(self, p: np.ndarray) -> np.ndarray:
        """Linear interpolation at level p (n, S), flat beyond the grid ends."""
        lv = self.levels
        p = np.clip(p, lv[0], lv[-1])
        j = np.clip(np.searchsorted(lv, p, side="right") - 1, 0, len(lv) - 2)
        w = ((p - lv[j]) / (lv[j + 1] - lv[j]))[..., None]
        lo = np.take_along_axis(self.q, j[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0]
        hi = np.take_along_axis(self.q, (j + 1)[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0]
        return lo + w * (hi - lo)

    def _interp_many(self, us: np.ndarray) -> np.ndarray:
        return np.stack([self._interp(np.full(self.q.shape[:2], u)) for u in us], axis=2)

    def reserve(self, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bonferroni: r_d = Q_d(1 - eps / 2). lambda_capped flags levels beyond the grid."""
        eps = np.broadcast_to(eps, self.q.shape[:2])
        p = 1.0 - eps / 2.0
        capped = p > self.levels[-1] + 1e-12
        return self._interp(p), capped

    def shortfall(self, r: np.ndarray) -> np.ndarray:
        es = np.maximum(self._u_values - r[:, :, None, :], 0.0).mean(axis=2)
        return (es / np.maximum(self.med, 1e-6)).sum(axis=-1)


def build_bank(forecaster, traces: np.ndarray, t_idx: np.ndarray, samples: int, seed: int):
    if forecaster.name == "QuantileMLP":
        return QuantileBank(forecaster.quantiles(traces, t_idx), forecaster.levels.numpy())
    return SampleBank(forecaster.sample(traces, t_idx, samples, seed))


# ---------------------------------------------------------------------------
# Validation-window recalibration (section 2.4)
# ---------------------------------------------------------------------------

@dataclass
class Recalibration:
    g: float
    coverage: float
    target: float
    table: List[Tuple[float, int]] = field(default_factory=list)


def recalibrate(bank_val, y_val: np.ndarray, eps_base: np.ndarray) -> Recalibration:
    """Choose g whose pooled joint coverage on the validation slots is closest to mean(1 - eps_base).

    Ties (equal absolute error, compared on integer counts) are broken by the
    smallest |log g|, then the smaller g.
    """
    eps_base = clip_eps(np.asarray(eps_base, dtype=float))
    n_items = y_val.shape[0] * y_val.shape[1]
    target = float(np.mean(1.0 - eps_base))
    best_key, best, table = None, None, []
    for g in G_GRID:
        r, _ = bank_val.reserve(clip_eps(g * eps_base)[None, :])
        count = int((y_val <= r).all(axis=-1).sum())
        table.append((float(g), count))
        err = abs(count - target * n_items)
        key = (round(err, 9), abs(np.log(g)), g)
        if best_key is None or key < best_key:
            best_key, best = key, (float(g), count)
    return Recalibration(g=best[0], coverage=best[1] / n_items, target=target, table=table)


# ---------------------------------------------------------------------------
# Relaxation ladder (section 2.5)
# ---------------------------------------------------------------------------

@dataclass
class SliceLadder:
    eps: np.ndarray            # (K,) clipped, strictly increasing risk levels
    raw: np.ndarray            # (K,) unclipped values of the kept levels
    r: np.ndarray              # (K, n, 2) reservations
    es: np.ndarray             # (K, n) unit-free expected shortfall
    capped: np.ndarray         # (K, n) lambda-capped / grid-capped flags


def build_ladders(bank, eps_raw: np.ndarray) -> List[SliceLadder]:
    """Ladder per slice from eps_s' = clip(eps_raw_s), levels clip(eps_s' * 1.5^k), k = 0..11.

    eps_raw is g * eps_s before the final clip; it is kept only to flag
    whether the level-0 decision was clipped (section 2.3).
    """
    ladders = []
    S = bank.med.shape[1]
    eps_raw = np.asarray(eps_raw, dtype=float)
    eps_prime = clip_eps(eps_raw)
    for s in range(S):
        raw_all = eps_prime[s] * LADDER_FACTOR ** np.arange(LADDER_LEVELS)
        raw_all[0] = eps_raw[s]
        clipped = clip_eps(eps_prime[s] * LADDER_FACTOR ** np.arange(LADDER_LEVELS))
        keep = np.concatenate([[True], np.diff(clipped) > 1e-15])        # merge equal consecutive levels
        eps_k, raw_k = clipped[keep], raw_all[keep]
        rs, ess, caps = [], [], []
        for e in eps_k:
            eps_mat = np.full(bank.med.shape[:2], EPS_MAX)
            eps_mat[:, s] = e
            r, capped = bank.reserve(eps_mat)
            rs.append(r[:, s])
            ess.append(bank.shortfall(r)[:, s])
            caps.append(capped[:, s])
        ladders.append(SliceLadder(eps_k, raw_k, np.stack(rs), np.stack(ess), np.stack(caps)))
    return ladders


# ---------------------------------------------------------------------------
# The RB controller
# ---------------------------------------------------------------------------

class RBPolicy:
    """Controller for run_episode's `plan` hook.

    ladders cover slots t_first .. t_first + n - 1. Earlier slots (no
    five-slot history) reserve the previous demand, as in study A.
    """

    name = "RB"
    uses_estimate = False
    mode = "jcso"

    def __init__(self, name: str, ladders: List[SliceLadder], t_first: int, traces: np.ndarray,
                 scale: float = 1.0) -> None:
        self.name = name
        self.ladders = ladders
        self.t_first = t_first
        self.traces = traces
        # Amendment 1: frontier lever. Multiplies every ladder level; recalibration
        # (section 2.4) is computed on the unscaled reservations and never sees it.
        self.scale = float(scale)

    def plan(self, sm: SystemModel, t: int, qp: np.ndarray) -> Tuple[np.ndarray, Placement, Dict]:
        S = sm.num_slices
        kappa = float(sm.ctrl["kappa"])
        corr = (1.0 + kappa * qp)[:, None]
        i = t - self.t_first
        if i < 0:
            tgt = self.traces[max(t - 1, 0)] * corr
            return tgt, place(sm, tgt, sm.deadline_order, "jcso"), {"relax_steps": 0, "levels": np.zeros(S, int)}
        level = np.zeros(S, dtype=int)

        def targets(lv):
            return self.scale * np.stack([self.ladders[s].r[lv[s], i] for s in range(S)]) * corr

        requested = targets(level)
        tgt = requested
        pl = place(sm, tgt, sm.deadline_order, "jcso")
        steps = 0
        while steps < MAX_RELAX_STEPS:
            trigger = any(pl.assigned[s] < 0 or np.any(pl.alloc[s] < tgt[s] - 1e-9) for s in range(S))
            if not trigger:
                break
            best, best_key = None, None
            for s in range(S):
                lad = self.ladders[s]
                if level[s] + 1 >= len(lad.eps):
                    continue
                delta = sm.slices[s].weight * (lad.es[level[s] + 1, i] - lad.es[level[s], i])
                key = (delta, -sm.slices[s].deadline_ms)          # tie: longest deadline first
                if best_key is None or key < best_key:
                    best, best_key = s, key
            if best is None:
                break
            level[best] += 1
            steps += 1
            tgt = targets(level)
            pl = place(sm, tgt, sm.deadline_order, "jcso")
        return requested, pl, {"relax_steps": steps, "levels": level.copy()}


def decision_flags(ladders: List[SliceLadder], levels_used: np.ndarray, t_first: int, t_idx: np.ndarray) -> Dict:
    """Fractions of slice-slot decisions clipped low, clipped high and lambda-capped (section 2.3)."""
    low = high = cap = n = 0
    for row, t in zip(levels_used, t_idx):
        i = t - t_first
        for s, k in enumerate(row):
            lad = ladders[s]
            low += lad.raw[k] < EPS_MIN
            high += lad.raw[k] > EPS_MAX
            cap += bool(lad.capped[k, i])
            n += 1
    return {"frac_clip_low": low / n, "frac_clip_high": high / n, "frac_lambda_capped": cap / n}

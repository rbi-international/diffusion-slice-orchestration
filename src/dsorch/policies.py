"""Reservation rules and node placement for all compared controllers.

Every JCSO-family controller (Persistence, MovingAverage, QuantileMLP, GAN and
Diffusion) shares one reservation rule (Eq. 20), one deadline-first order and
one isolation-aware placement rule (Eqs. 21-22). They differ only in the demand
estimate u_bar supplied to Eq. (20).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .system import SystemModel, candidate_coupling


@dataclass
class Placement:
    assigned: np.ndarray            # (S,) node index or -1
    alloc: np.ndarray               # (S, 2) realized allocation
    members: Dict[int, List[int]]   # node -> slices placed on it
    used: np.ndarray                # (N, 2) used capacity


def place(
    sm: SystemModel,
    targets: np.ndarray,
    order: List[int],
    mode: str,
) -> Placement:
    """Assign each slice to at most one node and cap the target by residual capacity.

    mode = "fit"    rank by two-dimensional residual fit only (Independent)
           "static" only nodes that hold the full target, ranked by fit (StaticReserve)
           "joint"  Eq. (21) score with the continuous coupling term, no screen (JointHeuristic)
           "jcso"   Eq. (21) score after the candidate screen Phi <= theta (JCSO family)
    """
    N = len(sm.nodes)
    S = sm.num_slices
    residual = np.array([[n.bw, n.cpu] for n in sm.nodes], dtype=float)
    members: Dict[int, List[int]] = {n.idx: [] for n in sm.nodes}
    assigned = -np.ones(S, dtype=int)
    alloc = np.zeros((S, 2))
    beta, gamma, scale = float(sm.ctrl["beta"]), float(sm.ctrl["gamma"]), float(sm.ctrl["fit_scale"])
    theta = sm.iso["theta"]
    for s in order:
        tgt = np.maximum(targets[s], 1e-9)
        best, best_score = -1, -np.inf
        for n in sm.nodes:
            r = residual[n.idx]
            if r[0] <= 0 or r[1] <= 0:
                continue
            fit = min(r[0] / tgt[0], r[1] / tgt[1])
            if mode == "fit":
                score = fit
            elif mode == "static":
                if r[0] < tgt[0] or r[1] < tgt[1]:
                    continue
                score = fit
            else:
                phi = candidate_coupling(s, members[n.idx], sm.rho)
                if mode == "jcso" and phi > theta:
                    continue
                score = scale * fit - beta * n.prop_ms - gamma * phi
            if score > best_score:
                best, best_score = n.idx, score
        if best >= 0:
            a = np.minimum(targets[s], residual[best])
            assigned[s] = best
            alloc[s] = a
            residual[best] -= a
            members[best].append(s)
    used = np.array([[n.bw, n.cpu] for n in sm.nodes]) - residual
    return Placement(assigned, alloc, members, used)


class Policy:
    name = "base"
    mode = "jcso"
    uses_estimate = False

    def targets(
        self, sm: SystemModel, u_obs: np.ndarray, qp: np.ndarray, est: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, List[int]]:
        raise NotImplementedError


class Independent(Policy):
    name = "Independent"
    mode = "fit"

    def __init__(self, margin: float) -> None:
        self.margin = margin

    def targets(self, sm, u_obs, qp, est):
        return self.margin * u_obs, list(range(sm.num_slices))


class StaticReserve(Policy):
    name = "StaticReserve"
    mode = "static"

    def __init__(self, fit_mean: np.ndarray, safety: float) -> None:
        self.reserve = safety * fit_mean

    def targets(self, sm, u_obs, qp, est):
        return self.reserve.copy(), sm.deadline_order


class JointHeuristic(Policy):
    name = "JointHeuristic"
    mode = "joint"

    def targets(self, sm, u_obs, qp, est):
        kappa = float(sm.ctrl["kappa"])
        return u_obs * (1.0 + kappa * qp)[:, None], sm.deadline_order


class JCSO(Policy):
    """a_tar = alpha_s * max(u_bar, u_obs) * (1 + kappa * Q) * budget_scale (Eq. 20)."""

    mode = "jcso"
    uses_estimate = True

    def __init__(self, name: str, budget_scale: float = 1.0) -> None:
        self.name = name
        self.budget_scale = budget_scale

    def targets(self, sm, u_obs, qp, est):
        kappa = float(sm.ctrl["kappa"])
        alpha = np.array([s.alpha for s in sm.slices])
        base = np.maximum(est, u_obs) if est is not None else u_obs
        tgt = alpha[:, None] * base * (1.0 + kappa * qp)[:, None] * self.budget_scale
        return tgt, sm.deadline_order

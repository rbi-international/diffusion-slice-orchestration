"""Slot-by-slot orchestration engine and demand-estimate precomputation."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .forecasters import Forecaster, lwma
from .policies import Policy, place
from .system import (
    SystemModel,
    fragmentation,
    isolation_index,
    latency_components,
    queue_pressure,
    update_backlog,
    utilization,
)


def jcso_estimates(
    forecaster: Forecaster, traces: np.ndarray, cfg: Dict, seed: int, samples: Optional[int] = None,
    quantile: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Demand estimate u_bar for every slot t >= k (history t-k..t-1).

    Generators: u_bar = w * Q_q(M samples) + (1 - w) * LWMA   (Eq. 17)
    QuantileMLP: u_bar = w * q_hat_q + (1 - w) * LWMA
    Point estimators: u_bar = point estimate.
    Returns the estimate tensor (T, S, 2), NaN for t < k, and the mean
    inference time per slice prediction in milliseconds.
    """
    c = cfg["control"]
    k = int(c["history"])
    q = float(quantile if quantile is not None else c["quantile"])
    M = int(samples if samples is not None else c["samples"])
    w = float(c["blend_generator"])
    T, S, _ = traces.shape
    t_idx = np.arange(k, T)
    est = np.full((T, S, 2), np.nan)
    t0 = time.perf_counter()
    if not forecaster.distributional:
        est[k:] = forecaster.point(traces, t_idx)
    else:
        if forecaster.name == "QuantileMLP":
            raw = forecaster.quantile_at(traces, t_idx, q)
        else:
            raw = np.quantile(forecaster.sample(traces, t_idx, M, seed), q, axis=2)
        anchor = np.stack(
            [np.stack([lwma(traces[t - k : t, s], c["lwma_a"], c["lwma_b"]) for s in range(S)]) for t in t_idx]
        )
        est[k:] = w * raw + (1.0 - w) * anchor
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / (len(t_idx) * S)
    return {"estimate": est, "ms_per_prediction": elapsed_ms}


def run_episode(
    sm: SystemModel,
    traces: np.ndarray,
    policy: Policy,
    cfg: Dict,
    estimate: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Simulate all slots and return one record per (slot, slice)."""
    T, S, _ = traces.shape
    info = cfg["protocol"]["information"]
    lat = sm.lat
    backlog = np.zeros((S, 2))
    rows = []
    plan_log = []
    for t in range(T):
        u = traces[t]
        u_obs = u if info == "request_observed" else traces[max(t - 1, 0)]
        qp = np.array([queue_pressure(backlog[s], sm.queue) for s in range(S)])
        est = None
        if policy.uses_estimate and estimate is not None and not np.isnan(estimate[t]).any():
            est = estimate[t]
        if hasattr(policy, "plan"):
            # study-B controllers (RB): targets are the requested reservations,
            # the placement already includes any capacity-aware relaxation
            targets, pl, extra = policy.plan(sm, t, qp)
            plan_log.append(extra)
        else:
            targets, order = policy.targets(sm, u_obs, qp, est)
            pl = place(sm, targets, order, policy.mode)
        _, gamma_t = isolation_index(pl.members, sm.rho, sm.iso["theta"])
        frag = fragmentation(pl.used, sm.nodes)
        util = utilization(pl.used, sm.nodes)
        for s in range(S):
            spec = sm.slices[s]
            node = pl.assigned[s]
            a = pl.alloc[s]
            admitted = bool(
                node >= 0
                and a[0] >= sm.iso["admission_min_ratio"] * u[s, 0]
                and a[1] >= sm.iso["admission_min_ratio"] * u[s, 1]
                and gamma_t >= sm.iso["admission_min_index"]
            )
            if admitted:
                phi = sum(sm.rho[s, j] for j in pl.members[node] if j != s)
                total, l_tx, l_edge, l_q, l_iso = latency_components(
                    u[s], a, spec, sm.nodes[node].prop_ms, backlog[s], phi, lat
                )
            else:
                total, l_tx, l_edge, l_q, l_iso = lat["penalty_mult"] * spec.deadline_ms, 0.0, 0.0, 0.0, 0.0
            rows.append(
                (t, spec.name, u[s, 0], u[s, 1], targets[s, 0], targets[s, 1], a[0], a[1], int(node),
                 admitted, total, l_tx, l_edge, l_q, l_iso, spec.deadline_ms,
                 (not admitted) or total > spec.deadline_ms, gamma_t, frag, util, qp[s])
            )
            backlog[s] = update_backlog(backlog[s], u[s], a, admitted, sm.queue)
    df = pd.DataFrame(
        rows,
        columns=["slot", "slice", "u_bw", "u_cpu", "tgt_bw", "tgt_cpu", "a_bw", "a_cpu", "node",
                 "admitted", "latency_ms", "l_tx", "l_edge", "l_queue", "l_iso", "deadline_ms",
                 "miss", "isolation", "fragmentation", "utilization", "queue_pressure"],
    )
    if plan_log:
        df.attrs["plan_log"] = plan_log
    return df

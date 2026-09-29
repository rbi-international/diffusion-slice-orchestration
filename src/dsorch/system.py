"""System model of the sliced cloud-edge network.

Equation numbers refer to the GAN-JCSO paper (Qiu and Zhang, 2026), whose
orchestration-level model is reused unchanged so that the only difference
between GAN-JCSO and Diffusion-JCSO is the conditional demand generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SliceSpec:
    name: str
    deadline_ms: float
    weight: float
    base_bw: float
    base_cpu: float
    alpha: float
    eta: float
    chi_tx: float
    chi_edge: float


@dataclass(frozen=True)
class Node:
    idx: int
    bw: float
    cpu: float
    prop_ms: float


@dataclass
class SystemModel:
    slices: List[SliceSpec]
    nodes: List[Node]
    rho: np.ndarray                     # (S, S) symmetric coupling, zero diagonal
    lat: Dict[str, float]
    queue: Dict[str, float]
    iso: Dict[str, float]
    ctrl: Dict[str, float]
    deadline_order: List[int] = field(default_factory=list)

    @property
    def num_slices(self) -> int:
        return len(self.slices)

    @classmethod
    def from_config(cls, cfg: dict, capacity_multiplier: float = 1.0) -> "SystemModel":
        slices = [SliceSpec(**s) for s in cfg["slices"]]
        names = [s.name for s in slices]
        rho = np.zeros((len(slices), len(slices)))
        for key, val in cfg["coupling"].items():
            a, b = key.split("-")
            i, j = names.index(a), names.index(b)
            rho[i, j] = rho[j, i] = float(val)
        n = cfg["nodes"]
        nodes = [
            Node(i, float(bw) * capacity_multiplier, float(cpu) * capacity_multiplier, float(p))
            for i, (bw, cpu, p) in enumerate(zip(n["bw"], n["cpu"], n["prop_ms"]))
        ]
        order = sorted(range(len(slices)), key=lambda s: slices[s].deadline_ms)
        return cls(
            slices=slices,
            nodes=nodes,
            rho=rho,
            lat={k: float(v) for k, v in cfg["latency"].items()},
            queue={k: float(v) for k, v in cfg["queue"].items()},
            iso={k: float(v) for k, v in cfg["isolation"].items()},
            ctrl={k: v for k, v in cfg["control"].items()},
            deadline_order=order,
        )


# ---------------------------------------------------------------------------
# Latency model, Eqs. (4)-(7) and (10)
# ---------------------------------------------------------------------------

def latency_components(
    demand: np.ndarray,
    alloc: np.ndarray,
    spec: SliceSpec,
    prop_ms: float,
    backlog: np.ndarray,
    same_node_coupling: float,
    lat: Dict[str, float],
) -> Tuple[float, float, float, float, float]:
    """Return (total, L_tx, L_edge, L_queue, L_iso) in milliseconds."""
    b, c = float(demand[0]), float(demand[1])
    b_hat, c_hat = float(alloc[0]), float(alloc[1])
    eps = lat["eps"]
    l_tx = spec.chi_tx * (
        prop_ms + lat["tau_b"] * b / max(b_hat, eps) + lat["nu_b"] * max(b - b_hat, 0.0)
    )
    l_edge = spec.chi_edge * (
        lat["tau_c"] * c / max(c_hat, eps) + lat["nu_c"] * max(c - c_hat, 0.0)
    )
    l_queue = lat["lambda_b"] * float(backlog[0]) + lat["lambda_c"] * float(backlog[1])
    l_iso = spec.eta * l_edge * same_node_coupling
    total = l_tx + l_edge + l_queue + l_iso
    return total, l_tx, l_edge, l_queue, l_iso


# ---------------------------------------------------------------------------
# Queue dynamics, Eq. (8), and queue pressure used in Eq. (20)
# ---------------------------------------------------------------------------

def update_backlog(
    backlog: np.ndarray, demand: np.ndarray, alloc: np.ndarray, served: bool, queue: Dict[str, float]
) -> np.ndarray:
    if served:
        return queue["delta"] * backlog + np.maximum(demand - alloc, 0.0)
    return backlog + queue["xi_unassigned"] * demand


def queue_pressure(backlog: np.ndarray, queue: Dict[str, float]) -> float:
    return float(min(1.0, np.abs(backlog).sum() / queue["q_max"]))


# ---------------------------------------------------------------------------
# Isolation, Eqs. (9), (11), (22)
# ---------------------------------------------------------------------------

def candidate_coupling(s: int, members: Sequence[int], rho: np.ndarray) -> float:
    """Phi_{s,n}: summed coupling of slice s with the slices already on the node (Eq. 22)."""
    return float(sum(rho[s, j] for j in members if j != s))


def isolation_index(node_members: Dict[int, List[int]], rho: np.ndarray, theta: float) -> Tuple[float, float]:
    """Return (mean pairwise coupling over co-located pairs, Gamma_t) as in Eq. (11)."""
    total, pairs = 0.0, 0
    for members in node_members.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                total += rho[members[a], members[b]]
                pairs += 1
    if pairs == 0:
        return 0.0, 1.0
    mean_phi = total / pairs
    return mean_phi, max(0.0, 1.0 - mean_phi / theta)


# ---------------------------------------------------------------------------
# Resource metrics, Eq. (12) and utilization
# ---------------------------------------------------------------------------

def fragmentation(used: np.ndarray, nodes: Sequence[Node]) -> float:
    """F_t = 1 - mean_n min(bw utilization, cpu utilization) (Eq. 12)."""
    vals = [min(used[n.idx, 0] / n.bw, used[n.idx, 1] / n.cpu) for n in nodes]
    return float(1.0 - np.mean(vals))


def utilization(used: np.ndarray, nodes: Sequence[Node]) -> float:
    bw = used[:, 0].sum() / sum(n.bw for n in nodes)
    cpu = used[:, 1].sum() / sum(n.cpu for n in nodes)
    return float(0.5 * (bw + cpu))

import numpy as np
import pytest

from dsorch.engine import run_episode
from dsorch.policies import JCSO, Independent, JointHeuristic, StaticReserve, place


def test_eq_20_reservation(sm):
    u = np.array([[40.0, 30.0], [18.0, 20.0], [12.0, 30.0]])
    est = np.array([[45.0, 25.0], [10.0, 25.0], [12.0, 33.0]])
    qp = np.array([0.5, 0.0, 1.0])
    tgt, order = JCSO("x").targets(sm, u, qp, est)
    alpha = np.array([1.35, 1.30, 1.18])
    exp = alpha[:, None] * np.maximum(u, est) * (1 + 0.18 * qp)[:, None]
    assert np.allclose(tgt, exp)
    assert order == [1, 2, 0]


@pytest.mark.parametrize("mode", ["fit", "static", "joint", "jcso"])
def test_placement_respects_capacity(sm, mode):
    rng = np.random.default_rng(0)
    for _ in range(200):
        targets = rng.uniform(5, 140, size=(3, 2))
        pl = place(sm, targets, [1, 2, 0], mode)
        assert np.all(pl.alloc <= targets + 1e-9)
        for n in sm.nodes:
            assert pl.used[n.idx, 0] <= n.bw + 1e-9
            assert pl.used[n.idx, 1] <= n.cpu + 1e-9
        if mode == "jcso":                       # screen forbids three slices on one node
            assert all(len(m) <= 2 for m in pl.members.values())
        if mode == "static":                     # static only accepts nodes that hold the full target
            for s in range(3):
                if pl.assigned[s] >= 0:
                    assert np.allclose(pl.alloc[s], targets[s])


def test_rejected_requests_get_penalty(sm, cfg, trace):
    df = run_episode(sm, trace, StaticReserve(trace[:100].mean(0), 1.25), cfg)
    rej = df[~df["admitted"]]
    assert len(rej) > 0
    assert np.allclose(rej["latency_ms"], 2.2 * rej["deadline_ms"])
    assert rej["miss"].all()


def test_admission_rule(sm, cfg, trace):
    df = run_episode(sm, trace, JointHeuristic(), cfg)
    adm = df[df["admitted"]]
    assert (adm["a_bw"] >= 0.85 * adm["u_bw"] - 1e-9).all()
    assert (adm["a_cpu"] >= 0.85 * adm["u_cpu"] - 1e-9).all()
    assert (adm["isolation"] >= 0.25).all()


def test_proactive_mode_uses_previous_request(sm, cfg, trace):
    import copy
    c = copy.deepcopy(cfg)
    c["protocol"]["information"] = "proactive"
    df = run_episode(sm, trace, Independent(1.02), c)
    row = df[(df["slot"] == 50) & (df["slice"] == "eMBB")].iloc[0]
    assert row["tgt_bw"] == pytest.approx(1.02 * trace[49, 0, 0])

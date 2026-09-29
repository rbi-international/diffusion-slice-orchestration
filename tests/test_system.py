"""System-model equations checked against hand calculations from the paper."""
import numpy as np
import pytest

from dsorch.system import (
    candidate_coupling, fragmentation, isolation_index, latency_components,
    queue_pressure, update_backlog, utilization,
)


def test_config_matches_paper_tables(sm):
    names = [s.name for s in sm.slices]
    assert names == ["eMBB", "URLLC", "mMTC"]
    assert [s.deadline_ms for s in sm.slices] == [6.8, 2.6, 4.0]
    assert [s.alpha for s in sm.slices] == [1.35, 1.30, 1.18]
    assert [s.eta for s in sm.slices] == [0.20, 0.35, 0.25]
    assert sm.rho[0, 1] == 0.42 and sm.rho[0, 2] == 0.31 and sm.rho[1, 2] == 0.36
    assert np.allclose(sm.rho, sm.rho.T) and np.all(np.diag(sm.rho) == 0)
    assert [n.bw for n in sm.nodes] == [122, 105, 95, 84]
    assert [n.cpu for n in sm.nodes] == [118, 132, 112, 98]
    assert sm.deadline_order == [1, 2, 0]          # URLLC, mMTC, eMBB


def test_latency_eqs_5_to_10(sm):
    urllc = sm.slices[1]
    u, a = np.array([20.0, 18.0]), np.array([16.0, 20.0])
    q = np.array([2.0, 1.0])
    total, tx, edge, lq, iso = latency_components(u, a, urllc, 1.9, q, 0.36, sm.lat)
    exp_tx = 0.72 * (1.9 + 0.42 * 20 / 16 + 0.014 * 4)
    exp_edge = 0.78 * (0.55 * 18 / 20 + 0.0)
    exp_q = 0.12 * 2 + 0.14 * 1
    exp_iso = 0.35 * exp_edge * 0.36
    assert tx == pytest.approx(exp_tx)
    assert edge == pytest.approx(exp_edge)
    assert lq == pytest.approx(exp_q)
    assert iso == pytest.approx(exp_iso)
    assert total == pytest.approx(exp_tx + exp_edge + exp_q + exp_iso)


def test_backlog_eq_8(sm):
    q, u, a = np.array([4.0, 2.0]), np.array([10.0, 8.0]), np.array([7.0, 9.0])
    assert np.allclose(update_backlog(q, u, a, True, sm.queue), 0.55 * q + np.array([3.0, 0.0]))
    assert np.allclose(update_backlog(q, u, a, False, sm.queue), q + 0.15 * u)


def test_queue_pressure_uses_l1_and_saturates(sm):
    assert queue_pressure(np.array([9.0, 9.0]), sm.queue) == pytest.approx(18 / 45)
    assert queue_pressure(np.array([40.0, 40.0]), sm.queue) == 1.0


def test_isolation_index_eq_11(sm):
    assert isolation_index({0: [0], 1: [1], 2: [2], 3: []}, sm.rho, 0.52) == (0.0, 1.0)
    phi, g = isolation_index({0: [0, 1], 1: [2], 2: [], 3: []}, sm.rho, 0.52)
    assert phi == pytest.approx(0.42) and g == pytest.approx(1 - 0.42 / 0.52)
    _, g3 = isolation_index({0: [0, 1, 2]}, sm.rho, 0.30)
    assert g3 == 0.0                               # clipped at zero


def test_proposition_1_monotone(sm):
    rho = sm.rho.copy()
    last = 2.0
    for scale in np.linspace(0, 2, 21):
        _, g = isolation_index({0: [0, 1]}, rho * scale, 0.52)
        assert g <= last + 1e-12
        last = g


def test_candidate_coupling_eq_22(sm):
    assert candidate_coupling(1, [], sm.rho) == 0.0
    assert candidate_coupling(1, [0, 2], sm.rho) == pytest.approx(0.42 + 0.36)


def test_fragmentation_eq_12(sm):
    idle = np.zeros((4, 2))
    assert fragmentation(idle, sm.nodes) == pytest.approx(1.0)
    full = np.array([[n.bw, n.cpu] for n in sm.nodes])
    assert fragmentation(full, sm.nodes) == pytest.approx(0.0)
    assert utilization(full, sm.nodes) == pytest.approx(1.0)

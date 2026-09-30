"""Rule RB (study B) checked against PREREGISTRATION_B v3.1, section 2."""
import numpy as np
import pytest

from dsorch.engine import run_episode
from dsorch.reservation import (
    EPS_MAX, EPS_MIN, G_GRID, LADDER_FACTOR, QuantileBank, RBPolicy, SampleBank,
    build_ladders, clip_eps, decision_flags, recalibrate,
)


def sample_bank(n=6, S=3, M=200, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(20, 60, size=(n, S, 1, 2))
    x = base * rng.lognormal(0, 0.2, size=(n, S, M, 2))
    return SampleBank(x)


def joint_cov(bank, r):
    return (bank.x <= r[:, :, None, :]).all(axis=-1).mean(axis=2)


def test_grid_constants():
    assert np.isclose(G_GRID, 1.0).sum() == 1 and 1.0 in G_GRID    # g = 1 exactly
    assert len(G_GRID) == 41 and np.isclose(G_GRID[0], 0.1) and np.isclose(G_GRID[-1], 10.0)
    assert list(clip_eps(np.array([0.001, 0.3, 0.9]))) == [EPS_MIN, 0.3, EPS_MAX]


@pytest.mark.parametrize("eps", [0.01, 0.05, 0.137, 0.3, 0.6])
def test_generator_reservation_is_smallest_covering_grid_point(eps):
    b = sample_bank()
    r, capped = b.reserve(np.full((6, 3), eps))
    assert not capped.any()
    cov = joint_cov(b, r)
    assert np.all(cov >= 1 - eps - 1e-12)
    lam = np.round((r - b.med) / b.h, 6)[..., 0]
    smaller = b.med + np.maximum(lam - 0.01, 0)[..., None] * b.h
    cov_smaller = joint_cov(b, smaller)
    assert np.all((cov_smaller < 1 - eps - 1e-12) | (lam == 0))    # one grid step less fails


def test_lambda_cap_is_flagged():
    b = sample_bank(n=1, S=1)
    b.x[0, 0, :5] *= 1000.0                                         # 5 of 200 samples far outside
    b.__init__(b.x)
    r, capped = b.reserve(np.array([[0.01]]))                      # needs 198 of 200 covered
    assert capped.all()
    assert np.allclose(r, b.med + 6.0 * b.h)


def test_quantile_bank_bonferroni_and_cap():
    levels = [0.05, 0.5, 0.9, 0.95, 0.975, 0.99, 0.995]
    q = np.tile(np.array(levels)[None, None, :, None] * 100, (2, 3, 1, 2))
    b = QuantileBank(q, levels)
    r, capped = b.reserve(np.full((2, 3), 0.1))                    # 1 - 0.05 = 0.95
    assert np.allclose(r, 95.0) and not capped.any()
    r, _ = b.reserve(np.full((2, 3), 0.03))                        # 0.985: halfway 0.975..0.99
    assert np.allclose(r, 98.5)
    r, capped = b.reserve(np.full((2, 3), 0.004))                  # beyond the grid
    assert np.allclose(r, 99.5) and capped.all()
    assert np.allclose(b.med, 50.0)


def test_shortfall_decreases_with_reservation():
    b = sample_bank()
    lo, _ = b.reserve(np.full((6, 3), 0.5))
    hi, _ = b.reserve(np.full((6, 3), 0.02))
    assert np.all(b.shortfall(hi) <= b.shortfall(lo) + 1e-12)


def test_recalibration_hits_target_and_breaks_ties_toward_g_one():
    b = sample_bank(n=20, seed=1)
    y = np.median(b.x, axis=2)                                      # every point covered for any g
    rec = recalibrate(b, y, np.array([0.1, 0.1, 0.1]))
    assert rec.g == 1.0                                             # all g tie, least correction wins
    rng = np.random.default_rng(3)
    y2 = b.x[np.arange(20)[:, None], np.arange(3)[None, :], rng.integers(0, 200, size=(20, 3))]
    rec2 = recalibrate(b, y2, np.array([0.3, 0.3, 0.3]))
    errs = {g: abs(c / 60 - rec2.target) for g, c in rec2.table}
    assert errs[rec2.g] == pytest.approx(min(errs.values()))


def test_recalibration_uses_clipped_base_levels():
    b = sample_bank(n=20, seed=2)
    y = np.median(b.x, axis=2)
    rec = recalibrate(b, y, np.array([5.0, 5.0, 5.0]))             # c * b_s far above 0.60 (R1)
    assert rec.target == pytest.approx(1 - EPS_MAX)


def test_ladder_levels_merge_and_reservations_fall():
    b = sample_bank()
    lads = build_ladders(b, np.array([0.02, 0.3, 0.6]))
    assert np.allclose(lads[0].eps[:3], [0.02, 0.03, 0.045])
    assert lads[0].eps[-1] == EPS_MAX and len(lads[0].eps) <= 12
    assert np.all(np.diff(lads[0].eps) > 0)                         # merged, strictly increasing
    assert len(lads[2].eps) == 1                                    # 0.60 already at the cap
    assert np.allclose(lads[1].eps, [0.3, 0.45, 0.6])               # 0.675 clipped to 0.60
    for lad in lads:
        assert np.all(np.diff(lad.r, axis=0) <= 1e-9)               # reservations shrink as risk grows
        assert np.all(np.diff(lad.es, axis=0) >= -1e-9)


def _rb_setup(sm, trace, cfg, eps, cap_mult=1.0):
    from dsorch.system import SystemModel
    sm2 = SystemModel.from_config(cfg, cap_mult)
    t_idx = np.arange(5, trace.shape[0])
    rng = np.random.default_rng(0)
    x = trace[t_idx][:, :, None, :] * rng.lognormal(0, 0.15, size=(len(t_idx), 3, 200, 2))
    bank = SampleBank(x)
    lads = build_ladders(bank, np.array(eps))
    return sm2, RBPolicy("test-RB", lads, 5, trace), lads


def test_requested_target_is_level_zero_and_budget_is_pre_relaxation(sm, cfg, trace):
    import copy
    c = copy.deepcopy(cfg)
    c["protocol"]["information"] = "proactive"
    sm2, pol, lads = _rb_setup(sm, trace, c, [0.05, 0.05, 0.05], cap_mult=0.5)
    df = run_episode(sm2, trace, pol, c)
    log = df.attrs["plan_log"]
    assert any(e["relax_steps"] > 0 for e in log)                   # tight capacity forces relaxation
    assert max(e["relax_steps"] for e in log) <= 12
    qp = df.pivot(index="slot", columns="slice", values="queue_pressure")[["eMBB", "URLLC", "mMTC"]].to_numpy()
    t = 150
    expect = np.stack([lads[s].r[0, t - 5] for s in range(3)]) * (1 + 0.18 * qp[t])[:, None]
    got = df[df.slot == t].set_index("slice").loc[["eMBB", "URLLC", "mMTC"], ["tgt_bw", "tgt_cpu"]].to_numpy()
    assert np.allclose(got, expect)                                 # recorded target = requested level 0


def test_relaxation_choice_and_tie_break(sm, cfg, trace):
    sm2, pol, lads = _rb_setup(sm, trace, cfg, [0.05, 0.05, 0.05], cap_mult=0.4)
    for lad in lads:                                                # equal shortfall everywhere: pure tie
        lad.es[:] = 0.0
    qp = np.zeros(3)
    _, _, extra = pol.plan(sm2, 150, qp)
    lv = extra["levels"]
    assert extra["relax_steps"] > 0
    assert lv[0] >= lv[2] >= lv[1]                              # eMBB first, then mMTC, then URLLC
    _, _, extra_free = RBPolicy("x", lads, 5, trace).plan(SystemModelBig(sm2), 150, qp)
    assert extra_free["relax_steps"] == 0


def SystemModelBig(sm2):
    from dataclasses import replace
    import copy
    big = copy.copy(sm2)
    big.nodes = [replace(n, bw=n.bw * 100, cpu=n.cpu * 100) for n in sm2.nodes]
    return big


def test_decision_flags_count_clipping():
    b = sample_bank()
    lads = build_ladders(b, np.array([0.005, 0.1, 0.9]))           # raw below and above the range
    used = np.zeros((6, 3), dtype=int)
    f = decision_flags(lads, used, 0, np.arange(6))
    assert f["frac_clip_low"] == pytest.approx(1 / 3)
    assert f["frac_clip_high"] == pytest.approx(1 / 3)


def test_scale_lever_multiplies_requested_target(sm, cfg, trace):
    """Amendment 1: requested target = s * r(eps_0) * (1 + kappa Q); ladder scaled too."""
    sm2, pol, lads = _rb_setup(sm, trace, cfg, [0.05, 0.05, 0.05])
    qp = np.array([0.2, 0.0, 0.5])
    base, _, _ = pol.plan(sm2, 150, qp)
    scaled, _, _ = RBPolicy("s", lads, 5, trace, scale=1.4).plan(sm2, 150, qp)
    assert np.allclose(scaled, 1.4 * base)
    expect = 1.4 * np.stack([lads[s].r[0, 145] for s in range(3)]) * (1 + 0.18 * qp)[:, None]
    assert np.allclose(scaled, expect)

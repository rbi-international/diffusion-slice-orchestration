"""The study-B analysis recovers known effects on synthetic frontiers."""
import numpy as np
import pandas as pd
import pytest

from dsorch.analysis_b import analyse, cell_tests
from dsorch.experiments_b import frontier_points

OFFSET = {"Diffusion-RB": 10.0, "GAN-RB": 15.0, "QuantileMLP-RB": 5.0, "Diffusion-JCSO": 10.0,
          "GAN-JCSO": 12.0, "QuantileMLP-JCSO": 8.0}


def synthetic(n_seeds=80, short_seeds=()):
    """miss = offset + seed noise - 0.2 * over_reservation; frontiers span 24-72 percent.

    Expected: H1 diff -5 (supported), H2 +5 (contradicted), H3 0 (not supported),
    H4 (0) - (3) = -3 (supported).
    """
    rng = np.random.default_rng(0)
    rows = []
    for w in ["heavy", "extreme"]:
        for cap in [1.0, 0.82]:
            for seed in range(1001, 1001 + n_seeds):
                for m, off in OFFSET.items():
                    noise = rng.normal(0, 2.0)
                    slope = 40.0
                    if m == "GAN-RB" and w == "heavy" and cap == 1.0 and seed in short_seeds:
                        slope = 25.0                      # frontier tops out at 45 percent
                    for lever in np.linspace(0.6, 1.8, 21):
                        orr = slope * lever
                        rows.append({"workload": w, "seed": seed, "capacity": cap, "method": m, "lever": lever,
                                     "over_reservation": orr, "miss_rate": off + noise - 0.2 * orr,
                                     "acceptance": 90.0, "p95_latency_ms": 5.0, "p99_latency_ms": 8.0,
                                     "utilization": 30.0, "realized_reservation_ratio": 1.2})
    return pd.DataFrame(rows)


def test_interpolation_matches_linear_frontier():
    pts = frontier_points(synthetic(n_seeds=3), [30, 40, 50], ["workload", "seed", "capacity", "method"])
    assert pts["miss_rate"].notna().all()
    one = pts[(pts.method == "QuantileMLP-RB")].groupby("budget")["miss_rate"].mean()
    assert one.diff().dropna().round(6).eq(-2.0).all()          # slope -0.2 per percentage point


def test_verdicts_recover_known_effects():
    res = analyse(synthetic(), n_seeds=80)
    v = res["verdicts"].set_index("hypothesis")
    assert v.loc["H1", "verdict"] == "supported"
    assert v.loc["H2", "verdict"] == "contradicted"
    assert v.loc["H3", "verdict"] == "not supported"
    assert v.loc["H4", "verdict"] == "supported"
    assert v.loc["H1", "mean_diff_pp"] == pytest.approx(-5.0, abs=0.5)
    assert v.loc["H4", "mean_diff_pp"] == pytest.approx(-3.0, abs=0.5)
    assert v.loc["H1", "boot_ci_low"] < v.loc["H1", "mean_diff_pp"] < v.loc["H1", "boot_ci_high"]
    assert (res["cell_tests"].groupby("hypothesis").size() == 12).all()


def test_missing_seeds_exclude_cell_only_for_affected_hypotheses():
    short = set(range(1001, 1011))                                # 10 of 80 seeds cannot reach 50 percent
    pts = frontier_points(synthetic(short_seeds=short), [30, 40, 50], ["workload", "seed", "capacity", "method"])
    t = cell_tests(pts, n_seeds=80)
    cell = (t.workload == "heavy") & (t.capacity == 1.0) & (t.budget == 50)
    by_h = t[cell].set_index("hypothesis")
    assert not by_h.loc["H1", "analysed"] and by_h.loc["H1", "n"] == 70
    assert not by_h.loc["H4", "analysed"]
    assert by_h.loc["H2", "analysed"] and by_h.loc["H3", "analysed"]   # GAN-RB not involved
    assert np.isnan(by_h.loc["H1", "p_holm"])
    assert t[~cell]["analysed"].all()

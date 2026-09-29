import numpy as np
import pytest

from dsorch.metrics import crps_samples, energy_score
from dsorch.stats import holm, paired_comparison


def test_crps_matches_brute_force():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(4, 30, 2))
    y = rng.normal(size=(4, 2))
    fast = crps_samples(s, y)
    brute = np.abs(s - y[:, None]).mean(1) - 0.5 * np.abs(s[:, :, None] - s[:, None, :]).mean((1, 2))
    assert np.allclose(fast, brute)


def test_energy_score_reduces_to_abs_error_for_point_mass():
    s = np.zeros((1, 5, 2))
    y = np.array([[3.0, 4.0]])
    assert energy_score(s, y)[0] == pytest.approx(5.0)


def test_holm_known_values():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_paired_comparison_sign():
    r = paired_comparison(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.5, 3.5, 5.0]))
    assert r["mean_diff"] == pytest.approx(-1.0)
    assert r["ci_low"] < -1.0 < r["ci_high"]

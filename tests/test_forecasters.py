import numpy as np
import pytest

from dsorch.forecasters import build, lwma, make_schedule


def test_gan_matches_table_5_parameter_counts(cfg, trace):
    f = build("GAN", {**cfg, "forecasters": {**cfg["forecasters"], "gan": {**cfg["forecasters"]["gan"], "epochs": 1}}})
    f.fit(trace[:100], 3)
    assert f.num_parameters() == {"online": 3506, "training_only": 3169}


def test_ddpm_capacity_comparable_to_gan(cfg, trace):
    c = {**cfg, "forecasters": {**cfg["forecasters"], "ddpm": {**cfg["forecasters"]["ddpm"], "epochs": 1}}}
    f = build("DDPM", c).fit(trace[:100], 3)
    assert f.num_parameters()["online"] == 3674


def test_schedules_reach_noise():
    for kind in ["cosine", "linear"]:
        ab = np.cumprod(1 - make_schedule(kind, 20))
        assert ab[-1] < 0.02, kind
    legacy = np.cumprod(1 - make_schedule("linear_legacy", 20))
    assert legacy[-1] == pytest.approx(0.8168, abs=1e-3)   # documents the earlier flaw


def test_lwma_weights_eq_17():
    h = np.tile(np.arange(1, 6, dtype=float)[:, None], (1, 2))
    w = 0.35 + 0.65 * np.arange(5) / 4
    assert np.allclose(lwma(h), (w * np.arange(1, 6)).sum() / w.sum())


@pytest.mark.parametrize("name", ["GAN", "DDPM"])
def test_sampling_is_deterministic_and_batch_invariant(cfg, trace, name):
    key = {"GAN": "gan", "DDPM": "ddpm"}[name]
    c = {**cfg, "forecasters": {**cfg["forecasters"], key: {**cfg["forecasters"][key], "epochs": 5}}}
    a = build(name, c).fit(trace[:100], 7).sample(trace, np.arange(100, 140), 12, 7)
    b = build(name, c).fit(trace[:100], 7).sample(trace, np.arange(100, 140), 12, 7)
    assert np.array_equal(a, b)
    single = build(name, c).fit(trace[:100], 7).sample(trace, np.array([120]), 12, 7)
    assert np.allclose(a[20], single[0], atol=1e-5)


@pytest.mark.parametrize("name", ["GAN", "DDPM", "QuantileMLP"])
def test_no_test_leakage(cfg, trace, name):
    """Corrupting the held-out slots must not change anything the model learned."""
    key = {"GAN": "gan", "DDPM": "ddpm", "QuantileMLP": "quantile_mlp"}[name]
    c = {**cfg, "forecasters": {**cfg["forecasters"], key: {**cfg["forecasters"][key], "epochs": 5}}}
    corrupted = trace.copy()
    corrupted[100:] *= 50.0
    f1 = build(name, c).fit(trace[:100], 3)
    f2 = build(name, c).fit(corrupted[:100], 3)
    assert np.array_equal(f1.scaler.lo, f2.scaler.lo) and np.array_equal(f1.scaler.hi, f2.scaler.hi)
    t = np.arange(20, 60)
    if name == "QuantileMLP":
        assert np.allclose(f1.quantiles(trace, t), f2.quantiles(corrupted, t))
    else:
        assert np.allclose(f1.sample(trace, t, 8, 3), f2.sample(corrupted, t, 8, 3))

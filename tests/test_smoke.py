"""End-to-end run of the smoke configuration."""
from pathlib import Path

import pandas as pd

from dsorch.config import load_config
from dsorch.experiments import exp_orchestration

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_orchestration(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    exp_orchestration(cfg, tmp_path, 1)
    df = pd.read_csv(tmp_path / "orchestration.csv")
    assert len(df) == 2 * 8                        # two information settings x eight controllers
    assert set(df["method"]) >= {"Diffusion-JCSO", "GAN-JCSO", "JointHeuristic"}
    assert df["miss_rate"].between(0, 100).all()
    assert df["acceptance"].between(0, 100).all()

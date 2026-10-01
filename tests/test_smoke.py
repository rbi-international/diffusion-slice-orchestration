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


CALLS = []


def _square(x):
    CALLS.append(x)
    return [{"x": x, "y": x * x}]


def test_run_parallel_checkpoint_resumes(tmp_path):
    from dsorch.experiments import run_parallel
    jobs = [(1,), (2,), (3,)]
    labels = ["a", "b", "c"]
    first = run_parallel(_square, jobs, 1, tmp_path, labels)
    (tmp_path / "b.pkl").unlink()                     # simulate a job lost to an interruption
    CALLS.clear()
    again = run_parallel(_square, jobs, 1, tmp_path, labels)
    assert CALLS == [2]                               # only the missing job is recomputed
    assert again == first


def test_checkpoint_dirs_differ_for_official_and_verify_runs():
    from pathlib import Path
    from dsorch.experiments_b import checkpoint_dir
    root = Path(__file__).resolve().parents[1]
    a = checkpoint_dir(root / "results" / "study_b")
    b = checkpoint_dir(root / "results" / "verify" / "study_b")
    assert a != b and a.name == "results__study_b" and b.name == "results__verify__study_b"


def test_study_b_manifest_records_effective_settings(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "smoke_b"
    subprocess.run([sys.executable, str(root / "scripts" / "run_experiment.py"), str(root / "configs" / "smoke_b.yaml"),
                    "--jobs", "1", "--out", str(out)], check=True, capture_output=True)
    cfg = json.loads((out / "manifest.json").read_text())["config"]
    assert cfg["protocol"]["information"] == "proactive"
    assert 0.995 in cfg["forecasters"]["quantile_mlp"]["levels"]

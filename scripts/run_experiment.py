"""Run one experiment from a YAML config.

Usage:
    python scripts/run_experiment.py configs/orchestration.yaml --jobs 2
"""
from __future__ import annotations

import os

# One math-library thread per worker process. Must be set before numpy or torch
# is imported; child processes inherit it. Without this, N workers each start one
# thread per CPU core and contend for the cores (severe slow-downs on Windows).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.config import load_config  # noqa: E402
from dsorch.experiments import EXPERIMENTS, apply_selected  # noqa: E402
from dsorch.experiments_b import EXPERIMENTS_B, study_b_config  # noqa: E402

EXPERIMENTS = {**EXPERIMENTS, **EXPERIMENTS_B}
from dsorch.repro import write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--seeds", type=int, nargs="*", help="override protocol seeds")
    ap.add_argument("--out", type=Path, help="write here instead of the config's output folder "
                    "(use this for verification reruns so the saved results are not overwritten)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_selected(cfg, ROOT)
    if args.seeds:
        cfg["protocol"]["seeds"] = args.seeds
    out = args.out.resolve() if args.out else ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    kind = cfg["experiment"]
    print(f"[{kind}] -> {out}  (jobs={args.jobs})", flush=True)
    t0 = time.time()
    EXPERIMENTS[kind](cfg, out, args.jobs)
    elapsed = time.time() - t0
    # study B applies its own settings inside the experiment (proactive information,
    # extended QuantileMLP grid); record the configuration that was actually used
    used = study_b_config(cfg) if kind in EXPERIMENTS_B else cfg
    write_manifest(out, used, {"experiment": kind, "wall_seconds": round(elapsed, 1)})
    print(f"[{kind}] done in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()

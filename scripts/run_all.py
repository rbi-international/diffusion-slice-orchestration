"""Reproduce every result in the paper, in dependency order.

Usage:
    python scripts/run_all.py --jobs 4

Order: validation (selects training lengths) -> orchestration -> forecast ->
frontier -> ablation -> b5g (skipped if the external trace is absent) ->
statistical analysis -> figures.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["validation", "orchestration", "forecast", "frontier", "ablation", "b5g"]


for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*", choices=STEPS, help="run a subset of experiments")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    steps = args.only or STEPS
    for step in steps:
        if step == "b5g" and not (ROOT / "data" / "external" / "public_b5g_trace.csv").exists():
            print("! skipping b5g: data/external/public_b5g_trace.csv not found (see data/README.md)")
            continue
        run([sys.executable, "scripts/run_experiment.py", f"configs/{step}.yaml", "--jobs", str(args.jobs)])
    run([sys.executable, "scripts/analyze.py"])
    if not args.skip_figures:
        run([sys.executable, "scripts/make_figures.py"])


if __name__ == "__main__":
    main()

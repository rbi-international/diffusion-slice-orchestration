"""Run one experiment from a YAML config.

Usage:
    python scripts/run_experiment.py configs/orchestration.yaml --jobs 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.config import load_config  # noqa: E402
from dsorch.experiments import EXPERIMENTS, apply_selected  # noqa: E402
from dsorch.repro import write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--seeds", type=int, nargs="*", help="override protocol seeds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_selected(cfg, ROOT)
    if args.seeds:
        cfg["protocol"]["seeds"] = args.seeds
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    kind = cfg["experiment"]
    print(f"[{kind}] -> {out.relative_to(ROOT)}  (jobs={args.jobs})", flush=True)
    t0 = time.time()
    EXPERIMENTS[kind](cfg, out, args.jobs)
    elapsed = time.time() - t0
    write_manifest(out, cfg, {"experiment": kind, "wall_seconds": round(elapsed, 1)})
    print(f"[{kind}] done in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()

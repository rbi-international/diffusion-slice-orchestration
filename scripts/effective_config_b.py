"""Write the effective configuration of the committed study-B runs.

The manifests of results/validation_b and results/study_b were written by
run_experiment.py before the study-B settings were applied inside the
experiment, so they show the base configuration (request-observed
information, 12-level QuantileMLP grid). The runs used study_b_config(cfg):
proactive information and the 15-level grid. This script derives that
configuration from the same config files, without rerunning anything, and
writes it next to each manifest as effective_config.json. The manifests are
left unchanged as the original record.

Usage:
    python scripts/effective_config_b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsorch.config import load_config  # noqa: E402
from dsorch.experiments_b import study_b_config  # noqa: E402

RUNS = {"results/validation_b": "configs/validation_b.yaml", "results/study_b": "configs/test_b.yaml"}


def main() -> None:
    for out, conf in RUNS.items():
        cfg = study_b_config(load_config(ROOT / conf))
        rec = {"note": "Effective configuration of this run, derived with dsorch.experiments_b.study_b_config "
                       f"from {conf}. manifest.json in this folder records the configuration before these "
                       "study-B settings were applied.",
               "information": cfg["protocol"]["information"],
               "quantile_mlp_levels": cfg["forecasters"]["quantile_mlp"]["levels"],
               "config": cfg}
        path = ROOT / out / "effective_config.json"
        path.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
        print("wrote", path.relative_to(ROOT), rec["information"], len(rec["quantile_mlp_levels"]), "levels")


if __name__ == "__main__":
    main()

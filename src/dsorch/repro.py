"""Seeding and provenance utilities."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def derive_seed(*parts: int) -> int:
    """Deterministic 32-bit seed from a tuple of non-negative integers.

    Used so that every random draw depends only on its logical position
    (run seed, slot, slice, method), never on call order.
    """
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def set_global_determinism(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def git_info(root: Path | None = None) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
    except Exception:  # not a git checkout
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}


def environment_info() -> Dict[str, Any]:
    import matplotlib
    import pandas
    import scipy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }


def write_manifest(out_dir: Path, config: Dict[str, Any], extra: Dict[str, Any] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git": git_info(),
        "environment": environment_info(),
        "config": config,
    }
    if extra:
        manifest.update(extra)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

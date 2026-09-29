"""Configuration loading with single-level inheritance.

An experiment config may contain ``extends: base.yaml``; the referenced file is
loaded first and the experiment values are deep-merged on top of it.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict in which ``override`` is recursively merged into ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = deep_merge(base, cfg)
    return cfg


def default_config() -> Dict[str, Any]:
    """Load configs/base.yaml relative to the repository root."""
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "configs" / "base.yaml")

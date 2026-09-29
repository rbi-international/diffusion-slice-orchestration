"""Demand forecasters. ``build`` returns an untrained instance by name."""
from __future__ import annotations

from typing import Dict

from .baselines import MovingAverage, Persistence, QuantileMLP
from .common import Forecaster, lwma
from .ddpm import ConditionalDDPM, make_schedule
from .gan import ConditionalGAN


def build(name: str, cfg: Dict, overrides: Dict | None = None) -> Forecaster:
    k = int(cfg["control"]["history"])
    S = len(cfg["slices"])
    fc = cfg["forecasters"]
    if name == "GAN":
        return ConditionalGAN(fc["gan"], S, k)
    if name == "DDPM":
        return ConditionalDDPM({**fc["ddpm"], **(overrides or {})}, S, k)
    if name == "QuantileMLP":
        return QuantileMLP(fc["quantile_mlp"], S, k)
    if name == "Persistence":
        return Persistence({}, S, k)
    if name == "MovingAverage":
        return MovingAverage({}, S, k)
    raise ValueError(f"unknown forecaster {name!r}")


__all__ = [
    "build",
    "Forecaster",
    "ConditionalGAN",
    "ConditionalDDPM",
    "QuantileMLP",
    "Persistence",
    "MovingAverage",
    "lwma",
    "make_schedule",
]

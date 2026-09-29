"""Conditional denoising diffusion probabilistic model (DDPM) demand sampler.

The denoiser epsilon_theta(x_k, k, c) is a two-hidden-layer MLP of width 48,
matching the GAN generator in depth, width and conditioning (3,674 parameters
with the default 8-d step embedding). Targets are the normalized next-slot
bandwidth-CPU vector mapped to [-1, 1].

Noise schedules
---------------
cosine         Nichol and Dhariwal (2021). With K = 20 the terminal signal
               fraction alpha_bar_K is effectively zero, so sampling from
               N(0, I) matches the end of the forward process.
linear         Linear beta from 1e-4 to ``linear_beta_end`` (0.35 by default).
linear_legacy  Linear beta from 1e-4 to 0.02, the setting of Ho et al. (2020)
               for K = 1000. With K = 20 it leaves alpha_bar_K = 0.82, so the
               forward process never reaches noise. Kept only as an ablation
               of the earlier implementation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .common import Forecaster, count_params, item_noise


def make_schedule(kind: str, steps: int, beta_start: float = 1e-4, beta_end: float = 0.35) -> np.ndarray:
    if kind == "cosine":
        s = 0.008
        t = np.arange(steps + 1, dtype=np.float64) / steps
        f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 1e-8, 0.999)
    if kind == "linear":
        return np.linspace(beta_start, beta_end, steps, dtype=np.float64)
    if kind == "linear_legacy":
        return np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    raise ValueError(f"unknown schedule {kind!r}")


class StepEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion step followed by a linear layer and SiLU."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        ang = k.float()[:, None] * freqs[None, :]
        return self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class Denoiser(nn.Module):
    def __init__(self, cond_dim: int, time_dim: int, hidden: int) -> None:
        super().__init__()
        self.emb = StepEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, k: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.emb(k), cond], dim=-1))


class ConditionalDDPM(Forecaster):
    name = "DDPM"
    model_id = 202

    def __init__(self, cfg, num_slices, history) -> None:
        super().__init__(cfg, num_slices, history)
        self.K = int(cfg["steps"])
        betas = make_schedule(
            cfg["schedule"], self.K, cfg.get("linear_beta_start", 1e-4), cfg.get("linear_beta_end", 0.35)
        )
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar_prev = np.concatenate([[1.0], alpha_bar[:-1]])
        if cfg.get("variance", "posterior") == "posterior":
            var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        else:
            var = betas.copy()
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alpha_bar = torch.tensor(alpha_bar, dtype=torch.float32)
        self.sigma = torch.tensor(np.sqrt(var), dtype=torch.float32)

    @property
    def terminal_alpha_bar(self) -> float:
        return float(self.alpha_bar[-1])

    def _fit(self, conds: np.ndarray, targets: np.ndarray, seed: int) -> None:
        c = self.cfg
        self.net = Denoiser(conds.shape[1], int(c["time_dim"]), int(c["hidden"]))
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"])
        X = torch.from_numpy(conds)
        Y0 = torch.from_numpy(targets) * 2.0 - 1.0
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        bs = int(c["batch_size"])
        for epoch in range(int(c["epochs"])):
            perm = rng.permutation(n)
            tot, nb = 0.0, 0
            for start in range(0, n, bs):
                idx = torch.from_numpy(perm[start : start + bs])
                cb, x0 = X[idx], Y0[idx]
                m = cb.shape[0]
                k = torch.randint(0, self.K, (m,))
                eps = torch.randn(m, 2)
                ab = self.alpha_bar[k][:, None]
                xk = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
                loss = ((self.net(xk, k, cb) - eps) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            self.loss_history.append({"epoch": epoch, "loss": tot / nb})
        self.net.eval()

    def _sample_norm(self, conds: np.ndarray, n: int, item_seeds: list[int]) -> np.ndarray:
        B = conds.shape[0]
        noise = item_noise(item_seeds, (self.K, n, 2))                   # (B, K, n, 2)
        x = noise[:, 0].reshape(B * n, 2)                                # x_K ~ N(0, I)
        cond = torch.from_numpy(conds)[:, None, :].expand(B, n, conds.shape[1]).reshape(B * n, -1)
        for k in range(self.K - 1, -1, -1):
            kk = torch.full((B * n,), k, dtype=torch.long)
            eps = self.net(x, kk, cond)
            mean = (x - self.betas[k] / (1.0 - self.alpha_bar[k]).sqrt() * eps) / self.alphas[k].sqrt()
            if k > 0:
                z = noise[:, self.K - k].reshape(B * n, 2)
                x = mean + self.sigma[k] * z
            else:
                x = mean
        return ((x + 1.0) / 2.0).reshape(B, n, 2).numpy()

    def num_parameters(self) -> dict:
        return {"online": count_params(self.net), "training_only": 0}

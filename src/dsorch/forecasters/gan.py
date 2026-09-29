"""Conditional GAN demand sampler, configured as in Table 5 of the GAN-JCSO paper.

Generator: 8-d noise + 13-d condition -> 48 -> 48 -> 2 (3,506 parameters).
Discriminator: 13-d condition + 2-d sample -> 48 -> 48 -> 1 (3,169 parameters).
Loss: conditional binary cross-entropy (Eq. 15) and the generator objective
of Eq. (16) with the weak L1 conditional constraint lambda_c = 0.25.
The generator output head is linear in the normalized space, identical to the
diffusion model, so neither generator is clipped to the fitting-segment range.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .common import Forecaster, count_params, item_noise


class Generator(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, z], dim=-1))


class Discriminator(nn.Module):
    def __init__(self, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, x], dim=-1))   # logit


class ConditionalGAN(Forecaster):
    name = "GAN"
    model_id = 101

    def _fit(self, conds: np.ndarray, targets: np.ndarray, seed: int) -> None:
        c = self.cfg
        cond_dim = conds.shape[1]
        self.latent_dim = int(c["latent_dim"])
        self.G = Generator(cond_dim, self.latent_dim, int(c["hidden"]))
        self.D = Discriminator(cond_dim, int(c["hidden"]))
        opt_g = torch.optim.Adam(self.G.parameters(), lr=c["lr"], betas=tuple(c["betas"]))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=c["lr"], betas=tuple(c["betas"]))
        bce = nn.BCEWithLogitsLoss()
        l1 = nn.L1Loss()
        X = torch.from_numpy(conds)
        Y = torch.from_numpy(targets)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        bs = int(c["batch_size"])
        for epoch in range(int(c["epochs"])):
            perm = rng.permutation(n)
            d_sum = g_sum = 0.0
            nb = 0
            for start in range(0, n, bs):
                idx = torch.from_numpy(perm[start : start + bs])
                cb, real = X[idx], Y[idx]
                m = cb.shape[0]
                ones, zeros = torch.ones(m, 1), torch.zeros(m, 1)
                fake = self.G(cb, torch.randn(m, self.latent_dim)).detach()
                d_loss = bce(self.D(cb, real), ones) + bce(self.D(cb, fake), zeros)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                fake = self.G(cb, torch.randn(m, self.latent_dim))
                g_loss = bce(self.D(cb, fake), ones) + c["lambda_c"] * l1(fake, real)
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                d_sum += d_loss.item()
                g_sum += g_loss.item()
                nb += 1
            self.loss_history.append({"epoch": epoch, "d_loss": d_sum / nb, "g_loss": g_sum / nb})
        self.G.eval()

    def _sample_norm(self, conds: np.ndarray, n: int, item_seeds: list[int]) -> np.ndarray:
        B = conds.shape[0]
        z = item_noise(item_seeds, (n, self.latent_dim))                 # (B, n, d)
        cond = torch.from_numpy(conds)[:, None, :].expand(B, n, conds.shape[1])
        out = self.G(cond.reshape(B * n, -1), z.reshape(B * n, -1))
        return out.reshape(B, n, 2).numpy()

    def num_parameters(self) -> dict:
        return {"online": count_params(self.G), "training_only": count_params(self.D)}

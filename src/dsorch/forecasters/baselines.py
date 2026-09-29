"""Non-generative demand estimators used as controls.

Persistence       next-slot demand equals the most recent observation.
MovingAverage     elementwise mean of the five most recent observations.
QuantileMLP       direct distributional regression: an MLP with the same
                  condition and hidden sizes as the generators, trained with
                  the pinball loss at a fixed grid of quantile levels. It tests
                  whether a generative sampler is needed at all.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .common import Forecaster, count_params


class Persistence(Forecaster):
    name = "Persistence"
    distributional = False
    model_id = 11

    def _fit(self, conds, targets, seed) -> None:
        pass

    def point(self, traces: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        return np.stack([traces[t - 1] for t in t_idx])               # (len, S, 2)


class MovingAverage(Forecaster):
    name = "MovingAverage"
    distributional = False
    model_id = 12

    def _fit(self, conds, targets, seed) -> None:
        pass

    def point(self, traces: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        return np.stack([traces[t - self.k : t].mean(axis=0) for t in t_idx])


class QuantileMLP(Forecaster):
    name = "QuantileMLP"
    model_id = 303

    def _fit(self, conds, targets, seed) -> None:
        c = self.cfg
        self.levels = torch.tensor(c["levels"], dtype=torch.float32)
        L = len(c["levels"])
        h = int(c["hidden"])
        self.net = nn.Sequential(
            nn.Linear(conds.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * L)
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"])
        X, Y = torch.from_numpy(conds), torch.from_numpy(targets)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        bs = int(c["batch_size"])
        for epoch in range(int(c["epochs"])):
            perm = rng.permutation(n)
            tot, nb = 0.0, 0
            for start in range(0, n, bs):
                idx = torch.from_numpy(perm[start : start + bs])
                pred = self.net(X[idx]).reshape(-1, 2, L)
                err = Y[idx][:, :, None] - pred
                loss = torch.maximum(self.levels * err, (self.levels - 1.0) * err).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            self.loss_history.append({"epoch": epoch, "loss": tot / nb})
        self.net.eval()

    def quantiles(self, traces: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        """Monotone (sorted) quantiles in original units, shape (len, S, L, 2)."""
        conds, _ = self.conditions_for(traces, t_idx)
        with torch.no_grad():
            q = self.net(torch.from_numpy(conds)).reshape(-1, 2, len(self.levels)).numpy()
        q = np.sort(q, axis=2).transpose(0, 2, 1)                        # (B, L, 2)
        q = self.scaler.inverse(q)
        return np.maximum(q, 0.0).reshape(len(t_idx), self.num_slices, len(self.levels), 2)

    def quantile_at(self, traces: np.ndarray, t_idx: np.ndarray, level: float) -> np.ndarray:
        q = self.quantiles(traces, t_idx)
        lv = self.levels.numpy()
        out = np.empty(q.shape[:2] + (2,))
        for d in range(2):
            flat = q[..., d].reshape(-1, len(lv))
            out[..., d] = np.array([np.interp(level, lv, row) for row in flat]).reshape(q.shape[:2])
        return out

    def _sample_norm(self, conds, n, item_seeds):
        raise NotImplementedError("QuantileMLP is evaluated through its quantiles, not samples")

    def num_parameters(self) -> dict:
        return {"online": count_params(self.net), "training_only": 0}

import random

import torch
import torch.nn as nn


class FreqMask(nn.Module):
    """Zero out up to `max_width` consecutive frequency bins."""

    def __init__(self, max_width: int = 30):
        super().__init__()
        self.max_width = max_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) — H is freq, W is time
        F = random.randint(1, self.max_width)
        f0 = random.randint(0, x.shape[2] - F)
        x = x.clone()
        x[:, :, f0:f0 + F, :] = 0.0
        return x


class TimeMask(nn.Module):
    """Zero out up to `max_width` consecutive time steps."""

    def __init__(self, max_width: int = 30):
        super().__init__()
        self.max_width = max_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) — H is freq, W is time
        T = random.randint(1, self.max_width)
        t0 = random.randint(0, x.shape[3] - T)
        x = x.clone()
        x[:, :, :, t0:t0 + T] = 0.0
        return x


class GaussianNoise(nn.Module):
    """Add zero-mean Gaussian noise with standard deviation `std`."""

    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std

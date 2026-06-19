"""CNN encoder/decoder and reward head for the RSSM (PlaNet architecture).

Dimensions follow the PlaNet paper appendix: encoder is 4 stride-2 convs with
channels (d, 2d, 4d, 8d) and kernel 4 (d=32 -> 1024-d embedding for 64x64);
decoder mirrors it with transposed convs (kernels 5,5,6,6); reward head is a
small ReLU MLP. All shapes are commented.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, image_shape: tuple[int, int, int], depth: int = 32) -> None:
        super().__init__()
        c, h, w = image_shape  # (C, H, W)
        self.net = nn.Sequential(
            nn.Conv2d(c, depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth, 2 * depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = self.net(torch.zeros(1, c, h, w))  # (1, 8d, h', w')
        self.embed_dim = int(dummy.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, embed_dim)
        return self.net(x).flatten(1)


class ConvDecoder(nn.Module):
    def __init__(
        self, feat_dim: int, image_shape: tuple[int, int, int], depth: int = 32
    ) -> None:
        super().__init__()
        c, _, _ = image_shape
        self._depth = depth
        self.fc = nn.Linear(feat_dim, 32 * depth)  # -> 1024 when depth=32
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * depth, depth, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, c, 6, stride=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, feat_dim) -> (N, C, 64, 64)
        x = self.fc(feat).view(-1, 32 * self._depth, 1, 1)
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, units: int, layers: int, out_dim: int) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, units), nn.ReLU()]
            d = units
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

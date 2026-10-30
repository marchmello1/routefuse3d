from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.01, inplace=True)
        x = self.norm2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.01, inplace=True)


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        blocks = []
        for deep, skip in zip(reversed(channels[1:]), reversed(channels[:-1])):
            blocks.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose3d(deep, skip, 2, stride=2),
                        "fuse": ResidualBlock(skip * 2, skip),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block["up"](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block["fuse"](torch.cat((x, skip), dim=1))
        return x


@dataclass
class ModelOutput:
    shared: torch.Tensor
    isolated: torch.Tensor
    contiguous: torch.Tensor
    morphology: torch.Tensor
    density: torch.Tensor | None = None
    strata: torch.Tensor | None = None


class RoutedResidualUNet(nn.Module):
    """Compact A10-compatible network with metric-specific output heads."""

    def __init__(
        self,
        base_channels: int = 32,
        projection_dim: int = 256,
        with_density_head: bool = False,
    ):
        super().__init__()
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            min(base_channels * 16, 320),
        )
        self.channels = channels
        self.stem = ResidualBlock(1, channels[0])
        self.down = nn.ModuleList(
            [ResidualBlock(channels[i], channels[i + 1], stride=2) for i in range(len(channels) - 1)]
        )
        self.decoder = Decoder(channels)
        self.shared_head = nn.Conv3d(channels[0], 1, 1)
        self.isolated_head = nn.Conv3d(channels[0], 1, 1)
        self.contiguous_head = nn.Conv3d(channels[0], 1, 1)
        # Channel 0 reconstructs appearance; channels 1-3 complete an
        # annotation-free distance field and coarse tube/blob descriptors.
        self.reconstruction_head = nn.Conv3d(channels[0], 4, 1)
        self.morphology_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels[-1] // 2, 1),
        )
        self.density_head = (
            nn.Sequential(
                nn.Linear(channels[-1], channels[-1] // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(channels[-1] // 2, 1),
            )
            if with_density_head
            else None
        )
        # Channels: isolated-sparse, isolated-dense,
        # contiguous-sparse, contiguous-dense.
        self.strata_head = (
            nn.Conv3d(channels[0], 4, 1) if with_density_head else None
        )
        self.projector = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(),
            nn.Linear(channels[-1], projection_dim),
        )
        self.latent_predictor = nn.Conv3d(channels[-1], 128, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        x = self.stem(x)
        skips.append(x)
        for layer in self.down:
            x = layer(x)
            if layer is not self.down[-1]:
                skips.append(x)
        return x, skips

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(x, 1).flatten(1)

    def latent_map(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.latent_predictor(bottleneck), dim=1)

    def forward_ssl_target(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encode(x)
        return self.latent_map(bottleneck)

    def forward_ssl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bottleneck, skips = self.encode(x)
        decoded = self.decoder(bottleneck, skips)
        reconstruction = self.reconstruction_head(decoded)
        embedding = F.normalize(self.projector(self.pool(bottleneck)), dim=1)
        return reconstruction, embedding, self.latent_map(bottleneck)

    def forward(
        self, x: torch.Tensor, ssl: bool = False, ssl_target: bool = False
    ) -> ModelOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if ssl_target:
            return self.forward_ssl_target(x)
        if ssl:
            return self.forward_ssl(x)
        bottleneck, skips = self.encode(x)
        decoded = self.decoder(bottleneck, skips)
        pooled = self.pool(bottleneck)
        return ModelOutput(
            shared=self.shared_head(decoded),
            isolated=self.isolated_head(decoded),
            contiguous=self.contiguous_head(decoded),
            morphology=self.morphology_head(pooled).squeeze(1),
            density=(
                self.density_head(pooled).squeeze(1)
                if self.density_head is not None
                else None
            ),
            strata=(
                self.strata_head(decoded) if self.strata_head is not None else None
            ),
        )


def strip_distributed_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state.items()}

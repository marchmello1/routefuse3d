"""Masked structural-completion objective, independent of nnSSL internals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from msc_targets import structural_targets


class MaskedStructuralCompletionLoss(nn.Module):
    def __init__(self, workers: int = 6):
        super().__init__()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def forward(
        self,
        prediction: torch.Tensor,
        groundtruth: torch.Tensor,
        mask: torch.Tensor,
        precomputed_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.ndim != 5 or prediction.shape[1] != 4:
            raise ValueError(f"Expected four predicted channels, got {prediction.shape}")
        if groundtruth.shape[1] != 1 or groundtruth.shape[0] != prediction.shape[0]:
            raise ValueError("Expected a single-channel input with matching batch size")

        if precomputed_targets is None:
            source = groundtruth.detach().float().cpu().numpy()[:, 0]
            targets = np.stack(list(self.pool.map(structural_targets, source)), axis=0)
            target = torch.from_numpy(targets).to(prediction.device, non_blocking=True)
        else:
            target = precomputed_targets.to(prediction.device, non_blocking=True)
        if target.shape != prediction.shape:
            raise ValueError(f"Target/prediction mismatch: {target.shape}, {prediction.shape}")

        # nnSSL's mask is 1 for visible blocks and 0 for hidden blocks.
        hidden = F.interpolate(mask.float(), size=prediction.shape[2:], mode="nearest") < 0.5
        hidden = hidden.expand_as(prediction)
        error = (prediction.float() - target).square()
        channel_weights = (0.25, 1.0, 1.0, 1.0)
        return sum(
            channel_weights[index] * error[:, index][hidden[:, index]].mean()
            for index in range(4)
        )

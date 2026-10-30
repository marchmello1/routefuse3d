"""Budgeted nnU-Net fine-tuning initialized from an MSC SparK encoder.

Set MSC_PRETRAINED_CHECKPOINT to the standalone train_msc_spark.py checkpoint.
Only encoder tensors are transferred. Decoder and segmentation head remain
freshly initialized. The standard nnU-Net preload path requires a full
segmentation network and therefore cannot load an encoder-only SSL artifact.
"""

from __future__ import annotations

import os

import torch
from torch.nn.parallel import DistributedDataParallel

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerSelmaFast import nnUNetTrainerSelmaFast


def load_msc_encoder(network: torch.nn.Module, checkpoint_path: str) -> tuple[int, int]:
    module = network.module if isinstance(network, DistributedDataParallel) else network
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["network_weights"]
    target = module.state_dict()
    encoder_keys = [key for key in target if key.startswith("encoder.")]
    if not encoder_keys:
        raise ValueError("Downstream model has no encoder tensors")
    incompatible = [
        key for key in encoder_keys
        if key not in source or source[key].shape != target[key].shape
    ]
    if incompatible:
        raise ValueError(f"MSC encoder incompatible at {len(incompatible)} tensors: {incompatible[:8]}")
    for key in encoder_keys:
        target[key] = source[key]
    module.load_state_dict(target, strict=True)
    return len(encoder_keys), sum(target[key].numel() for key in encoder_keys)


class nnUNetTrainerSelmaMSC(nnUNetTrainerSelmaFast):
    def initialize(self):
        super().initialize()
        if getattr(self, "_msc_loaded", False):
            return
        checkpoint_path = os.environ.get("MSC_PRETRAINED_CHECKPOINT")
        if checkpoint_path:
            tensors, parameters = load_msc_encoder(self.network, checkpoint_path)
            self.print_to_log_file(
                f"MSC encoder initialized: {tensors} tensors, {parameters} parameters from {checkpoint_path}"
            )
        self._msc_loaded = True

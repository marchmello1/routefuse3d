"""MSC experiment using nnSSL's sparse ResEnc-L pretraining machinery.

Install this module inside nnssl.training.nnsslTrainer before invoking
``nnssl_train``. The upstream nnSSL package is licensed CC-BY-SA-4.0 and must
retain its own license and attribution when distributed.
"""

from __future__ import annotations

import torch

from msc_loss import MaskedStructuralCompletionLoss
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer


class MSCResEncLTask1Trainer(SparkMAETrainer):
    """62,500 update Task-1 paper schedule: 250 epochs x 250 iterations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_output_channels = 4
        self.total_batch_size = 6
        self.num_epochs = 250
        self.num_iterations_per_epoch = 250
        self.mask_percentage = 0.75

    @staticmethod
    def mask_creation(batch_size, patch_size, mask_percentage, rng_seed=None, block_size=16):
        ratio = float(torch.empty(()).uniform_(0.6, 0.9))
        return BaseMAETrainer.mask_creation(
            batch_size, patch_size, ratio, rng_seed=rng_seed, block_size=block_size
        )

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ):
        return super().build_architecture_and_adaptation_plan(
            config_plan, num_input_channels, 4
        )

    def build_loss(self):
        return MaskedStructuralCompletionLoss()

    def train_step(self, batch: dict) -> dict:
        # nnSSL's ZScoreNormalization supplies a normalized input, but does not
        # enforce the paper's intensity range. Keep its batch-owned tensor
        # untouched because the data loader may reuse it.
        bounded = dict(batch)
        bounded["data"] = batch["data"].clamp(-5.0, 20.0)
        return super().train_step(bounded)


class MSCResEncLTask2Trainer(MSCResEncLTask1Trainer):
    """125,000 update Task-2 paper schedule, using the same four-channel loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 500

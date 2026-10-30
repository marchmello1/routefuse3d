"""Budgeted ResEnc-L fold gate for the SELMA3D labeled set.

This is a supervised-only nnU-Net trainer. It keeps nnU-Net's architecture,
augmentation, loss, optimizer and validation code, and shortens the schedule
to produce a measured fold gate before the challenge deadline.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSelmaFast(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 100
        self.num_iterations_per_epoch = 50
        self.num_val_iterations_per_epoch = 10
        self.save_every = 5

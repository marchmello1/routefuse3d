"""Longer ResEnc-L schedule sized to finish before the preliminary deadline.

Same as nnUNetTrainerSelmaFast except 200 epochs x 250 iterations (40x the
iterations), so an isolated specialist is not judged on an undertrained gate.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSelmaLong(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 200
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 20
        self.save_every = 5

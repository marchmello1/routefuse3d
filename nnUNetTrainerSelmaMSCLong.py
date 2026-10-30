"""MSC-encoder-initialized ResEnc-L with the SelmaLong schedule (200 x 250).

Paired with nnUNetTrainerSelmaLong on identical splits, so the difference is
the SSL initialization alone. Set MSC_PRETRAINED_CHECKPOINT as for
nnUNetTrainerSelmaMSC.
"""

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerSelmaMSC import nnUNetTrainerSelmaMSC


class nnUNetTrainerSelmaMSCLong(nnUNetTrainerSelmaMSC):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 200
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 20
        self.save_every = 5

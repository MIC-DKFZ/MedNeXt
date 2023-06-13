import torch
import torch.nn as nn
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper
import os

class nnUNetTrainerV2_CheckPool(nnUNetTrainerV2):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.save_every = 10
    
    def save_checkpoint(self, fname, save_optimizer=True):
        super(nnUNetTrainerV2_CheckPool, self).save_checkpoint(fname, save_optimizer)
        if ((self.epoch+1) % self.save_every) == 0:        
            state_dict = self.network.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            n_ckpt = (self.epoch)
            save_this = {
                'epoch': self.epoch,
                'state_dict': state_dict,
            }
            os.makedirs(os.path.join(self.output_folder, 'intermediate_checkpoints'), exist_ok=True)
            fname = os.path.join(self.output_folder, 'intermediate_checkpoints', f"intermediate_{n_ckpt}.model.pkl")
            torch.save(save_this, fname)


class nnUNetTrainerV2_CheckPool_AdamW(nnUNetTrainerV2_CheckPool):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None
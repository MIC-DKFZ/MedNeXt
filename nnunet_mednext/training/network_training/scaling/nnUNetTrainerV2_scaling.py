import torch
import os
import torch.nn as nn
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt \
                                                import nnUNetTrainerV2_Optim_and_LR

class nnUNetTrainerV2_MedNeXt_GRN_L_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            # exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [6,6,6,6,4,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block',
            grn=True
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_GRN_L_kernel3_500batches(nnUNetTrainerV2_MedNeXt_GRN_L_kernel3):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_batches_per_epoch = 500


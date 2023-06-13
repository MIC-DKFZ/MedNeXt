import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.custom_modules.custom_networks.TransBTS.TransBTS_downsample8x_skipconnection \
    import BTS as BTS_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class BTS(BTS_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['num_classes']
        self.do_ds = False       # Already added this in the main class


class nnUNetTrainerV2_TransBTS(nnUNetTrainerV2_noDeepSupervision):   
    
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4
        
    def initialize_network(self):
        
        _conv_repr=True
        _pe_type="learned"
        patch_dim = 8
        aux_layers = [1, 2, 3, 4]
        
        self.network = BTS(
            img_dim=128,
            patch_dim=patch_dim,
            num_channels=self.num_input_channels,
            num_classes=self.num_classes,
            embedding_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=4096,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            conv_patch_representation=_conv_repr,
            positional_encoding_type=_pe_type,
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
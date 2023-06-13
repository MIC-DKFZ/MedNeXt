import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.custom_modules.custom_networks.UXNet3D. \
    network_backbone import UXNET as UXNET_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision \
    import nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class UXNET(UXNET_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['out_chans']
        self.do_ds = False        


class nnUNetTrainerV2_3DUXNet(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4
    
    def initialize_network(self):
        
        self.network = UXNET(
            in_chans=self.num_input_channels,
            out_chans=self.num_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None

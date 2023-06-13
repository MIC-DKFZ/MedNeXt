import torch
import torch.nn as nn
from monai.networks.nets.unetr import UNETR as UNETR_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class UNETR(UNETR_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['out_channels']
        self.do_ds = False


class nnUNetTrainerV2_UNETR(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4
        # self.oversample_foreground_percent = 1.0
    
    def initialize_network(self):
        self.network = UNETR(
                in_channels=self.num_input_channels, 
                out_channels=self.num_classes, 
                img_size=(128, 128, 128),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed='perceptron',
                norm_name='instance',
                conv_block=True,
                res_block=True,
                dropout_rate=0.0
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
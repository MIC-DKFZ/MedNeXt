import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.custom_modules.custom_networks.nnFormer.nnFormer_tumor import nnFormer as nnFormer_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


# Wrapper using SegmentationNetwork object
class nnFormer(nnFormer_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['num_classes']
        self.do_ds = False


class nnUNetTrainerV2_nnFormerTumor(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.initial_lr = 5e-5

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = nnFormer(
            crop_size=[128, 128, 128],
            embedding_dim=96,
            input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            conv_op=nn.Conv3d,
            depths=[2,2,2,2],
            num_heads=[3,6,12,24],
            patch_size=[4,4,4],
            window_size=[4,4,8,4],
            deep_supervision=False,
            dummy=False
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None


class nnUNetTrainerV2_nnFormerTumor_Dummy(nnUNetTrainerV2_nnFormerTumor):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = nnFormer(
            crop_size=[128, 128, 128],
            embedding_dim=96,
            input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            conv_op=nn.Conv3d,
            depths=[2,2,2,2],
            num_heads=[3,6,12,24],
            patch_size=[4,4,4],
            window_size=[4,4,8,4],
            deep_supervision=False,
            dummy=True
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
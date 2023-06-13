import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.custom_modules.custom_networks.SwinUNETR.swinunetr_2d import SwinUNETR2D as SwinUNETR2D_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


# Wrapper using SegmentationNetwork object
class SwinUNETR2D(SwinUNETR2D_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv2d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['out_channels']
        self.do_ds = False


class nnUNetTrainerV2_SwinUNETR2D_Optimizer_and_LR(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.initial_lr = 1e-3
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None


class nnUNetTrainerV2_SwinUNETR2D_Pretrained_ReuseEmb(nnUNetTrainerV2_SwinUNETR2D_Optimizer_and_LR):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = SwinUNETR2D(
                            img_size=384,
                            in_channels=self.num_input_channels,
                            out_channels=self.num_classes,
                            feature_size=128,
                            spatial_dims=2,
                            use_checkpoint=False,
                            dummy=False,
                            pretrained = True,
                            reuse_embedding = True,
                            swin_variant='swin_base_patch4_window12_384_in22k'
                        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


# class nnUNetTrainerV2_SwinUNETR2D_Pretrained_TrainEmb(nnUNetTrainerV2_SwinUNETR2D_Optimizer_and_LR):

#     def initialize_network(self):
#         """
#         changed deep supervision to False
#         :return:
#         """
#         self.network = SwinUNETR2D(
#                             img_size=448,
#                             in_channels=self.num_input_channels,
#                             out_channels=self.num_classes,
#                             feature_size=128,
#                             spatial_dims=2,
#                             use_checkpoint=False,
#                             dummy=False,
#                             pretrained = True,
#                             reuse_embedding = False
#                         )

#         if torch.cuda.is_available():
#             self.network.cuda()
#         self.network.inference_apply_nonlin = softmax_helper


class nnUNetTrainerV2_SwinUNETR2D_NoPretrained(nnUNetTrainerV2_SwinUNETR2D_Optimizer_and_LR):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = SwinUNETR2D(
                            img_size=384,
                            in_channels=self.num_input_channels,
                            out_channels=self.num_classes,
                            feature_size=128,
                            spatial_dims=2,
                            use_checkpoint=False,
                            dummy=False,
                            pretrained = False,
                            reuse_embedding = True,
                            swin_variant='swin_base_patch4_window12_384_in22k'
                        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

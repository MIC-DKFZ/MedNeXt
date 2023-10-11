import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.custom_modules.custom_networks.TransUnet import vit_seg_configs as configs
from nnunet_mednext.network_architecture.custom_modules.custom_networks.TransUnet.vit_seg_modeling import TransUNet as TransUNet_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


# Wrapper using SegmentationNetwork object
class TransUNet(TransUNet_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv2d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['num_classes']
        self.do_ds = False


class nnUNetTrainerV2_TransUNet(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.initial_lr = 1e-4
    
    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        config_vit = configs.get_r50_b16_config()
        config_vit.n_classes = self.num_classes
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(512/16), int(512/16))

        self.network = TransUNet(
                config_vit,
                img_size=512, 
                in_channels=3 if self.num_input_channels==1 else self.num_input_channels,   # IDK some original paper design decision 
                dummy=False, 
                num_classes=self.num_classes
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


class nnUNetTrainerV2_TransUNet_Dummy(nnUNetTrainerV2_TransUNet):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        config_vit = configs.get_r50_b16_config()
        config_vit.n_classes = self.num_classes 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(512/16), int(512/16))

        self.network = TransUNet(
                config_vit,
                img_size=512, 
                in_channels=3 if self.num_input_channels==1 else self.num_input_channels,   # IDK some original paper design decision 
                dummy=True, 
                num_classes=self.num_classes
            )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


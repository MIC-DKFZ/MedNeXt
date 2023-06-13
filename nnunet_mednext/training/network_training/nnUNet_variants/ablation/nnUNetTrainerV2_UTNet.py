import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.custom_modules.custom_networks.UTNet.utnet import UTNet as UTNet_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


# Wrapper using SegmentationNetwork object
class UTNet(UTNet_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv2d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['num_classes']
        self.do_ds = False


class nnUNetTrainerV2_UTNet(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.initial_lr = 5e-4
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = UTNet(
                        in_chan=self.num_input_channels, 
                        base_chan=32, 
                        num_classes=self.num_classes,
                        block_list='1234', 
                        num_blocks=[1,1,1,1], 
                        num_heads=[2,4,8,16],
                        dummy=False,
                    )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


class nnUNetTrainerV2_UTNet_Dummy(nnUNetTrainerV2_UTNet):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = UTNet(
                        in_chan=self.num_input_channels, 
                        base_chan=32, 
                        num_classes=self.num_classes,
                        block_list='1234', 
                        num_blocks=[1,1,1,1], 
                        num_heads=[2,4,8,16],
                        dummy=True,
                    )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

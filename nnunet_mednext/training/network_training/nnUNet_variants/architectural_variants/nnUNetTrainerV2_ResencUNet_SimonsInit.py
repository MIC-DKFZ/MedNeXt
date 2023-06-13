#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn

from nnunet_mednext.network_architecture.custom_modules.conv_blocks import BasicResidualBlock
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import \
    nnUNetTrainerV2_ResencUNet


def init_last_bn_before_add_to_0(module):
    if isinstance(module, BasicResidualBlock):
        module.norm2.weight = nn.init.constant_(module.norm2.weight, 0)
        module.norm2.bias = nn.init.constant_(module.norm2.bias, 0)


class nnUNetTrainerV2_ResencUNet_SimonsInit(nnUNetTrainerV2_ResencUNet):
    """
    SimonsInit = Simon Kohl's suggestion of initializing each residual block such that it adds nothing
    (weight and bias initialized to zero in last batch norm)
    """
    def initialize_network(self):
        ret = super().initialize_network()
        self.network.apply(init_last_bn_before_add_to_0)
        return ret


class nnUNetTrainerV2_ResencUNet_SimonsInit_adamw(nnUNetTrainerV2_ResencUNet_SimonsInit):
    
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None
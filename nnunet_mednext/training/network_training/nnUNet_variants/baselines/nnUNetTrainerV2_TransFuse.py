import torch
import torch.nn as nn
from nnunet_mednext.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from nnunet_mednext.network_architecture.custom_modules.custom_networks.TransFuse.TransFuse import TransFuse_S as TransFuse_S_Orig
from nnunet_mednext.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


# Wrapper using SegmentationNetwork object
class TransFuse_S(TransFuse_S_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv2d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['num_classes']
        self.do_ds = False


class nnUNetTrainerV2_TransFuse(nnUNetTrainerV2_noDeepSupervision):

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

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data) # single o/p in eval mode, 3 in train mode
                del data
                l = 0
                if self.network.training:
                    l_weights = [0.2, 0.3, 0.5]
                    for i in range(len(output)):
                        l+= l_weights[i]*self.loss(output[i], target)
                
                    output = output[-1]     # 3 outputs only triggered during training. only last needed after loss
                else:
                    l+= self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()


class nnUNetTrainerV2_TransFuse_S(nnUNetTrainerV2_TransFuse):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network =  TransFuse_S(
                            img_size=512, 
                            in_chans=self.num_input_channels, 
                            dummy=False, 
                            num_classes=self.num_classes
                        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


class nnUNetTrainerV2_TransFuse_S_Dummy(nnUNetTrainerV2_TransFuse):

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network =  TransFuse_S(
                            img_size=512, 
                            in_chans=self.num_input_channels, 
                            dummy=True, 
                            num_classes=self.num_classes
                        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

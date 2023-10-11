from typing import Tuple

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
# from lib.utils.tools.logger import Logger as Log
# from lib.models.tools.module_helper import ModuleHelper
from nnunet_mednext.network_architecture.custom_modules.custom_networks.UXNet3D.uxnet_encoder import uxnet_conv

# This used to be in ModuleHelper in the original repo
def BNReLU(num_features, bn_type=None, **kwargs):
    if bn_type == 'torchbn':
        return nn.Sequential(
            nn.BatchNorm3d(num_features, **kwargs),
            nn.ReLU()
        )

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        # Log.info
        print('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class UXNET(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        enc1 = self.encoder1(x_in)
        # print(enc1.size())
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        # print(enc2.size())
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        # print(enc3.size())
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        # print(enc4.size())
        # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        
        # feat = self.conv_proj(dec4)
        
        return self.out(out)


if __name__ == "__main__":

    import torch 

    network = UXNET(
            in_chans=1,
            out_chans=3,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )#.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1,1,96,96,96), requires_grad=False)#.cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())
    
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 1, 128, 128, 128)).cuda()
        print(network(x).shape)
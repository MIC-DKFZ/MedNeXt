# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from einops.layers.torch import Rearrange
from einops.einops import repeat
import timm
import math

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import


from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


class UnetrUpBlock_noUp(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            in_channels,
            kernel_size=upsample_kernel_size,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        print(out.shape, skip.shape)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SwinUNETR2D(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        dummy = False,
        pretrained = False,
        reuse_embedding=False,
        swin_variant='swin_base_patch4_window12_384_in22k'
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        # if feature_size % 12 != 0:
        #     raise ValueError("feature_size should be divisible by 12.")

        # self.normalize = normalize

        # self.swinViT = SwinTransformer(
        #     in_chans=in_channels,
        #     embed_dim=feature_size,
        #     window_size=window_size,
        #     patch_size=patch_size,
        #     depths=depths,
        #     num_heads=num_heads,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_rate=drop_rate,
        #     attn_drop_rate=attn_drop_rate,
        #     drop_path_rate=dropout_path_rate,
        #     norm_layer=nn.LayerNorm,
        #     use_checkpoint=use_checkpoint,
        #     spatial_dims=spatial_dims,
        #     downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        #     dummy=dummy
        # )
        if pretrained:
            print('Initializing with Pretrained Swin-Transformer')
        else:
            print('Initializing with Swin-Transformer without pretraining')
        
        self.swinViT = timm.create_model(
                            swin_variant, 
                            pretrained=pretrained, 
                            drop_rate=drop_rate, 
                            attn_drop_rate=attn_drop_rate, 
                            img_size=img_size[0]
                        )
        
        if reuse_embedding:
            print("Reusing embedding by adding new Transpose to compensate for extra 2x downsampling")
        else:
            print("Training new SwinUNETR-like embedding. Configuring Pretrained network to compensate.")

        self.reuse_embedding = reuse_embedding
        if not self.reuse_embedding:
            self.non_pretrained_embed_layer = PatchEmbed(
                                patch_size=patch_size,
                                in_chans=in_channels*3,
                                embed_dim=feature_size,
                                norm_layer=None,  # type: ignore
                                spatial_dims=spatial_dims,
                            )

        spatial_reduction_den = 32 if self.reuse_embedding else 16
        self.final_patch_merging = timm.models.swin_transformer.PatchMerging(
                                        input_resolution = [i//spatial_reduction_den for i in img_size],
                                        dim = 1024, 
                                        out_dim=1024, 
                                        norm_layer=nn.LayerNorm
                                    )
        self.swinViT.patch_embed.img_size = ensure_tuple_rep(img_size[0], 2)

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.upsample_final = nn.ConvTranspose2d(
                                feature_size, feature_size, 
                                kernel_size=2, stride=2, padding=0
                                )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)


    def forward(self, x_in):
        # hidden_states_out = self.swinViT(x_in, self.normalize)
        x_orig = x_in
        # print(x_in.shape)
        # b_dim = x_in.shape[0]
        x_in = repeat(x_in, "b c h w -> b (c repeat) h w", repeat=3)
        # print(x_in.shape)

        # hidden_states_out = self.swinViT(x_in)
        hidden_states_out = []

        # embedding_patch_size = 4
        if self.reuse_embedding:
            x = self.swinViT.patch_embed(x_in)       # 4x4 patches
            h = int(x.shape[1]**0.5)
            x_stage = rearrange(x, "b (h w) c -> b c h w", h=h)
        else:
            x_stage = self.non_pretrained_embed_layer(x_in)
            # print(x_stage.shape)
            x = rearrange(x_stage, "b c h w -> b (h w) c")
        # print(x.shape)
        # print(x_stage.shape)
        hidden_states_out.append(x_stage)
        # print('loop begins')
        for idx, l in enumerate(self.swinViT.layers):
            # print(x.shape)
            # l.input_resolution = [i*2 for i in l.input_resolution]
            # print(l.input_resolution, x.shape) 
            x = l(x) 
            # x = rearrange(x, "(b w) c l h -> b c l h w", w=w_dim)
            # print(x.shape)
            if idx==3:
                # print(x.shape)
                x = self.final_patch_merging(x)
    
            h = int(x.shape[1]**0.5)
            x_stage = rearrange(x, "b (h w) c -> b c h w", h=h)
            # print(x_stage.shape)
            hidden_states_out.append(x_stage)
            # print(x.shape)
            # print()
            # exit(0)

        enc0 = self.encoder1(x_orig)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        # print(dec4.shape, hidden_states_out[3].shape)
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        # print(dec3.shape, enc3.shape)
        dec2 = self.decoder4(dec3, enc3)
        # print(dec2.shape, enc2.shape)
        dec1 = self.decoder3(dec2, enc2)
        # print(dec1.shape, enc1.shape)
        dec0 = self.decoder2(dec1, enc1)
        # print(dec0.shape)
        if self.reuse_embedding:
            dec0 = self.upsample_final(dec0)
        # print(dec0.shape, enc0.shape)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


    # def project_and_reshape(self, x_seq, b): 

    #     _, seq_len, _ = x_seq.shape
    #     n = int(math.sqrt(seq_len))
    #     print(n, x_seq.shape)
    #     x_seq = rearrange(x_seq, '(b d) (h w) c -> b c h w d',b=b, h=n)
    #     print(x_seq.shape)
    #     return x_seq


if __name__ == "__main__":

    model = SwinUNETR2D(img_size=512, in_channels=1, out_channels=14,
                        spatial_dims=2, dummy=False, feature_size=128,
                        reuse_embedding=True, pretrained=True)
    x = torch.zeros((1,1,512,512))
    # print(model)
    with torch.no_grad():
        print(model(x).shape)
    
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(count_parameters(model))

    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import parameter_count_table

    # # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    # # x = torch.zeros((1,1,128,128,128)).cuda()
    # print(parameter_count_table(model,1))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CoTr.network_architecture import CNNBackbone
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from CoTr.network_architecture.DeTrans.DeformableTrans import DeformableTransformer
from CoTr.network_architecture.DeTrans.position_encoding import build_position_encoding

class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=(1, 1, 1),padding=(0, 0, 0),dilation=(1, 1, 1),bias=False,weight_std=False):
        super(Conv3dBlock,self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out

class U_ResTran3D(SegmentationNetwork):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, 
                weight_std=False, in_channels=1, dummy=False):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes
        self.dummy = dummy

        self.upsamplex2 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

        self.transposeconv_stage2 = nn.ConvTranspose3d(384, 384, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(384, 192, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(192, 64, kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.stage2_de = ResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(192, 192, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        self.ds2_cls_conv = nn.Conv3d(384, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(192, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds0_cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)

        self.cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = CNNBackbone.Backbone(depth=9, norm_cfg=norm_cfg, activation_cfg=activation_cfg, 
                                weight_std=weight_std, in_channels=in_channels)
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))

        self.position_embed = build_position_encoding(mode='v2', hidden_dim=384)
        if not self.dummy:
            self.encoder_Detrans = DeformableTransformer(d_model=384, dim_feedforward=1536, dropout=0.1, activation='gelu', num_feature_levels=2, nhead=6, num_encoder_layers=6, enc_n_points=4)
            total = sum([param.nelement() for param in self.encoder_Detrans.parameters()])
            print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())

        return x_fea, masks, x_posemb


    def forward(self, inputs):
        # # %%%%%%%%%%%%% CoTr
        x_convs = self.backbone(inputs)
        x_fea, masks, x_posemb = self.posi_mask(x_convs)
        # print(type(x_fea), type(masks), type(x_posemb))
        # print(x_fea.shape, masks.shape, x_posemb.shape)
        if not self.dummy:
            x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)
        else:
            src_flatten = []
            spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(x_fea, masks, x_posemb)):
                bs, c, d, h, w = src.shape
                spatial_shape = (d, h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                # mask = mask.flatten(1)
                # pos_embed = pos_embed.flatten(2).transpose(1, 2)
                # lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                # lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                # mask_flatten.append(mask)
            x_trans = torch.cat(src_flatten, 1)
        # print(list(x_trans))

        # # Single_scale
        # # x = self.transposeconv_stage2(x_trans.transpose(-1, -2).view(x_convs[-1].shape))
        # # skip2 = x_convs[-2]
        # Multi-scale   
        x = self.transposeconv_stage2(x_trans[:, x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]::].transpose(-1, -2).view(x_convs[-1].shape)) # x_trans length: 12*24*24+6*12*12=7776
        skip2 = x_trans[:, 0:x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]].transpose(-1, -2).view(x_convs[-2].shape)
        
        x = x + skip2
        x = self.stage2_de(x)
        ds2 = self.ds2_cls_conv(x)

        x = self.transposeconv_stage1(x)
        skip1 = x_convs[-3]
        x = x + skip1
        x = self.stage1_de(x)
        ds1 = self.ds1_cls_conv(x)

        x = self.transposeconv_stage0(x)
        skip0 = x_convs[-4]
        x = x + skip0
        x = self.stage0_de(x)
        ds0 = self.ds0_cls_conv(x)


        result = self.upsamplex2(x)
        result = self.cls_conv(result)

        return [result, ds0, ds1, ds2]


class ResTranUnet(nn.Module):
    """
    ResTran-3D Unet
    """
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, 
                num_classes=None, weight_std=False, deep_supervision=False,
                in_channels=1, dummy=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, img_size, 
                        num_classes, weight_std, in_channels, dummy) # U_ResTran3D

        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg=='BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg=='SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg=='GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg=='IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.U_ResTran3D(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]


if __name__ == "__main__":

    model = ResTranUnet(img_size=128, in_channels=1, num_classes=9, dummy=False).cuda()
    x = torch.zeros((1,1,128,128,128)).cuda()

    with torch.no_grad():
        print(model(x).shape)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1,1,128,128,128)).cuda()
    print(parameter_count_table(model,1))
    # flops = FlopCountAnalysis(model, input)
    # print(flops.total())
    # with torch.no_grad():
    #     print(model(x).shape)
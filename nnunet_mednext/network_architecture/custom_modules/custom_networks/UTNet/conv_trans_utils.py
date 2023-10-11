import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x): 
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out 

class Mlp(nn.Module):
    def __init__(self, in_ch, hid_ch=None, out_ch=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        hid_ch = hid_ch or in_ch

        self.fc1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hid_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu, 
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x): 
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out

class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True, dummy=False):
        super().__init__()
        self.dummy = dummy

        if not self.dummy:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):

        if not self.dummy:
            out = self.bn1(x)
            out, q_k_attn = self.attn(out)
            
            out = out + x
            residue = out
        else:
            residue = x.clone()
            out = x

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out

class BasicTransDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True, dummy=False):
        super().__init__()
        self.dummy = dummy
        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        if not self.dummy:
            self.attn = LinearAttentionDecoder(in_ch, out_ch, heads=heads, dim_head=out_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
            
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):

        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        #x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)
        print(residue.shape ,x1.shape, x2.shape)
        if not self.dummy:
            out, q_k_attn = self.attn(x2, x1)
        else:
            out = x2
        out = out + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out




########################################################################
# Transformer components

class LinearAttention(nn.Module):
    
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        
        # depthwise conv is slightly better than conv1x1
        #self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
       
        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim*3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):

        B, C, H, W = x.shape

        #B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias
            #rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            #q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

class LinearAttentionDecoder(nn.Module):
    
    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        
        # depthwise conv is slightly better than conv1x1
        #self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
       
        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, q, x):

        B, C, H, W = x.shape # low-res feature shape
        BH, CH, HH, WH = q.shape # high-res feature shape

        k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
        q = self.to_q(q) #BH, inner_dim, HH, WH

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=HH, w=WH)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(HH, WH)
            q_k_attn += relative_position_bias
            #rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, HH, WH, self.dim_head)
            #q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
       
        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

class RelativePositionEmbedding(nn.Module):
    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape

        self.key_rel_w = nn.Parameter(torch.randn((2*self.shape-1, dim))*0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2*self.shape-1, dim))*0.02)

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None] # h, h
        relative_coords += self.shape - 1 # shift to start from 0
        
        self.register_buffer('relative_position_index', relative_coords)



    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):
        
        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k) # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            # self_relative_position_index origin shape: w, w
            # after repeat: W, w
            relative_index= torch.repeat_interleave(self.relative_position_index, W//self.shape, dim=0) # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index) # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == 'w':
            rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')

        elif case == 'h':
            rel_logits = rearrange(rel_logits, 'b heads W w H h -> b heads (H W) (h w)')

        return rel_logits




class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
                torch.randn((2*h-1) * (2*w-1), num_heads)*0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, h, w
        coords_flatten = torch.flatten(coords, 1) # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1) # hw, hw
    
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h, self.w, self.h*self.w, -1) #h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H//self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W//self.w, dim=1) #HW, hw, nH
        
        relative_position_bias_expanded = relative_position_bias_expanded.view(H*W, self.h*self.w, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias_expanded


###########################################################################
# Unet Transformer building block

class down_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, maxpool=True, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True, dummy=False):

        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        attn_block = BasicTransBlock

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))
        
        assert num_block > 0
        for i in range(num_block):
            block_list.append(attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos, dummy=dummy))
        self.blocks = nn.Sequential(*block_list)

        
    def forward(self, x):
        
        out = self.blocks(x)


        return out

class up_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True, dummy=False):
        super().__init__()
 
        self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos, dummy=dummy)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock
        attn_block = BasicTransBlock
        
        block_list = []

        for i in range(num_block):
            block_list.append(attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos, dummy=dummy))

        block_list.append(block(2*out_ch, out_ch, stride=1))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # x1: low-res feature, x2: high-res feature
        out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2], dim=1)
        out = self.blocks(out)

        return out

class block_trans(nn.Module):
    def __init__(self, in_ch, num_block, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True, dummy=False):

        super().__init__()

        block_list = []

        attn_block = BasicTransBlock

        assert num_block > 0
        for i in range(num_block):
            block_list.append(attn_block(in_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos, dummy=dummy))
        self.blocks = nn.Sequential(*block_list)

        
    def forward(self, x):
        
        out = self.blocks(x)


        return out

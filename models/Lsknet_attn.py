from audioop import bias
from json import decoder
from os import path
from re import A
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)
# from ..builder import ROTATED_BACKBONES
# from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer
from torch.fft import fft2, ifft2
import torch.nn.functional as F
from einops import rearrange
# import einops


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.co
m/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim,use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3,4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(x.shape,self.weight.shape,self.bias.shape)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class Layer_norm_process(nn.Module):  #n, h, w, d, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(1,1,1,c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(1,1,1,c), requires_grad=True)
        self.eps = eps
    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature
        
def block_images_einops(x, patch_size):  #n, h, w, d, c
  """Image to patches."""
  batch, channels, height, width, depth = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  grid_depth = depth // patch_size[2]
  x = rearrange(
      x, "n  c (gh fh) (gw fw) (gd fd) -> n  c (gh gw gd) (fh fw fd)",
      gh=grid_height, gw=grid_width, gd=grid_depth, fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = rearrange(
      x, "n  c (gh gw gd) (fh fw fd) -> n c (gh fh) (gw fw) (gd fd)",
      gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
  return x

class Global_attn(nn.Module):  #input shape: n  c (gh gw gd) (fh fw fd)
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self,c ,n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.intermediate_layernorm = Layer_norm_process(self.c)
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)
    def forward(self, x):
        x = x.permute(0,2,3,1) #n (gh gw gd) (fh fw fd)  c
        v = self.intermediate_layernorm(x)
        v = v.permute(0, 3, 2, 1)  #n, c, (fh fw fd) (gh gw gd)
        v = self.Dense_0(v)  #apply fc on the last dimension (gh gw)
        v = v.permute(0, 1, 3, 2)  #n, c, (gh gw gd),(fh fw fd) 
        return v


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.grn = GRN(hidden_features)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim,img_size,patch_size = [12,12,12]):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        conv0 = 5
        conv_spatial = 7
        self.conv0 = nn.Conv3d(dim, dim, conv0, padding=conv0//2, groups=dim)
        # self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv_spatial = nn.Conv3d(dim, dim, conv_spatial, padding=conv_spatial//2, groups=dim)
        self.conv1 = nn.Conv3d(dim, dim//2, 1)
        self.conv2 = nn.Conv3d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv3d(2, 2, 7, padding=3)
        self.attn = Global_attn(2, self.patch_size[0]*self.patch_size[1]*self.patch_size[2])
        self.attn_conv = nn.Conv3d(2, 2, 1)
        self.conv = nn.Conv3d(dim//2, dim, 1)
        
    def forward(self, x):
        # print(x.shape)   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True) 
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg)
        _, _, h, w, d = sig.shape
        gh, gw, gd = h // self.patch_size[0], w // self.patch_size[1], d // self.patch_size[2]
        sig = block_images_einops(sig, patch_size=(gh, gw, gd))  #n (gh gw)(fh fw) c
        sig = self.attn(sig)
        sig = unblock_images_einops(sig, grid_size=(self.patch_size[0], self.patch_size[1],self.patch_size[2]), 
                                    patch_size=(gh, gw, gd))
        sig = self.attn_conv(sig)

        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn



class Attention(nn.Module):
    def __init__(self, d_model,img_size,patch_size):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model,img_size,patch_size)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, img_size,patch_size,mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm3d(dim)
            self.norm2 = nn.BatchNorm3d(dim)
        self.attn = Attention(dim,img_size,patch_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        # self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
        #                       padding=1)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size// 2)

        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm3d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = self.norm(x)        
        return x, H, W, D

class LSKNet_attn(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[48, 96, 192, 768],
                mlp_ratios=[4, 4, 4, 4], patch_size=[6,6,6],drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__()
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        img_size=img_size[0]
        for i in range(num_stages):
            # patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)),
            #                                 patch_size=3,
            #                                 stride=2,
            #                                 in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                                 embed_dim=embed_dims[i], norm_cfg=norm_cfg)
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)


            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                img_size = img_size // (2 ** (i + 1)),
                patch_size = patch_size,
                 mlp_ratio=mlp_ratios[i], 
                 drop=drop_rate, 
                 drop_path=dpr[cur + j],
                 norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)




    def init_weights(self):
        print('init cfg')
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv3d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet_attn, self).init_weights()
            
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        ens = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W, D = patch_embed(x)
            for blk in block:
                # print(x.shape)
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, D, -1).permute(0, 4, 1, 2,3).contiguous()
            ens.append(x)
            
        return ens

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict
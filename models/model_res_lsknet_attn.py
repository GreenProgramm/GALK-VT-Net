from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)
import math
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from .unetr_block import UnetrUpBlock

from .Lsknet_attn import LSKNet_attn

rearrange, _ = optional_import("einops", name="rearrange")


class Res_lsknet_attn(nn.Module):
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
        feat_size=[64, 128, 256, 512],
        patch_size = [6,6,6],
        depths=[3, 3, 6, 4],
        norm_name: Union[Tuple, str] = "instance",
        hidden_size: int = 768,
        spatial_dims: int = 3,
        res_block:bool = True,
    ) -> None:
        

        super().__init__()
            
        self.in_chans = in_channels
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.lsknet_attn = LSKNet_attn(
            embed_dims = self.feat_size,
            img_size=img_size, 
            patch_size = patch_size,
            mlp_ratios=[4, 4, 4, 4],
            depths=depths,
            in_chans=in_channels,
        )

        self.inconv = nn.Conv3d(self.in_chans, self.feat_size[0], 7, padding=3, groups=self.in_chans)

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            skip_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            skip_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            skip_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            skip_channels=self.feat_size[0],
            out_channels=self.out_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.out_channels, 
            out_channels=self.out_channels)
        
    
        
    # def load_params(self, model_dict):
    #         store_dict = self.backbone.state_dict()
    #         for key in model_dict.keys():
    #             if 'out' not in key:
    #                 store_dict[key] = model_dict[key]
    #         self.backbone.load_state_dict(store_dict)
    #         print('Use pretrained weights')


    def forward(self, x_in):
        # print(x_in.shape, task_id.shape)
        hidden_states_out = self.lsknet_attn(x_in)
        # for feat in hidden_states_out:
        #     print(feat.shape)
        enc1 = self.inconv(x_in)
        # enc2 = self.encoder2(hidden_states_out[0])
        # enc3 = self.encoder3(hidden_states_out[1])
        # enc4 = self.encoder4(hidden_states_out[2])
        # enc_hidden = self.encoder5(hidden_states_out[3])
        # print(enc1.shape, enc2.shape, enc3.shape, enc4.shape,enc_hidden.shape)
        # torch.Size([6, 48, 96, 96, 96]) torch.Size([6, 96, 48, 48, 48]) torch.Size([6, 192, 24, 24, 24]) 
        # torch.Size([6, 384, 12, 12, 12]) torch.Size([6, 512, 6,6, 6])

        # print(enc_hidden.shape, enc4.shape)
        dec3 = self.decoder4(hidden_states_out[3], hidden_states_out[2])
        dec2 = self.decoder3(dec3, hidden_states_out[1])
        dec1 = self.decoder2(dec2, hidden_states_out[0])
        dec0 = self.decoder1(dec1, enc1)
        # out = self.decoder1(dec0)
        out = self.out(dec0)
        # print(dec3.shape, dec2.shape, dec1.shape, dec0.shape, out.shape)
        # torch.Size([6, 384, 4, 4, 4]) torch.Size([6, 192, 8, 8, 8]) torch.Size([6, 96, 16, 16, 16]) 
        # torch.Size([6, 48, 32, 32, 32]) torch.Size([6, 48, 64, 64, 64])
        
        return out
        # return dec,img_embeding,out


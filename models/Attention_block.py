from audioop import bias
from collections import OrderedDict
from turtle import shape
from typing import Tuple, Union
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop_path, trunc_normal_ #drop,
from einops import rearrange
from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck

import math
from timm.models.vision_transformer import VisionTransformer

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, v_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim


        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # print(tensor.shape)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()
        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output

        # return attn_output, attn_weights

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(
            self.positional_embedding[1:, ].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3,
                                                                                                                   1,
                                                                                                                   2),
            size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H * W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop_path=0.,
                 bias = False,):
        super().__init__()
        # out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias = bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias = bias)
        # self.drop = nn.Dropout(drop)
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MLP_gddiff(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
    
class TransformerLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = self.norm1(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class Self_token_attn(nn.Module):
    def __init__(self,
                 embed_dim=384,
                 transformer_heads=32,
                 output_dim=384,
                 spacial_dim = 64,
                 out_channels = 15,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim,bias = True)
        self.q_proj = nn.Linear(embed_dim, embed_dim,bias = True)
        self.v_proj = nn.Linear(embed_dim, embed_dim,bias = True)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim,bias = True)
        self.num_heads = transformer_heads
        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 3 + 1, 
                                                            embed_dim) / embed_dim ** 0.5)
        self.out_channels = out_channels
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):
        # B, C,H,W,D = x.shape
        # x = self.in_proj(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)  #BTHWD
        B, C,H,W,D = x.shape
        x = x.reshape(B,C,H*W*D).permute(2, 0, 1)  # BCHWD -> (HWD)BC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HWD+1)BC
        cls_pos = self.positional_embedding[0:1, :]
        # print(self.positional_embedding[1:, ].shape)
        spatial_pos = F.interpolate(
            self.positional_embedding[1:, ].reshape(1,
            self.spacial_dim, self.spacial_dim,self.spacial_dim, self.embed_dim).permute(0,4,1,2,3),
            size=(H, W, D), mode='trilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H * W * D).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
        x = x + positional_embedding[:, None, :]  #N,B,C
        # print(text.shape, visual.shape)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        x = x.permute(1,0,2)
        token = x[:, 0:1, :]
        img_embdding = x[:, 1:, :].reshape(B, -1, H, W,D)
        # print(token, img_embdding)
        return token, img_embdding

class Ca_attn(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=1,
                 q_dim=384,
                 kv_dim=384,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.q_proj = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, transformer_width),
            # nn.LayerNorm(transformer_width),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(kv_dim),
            nn.Linear(kv_dim, transformer_width),
        )

        self.cattn = nn.ModuleList([
            TransformerLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, kv_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, x):
        # print(q.shape, x.shape)
        # B, N1, C1 = q.shape
        # B,N2,C2 = x.shape
        # q = q + self.positional_embedding
        k= v = self.kv_proj(x)
        x = self.q_proj(q)

        for layer in self.cattn:
            x = layer(x, k)

        return self.out_proj(x)

class Attention_gddffi(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        N, C = q.shape
        assert k.shape == v.shape
        M, C = k.shape
        q = self.q_proj(q).reshape(N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('nkc,mkc->knm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('knm,mkc->nkc', attn, v).reshape(N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_DwConv(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.q_proj = nn.Conv3d(in_channels=dim, out_channels=dim, 
                                kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv3d(in_channels=dim, out_channels=dim, 
                                kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv3d(in_channels=dim, out_channels=dim, 
                                kernel_size=1, stride=1,bias=qkv_bias)

        self.q = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, 
                            padding=1,stride=1,groups=dim, bias=qkv_bias)
        self.k = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, 
                            padding=1,stride=1,groups=dim, bias=qkv_bias)
        self.v = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, 
                            padding=1,stride=1,groups=dim, bias=qkv_bias)

    
    def forward(self, q, k, v):
        B,N,H,W,D = q.shape
        assert k.shape == v.shape
        
        q = self.q(self.q_proj(q))
        k = self.k(self.k_proj(k))
        v = self.v(self.v_proj(v))

        attn = q * k
        attn = attn.softmax(dim=-1)
        x = attn * v
        return x

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory1,memory2, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory1,memory2, pos=pos, query_pos=query_pos)

        return output
    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, q_dim, k_dim, v_dim, embed_dim,embed_dim_Mul, num_heads, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="gelu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=q_dim,
                                       v_dim=q_dim,
                                       embed_dim=embed_dim_Mul,
                                       num_heads=num_heads,
                                       )
        self.multihead_attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       v_dim=v_dim,
                                       embed_dim=embed_dim_Mul,
                                       num_heads=num_heads,
                                       )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(embed_dim) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory1, memory2, pos = None, query_pos = None):
        if len(tgt.size()) == 3:
            bsz, tgt_len, embed_dim = tgt.size()
            # bsz, tgt_len_k, embed_dim = k.size()
            # bsz, tgt_len_v, embed_dim = v.size()
        elif len(tgt.size()) == 2:
            tgt_len, embed_dim = tgt.size()
            bsz = memory1.shape[0]
            tgt = tgt.expand(bsz, tgt_len, embed_dim)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(q=self.with_pos_embed(tgt2, query_pos),
                                   k=self.with_pos_embed(memory1, pos),
                                   v=self.with_pos_embed(memory2, pos),)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    
# class Attention_dwconv(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super().__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
#         self.q_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
#         self.k_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
#         self.v_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        
#         self.q_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.k_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.v_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         # self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
#         # self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
#         self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        
#     def forward(self, q,k,v):
#         b,c,h,w,d = q.shape
#         # qkv = self.qkv_dwconv(self.qkv(x))
#         # q,k,v = qkv.chunk(3, dim=1) 
#         q = self.q_dwconv(self.q_proj(q))
#         k = self.k_dwconv(self.k_proj(k))
#         v = self.v_dwconv(self.v_proj(v))
        
#         q = rearrange(q, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w d-> b head c (h w d)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w d-> b head c (h w d)', head=self.num_heads)
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#         out = (attn @ v)
#         out = rearrange(out, 'b head c (h w d) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)
#         out = self.project_out(out)
#         return out

    
class Ca_atten_dwconv(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super().__init__()
        # with torch.no_grad():
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        self.q_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v_dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, q,k,v):
        b,c,h,w,d = q.shape
        q = self.q_dwconv(self.q_proj(q))
        k = self.k_dwconv(self.k_proj(k))
        v = self.v_dwconv(self.v_proj(v))
        
        q = rearrange(q, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w d-> b head c (h w d)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w d-> b head c (h w d)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w d) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)
        out = self.project_out(out)
        return out

class VisionLanguageAlign(nn.Module):
    def __init__(
        self, embed_dim, embed_dim_language, prior_prob=0.01, log_scale=0.0, clamp_dot_product=True
    ):
        super().__init__()
        # initialize the bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # dot product soft token head
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(
            embed_dim_language, embed_dim, bias=True
        )  # 768 -> 256
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(embed_dim_language), requires_grad=True)  # (768，)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)  # size (1,)

        self.clamp_dot_product = clamp_dot_product

    def forward(self, x, embedding):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, L, 768)
        """
        embedding = embedding.to(x.dtype)

        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)  # (bs, L, 768) L is maximum sentence length
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)  # 768 -> 256
        dot_product_proj_tokens_bias = (
            torch.matmul(embedding, self.bias_lang) + self.bias0
        )  # (bs, L, 768) x (768, ) + (1, ) -> (bs, L)

        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        A = dot_product_proj_queries.shape[1]  # num_query
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)  # (bs, num_query, L)

        dot_product_logit = (
            torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
            / self.log_scale.exp()
        ) + bias  # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)
        if self.clamp_dot_product:
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        return dot_product_logit
    
# class Multi_Scale_Visual_Text_Fusion(nn.Module):
#     def __init__(
#         self, visual_dim,visual_size, text_dim, embed_dim, num_heads,GN_num, dropout=0.1, no_norm = False,
#     ):
#         super().__init__()
#         self.GAP_h = nn.Sequential(
#             nn.GroupNorm(GN_num, visual_dim),
#             nn.LeakyReLU(inplace=True),
#             torch.nn.AdaptiveAvgPool3d((visual_size[0],1,1)),
#             nn.Conv3d(visual_dim, visual_dim, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
#         )
#         self.GAP_w = nn.Sequential(
#             nn.GroupNorm(GN_num, visual_dim),
#             nn.LeakyReLU(inplace=True),
#             torch.nn.AdaptiveAvgPool3d((1,visual_size[1],1)),
#             nn.Conv3d(visual_dim, visual_dim, kernel_size=(1,3,1), stride=1, padding=(0,1,0))
#         )
#         self.GAP_d = nn.Sequential(
#             nn.GroupNorm(GN_num, visual_dim),
#             nn.LeakyReLU(inplace=True),
#             torch.nn.AdaptiveAvgPool3d((1,1,visual_size[2])),
#             nn.Conv3d(visual_dim, visual_dim, kernel_size=(1,1,3), stride=1, padding=(0,0,1))
#         )
#         self.multihead_attn_h = MultiHeadAttention(q_dim=visual_dim,
#                                        k_dim=text_dim,
#                                        v_dim=text_dim,
#                                        embed_dim=embed_dim,
#                                        num_heads=num_heads,
#                                        )
#         self.multihead_attn_w = MultiHeadAttention(q_dim=visual_dim,
#                                        k_dim=text_dim,
#                                        v_dim=text_dim,
#                                        embed_dim=embed_dim,
#                                        num_heads=num_heads,
#                                        )
#         self.multihead_attn_d = MultiHeadAttention(q_dim=visual_dim,
#                                        k_dim=text_dim,
#                                        v_dim=text_dim,
#                                        embed_dim=embed_dim,
#                                        num_heads=num_heads,
#                                        )
#         self.msvtf_conv = nn.Conv3d(visual_dim, visual_dim, kernel_size=1, stride=1, padding=0)
#     def forward(self, visual, text): # visual:b,c,h,w,d   text:n_class,c
#         B,C,H,W,D = visual.shape
#         visual_h = self.GAP_h(visual).reshape(B,C,H).permute(0,2,1)  #b,c,h,1,1  -> b,c,h -> b,h,c
#         visual_w = self.GAP_w(visual).reshape(B,C,W).permute(0,2,1)  #b,c,1,w,1 -> b,c,w -> b,w,c
#         visual_d = self.GAP_d(visual).reshape(B,C,D).permute(0,2,1)  #b,c,1,1,d  -> b,c,d -> b,d,c
#         vt_h = self.multihead_attn_h(visual_h,text,text).permute(0,2,1).reshape(B,C,H,1,1)  #b,h,c ->  b,c,h,1,1
#         vt_w = self.multihead_attn_w(visual_w,text,text).permute(0,2,1).reshape(B,C,1,W,1)  #b,w,c ->  b,c,1,w,1
#         vt_d = self.multihead_attn_d(visual_d,text,text).permute(0,2,1).reshape(B,C,1,1,D)  #b,d,c -> b,c,1,1,d
#         visual = vt_h + vt_w + vt_d
#         out = self.msvtf_conv(visual)
#         return out

class Visual_Text_Fusion(nn.Module):
    def __init__(
        self, visual_dim,text_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(q_dim=visual_dim,
                                       k_dim=text_dim,
                                       v_dim=text_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       )
    def forward(self, visual, text): # H,W,D:1,img_size   text:n_class,c
        b,c,h,w,d = visual.shape
        v = rearrange(visual, 'b c h w d -> b c (h w d)').permute(0,2,1)  #b,n,c
        t = rearrange(text, 'b c h w d -> b c (h w d)').permute(0,2,1)    #b,n,c
        vt = self.multihead_attn(v,t,t).permute(0,1,2)    #b,n,c
        out = rearrange(vt, 'b c (h w d) -> b c h w d',  h=h, w=w, d=d)
        return out
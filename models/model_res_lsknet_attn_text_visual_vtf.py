from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from .model_res_lsknet_attn_text_visual_model import Res_lsknet_attn_text_visual_model
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from .Attention_block import TransformerDecoderLayer,TransformerDecoder,MultiHeadAttention,Self_token_attn,Ca_atten_dwconv


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



class Res_lsknet_attn_text_visual_vtf(nn.Module):
    def __init__(self, img_size, in_channels=1,out_channels=13, text_embeddings_path = None,Training = True ):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.out_channels = out_channels
        self.backbone = Res_lsknet_attn_text_visual_model(img_size=img_size, 
                    in_channels = 1,
                    out_channels=self.out_channels
                    )

       
        self.text_embedding_path = text_embeddings_path          
        if self.text_embedding_path is None:
            self.text_embedding = nn.Parameter(torch.zeros(self.out_channels,512))
            nn.init.normal_(self.text_embedding, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embedding', torch.randn(self.out_channels,512))
            loaded = torch.load(self.text_embedding_path, map_location='cuda')
            self.text_embedding[:, :] = loaded[:, :]
        
        self.out_backbone = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, groups=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1)
            )
        

        # self.out_backbone = nn.Sequential(
        #         nn.GroupNorm(self.out_channels, self.out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1)
        #     )
            
        self.GAP = nn.Sequential(
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
            )
        
#         self.self_attn = Self_token_attn(embed_dim=self.out_channels,
#                                          transformer_heads=self.out_channels,
#                                          output_dim=self.out_channels,
#                                          spacial_dim = 1,
#                                          out_channels = self.out_channels,
#                                         )

                
#         decoder_layer = TransformerDecoderLayer(q_dim = 512, 
#                                                 k_dim = self.out_channels, 
#                                                 v_dim = self.out_channels, 
#                                                 embed_dim = 512, 
#                                                 embed_dim_Mul = 768,
#                                                 num_heads = 32, 
#                                                 dim_feedforward=2048, 
#                                                )

#         self.multihead_attn = TransformerDecoder(decoder_layer,3)
        self.text_linear = MLP(512, 512,1,1)
        
        # self.conv_visual = nn.Conv3d(1,1,kernel_size=1,stride=1,padding=0)
        # self.conv_prompt = nn.Conv3d(self.out_channels,self.out_channels,kernel_size=3,stride=1,padding=1)
        
        for i in range(self.out_channels):
            # self.__setattr__('conv_visual_{}'.format(i),nn.Conv3d(in_channels=1, out_channels=1,kernel_size=1,padding=0))
            # self.__setattr__('conv_text_{}'.format(i),nn.Conv3d(in_channels=1, out_channels=1,kernel_size=1,padding=0))
            self.__setattr__('vtf_{}'.format(i),Ca_atten_dwconv(dim=1,num_heads=1))
            # self.__setattr__('conv_out_{}'.format(i),nn.Conv3d(in_channels=1, out_channels=1,kernel_size=1,padding=0))

            

        self.Training = Training
        
    # def load_params(self, model_dict):
    #     num_block = ['1','2','3','4']
    #     store_dict = self.backbone.state_dict()
    #     # print('Use pretrained weights','backbone.inconv')
    #     for key in model_dict.keys():
    #         if 'inconv' in key:
    #             store_dict[key] = model_dict[key]
    #             # print('Use pretrained weights',key)
    #         for block in num_block:
    #             block_key = 'lsknet_attn.patch_embed'+block
    #             block_b = 'lsknet_attn.block'+block
    #             block_norm = 'lsknet_attn.norm'+block
    #             block_decoder = 'decoder'+block
    #             if  block_key in key or block_b in key or block_norm in key or  block_decoder in key:
    #                 store_dict[key] = model_dict[key]
    #                 # print('Use pretrained weights',key)
    #     self.backbone.load_state_dict(store_dict)
    #     print('Use pretrained weights')
                                                                                                                                                   
    def load_params(self, model_dict):
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def forward(self, x_in):
        # with torch.no_grad():
        out_backbone = self.backbone(x_in)
        out_backbone = self.out_backbone(out_backbone)
        B,C,H,W,D = out_backbone.shape

        # out_gap = self.GAP(out_backbone)  #b,c,1,1,1
        # token = out_gap.squeeze(-1).squeeze(-1).permute(0,2,1)
        # token,img = self.self_attn(out_gap)
        text_embedding = self.text_embedding.unsqueeze(0).expand(B,-1,-1)
        # text_embedding_token = self.multihead_attn(text_embedding,token,token)
        # text_embedding = torch.mean(text_embedding_token,dim=-1,keepdim=True)  #b,num_class,1
        text_embedding = self.text_linear(text_embedding).unsqueeze(-1).unsqueeze(-1)
        text_embedding = text_embedding.repeat(1,1,H,W,D)
        # text_embedding = torch.mean(text_embedding_token,dim=-1,keepdim=True).unsqueeze(-1).unsqueeze(-1)#b,num_class,num_class,1,1
        # text_embedding = text_embedding.repeat(1,1,H,W,D)

        # visual =  torch.softmax(self.conv_visual(img), dim=1)
        # visual = torch.mean(img,dim=1,keepdim=True)
        # visual = self.conv_visual(visual).sigmoid()
        # # prompt = torch.einsum('bnc,bchwd ->bnhwd',text_embedding, out_backbone)
        # prompt = text_embedding * visual
        # prompt = self.conv_prompt(prompt)

        out_list = []
        for i in range(self.out_channels):                                                  
            
            out_backbone_ = out_backbone[:,i,:].unsqueeze(1)
            prompt_ = text_embedding[:,i,:].unsqueeze(1)
            
            # out_backbone_ = out_backbone_ / out_backbone_.norm(dim=1, keepdim=True)
            
            # prompt_conv_ein = self.__getattr__('prompt_conv_ein{}'.format(i))
            # class_ein = torch.einsum('bchwd,bnhwd->bnhwd', out_backbone_, prompt_conv_ein(prompt_)) * out_backbone_
            # conv_visual = self.__getattr__('conv_visual_{}'.format(i))
            # conv_text = self.__getattr__('conv_text_{}'.format(i))
            # visual = conv_visual(out_backbone_)
            # text = conv_text(prompt_)
            # visual_text = out_backbone_ * prompt_
            vtf = self.__getattr__('vtf_{}'.format(i))
            visual_text = vtf(out_backbone_,prompt_,prompt_)
            # conv_out = self.__getattr__('conv_out_{}'.format(i))
            # out = conv_out(visual_text)
            out_list.append(visual_text)
        out = torch.cat(out_list,dim=1) + out_backbone
        # print(text_visual.shape, out.shape)
        if self.Training:
            return out
        else:
            return out
        
    
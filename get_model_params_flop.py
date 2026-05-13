from email.mime import image
from turtle import xcor
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis, parameter_count_table, jit_handles
from typing import Any, Callable, List, Optional, Union
from numbers import Number
from numpy import prod
import numpy as np
from thop import profile
import math
import os
from ptflops import get_model_complexity_info
from models.model_res_lsknet_attn_text_visual_test  import Res_lsknet_attn_text_visual

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")
main_data_path = '/root/autodl-tmp/'
main_path = '/root/autodl-tmp/data/'
pretrained_path_text = main_path+'pretrained/clip_text_amos.pth'
pretrained_path_visual = main_path+'pretrained/Res_lsknet_attn_amos.pth'

for i in range(4):
    in_chan = 1
    print('in chan',in_chan)
    x = torch.randn((2,in_chan,96,96,96))
    model = Res_lsknet_attn_text_visual(img_size=(96,96,96),
                    in_channels=1,
                    out_channels=15,
                    text_embeddings_path = pretrained_path_text,
                    )

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (in_chan,96,96,96), as_strings=True, print_per_layer_stat=False)
    print(parameter_count_table(model))
    print("FLOPs: {}".format(macs))
    print('Params: {}'.format(params))
    print('--------------------------------------\n')
    
    flops, param = profile(model, inputs=(x,))
    params =sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('flops:{} G'.format(flops/ 1e9))
    # print('flops:{} M'.format( param/ 1e6))
    print('flops:{} M'.format( params/ 1e6))
    # print('--------------------------------------')

    fca1 = FlopCountAnalysis(model, x)
    parm = parameter_count_table(model)
    print('flops:{} G'.format(fca1.total()/ 1e9))
    print('flops:{} M'.format(parm.total()))
    print('-------------------------------\n')


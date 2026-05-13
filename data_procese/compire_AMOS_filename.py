from pickle import TRUE
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import glob
import warnings

filename = ['Training','Valing','Testing']
for i in range(3):
    name = filename[i]
    print(name)
    img_path = '/media/xd/date/muzhaoshan/AMOS2022/'+name+'/images/'
    file_list_img = sorted(glob.glob(os.path.join(img_path, '*.npy')))

    mask_path = '/media/xd/date/muzhaoshan/AMOS2022/'+name+'/images/'
    file_list_mask = sorted(glob.glob(os.path.join(mask_path, '*.npy')))

    if(len(file_list_img) != len(file_list_mask)):
        print('data file error!')
        exit(0)

    name_list = []
    for j in range(len(file_list_img)):
        img_name = file_list_img[j][-13:-4]
        mask_name = file_list_mask[j][-13:-4]
        if(img_name == mask_name):
            print(img_name,mask_name)
        else:
            print('data file error!')
            exit(0)


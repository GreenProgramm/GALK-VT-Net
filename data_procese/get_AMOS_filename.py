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
warnings.filterwarnings("ignore")
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    EnsureChannelFirstd,
    RandZoomd,
    RandCropByLabelClassesd,
)
import SimpleITK as sitk
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
from monai.data import CacheDataset, DataLoader, decollate_batch
import logging


file_name = 'AMOS_total.txt'
# file_list = open(os.path.join('/media/xd/date/muzhaoshan/AMOS2022/imagesTr/', file_name)).readlines()
main_path = '/media/xd/date/muzhaoshan/AMOS2022/imagesTr/'
file_list = sorted(glob.glob(os.path.join(main_path, '*.nii.gz')))
print(len(file_list))
name_list = []
for patient_name in file_list:
    patient_name = patient_name[-16:-7]
    name_list.append(patient_name)
    print(patient_name)
new_name_list = list(set(name_list))
new_name_list.sort()
print(len(new_name_list))
print(new_name_list)
name = []
for file in new_name_list:
            #print(name_64)
        with open(file_name, 'a+') as f:  # 设置文件对象
            f.write(file)  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))

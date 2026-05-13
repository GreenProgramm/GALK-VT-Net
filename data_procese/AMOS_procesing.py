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
logging.disable(logging.WARNING)

def resampleVolume(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    # print(data_dict)

    # print('raw shape',data_dict['image'].shape,data_dict['label'].shape)

    # 添加通道
    ensureChannelFirstd = EnsureChannelFirstd(keys=["image", "label"])
    data_dict = ensureChannelFirstd(data_dict)

    # print('raw shape',data_dict['image'].shape,data_dict['label'].shape)

    orientationd = Orientationd(keys=["image", "label"], axcodes="RAS")
    data_dict = orientationd(data_dict)

    spacingd = Spacingd(
            keys=["image", "label"],
            pixdim=(1.5,1.5,2.0),
            mode=("bilinear", "nearest"),
        )
    data_dict = spacingd(data_dict)

    scaleIntensityRanged = ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=275,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    data_dict = scaleIntensityRanged(data_dict)
    
    data_image, data_label = data_dict["image"].squeeze().permute(1,0,2), data_dict["label"].squeeze().permute(1,0,2)
    # print(data_label.shape)
    # # z = data_label.shape[-1]
    # # print(total_slice)
    # total_slice = total_slice + data_label.shape[-1]
    return data_image, data_label

main_path = r'D:\Project\multi-organ seg\amos'
file_list_train = open(os.path.join(main_path, 'AMOS_train.txt')).readlines()
file_list_val = open(os.path.join(main_path, 'AMOS_val.txt')).readlines()
file_list_test = open(os.path.join(main_path, 'AMOS_val.txt')).readlines()
input_path = main_path + '/imagesTr/'
for i in range(3):
    if i==0:
        file_list = file_list_train
        save_file = 'Training'
    elif i==1:
        file_list = file_list_val
        save_file = 'Valing'
    elif i==2:
        file_list = file_list_test
        save_file = 'Testing'
    
    save_path_img = main_path + '/'+save_file+'/images/'
    save_path_label = save_path_img.replace('images','masks')
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)
    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)
    for patient_name in file_list:
        patient_name = patient_name.strip('\n')
        print(patient_name)
        data_image_str = input_path + patient_name + '.nii.gz'
        data_masks_str = input_path.replace('images', 'labels')  + patient_name + '.nii.gz'
        data_dicts = {'image':data_image_str, 'label': data_masks_str}

        image,label = resampleVolume(data_dict=data_dicts)
        # preprossing_data(data_dicts,save_path,patient_name,save_path)

        # image_data = sitk.Image(sitk.ReadImage(data_image_str))
        # mask_data = sitk.Image(sitk.ReadImage(data_masks_str)) 

        # image = sitk.GetArrayFromImage(image_data).transpose(1,2,0)
        # label = sitk.GetArrayFromImage(mask_data).transpose(1,2,0)
        image =np.flipud(image)
        image =np.fliplr(image)

        label = np.flipud(label)
        label = np.fliplr(label)
        # label = generate_label(label)
        # print(image.shape,label.shape)
        # print(np.unique(image),np.unique(label))
        np.save(save_path_img +patient_name+'.npy',image)
        np.save(save_path_label +patient_name+'.npy',label)

        show =False
        if show:
            index = 0
            for i in range(label.shape[-1]):
                if len(np.unique(label[:,:,i]))>5:
                    index = i
                    break;
            plt.figure()
            slice_index = index
            # print(nor_image[:,:,slice_index])
            print(slice_index)
            show_num = 16
            sub_plt = int(pow(show_num,0.5))
            for i in range(show_num):
                num = slice_index+i
                x = image[:,:,num]
                plt.subplot(sub_plt,sub_plt,i+1)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(x,cmap='gray')

            plt.figure()
            for i in range(0,show_num):
                num = slice_index+i
                x = label[:,:,num]
                plt.subplot(sub_plt,sub_plt,i+1)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(x,cmap='gray')
            plt.show()
                


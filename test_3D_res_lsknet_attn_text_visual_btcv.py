from operator import mod
from typing import overload
import pandas as pd
import scipy.io as scio
import math
import scipy.io as sc
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.metrics import Train_index, Train_index_gpu, Train_index_gpu_no_background
from torch.utils.data import Dataset, DataLoader
import glob
from models.model_res_lsknet_attn_text_visual  import Res_lsknet_attn_text_visual
from utils.dataloader import get_loader
import os
import time
from utils.getlogger import get_logger
import argparse
import sys
from thop import profile
from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
from clip import clip
from ptflops import get_model_complexity_info
from monai.inferers import sliding_window_inference


Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

#===============================================init======================================
parser = argparse.ArgumentParser()
parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')
parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
parser.add_argument('--phase', default='train', help='train or validation or test')
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--dataset_type',type=str, default='BTCV')
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--num_classes', type=float, default=13)
parser.add_argument('--save_pred', type=str, default='True')
parser.add_argument('--image_size', type=int, default=96)
parser.add_argument('--window_size', type=int, default=8)
parser.add_argument('--load_model_num', type=int, default=-1)
parser.add_argument('--re_type', type=str, default='True')
parser.add_argument('--data_root_path', type=str, default='/home/share/zym/Multi-organ_seg/all_data/btcv/Testing')

args = parser.parse_args()

'''
CUDA_VISIBLE_DEVICES=0 python test_3D_res_lsknet_attn_text_visual_btcv_hd95.py  --loss_type CE+Dice_config_3D_res_lsknet_attn_text_visual_btcv_0
'''

torch.cuda.empty_cache()
#===============================================path======================================
main_data_path = '/root/autodl-tmp/'
main_path = '/home/share/zym/Multi-organ_seg/code/data/'
pretrained_path_text = main_path+'pretrained/clip_text_btcv.pth'
pretrained_path_swin = main_path+'pretrained/Res_lsknet_attn_btcv.pth'
#======================================parser args==========================================
num_classes = args.num_classes
dataset_type = args.dataset_type
patch_size = args.patch_size
loss_type = args.loss_type
load_model_num = args.load_model_num
print('patch size',patch_size)
if patch_size==0:
    print('error show type!')
    import sys
    sys.exit(0)
if args.save_pred=='True':    
    save_pred = 1
elif args.save_pred=='False':
    save_pred = 0
else:
    print('error show type!')
    sys.exit(0)
image_size = args.image_size
window_size = (args.window_size,args.window_size)
# stride_size = [image_size,image_size]

#======================================input path==========================================
in_chan = patch_size
data_type = 'crop_'+str(image_size)+'_patch_'+str(in_chan)

save_path = main_path + 'test_result/'+ dataset_type+'/'+data_type +'_'+loss_type+'/'
if not os.path.exists(save_path):
        os.makedirs(save_path)
        
save_path_pred = save_path+'/pred/'
if not os.path.exists(save_path_pred):
        os.makedirs(save_path_pred)
        
weigth_input_path = main_path +'/work_dirs/weight/'+dataset_type+'_'+data_type+'_'+loss_type+'/'

logger_path = main_path + 'test_result/logger/'+dataset_type+'_'+data_type+'_'+loss_type+'/'
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logger = get_logger(logger_path+'single_patient_'+str(Time)+'.log',
                    verbosity=1, name=__name__)

logger.info(loss_type)


#======================================label and indicator==========================================
# indicator_list=['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
indicator_list=['Dice','ACC','Iou','F_score','Precision','Recall']
# label_list=["Aorta","Gallbladder","Kidney(L)","Kidney(R)","Liver","Pancreas","Spleen","Stomach","Avge"]  #avg
label_list=['Sple', 'Rkid', 'Lkid', 'Gall', 'Esop', 
                'Liver', 'Stom', 'Arot', 'InVC', 'Vein',
                'Panc', 'RAGd','LAGd',"Avge"]
    # print(len(label_list))
# print(len(label_list))

#======================================load model==========================================
weigth_path = weigth_input_path + 'best_model.pth'
logger.info(weigth_path)

# ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
#                 'Liver', 'Stomach', 'Arota',  'Inferior Vena Cava',
#                 'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland',
#                 'Duodenum','Bladder','Prostate']
model = Res_lsknet_attn_text_visual(img_size=(96,96,96),
                    in_channels=1,
                    out_channels=num_classes,
                    text_embeddings_path = pretrained_path_text,
                    )
logger.info('model {}'.format(model))
checkpoint = torch.load(weigth_path)
model.load_state_dict(checkpoint['model'])
# new_state_dict = {}
# for k, v in checkpoint['model'].items():
#     name = 'backbone.'+k # remove `module.`
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)
model.cuda()
model.Training = False
# #======================================params flops==========================================
macs, params = get_model_complexity_info(model, (in_chan,image_size,image_size,image_size), as_strings=True, print_per_layer_stat=False)
# print(parameter_count_table(model))
logger.info("FLOPs: {}".format(macs))
logger.info('Params: {}'.format(params))

model.eval()
metric_list = 0.0
args.phase = 'test'
test_loader,data_dicts_test = get_loader(args,logger)
test_len = len(test_loader)
with torch.no_grad():
    # for image_input, label_input in tqdm(test_loader):      #single case slice
    # index = 0
    # for batch in tqdm(test_loader):
    for index, batch in enumerate(test_loader):
        image, label = batch["image"], batch["label"]
        image_input, label_input = image.float().cuda(), label.float().squeeze(1).cuda()
        #input:B,C,H,W
        # print(image_input.shape,label_input.shape)
        # raw_shape = image_input.shape
        output = sliding_window_inference(image_input, (96, 96, 96), 2, model,overlap = 0.75)
        # image_crop_list = sliding_window_inference_crop(image_input,(image_size,image_size,image_size), 
        #                                                 sw_batch_size=1,overlap = 0.5)
        # img_list = []
        # # print(len(image_crop_list))
        # for img in image_crop_list:
        #     # print(img.shape)
        #     out,_,_ = model(img)
        #     img_list.append(out)
        # output = sliding_window_inference_re(image_input,img_list,(image_size,image_size,image_size),
                                                # sw_batch_size=1,overlap = 0.01)
        # pred = torch.argmax(torch.softmax(output, dim=1), dim=1) #1,H,W
        pred = torch.sigmoid(output)
        # print(pred.shape,label_input.shape)
        # print(np.unique(pred),np.unique(mask))
        test_name = data_dicts_test[index]["image"][-8:-4]
        # print(data_dicts_test[index]["image"],test_name)
        if save_path_pred is not None:
            pred_save = (pred.squeeze(0) > 0.5).astype(torch.uint8)
            pred_ = torch.zeros(pred_save.shape[1],pred_save.shape[2],pred_save.shape[3]).cuda()
            for i in range(pred_save.shape[0]):
                label_index = i
                save = pred_save[i,:]
                save[save == 1] +=  label_index
                pred_ += save
            prd_itk = pred_.permute(2,0,1).cpu().detach().numpy()
            prd_itk = sitk.GetImageFromArray(prd_itk)
            sitk.WriteImage(prd_itk, save_path_pred + '/' + test_name + "_pred.nii")
            print('save '+test_name + "_pred.nii")

        #=================single indicators===============
        metric_ = Train_index_gpu_no_background(pred, label_input, num_classes,'test')    #single case indicators
        logger.info('case %s mean_dice %f ' % (test_name, np.mean(metric_, axis=0)[0]))
        metric_list += np.array(metric_)  # count all case indicators
metric_list = metric_list / test_len  #mean all case indicators
# print(len(metric_list))
# =================mean all case indicators===============
logger.info('total %d case mean'%(test_len))
logger.info('\t\tmean_dice\tmacc\t\tmIou\t\tmF_score\tmPrecision\tmRecall ')
for i in range(num_classes):
    ii = i
    logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f' %
                                (label_list[ii][:4], metric_list[ii][0],
                                    metric_list[ii][1], metric_list[ii][2],
                                    metric_list[ii][3], metric_list[ii][4],
                                    metric_list[ii][5]))
    #=================avg mean class(所有病例的平均)===============
Index = np.mean(metric_list, axis=0)
logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f' %
            (label_list[-1], round(Index[0], 6),
                round(Index[1], 6), round(Index[2], 6),
                round(Index[3], 6), round(Index[4], 6),
                round(Index[5], 6)))
best_metric_list = []
best_metric_list = metric_list.tolist()
best_metric_list.append(Index.tolist())
best_metric_list = np.array(best_metric_list)
best_metric_list = np.round(best_metric_list,6)
logger.info('best dice hd95 {}  {}'.format(Index[0], Index[1]))
# logger.info('best epoch  {}'.format(best_epoch))

print('\n')
# print(best_metric_list)
if save_pred: 
    save_csv = pd.DataFrame(best_metric_list, columns=indicator_list)
    save_csv.insert(0, 'Class', label_list)
    # save_csv.insert(len(indicator_list)+1, 'Best_eopch', best_epoch)
    save_csv.to_csv(save_path+dataset_type+'.csv',index=False,sep=',')








from itertools import zip_longest
import torch
import argparse
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from utils.metrics import Multi_DiceLoss,Multi_BCELoss
from torch.nn.modules.loss import CrossEntropyLoss
from models.model_res_lsknet_attn_text_visual  import Res_lsknet_attn_text_visual
from utils.dataloader import get_loader
# import cv2
from functools import partial
from random import randint
import random
import timeit
from thop import profile
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
import time
from utils.getlogger import get_logger
import sys
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from torchvision import transforms
from thop import profile
from val_3D_amos import val
import pandas as pd
import torch.nn.functional as F
from clip import clip
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', default=1, help='batch size')
# parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
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
parser.add_argument('--cache_rate', default=1.0, type=float, help='The percentage of cached data in total')
parser.add_argument('--phase', default='train', help='train or validation or test')
parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.00001, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--resume', type=str, default='False')
parser.add_argument('--dataset_type', type=str, default='AMOS')
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=15)
parser.add_argument('--load_model_num', type=int, default=14)
parser.add_argument('--save_model_epoch', type=int, default=1500)
parser.add_argument('--image_size', type=int, default=96)
parser.add_argument('--pretrained', type=str, default='False')
parser.add_argument('--start_val', type=int, default=300)
parser.add_argument('--val_epoch_gap', type=int, default=50)
parser.add_argument('--data_root_path', type=str, default='/home/share/zym/Multi-organ_seg/all_data/amos/Training')

args = parser.parse_args()

'''
CUDA_VISIBLE_DEVICES=0 python train_3D_res_lsknet_attn_text_visual_amos.py --loss_type CE+Dice_config_3D_res_lsknet_attn_text_visual_amos_0
'''

#================================random seed===============================
seed = random.randint(0, 2025)
# seed = 1779
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

#================================path===============================
main_data_path = '/home/share/zym/Multi-organ_seg/'
main_path = '/home/share/zym/Multi-organ_seg/code/data/'
pretrained_path_text = main_path+'pretrained/clip_text_amos.pth'
pretrained_path_visual = main_path+'pretrained/Res_lsknet_attn_amos.pth'
#================================ parser args===============================
dataset_type = args.dataset_type
num_classes = args.num_classes
patch_size = args.patch_size
image_size = args.image_size
start_epoch = args.load_model_num
# window_size = (args.window_size,args.window_size)
print('patch size',patch_size)
print('num_classes',num_classes)
if patch_size==0:
    print('error patch size!')
    import sys
    sys.exit(0)
if args.resume=='True' :
    resume = 1
#    start_epoch= 0
elif args.resume=='False':
    resume = 0
    start_epoch= 0
else:
    print('error break_point type!')
    sys.exit(0)

if args.pretrained=='True' :
    pretrained = pretrained_path_text
elif args.pretrained=='False':
    pretrained = None
else:
    print('error pretrained type!')
    sys.exit(0)

in_chan = patch_size
data_type = 'crop_'+str(image_size)+'_patch_'+str(in_chan)
# data_type = 'text_crop_'+str(image_size)+'_patch_'+str(in_chan)
max_dice = 0
#================================ input path ====================================

train_path = main_data_path +'train_data/'+ data_type+'/'
loss_type  = args.loss_type
save_path = main_path + 'test_result/'+ dataset_type+'/'+data_type +'_'+loss_type+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

record_type = data_type + '_'+loss_type      #save train loss logger ==> jpg
data_type = dataset_type+'_'+data_type
weight_patch = main_path +'/work_dirs/weight/'+data_type+'_'+loss_type+'/'
if not os.path.exists(weight_patch):
    os.makedirs(weight_patch)
if not os.path.exists(weight_patch.replace('weight','logger')):
    os.makedirs(weight_patch.replace('weight','logger'))
logger = get_logger(filename=weight_patch.replace('weight','logger')+str(Time)+'.log', 
                    verbosity=1, name=__name__)
# traceback.print_stack()

#================================model per====================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Res_lsknet_attn_text_visual(img_size=(96,96,96),
                    in_channels=1,
                    out_channels=num_classes,
                    text_embeddings_path = pretrained_path_text,
                    )
model.load_params(torch.load(pretrained_path_visual)["model"])
logger.info('model {}'.format(model))
# model = nn.DataParallel(model)
model.to(device)

##=========configs===========##
logger.info('Data Type {}'.format(data_type))
logger.info('max epoch {}'.format(args.epochs))
logger.info('batch size {}'.format(args.batch_size))
logger.info('learning rate {}'.format(args.learning_rate))
# logger.info('learning decay {}'.format(args.learning_decay))
logger.info('warmup epochs {}'.format(args.warmup_epochs))
logger.info('loss type {}'.format(loss_type))
logger.info('patch size {}'.format(in_chan))
logger.info('dataset type {}'.format(dataset_type))
logger.info('num_classes {}'.format(num_classes))
logger.info('save_model_epoch {}'.format(args.save_model_epoch))
logger.info('image_size {}'.format(image_size))
logger.info('resume {}'.format(args.resume))
logger.info("Random Seed: {}".format(seed))
#================================load data====================================
logger.info('train path {}'.format(train_path))
trianloader = get_loader(args,logger)

    
#================================model optimizer====================================
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# criterion_ce_ = CrossEntropyLoss()
# criterion_dice = DiceLoss_sy(num_classes)
# criterion_ce = CrossEntropyLoss()
criterion_ma_loss = nn.SmoothL1Loss()
criterion_dice = Multi_DiceLoss(num_classes=num_classes).to(device)
criterion_promt_ce = Multi_BCELoss(num_classes=num_classes).to(device)
criterion_ce = Multi_BCELoss(num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.warmup_epochs, gamma=0.99)
# scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
# #======================================params and flops==========================================
from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(model, (in_chan,image_size,image_size,image_size), as_strings=True, print_per_layer_stat=False)
# print(parameter_count_table(model))
logger.info("FLOPs: {}".format(macs))
logger.info('Params: {}'.format(params))






log_dict = {'celoss': [],'diceloss': [],'loss': [],'dice': [],'epoch':[] }  #log

#================================resume train====================================
if resume:
    logger.info('==============using checkpoint!================')
    path_checkpoint = weight_patch+'best_model.pth' 
    checkpoint = torch.load(path_checkpoint) 
    model.load_state_dict(checkpoint['model'])  

    optimizer.load_state_dict(checkpoint['optimizer'])
    df = pd.read_csv(save_path+dataset_type+".csv")
    start_epoch = df.iloc[-1]["Best_epoch"]
    
    max_dice = checkpoint['best_dice'] 
    logger.info("resume best dice: {}".format(max_dice))

#================================training====================================


logger.info('start training!')
for epoch in range(start_epoch+1,args.epochs+1):
    print('epoch',epoch)
    epoch_running_loss = 0
    train_idx = 0.0
    step_loss_dis_loss = 0
    step_loss_ce = 0
    step_loss_dice = 0
    step_loss_ma_loss = 0
    step_loss_dice_ = 0
    step_loss = 0
    trainloader_len = len(trianloader)
    model.train()
    # for image, label,text in tqdm(trianloader):
    for  batch in tqdm(trianloader):
        image, label = batch["image"], batch["label"]
        image = image.float()
        label = label.float()
        
        # image, label = image.to(device), label.to(device)
        image, label = image.to(device), label.squeeze(1).to(device)
        # print(image.shape,label.shape)
        # image = torch.flatten(image,0,1)
        # label = torch.flatten(label,0,1)

        output = model(image)
        
        # ious = get_iou(out_backbone, label).detach()
        # ious = mynorm(ious)
        # text_embedding = torch.mean(text_embedding,dim=-1,keepdim=True)
        # text_embedding = text_embedding.squeeze(-1)
        # ma_loss = criterion_ma_loss(text_embedding, ious)

        # print(output.shape,label.shape)
        # print(np.unique(label.cpu().detach().numpy()))
        loss_ce = criterion_ce(output, label)
        loss_dice = criterion_dice(output, label)
        # loss_prompt_ce = criterion_ce(prompt, label)
        # loss_dice = criterion_dice(output, label, softmax=True)
        # # loss = loss_ce + loss_dice
        # print(loss_ce.item(),loss_dice.item())
        loss = loss_ce + loss_dice
        # loss = loss_function(output, label)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================loss====================
        # step_loss_dis_loss += dis_loss.item()
        step_loss_ce += loss_ce.item()
        step_loss_dice += loss_dice.item()
        # step_loss_ma_loss += ma_loss.item()
        # step_loss_dice_ += loss_dice_.item()
        #step_loss_fl += loss_fl.item()
        
        step_loss += loss.item()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # ===================index====================
        # print(train_index)
    # scheduler.step()
    # step_loss_ma_loss /= trainloader_len
    step_loss_ce /= trainloader_len
    step_loss_dice /= trainloader_len
    step_loss /= trainloader_len
    # print(step_loss_ce_ / trainloader_len,step_loss_dice_ / trainloader_len)
    logger.info('epoch [{}/{}], lr:{:.8f}, CrossEntropyLoss:{:.6f}, Diceloss:{:.6f}, Loss:{:.6f}'
                .format(epoch, args.epochs,lr,step_loss_ce,step_loss_dice,step_loss))
    # logger.info('epoch [{}/{}], lr:{:.8f}, Loss:{:.6f}'
    #             .format(epoch, args.epochs,lr,step_loss))
    # logger.info('epoch [{}/{}], lr:{:.8f}, CrossEntropyLoss:{:.6f}, Diceloss:{:.6f},Mas_loss:{:.6f}, Loss:{:.6f}'
    #             .format(epoch, args.epochs,lr,step_loss_ce,step_loss_dice,step_loss_ma_loss,step_loss))

    checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
   
    # ================================testing====================================
    # print(max_dice)
    if epoch >= args.start_val or epoch % args.val_epoch_gap==0:
        model.Training = False
        max_dice = val(args=args,model=model,logger=logger, epoch=epoch,\
                                    checkpoint=checkpoint,main_data_path=main_data_path,\
                                    save_path=save_path,max_dice=max_dice,weight_patch=weight_patch,\
                                    device=device,save_pred=None)
        model.Training = True
    
     #================================save model====================================
    
    if epoch % args.save_model_epoch == 0:
        logger.info('save epoch {} model'.format(epoch))
        best_dice_dict={"best_dice":max_dice}
        checkpoint.update(best_dice_dict)
        torch.save(checkpoint, weight_patch + str(epoch) + ".pth")

    
    # ================================log traning====================================
    log_dict['celoss'].append(float(step_loss_ce))
    log_dict['diceloss'].append(float(step_loss_dice))
    log_dict['loss'].append(float(step_loss))
    log_dict['epoch'].append(epoch)

    
plt.figure()
plt.plot(log_dict['epoch'],log_dict['celoss'],\
        log_dict['epoch'],log_dict['diceloss'],\
        log_dict['epoch'],log_dict['loss'])
label = ['loss_ce','loss_dice','loss']
plt.legend(label,loc='best')
record_path = main_path + 'record/save_each_eopch/'
if not os.path.exists(record_path):
    os.makedirs(record_path)
plt.savefig(record_path +data_type+'_save_each_eopch_'+loss_type+'.jpg')

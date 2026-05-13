from heapq import merge
import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from medpy import metric
import time
# from .generate_labe import generate_label

def convert_labels(labels):
        labels_new = []
        for i in range(1, 14):
            one_tensor  = labels == i
            labels_new.append(one_tensor.unsqueeze(1))
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new.float()

def generate_label(input_lbl, num_classes,):
    c,h,w,d = input_lbl.shape
    # input_lbl = input_lbl.squeeze(1) #B,h,w,d
    result = torch.zeros(num_classes,h,w,d).cuda()
    input_lbl = input_lbl.long()

    ## generate binary cross entropy label and assign -1 to ignored organ
    organ_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    for i in range(num_classes):
        if (i) not in organ_list:
            result[i,] = -1
        else:
            # result[i,] = (input_lbl ==  (i+1))
            result[i,] = (input_lbl ==  (i))
    return result

def one_hot_encoder(input_tensor,num_classes):
        tensor_list = []
        for i in range(num_classes):
            temp_prob = input_tensor == i + 1# * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        # print(predict.shape,target.shape)
        target = one_hot_encoder(target,self.num_classes)
        # print(predict.shape, target.shape)
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'
        total_loss = 0
        # for b in range(B):
        for i in range(self.num_classes):
            ce_loss = self.criterion(predict[:, i], target[:, i])
            total_loss += ce_loss
        return total_loss/self.num_classes

class Multi_DiceLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(Multi_DiceLoss, self).__init__()
        self.n_classes = num_classes
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        target = one_hot_encoder(target,self.n_classes)
        # print(np.unique(target.cpu().detach().numpy()))
        if weight is None:
            weight = [1] * self.n_classes
        # print(inputs.shape, target.shape)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # class_wise_dice = []
        loss = 0.0
        # print(inputs.shape,target.shape,self.n_classes)
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceLoss_sy(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_sy, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True,one_hot = True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        # print(np.unique(target.cpu().detach().numpy()))
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # class_wise_dice = []
        loss = 0.0
        # print(inputs.shape,target.shape,self.n_classes)
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        # dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return  hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 0
    else:
        return 0

def Train_index(pred, gt,Type = 'train'):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    pred = np.where(pred > 0.5, 1., 0.)
    # pred = pred + 1
    # gt = gt.astype(float)
    #x,y,z =  y_ture.shape
    TP = np.sum(gt * pred)
    FP = np.sum(pred * (1 - gt))
    FN = np.sum((1 - pred) * (gt))
    TN = np.sum((1 - gt) * (1 - pred))

    #print(TP,FP,FN)
    #Recall not nan
    if (TP + FN) == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    #Precision not nan
    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    #F_score not nan
    if (Recall + Precision) == 0:
        F_score = 0
    else:
        F_score = (2 * Recall * Precision) / (Recall + Precision)
    #ACC not nan
    if (TP + FP + FN ) ==0:
        Iou = 0
    else:
        Iou = TP / (TP + FP + FN )

    ACC = (TP+TN) /(TP+TN+FP+FN)
    #print(ACC)
    #ACC = np.sum(y_pred == y_ture) / (x*y*z)
    if Recall is None:
        Recall = 0
    elif Precision is None:
        Precision = 0
    elif F_score is None:
        F_score = 0

    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        if Type == 'test':
            hd95 = metric.binary.hd95(pred, gt)
    elif pred.sum() > 0 and gt.sum()==0:
        dice =  0
        hd95 = 0
    else:
        dice =  0
        hd95 = 0
    # print(TP,TN,FP,FN)
    # hd95 = calculate_metric_percase(pred, gt)
    # print(Type)
    if Type == 'train':
        return round(dice,6), round(ACC,6), round(Iou,6), \
            round(F_score,6), round(Precision,6), round(Recall,6)
    elif Type == 'test':
        return round(dice,6), round(hd95,6),round(ACC,6), round(Iou,6), \
            round(F_score,6), round(Precision,6), round(Recall,6)


def dice_score(preds, labels,Type = 'train'):  # on GPU
    ### preds: w,h,d; label: w,h,d
    pred = preds
    gt = labels
    # print(pred.shape,gt.shape)
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    # preds = torch.where(preds > 0.5, 1., 0.)
    # preds = torch.where(preds > 1, 1., 0.)
    # labels = torch.where(labels > 0, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)
    # print(np.unique(predict.cpu().detach().numpy()),np.unique(target.cpu().detach().numpy()))

    TP = torch.sum(torch.mul(predict, target)).item()
    FN = torch.sum(torch.mul(predict!=1, target)).item()
    FP = torch.sum(torch.mul(predict, target!=1)).item()
    TN = torch.sum(torch.mul(predict!=1, target!=1)).item()
    
    den = (torch.sum(predict) + torch.sum(target) + 1).item()

    # print(TP,FN,FP,TN,den)


    if (TP + FN) == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    #Precision not nan
    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    #F_score not nan
    if (Recall + Precision) == 0:
        F_score = 0
    else:
        F_score = (2 * Recall * Precision) / (Recall + Precision)
    #ACC not nan
    if (TP + FP + FN ) ==0:
        Iou = 0
    else:
        Iou = TP / (TP + FP + FN )
    
    # if (TP+TN+FP+FN) == 0:
    #     ACC = 0
    # else:
    ACC = (TP+TN) /(TP+TN+FP+FN)
    #print(ACC)
    #ACC = np.sum(y_pred == y_ture) / (x*y*z)
    if Recall is None:
        Recall = 0
    elif Precision is None:
        Precision = 0
    elif F_score is None:
        F_score = 0

    dice = (2 * TP +1) / den
    # print(TP,TN,FP,FN)
    hd95 = 0
    # print(Type)
    if Type == 'train':
        return round(dice,6), round(ACC,6), round(Iou,6), \
            round(F_score,6), round(Precision,6), round(Recall,6)
    elif Type == 'test':
        # print(pred.cpu().detach().numpy().shape, gt.cpu().detach().numpy().shape)
        if len(torch.unique(preds)) > 1 and len(torch.unique(labels))  > 1:
            hd95 = metric.binary.hd95(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
        else:
            hd95 = 0.000000001
        return round(dice,6), round(hd95,6),round(ACC,6), round(Iou,6), \
            round(F_score,6), round(Precision,6), round(Recall,6)

# def Train_index_sy_gpu(image, label, classes,Type = 'train'):
#     Type = Type
#     metric_list = []
#     image,label = image.squeeze(0),label.squeeze(0)
#     label = generate_label(label,classes)
#     start = time.time()
#     # print(image.shape,label.shape)
#     for i in range(1,classes):
#         metric_list.append(dice_score(image[i], label[i],Type = Type))
#         # metric_list.append(dice_score(image==i+1, label==i+1,Type = Type))
#     end = time.time()
#     print('calculating time(s): ',end-start)

#     return metric_list

def Train_index_sy_gpu(image, label, classes,Type = 'train'):
    Type = Type
    metric_list = []
    start = time.time()
    # print(image.shape,label.shape)
    for i in range(1,classes):
        metric_list.append(dice_score(image == i, label == i,Type = Type))
    end = time.time()
    print('calculating time(s): ',end-start)

    return metric_list

def Train_index_sy_gpu_no_background(image, label, classes,Type = 'train'):
    Type = Type
    metric_list = []
    start = time.time()
    # print(torch.unique(image))
    # image = image + 1
    # print(torch.unique(image))
    # print(image.shape,label.shape)
    image = (image > 0.5).float()
    # print(torch.unique(image))
    # image = one_hot_encoder(image,classes)
    label = one_hot_encoder(label,classes)
    # print(image.shape,label.shape)
    for i in range(classes):
        # metric_list.append(dice_score(image == i+1, label == i+1,Type = Type))
        metric_list.append(dice_score(image[:,i], label[:,i],Type = Type))
    end = time.time()
    print('calculating time(s): ',end-start)

    return metric_list

def Train_index_sy_gd(output, label, classes):
    from models.diffusion_model.Diff_UNet.BTCV.light_training.evaluation.metric import dice, hausdorff_distance_95
    output = (output > 0.5).float().cpu().numpy()
    label = convert_labels(label)
    target = label.cpu().numpy()
    print(output.shape,target.shape)
    dices = []
    hd = []
    c = classes
    for i in range(0, c):
        pred_c = output[:, i]
        target_c = target[:, i]

        dices.append(dice(pred_c, target_c))
    print(sum(dices)/c)
    # return dices


def Train_index_sy(image, label, classes,Type = 'train'):
    Type = Type
    metric_list = []
    # label = generate_label(label,classes)
    start = time.time()
    # print(image.shape,label.shape)
    for i in range(1,classes):
        metric_list.append(Train_index(image == i, label == i,Type = Type))
    end = time.time()
    print('calculating time(s): ',end-start)
    # print(len(metric_list))
  
    return metric_list
import pandas as pd
import torch
import numpy as np
from utils.metrics import Train_index, Train_index_gpu, Train_index_gpu_no_background
import os
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from utils.dataloader import get_loader

def val(args, model, logger, epoch, checkpoint, main_data_path, \
        save_path, max_dice, weight_patch, device, save_pred=None):
    torch.cuda.empty_cache()
    logger.info('\n')
    logger.info('================testing================')

    # ======================================parser args==========================================
    num_classes = args.num_classes
    dataset_type = args.dataset_type
    image_size = args.image_size
    in_chan = args.patch_size
    data_type = 'crop_' + str(image_size) + '_patch_' + str(in_chan)
    # ======================================input path==========================================
    # input path
    test_path = main_data_path + 'test_data/' + data_type + '/'
    logger.info('data input patch:{}'.format(test_path))

    # ======================================label and indicator==========================================
    # indicator_list=['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
    indicator_list = ['Dice', 'ACC', 'Iou', 'F_score', 'Precision', 'Recall']

    # ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
    #             'Liver', 'Stomach', 'Arota',  'Inferior Vena Cava',
    #             'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland','Duodenum','Bladder','Prostate','Background']
    label_list = ['kys', 'KiT', "Avge"]
    # print(len(label_list))

    # =======================================metric========================================
    # post_label = AsDiscrete(to_onehot=num_classes)
    # post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # ======================================load model==========================================
    # weigth_path = weight_patch + str(epoch)+'.pth'
    # logger.info(weigth_path)

    # model = DUM(out_channels = num_classes)
    # checkpoint = torch.load(weigth_path)
    # # model.load_state_dict(checkpoint['model'],strict=False)
    # model.load_state_dict(checkpoint['model'])
    # model.to(device)

    # ======================================testing==========================================
    best_epoch = 0
    metric_list = 0.0
    model.eval()
    args.phase = 'test'
    test_loader, data_dicts_test = get_loader(args, logger)
    test_len = len(test_loader)
    with torch.no_grad():
        # for image_input, label_input in tqdm(test_loader):      #single case slice
        # index = 0
        # for batch in tqdm(test_loader):
        for index, batch in enumerate(test_loader):
            image, label = batch["image"], batch["label"]
            image_input, label_input = image.float().to(device), label.float().squeeze(1).to(device)
            # input:B,C,H,W
            # print(image_input.shape,label_input.shape)
            # raw_shape = image_input.shape
            output = sliding_window_inference(image_input, (96, 96, 96), 4, model, overlap=0.75)
            # pred = torch.argmax(torch.softmax(output, dim=1), dim=1) #1,H,W
            # pred = torch.softmax(output,dim=1)
            pred = torch.sigmoid(output)
            # pred = threshold_organ(pred_sigmoid)
            # pred = pred.cuda()
            # print(pred.shape,label_input.shape)
            # print(np.unique(pred),np.unique(mask))
            test_name = data_dicts_test[index]["image"][-8:-4]
            # print(data_dicts_test[index]["image"],test_name)
            if save_pred is not None:
                prd_itk = sitk.GetImageFromArray(pred.astype(np.float32))
                # sitk.WriteImage(prd_itk, save_path + '/' + test_name + "_pred.nii")

            # =================single indicators===============
            # metric_ = Train_index_sy_gpu_no_background(pred, label_input, num_classes,'train')    #single case indicators
            ###metric_ = Train_index_sy_gpu_no_background(pred, label_input, num_classes,'train')
            metric_ = Train_index_gpu_no_background(pred, label_input, num_classes, 'train')
            logger.info('case %s mean_dice %f ' % (test_name, np.mean(metric_, axis=0)[0]))
            metric_list += np.array(metric_)  # count all case indicators
    metric_list = metric_list / test_len  # mean all case indicators
    # metric_list.tolist()
    # print(metric_list.shape)
    # metric_list[-2] = (metric_list[-2] + metric_list[-1]) /2
    # metric_list = np.delete(metric_list,-1,axis=0)
    # =================mean all case indicators===============
    logger.info('total %d case mean' % (test_len))
    logger.info('\t\tmean_dice\tmacc\t\tmIou\t\tmF_score\tmPrecision\tmRecall ')
    for i in range(num_classes):
        ii = i
        logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f' %
                    (label_list[ii][:4], metric_list[ii][0],
                     metric_list[ii][1], metric_list[ii][2],
                     metric_list[ii][3], metric_list[ii][4],
                     metric_list[ii][5]))
    # =================avg mean class(所有病例的平均)===============
    Index = np.mean(metric_list, axis=0)
    logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f' %
                (label_list[-1], round(Index[0], 6),
                 round(Index[1], 6), round(Index[2], 6),
                 round(Index[3], 6), round(Index[4], 6),
                 round(Index[5], 6)))
    best_metric_list = []
    # ================================save best dice model====================================
    logger.info('max_dice %f ' % (round(max_dice, 6)))
    if max_dice < Index[0]:
        max_dice = Index[0]
        best_metric_list = metric_list.tolist()
        best_metric_list.append(Index.tolist())
        best_epoch = epoch
        best_metric_list = np.array(best_metric_list)
        best_metric_list = np.round(best_metric_list, 6)
        logger.info('best dice epoch {}'.format(epoch))
        # logger.info('best dice hd95 {}  {}'.format(Index[0], Index[1]))
        logger.info('best dice {}'.format(Index[0].round(6)))

        best_dice_dict = {"best_dice": max_dice}
        checkpoint.update(best_dice_dict)
        if not os.path.exists(weight_patch):
            os.makedirs(weight_patch)
        torch.save(checkpoint, weight_patch + 'best_model.pth')
        logger.info('save model, best epoch {} '.format(epoch))

        # print(best_metric_list)
        save_csv = pd.DataFrame(best_metric_list, columns=indicator_list)
        save_csv.insert(0, 'Class', label_list)
        save_csv.insert(len(indicator_list) + 1, 'Best_epoch', best_epoch)
        save_csv.to_csv(save_path + dataset_type + '.csv', index=False, sep=',')
    print('\n')
    return max_dice







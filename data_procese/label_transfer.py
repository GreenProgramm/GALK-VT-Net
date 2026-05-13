import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum
import os


def generate_label(input_lbl, num_classes,):
    """
    Convert class index tensor to one hot encoding tensor with -1 (ignored).
    Args:
         input: A tensor of shape [bs, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    Comment: spleen to 0
    """
    # shape = np.array(input_lbl.shape)
    # shape[1] = num_classes
    # shape = tuple(shape)
    b,h,w = input_lbl.shape
    result = np.zeros((b,num_classes,h,w))
    # input_lbl = input_lbl.long()

    ## generate binary cross entropy label and assign -1 to ignored organ
    B = result.shape[0]
    organ_list = [1,2,3,4,5,6,7,8]
    for b in range(B):
        for i in range(num_classes):
            if (i+1) not in organ_list:
                result[b, i] = -1
            else:
                result[b, i] = (input_lbl[b] ==  (i+1))
    return result

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")

main_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/re_nor/images/0002.npy'
data = np.load(main_path).astype(float)
print(data.shape)
data = torch.from_numpy(data).cuda()
mask_dec4 = F.interpolate(data.unsqueeze(0).unsqueeze(0), size=(300,300, 100), mode='trilinear', align_corners=True)
img = mask_dec4[0,0,:,:,70].cpu().detach().numpy()
print(img.shape)
plt.figure()
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.imshow(img,cmap = "gray")

# out = generate_label(np.expand_dims(img,axis=0),8)
# print(out.shape)
# plt.figure()
# for i in range(out.shape[1]):
#     plt.subplot(3,3,i+1,facecolor = "white")
#     plt.axis('off')
#     plt.xticks([])
#     plt.yticks([])
#     img = np.squeeze(out[:,i,:,:])
#     print(np.unique(img))
#     plt.imshow(img,cmap = "gray")

plt.show()



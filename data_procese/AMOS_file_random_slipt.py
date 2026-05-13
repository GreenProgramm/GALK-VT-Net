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

def FL(data):
    data = list(data)
    data_new = set(data)
    if len(data_new) == len(data):
        print('no chongfu!')
    else:
        print('have chongfu!')


file_name = 'AMOS_total.txt'
total = []
with open(file_name,"r") as f:  # 设置文件对象
    for ann in f.readlines():
        data = ann.strip('\n')        
        total.append(data)

total = np.array(total)
sample_num = 40
val_test_index = np.random.choice(total.shape[0],sample_num,replace=False)
val_test_index.sort()
val_test = total[val_test_index]


total_index = np.arange(total.shape[0])
train_index = np.delete(total_index,val_test_index)
train = total[train_index]
print(train,len(train))
name = []
file_name = 'AMOS_train.txt'
for file in train:
            #print(name_64)
        with open(file_name, 'a+') as f:  # 设置文件对象
            f.write(file)  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))


print(val_test,len(val_test))
print('\n')
#val
sample_num = 20
val_index = np.random.choice(val_test.shape[0],sample_num,replace=False)
val_index.sort()
val = val_test[val_index]
print(val,len(val))
name = []
file_name = 'AMOS_val.txt'
for file in val:
            #print(name_64)
        with open(file_name, 'a+') as f:  # 设置文件对象
            f.write(file)  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))

test_index = np.arange(val_test.shape[0])
test_index = np.delete(test_index,val_index)
test = val_test[test_index]
print(test,len(test))
name = []
file_name = 'AMOS_test.txt'
for file in test:
            #print(name_64)
        with open(file_name, 'a+') as f:  # 设置文件对象
            f.write(file)  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))

FL(train)
FL(val)
FL(test)
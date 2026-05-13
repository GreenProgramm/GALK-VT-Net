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
    RandZoomd,
    RandCropByLabelClassesd,
    EnsureChannelFirstd,
    RandAffined,
)
# import logging
# logging.disable(logging.WARNING)
import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import glob
import os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
# from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset, load_decathlon_datalist
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['post_label'] = data[0]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key not in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_
def generate_label(input_lbl, num_classes=8,):
    """
    Convert class index tensor to one hot encoding tensor with -1 (ignored).
    Args:
         input: A tensor of shape [bs, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    Comment: spleen to 0
    """
    _,_,h,w,d = input_lbl.shape
    result = torch.zeros(num_classes,1,1,h,w,d).cuda()
    input_lbl = input_lbl.long()

    ## generate binary cross entropy label and assign -1 to ignored organ
    organ_list = [1,2,3,4,5,6,7,8]
    for i in range(num_classes):
        if (i+1) not in organ_list:
            result[i,] = -1
        else:
            result[i,] = (input_lbl ==  (i+1))
    return result

def get_loader(args,logger):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        ]
    )

    # # print(args.roi_x, args.roi_y, args.roi_z)
    # train_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image", "label"]), #0
    #         EnsureChannelFirstd(keys=["image", "label"]),
    #         Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         CropForegroundd(keys=["image", "label"], source_key="image"),
    #         # RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest']), # 7
    #         RandCropByPosNegLabeld(
    #             keys=["image", "label"],
    #             label_key="label",
    #             spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
    #             pos=1,
    #             neg=1,
    #             num_samples=args.num_samples,
    #             image_key="image",
    #             image_threshold=0,
    #         ), # 8
    #          RandShiftIntensityd(
    #             keys=["image"],
    #             offsets=0.10,
    #             prob=0.20,
    #         ),
    #         RandAffined(
    #                 keys=['image', 'label'],
    #                 mode=('bilinear', 'nearest'),
    #                 prob=1.0, spatial_size=(args.roi_x, args.roi_y, args.roi_z),
    #                 rotate_range=(0, 0, np.pi / 30),
    #                 scale_range=(0.1, 0.1, 0.1)
    #                 ),
    #         RandFlipd(
    #             keys=["image", "label"],
    #             spatial_axis=[0],
    #             prob=0.10,
    #         ),
    #         RandFlipd(
    #             keys=["image", "label"],
    #             spatial_axis=[1],
    #             prob=0.10,
    #         ),
    #         RandFlipd(
    #             keys=["image", "label"],
    #             spatial_axis=[2],
    #             prob=0.10,
    #         ),
    #         RandRotate90d(
    #             keys=["image", "label"],
    #             prob=0.10,
    #             max_k=3,
    #         ),
    #         RandShiftIntensityd(
    #             keys=["image"],
    #             offsets=0.10,
    #             prob=0.20,
    #         ),
    #
    #         ToTensord(keys=["image", "label"]),
    #     ]
    # )
    #
    # val_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image", "label"]),
    #         EnsureChannelFirstd(keys=["image", "label"]),
    #         # Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         CropForegroundd(keys=["image", "label"], source_key="image"),
    #         ToTensord(keys=["image", "label"]),
    #     ]
    # )
    # test_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image", "label"]),
    #         EnsureChannelFirstd(keys=["image", "label"]),
    #         # Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         CropForegroundd(keys=["image", "label"], source_key="image"),
    #         ToTensord(keys=["image", "label"]),
    #     ]
    # )


    if args.phase == 'train':
        data_dir = "../abdomen/"
        split_json = "/dataset_0.json"

        datasets = data_dir + split_json
        datalist = load_decathlon_datalist(datasets, True, "training")
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=24,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        # train_samples = {}
        # train_img = sorted(glob.glob(os.path.join(args.data_root_path, 'images/', '*.npy')))
        # train_label = sorted(glob.glob(os.path.join(args.data_root_path, 'masks/', '*.npy')))
        # # print(train_img[0])
        # train_samples['images'] = train_img
        # train_samples['labels'] = train_label
        # data_dicts_train= [
        # {"image": image_name, "label": label_name}
        # for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
        # ]
        # # data_dicts_train= [
        # # {"image": train_samples['images'][0], "label": train_samples['labels'][0]}
        # # ]
        # print('train len {}'.format(len(data_dicts_train)))
        # logger.info('train len {}'.format(len(data_dicts_train)))
        #
        # train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        #
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        #                             collate_fn=list_data_collate)
        return train_loader


    if args.phase == 'val':
        data_dir = "../abdomen/"
        split_json = "/dataset_0.json"

        datasets = data_dir + split_json
        val_files = load_decathlon_datalist(datasets, True, "validation")
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        test_samples = {}
        test_img = sorted(glob.glob(os.path.join(args.data_root_path.replace('Training','Valing'), 'images/', '*.npy')))
        test_label = sorted(glob.glob(os.path.join(args.data_root_path.replace('Training','Valing'), 'masks/', '*.npy')))
        test_samples['images'] = test_img
        test_samples['labels'] = test_label
        data_dicts_test= [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(test_samples['images'], test_samples['labels'])
        ]
        # print('val len {}'.format(len(data_dicts_test)))
        logger.info('val len {}'.format(len(data_dicts_test)))

        return val_loader,data_dicts_test
#        if args.cache_dataset:
#            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
#        else:
#            test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
#        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=list_data_collate)
#        return test_loader,data_dicts_test
    if args.phase == 'test':

        test_samples = {}
        test_img = sorted(glob.glob(os.path.join(args.data_root_path.replace('Training','Testing'), 'images/', '*.npy')))
        test_label = sorted(glob.glob(os.path.join(args.data_root_path.replace('Training','Testing'), 'masks/', '*.npy')))
        test_samples['images'] = test_img
        test_samples['labels'] = test_label
        data_dicts_test= [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(test_samples['images'], test_samples['labels'])
        ]
        # print('val len {}'.format(len(data_dicts_test)))
        logger.info('test len {}'.format(len(data_dicts_test)))


        if args.cache_dataset:
            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return test_loader,data_dicts_test

if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()
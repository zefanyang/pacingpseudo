#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/18/2021 4:34 PM
# @Author: yzf
import copy
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from datasets.augmentations import *

class ACDCDataset(Dataset):
    classnames = {
        0: "background",
        1: "right ventricle",
        2: "myocardium",
        3: "left ventricle",
        4: "unknown"
    }
    def __init__(self,
                 file_ls,
                 num_classes,
                 dtype=np.float32):
        super(ACDCDataset, self).__init__()
        self.file_ls = file_ls
        self.num_classes = num_classes
        self.dtype = dtype

    def __len__(self):
        return len(self.file_ls)

    def __getitem__(self, idx):
        names_to_data = _load_npz(self.file_ls[idx], self.num_classes, self.dtype)
        return names_to_data

class ACDCTwoStream(ACDCDataset):
    def __init__(self,
                 file_ls,
                 num_classes,
                 base_transforms=None,
                 strong_transforms=None,
                 do_strong=False,):
        super(ACDCTwoStream, self).__init__(file_ls, num_classes)
        self.base_transforms = base_transforms
        self.strong_transforms = strong_transforms
        self.do_strong = do_strong
        self.to_torch_format = \
            ToTorchTensor(
                num_classes=num_classes,
                one_hot_encoding=True,
            )

    def __getitem__(self, item):
        """We make this function return outputs containing data under base transforms and
        data under further strong transforms"""
        names_to_data = super(ACDCTwoStream, self).__getitem__(item)

        assert self.base_transforms is not None and isinstance(self.base_transforms, list)
        # Do base transforms
        for b_trans in self.base_transforms:
            # The names_to_data is mutable and changed by the function, so there is actually no need to assign the value.
            names_to_data = b_trans(names_to_data)

        # Do strong transforms
        if self.do_strong:
            data_strong = copy.deepcopy(names_to_data)
            for s_trans in self.strong_transforms:
                if isinstance(s_trans, Mixup):
                    data_strong = s_trans(data_strong, self.file_ls)
                else:
                    data_strong = s_trans(data_strong)
            data_strong = self.to_torch_format(data_strong)

        # Base data to tensor
        names_to_data = self.to_torch_format(names_to_data)

        # Merge strong data
        if self.do_strong:
            # The update fn returns None ...
            names_to_data.update({
                'image_strong': data_strong['image'],
                'label_strong': data_strong['label'],
                'scribble_strong': data_strong['scribble'],
            })

        return names_to_data

def _load_npz(file, num_classes, as_dtyp=np.float32):
    data = np.load(file)
    uid = str(data['uid'])
    img = data['img'].astype(as_dtyp)  # h, w
    lab = data['lab'].astype(as_dtyp)  # h, w
    scb = data['scb'].astype(as_dtyp)  # h, w
    names_to_data = {
        'uid': uid,
        'image': img,
        'label': lab,
        'scribble': scb,
    }
    return names_to_data

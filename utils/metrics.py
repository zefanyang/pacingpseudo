#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/21/2021 1:41 PM
# @Author: yzf
import numpy as np

def compute_dice(input, target):
    """
    Compute Dice values over classes of a single sample

    :param input: softmax values of a shape C×H×W
    :param target: C×H×W
    :return:
    """
    assert input.shape == target.shape
    eps = 1e-5
    no_classes = input.shape[0]
    input_ = np.argmax(input, axis=0)
    input_ = _to_one_hot_encoding(input_, no_classes)

    dice_ls = []
    for c in range(no_classes):
        input_c = input_[c].reshape(-1,)
        target_c = target[c].reshape(-1,)

        input_empty = np.all(input_c == 0)
        target_empty = np.all(target_c == 0)
        if input_empty and target_empty:
            dice_ls.append(np.nan)
        else:
            up_part = 2* np.sum(input_c * target_c)
            down_part = np.sum(input_c) + np.sum(target_c) + eps
            dice_ls.append(up_part / down_part)
    return dice_ls

def _to_one_hot_encoding(image, no_classes=None):
    if no_classes is None:
        no_classes = np.unique(image).size
    image_one_hot = np.zeros((no_classes, *image.shape))
    for c in range(no_classes):
        image_one_hot[c][image == c] = 1
    return image_one_hot

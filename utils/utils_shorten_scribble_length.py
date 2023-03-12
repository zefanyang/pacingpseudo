#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2/25/2022 8:37 PM
# @Author: yzf
import math
import numpy as np
import torch
import torch.nn.functional as F

# Define endpoint elements
element1 = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
element2 = np.rot90(element1, k=1)
element3 = np.rot90(element1, k=2)
element4 = np.rot90(element1, k=3)
element5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
element6 = np.rot90(element5, k=1)
element7 = np.rot90(element5, k=2)
element8 = np.rot90(element5, k=3)
element_ls = [element1, element2, element3, element4, element5, element6, element7, element8]
kernels = []
for e in element_ls:
    e[e == 0] = 1000
    kernels.append(torch.from_numpy(e.copy()[None][None]).float())

def one_hot(mask, num_classes):
    h, w = mask.shape
    mask_oh = np.zeros((num_classes, h, w))
    for c in range(num_classes):
        mask_oh[c][mask==c] = 1
    return mask_oh

def delete_endpoints(image, unknown, length, ratio):
    """

    :param image: torch.tensor, (1, 1, h, w)
    :param unknown: torch.tensor, (1, 1, h, w)
    :param length: int
    :param ratio: float
    """
    _, _, h, w = image.shape
    # while image.sum() > int(length * ratio):
    while True:
        endpoints = detect_endpoints(image)

        # If there is no endpoint, assign the first foreground pixel.
        if not endpoints.sum():
            _, _, row, col = np.where(image==1)
            endpoints[0][0][row[0]][col[0]] = 1.

        # Delete the endpoint
        flag2 = False
        _, _, row, col = np.where(endpoints==1)
        for i, j in zip(row, col):
            if image.sum() > math.ceil(length*ratio):
                image[0][0][i][j] = 0.
                unknown[0][0][i][j] = 1.
            else:
                flag2 = True
                break
        # Break the while loop
        if flag2:
            break

def detect_endpoints(image):
    """

    :param image: (1, 1, h, w)
    :return: (1, 1, h, w)
    """
    _, _, h, w = image.shape
    endpoints = torch.zeros((1, 1, h, w))
    for kernel in kernels:
        output = F.conv2d(image, kernel, padding=1)
        endpoints += (output == 2).float()
    return endpoints
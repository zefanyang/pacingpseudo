#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 10/30/2021 12:48 PM
# @Author: yzf
"""Data augmentation configurations for ACDC heart dataset."""
from collections import namedtuple
from datasets.augmentations import *

NUM_CLASSES = 4
IGNORED_INDEX = 4
INPUT_SIZE = (224, 224)
STRENGTH = 1.
Transforms = namedtuple('Transforms', ['base_transforms', 'strong_transforms'])

# Base transformation
base_transforms = [
        MeanStdNorm(),
        Scaling(
            scale_range=(0.7, 1.4),
            num_classes=NUM_CLASSES,
            image_scale_order=3,
            label_scale_order=1,
            p=0.2,
        ),
        ElasticTransform(
            sigma_range=(9., 13.),
            alpha_range=(0., 200.),
            img_order=3,
            lab_order=0,
            mode='nearest',
            clip=True,
            p=0.2,
        ),
        RandomRotation(
            degree_range=(-30, 30),
            image_interp_order=3,
            image_padding_val=0,
            label_interp_order=0,
            label_padding_val=IGNORED_INDEX,
            p=0.2,
        ),
        Mirroring(
            axis=0,
            p=0.5
        ),
        Mirroring(
            axis=1,
            p=0.5
        ),
        GaussianNoise(
            noise_scale_range=(0, 0.1),
            p=0.15,
        ),
        MeanStdNorm(),
        RandomCrop(
            crop_size=INPUT_SIZE,
            image_padding_value=0,
            label_padding_value=IGNORED_INDEX,
            p=1.
        ),
    ]

class TransformsColor(object):
    def __init__(self, strength=1.):
        self.base_transforms = base_transforms
        self.strong_transforms = self.get_strong_transforms(strength)

    @staticmethod
    def get_strong_transforms(strength):
        color = [
                Brightness(
                    scale_range=(-strength*0.8, strength*0.8),
                    p=0.8
                ),
                Contrast(
                    scale_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    p=0.8
                ),
                GammaAugmentation(
                    gamma_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    retain_stats=True,
                    invert_data=False,
                    p=0.8
                ),
            ]
        return color

class TransformsColorBlur(object):
    def __init__(self, strength=1.):
        self.base_transforms = base_transforms
        self.strong_transforms = self.get_strong_transforms(strength)

    @staticmethod
    def get_strong_transforms(strength):
        color = [
                Brightness(
                    scale_range=(-strength*0.8, strength*0.8),
                    p=0.8
                ),
                Contrast(
                    scale_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    p=0.8
                ),
                GammaAugmentation(
                    gamma_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    retain_stats=True,
                    invert_data=False,
                    p=0.8
                ),]
        blur = [GaussianBlur(kernel_scale_range=(1, 1.5), p=0.8)]
        return color + blur

class TransformsColorMixup(object):
    def __init__(self, strength=1.):
        self.base_transforms = base_transforms
        self.strong_transforms = self.get_strong_transforms(strength)

    @staticmethod
    def get_strong_transforms(strength):
        color = [
                Brightness(
                    scale_range=(-strength*0.8, strength*0.8),
                    p=0.8
                ),
                Contrast(
                    scale_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    p=0.8
                ),
                GammaAugmentation(
                    gamma_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    retain_stats=True,
                    invert_data=False,
                    p=0.8
                ),]
        mixup = [Mixup(lam_range=(0.8, 1.), p=0.8)]
        return color + mixup

class TransformsColorLow(object):
    def __init__(self, strength=1.):
        self.base_transforms = base_transforms
        self.strong_transforms = self.get_strong_transforms(strength)

    @staticmethod
    def get_strong_transforms(strength):
        color = [
                Brightness(
                    scale_range=(-strength*0.8, strength*0.8),
                    p=0.8
                ),
                Contrast(
                    scale_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    p=0.8
                ),
                GammaAugmentation(
                    gamma_range=(max(0., 1-strength*0.8), 1+strength*0.8),
                    retain_stats=True,
                    invert_data=False,
                    p=0.8
                ),]
        low = [SimulationLowRes(downscale_range=(1.5, 2), down_order=0, up_order=3, p=0.8,)]
        return color + low
#
# class TransformsColorCutout(object):
#     def __init__(self, strength=1.):
#         self.base_transforms = base_transforms
#         self.strong_transforms = self.get_strong_transforms(strength)
#
#     @staticmethod
#     def get_strong_transforms(strength):
#         color = [
#                 Brightness(
#                     scale_range=(-strength*0.8, strength*0.8),
#                     p=0.8
#                 ),
#                 Contrast(
#                     scale_range=(max(0., 1-strength*0.8), 1+strength*0.8),
#                     p=0.8
#                 ),
#                 GammaAugmentation(
#                     gamma_range=(max(0., 1-strength*0.8), 1+strength*0.8),
#                     retain_stats=True,
#                     invert_data=False,
#                     p=0.8
#                 ),]
#         cutout = [Cutout(length=32, p=0.8)]
#         return color + cutout
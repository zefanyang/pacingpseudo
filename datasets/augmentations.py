#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 10/22/2021 8:23 PM
# @Author: yzf
import numpy as np
import cv2
import scipy.ndimage
import skimage.transform
import torch

class MeanStdNorm(object):
    """Standard normalization"""
    def __init__(self,):
        self.eps = 1e-8

    def __call__(self, names_to_data):
        image = names_to_data['image']
        mean_ = np.mean(image)
        std_ = np.std(image)
        names_to_data['image'] = (image - mean_) / (std_ + self.eps)
        return names_to_data

class Cutout(object):
    """Cutout"""
    def __init__(self, length=32, p=0.2):
        self.length = length
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            h, w = image.shape
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length//2, 0, h)
            y2 = np.clip(y + self.length//2, 0, h)
            x1 = np.clip(x - self.length//2, 0, w)
            x2 = np.clip(x + self.length//2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            image  = image * mask

            names_to_data['image'] = image
            return names_to_data

class Mixup(object):
    """Mixup"""
    def __init__(self, lam_range=(0.8, 1.), p=0.2):
        self.lam_range = lam_range
        self.p = p
        return

    def __call__(self, names_to_data, file_ls):
         if not np.random.uniform() < self.p:
             return names_to_data
         else:
             lam = np.random.uniform(self.lam_range[0], self.lam_range[1])

             image1 = names_to_data.get('image')
             file = np.load(np.random.choice(file_ls))
             image2 = file.get('img').astype(np.float32)
             if not image1.shape == image2.shape:
                 h, w = image1.shape
                 image2 = self.center_crop(image2, h, w)
             image2 = (image2 - image2.mean()) / max(image2.std(), 1e-8)
             image = image1 * lam + image2 * (1-lam)

             names_to_data['image'] = image
             return names_to_data

    @staticmethod
    def center_crop(image, h, w):
        """Not optimal implementation"""
        h0, w0 = image.shape
        y, x = h0//2, w0//2
        return image[y-h//2: y+h//2, x-w//2: x+w//2]

class GaussianBlur(object):
    """Gaussian blur"""
    def __init__(self, kernel_scale_range=(0.5, 1.5), p=0.2):
        self.kernel_scale_range = kernel_scale_range
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            scale = np.random.uniform(self.kernel_scale_range[0], self.kernel_scale_range[1])
            names_to_data['image'] = scipy.ndimage.gaussian_filter(image, scale, order=0)
            return names_to_data

class Brightness(object):
    """Adjusting brightness"""
    def __init__(self, scale_range=(-0.1, 0.1), p=0.15):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            names_to_data['image'] = image + scale
            return names_to_data

class Contrast(object):
    """Adjusting contrast"""
    def __init__(self, scale_range=(0.65, 1.5), p=0.15):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            mean_ = np.mean(image)
            max_ = np.max(image)
            min_ = np.min(image)
            names_to_data['image'] = np.clip((image-mean_)*scale + mean_, min_, max_)
            return names_to_data

class GammaAugmentation(object):
    """Gamma augmentation"""
    def __init__(self, gamma_range=(0.7, 1.5), retain_stats=True, invert_data=False, p=0.15):
        self.eps = 1e-8
        self.gamma_range = gamma_range
        self.retain_stats = retain_stats
        self.invert_data = invert_data
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            if self.invert_data:
                image = -image

            mean_ = np.mean(image)
            std_ = np.std(image)
            max_ = np.max(image)
            min_ = np.min(image)

            if np.random.uniform() < 0.5 and self.gamma_range[0] < 1.:  # bias the uniform selection of gamma, as gamma smaller and larger than 1 have different properties.
                gamma = np.random.uniform(self.gamma_range[0], 1.)  # half-open interval [0.7, 1)
            else:
                gamma = np.random.uniform(max(1., self.gamma_range[0]), self.gamma_range[1])

            image = np.power((image - min_) / (max_ - min_ + self.eps), gamma)

            if self.retain_stats:
                image = (image - np.mean(image)) / (np.std(image) + self.eps)
                image = image * std_ + mean_
            if self.invert_data:
                image = -image
            names_to_data['image'] = image
            return names_to_data

class SimulationLowRes(object):
    """Simulation of low resolution image"""
    def __init__(self, downscale_range=(1, 2), down_order=0, up_order=3, clip=True, p=0.25):
        self.downscale_range = downscale_range
        self.down_order = down_order
        self.up_order = up_order,
        self.clip = clip
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            h, w = image.shape
            scale = np.random.uniform(self.downscale_range[0], self.downscale_range[1])
            new_h = round(1 / scale * h)
            new_w = round(1 / scale * w)
            image = skimage.transform.resize(image, (new_h, new_w), self.down_order, clip=self.clip)
            image = skimage.transform.resize(image, (h, w), self.up_order, clip=self.clip)
            names_to_data['image'] = image
            return names_to_data

class Scaling(object):
    """Resizing"""
    def __init__(self, scale_range=(0.7, 1.4),
                 num_classes=4,
                 image_scale_order=3,
                 label_scale_order=1,
                 clip=True,
                 p=0.2):
        self.scale_range = scale_range
        self.num_classes = num_classes
        self.image_scale_order = image_scale_order
        self.label_scale_order = label_scale_order
        self.clip = clip
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image, label, scb = names_to_data['image'], names_to_data['label'], names_to_data['scribble']
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            h, w = image.shape
            new_h, new_w = round(scale*h), round(scale*w)
            image = skimage.transform.resize(image, (new_h, new_w), order=self.image_scale_order, clip=self.clip)

            label = to_one_hot_encoding(label, self.num_classes)
            label_ls = []
            for lab in label:
                lab = skimage.transform.resize(lab, (new_h, new_w), order=self.label_scale_order, clip=self.clip)
                label_ls.append(lab)
            label = np.argmax(label_ls, 0)
            scb = to_one_hot_encoding(scb, self.num_classes+1)  # CAREFUL
            scb_ls = []
            for s in scb:
                s = skimage.transform.resize(s, (new_h, new_w), order=self.label_scale_order, clip=self.clip)
                scb_ls.append(s)
            scb = np.argmax(scb_ls, 0)

            names_to_data['image'], names_to_data['label'], names_to_data['scribble'] = image, label, scb
            return  names_to_data

class ElasticTransform(object):
    """Elastic transformation"""
    def __init__(self,
                 sigma_range=(9., 13.),
                 alpha_range=(0., 200.),
                 img_order=3,
                 lab_order=0,
                 mode='nearest',
                 clip=True,
                 p=0.2,
                 ):
        self.sigma_range = sigma_range
        self.alpha_range = alpha_range
        self.img_order = img_order
        self.lab_order = lab_order
        self.clip = clip
        self.mode = mode
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image, label, scb = names_to_data['image'], names_to_data['label'], names_to_data['scribble']
            h, w = image.shape
            if self.clip:
                min_ = image.min()
                max_ = image.max()
            # width of the Gaussian filter and multiplier
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            # displacement
            dx = scipy.ndimage.gaussian_filter(np.random.rand(h, w)*2-1, sigma) * alpha
            dy = scipy.ndimage.gaussian_filter(np.random.rand(h, w)*2-1, sigma) * alpha
            # mesh grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_dx, y_dy = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
            # map coordinates -- o(x, y) = f(i(c(x, y)))
            image = scipy.ndimage.map_coordinates(image, (y_dy, x_dx), order=self.img_order, mode=self.mode).reshape(h, w)
            if self.clip:
                image = np.clip(image, min_, max_)
            label = scipy.ndimage.map_coordinates(label, (y_dy, x_dx), order=self.lab_order, mode=self.mode).reshape(h, w)
            scb = scipy.ndimage.map_coordinates(scb, (y_dy, x_dx), order=self.lab_order, mode=self.mode).reshape(h, w)
            # update data
            names_to_data['image'], names_to_data['label'], names_to_data['scribble'] = image, label, scb
        return names_to_data

class RandomRotation(object):
    """Random rotation"""
    def __init__(self, degree_range=(-180, 180),
                 image_interp_order=3,
                 image_padding_val=0,
                 label_interp_order=0,
                 label_padding_val=4,
                 p=0.2):
        self.inter_table = {
            0: cv2.INTER_NEAREST,
            1: cv2.INTER_LINEAR,
            3: cv2.INTER_CUBIC,
        }
        self.degree_range = degree_range
        self.image_interp_order = image_interp_order
        self.image_padding_val = image_padding_val
        self.label_interp_order = label_interp_order
        self.label_padding_val = label_padding_val  # CAREFUL
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image, label, scb = names_to_data['image'], names_to_data['label'], names_to_data['scribble']
            h, w = image.shape
            degree = np.random.uniform(self.degree_range[0], self.degree_range[1], )
            matrix = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=degree, scale=1.)
            image = cv2.warpAffine(image, matrix, (w, h), flags=self.inter_table[self.image_interp_order],
                                   borderValue=self.image_padding_val)
            label = cv2.warpAffine(label, matrix, (w, h), flags=self.inter_table[self.label_interp_order],
                                   borderValue=self.label_padding_val)
            scb = cv2.warpAffine(scb, matrix, (w, h), flags=self.inter_table[self.label_interp_order],
                                 borderValue=self.label_padding_val)

            names_to_data['image'] = image
            names_to_data['label'] = label
            names_to_data['scribble'] = scb
            return names_to_data

class Rotation90(object):
    """Rotating image 90 degrees"""
    def __init__(self, rot_choices=(1, 2, 3), axes=(0, 1), p=0.2):
        self.rot_choices = rot_choices
        self.axes = axes
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image, label, scb = names_to_data['image'], names_to_data['label'], names_to_data['scribble']
            num_rots = np.random.choice(self.rot_choices)
            names_to_data['image'] = np.rot90(image, num_rots, axes=self.axes)
            names_to_data['label'] = np.rot90(label, num_rots, axes=self.axes)
            names_to_data['scribble'] = np.rot90(scb, num_rots, axes=self.axes)
            return names_to_data

class Mirroring(object):
    """Flipping"""
    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image, label, scb = names_to_data['image'], names_to_data['label'], names_to_data['scribble']
            names_to_data['image'] = np.flip(image, self.axis)
            names_to_data['label'] = np.flip(label, self.axis)
            names_to_data['scribble'] = np.flip(scb, self.axis)
            return names_to_data

class GaussianNoise(object):
    """Additive Gaussian Noise"""
    def __init__(self, noise_scale_range=(0, 0.1), p=0.15):
        self.noise_scale_range = noise_scale_range
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image = names_to_data['image']
            scale = np.random.uniform(self.noise_scale_range[0], self.noise_scale_range[1])
            names_to_data['image'] = image + np.random.normal(0., scale, size=image.shape)
            return names_to_data

class RandomCrop(object):
    """Randomly cropping"""
    def __init__(self, crop_size, image_padding_value=0, label_padding_value=4, p=1.):
        self.crop_size = crop_size
        self.image_padding_value = image_padding_value
        self.label_padding_value = label_padding_value  # CAREFUL Use the ignored value
        self.p = p

    def __call__(self, names_to_data):
        if not np.random.uniform() < self.p:
            return names_to_data
        else:
            image, label, scb = names_to_data['image'], names_to_data['label'], names_to_data['scribble']
            h, w = image.shape
            crop_h, crop_w = self.crop_size
            w_margin = w - crop_w
            h_margin = h - crop_h

            # Crop image if image width is larger than canvas width, otherwise embed image.
            if w_margin > 0:
                image_left = np.random.randint(w_margin + 1)  # CAREFUL
                canvas_left = 0
            else:
                image_left = 0
                canvas_left = np.random.randint(abs(w_margin) + 1)
            if h_margin > 0:
                image_top = np.random.randint(h_margin + 1)
                canvas_top = 0
            else:
                image_top = 0
                canvas_top = np.random.randint(abs(h_margin) + 1)

            # Patch size is always smaller or equal to crop size.
            patch_w = min(w, crop_w)
            patch_h = min(h, crop_h)
            image_canvas = np.zeros(self.crop_size, np.float32) + self.image_padding_value
            label_canvas = np.zeros(self.crop_size, np.float32) + self.label_padding_value
            scb_canvas = np.zeros(self.crop_size, np.float32) + self.label_padding_value

            image_canvas[canvas_top: canvas_top + patch_h, canvas_left: canvas_left + patch_w] = \
                image[image_top: image_top + patch_h, image_left: image_left + patch_w]
            label_canvas[canvas_top: canvas_top + patch_h, canvas_left: canvas_left + patch_w] = \
                label[image_top: image_top + patch_h, image_left: image_left + patch_w]
            scb_canvas[canvas_top: canvas_top + patch_h, canvas_left: canvas_left + patch_w] = \
                scb[image_top: image_top + patch_h, image_left: image_left + patch_w]
            names_to_data['image'], names_to_data['label'], names_to_data['scribble'] = image_canvas, label_canvas, scb_canvas

            # Get the valid region mask
            valid_mask = np.zeros(self.crop_size, np.float32)
            valid_mask[canvas_top: canvas_top + patch_h, canvas_left: canvas_left + patch_w] = 1.
            names_to_data['valid_mask'] = valid_mask
            return names_to_data

class ToTorchTensor(object):
    """To torch tensor"""
    def __init__(self, num_classes, one_hot_encoding=True):
        self.num_classes = num_classes
        self.one_hot_encoding = one_hot_encoding

    def __call__(self, names_to_data):
        image = names_to_data['image']
        label = names_to_data['label']
        scb = names_to_data['scribble']

        image = image[None]
        if self.one_hot_encoding:
            label = to_one_hot_encoding(label, self.num_classes)
            scb = to_one_hot_encoding(scb, self.num_classes+1)

        names_to_data['image'] = torch.from_numpy(image)
        names_to_data['label'] = torch.from_numpy(label)
        names_to_data['scribble'] = torch.from_numpy(scb)

        # Validation does not require valid mask.
        valid_mask = names_to_data.get('valid_mask')
        if valid_mask is not None:
            valid_mask = valid_mask[None]
            names_to_data['valid_mask'] = torch.from_numpy(valid_mask)
        return names_to_data

def to_one_hot_encoding(image, no_classes=None, dtype=np.float32):
    """
    Transform hard-coding images to one-hot encodings.

    :param image: label
    :param no_classes: number of classes, namely number of class channels
    :return: image of a shape (NumClasses, H, W)
    """
    if no_classes is None:
        no_classes = np.unique(image).size
    image_one_hot = np.zeros((no_classes, *image.shape), dtype=dtype)
    for c in range(no_classes):
        image_one_hot[c][image == c] = 1
    return image_one_hot
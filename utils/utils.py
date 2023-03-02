#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/20/2021 11:32 PM
# @Author: yzf
import math

def linear_lr_decay(optimizer, step, num_steps, base_lr):
    """
    new_lr = (1-step/num_steps)*base_lr

    :param optimizer:
    :param step:
    :param num_steps:
    :param base_lr:
    :return:
    """
    new_lr = (1-step/num_steps) * base_lr
    for param in optimizer.param_groups:
        param['lr'] = new_lr
    return optimizer, new_lr

def cosine_lr_decay(optimizer, step, num_steps, base_lr):
    """
    new_lr = 1/2*(1+cos(t*pi/T))*base_lr

    :param optimizer:
    :param step:
    :param num_steps:
    :param base_lr:
    :return:
    """
    new_lr = 0.5 * (1+math.cos(step*math.pi/num_steps)) * base_lr
    for param in optimizer.param_groups:
        param['lr'] = new_lr
    return optimizer, new_lr

def poly_lr_decay(optimizer, step, num_steps, base_lr, gamma=0.9):
    """
    Polynomial learning rate decay policy

    :param optimizer:
    :param step: current epoch/iteration index
    :param num_steps: number of epochs/iterations
    :param base_lr: base learning rate
    :param gamma: power
    :return:
    """
    new_lr = base_lr * (1 - step / num_steps) ** gamma
    for param in optimizer.param_groups:  # `param_groups` is an attribute of `optimizer` object.
        param['lr'] = new_lr
    return optimizer, new_lr

def gaussian_ramp_up(t, base_value, max_t=80, scale=5.):
    """

    :param t:
    :param max_t:
    :param base_value:
    :param scale:
    :return:
    """
    if t < max_t:
        return base_value * math.exp(-scale*(1-t/max_t))
    else:
        return base_value

class AvgMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
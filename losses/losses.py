#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/20/2021 10:22 PM
# @Author: yzf
import torch
import torch.nn as nn
import torch.nn.functional as F

def entropy_minimization_loss(input, valid_mask=None):
    """

    :param input: logits (N, C, H, W)
    :param valid_mask: binary mask, (N, 1, H, W)
    :return:
    """
    input_log_softmax = F.log_softmax(input, 1)
    input_softmax = F.softmax(input, 1)
    loss = - input_softmax * input_log_softmax
    if valid_mask is not None:
        loss *= valid_mask
        loss = loss.sum() / max(valid_mask.sum(), 1e-8)
    else:
        loss = loss.mean()
    return loss

def cross_entropy_loss(input, target):
    """

    :param input: logits (N, C, H, W) or (N, C)
    :param target: torch.long, (N, H, W) or (N)
    :return:
    """
    return F.cross_entropy(input=input, target=target)

def partial_cross_entropy_loss(input, target, ignore_index):
    """

    :param input: logits (N, C, H, W)
    :param target: torch.long, hard label (N, H, W)
    :param ignore_index: value indicating ignore regions
    :return:
    """
    return F.cross_entropy(input=input, target=target, ignore_index=ignore_index)

def soft_label_cross_entropy_loss(input, target, valid_mask=None):
    """

    :param input: logits, torch.float32, (N, C, H, W)
    :param target: probability distribution, torch.float32, (N, C, H, W)
    :param valid_mask: binary mask, torch.bool, (N, 1, H, W)
    :return:
    """

    input_log_softmax = F.log_softmax(input, 1)
    # loss = - (target * input_log_softmax).sum() / target.sum()
    loss = - target * input_log_softmax
    if valid_mask is not None:
        loss *= valid_mask
        loss = loss.sum() / max(valid_mask.sum(), 1e-8)
    else:
        loss = loss.mean()
    return loss

def l1_loss(input, target, valid_mask=None):
    """
    L1 loss

    :param input: probability distribution (N, C, H, W)
    :param target: probability distribution (N, C, H, W)
    :param valid_mask: binary mask, torch.bool, (N, 1, H, W)
    :return:
    """
    loss = torch.sum(torch.abs(input - target), 1, keepdim=True)
    if valid_mask is not None:
        loss = loss * valid_mask
        loss = loss.sum() / max(valid_mask.sum(), 1e-8)
    else:
        loss = loss.mean()
    return loss

def l2_loss(input, target, valid_mask=None):
    """
    L2 loss or mean square error

    :param input:  probability distribution (N, C, H, W)
    :param target:  probability distribution (N, C, H, W)
    :param valid_mask: binary mask, torch.bool, (N, 1, H, W)
    :return:
    """
    loss = torch.sum(torch.pow(input - target, 2), 1, keepdim=True)
    if valid_mask is not None:
        loss = loss * valid_mask
        loss = loss.sum() / max(valid_mask.sum(), 1e-8)
    else:
        loss = loss.mean()
    return loss

def kl_loss(input, target, valid_mask=None):
    """
    Kullback-Leibler (KL) divergence

    :param input: logits, torch.float32, (N, C, H, W)
    :param target: logits, torch.float32, (N, C, H, W)
    :param valid_mask: binary mask, torch.bool, (N, 1, H, W)
    :return:
    """
    input_ll = F.log_softmax(input, dim=1)
    target_ll = F.log_softmax(target, dim=1)
    loss = F.kl_div(input_ll, target_ll, log_target=True, reduction='none')

    if valid_mask is not None:
        loss *= valid_mask
        loss = loss.sum() / max(valid_mask.sum(), 1e-8)
    else:
        loss = loss.mean()
    return loss

def bidirectional_kl_loss(input, target, valid_mask=None):
    """
    Bidirectional Kullback-Leibler (KL) divergence

    :param input: logits, torch.float32, (N, C, H, W)
    :param target: logits, torch.float32, (N, C, H, W)
    :param valid_mask: binary mask, torch.bool, (N, 1, H, W)
    :return:
    """
    # "ll" is short for log likelihood.
    input_ll = F.log_softmax(input, dim=1)
    target_ll = F.log_softmax(target, dim=1)

    p_loss = F.kl_div(input_ll, target_ll, log_target=True, reduction='none')
    q_loss = F.kl_div(target_ll, input_ll, log_target=True, reduction='none')

    if valid_mask is not None:
        p_loss *= valid_mask
        q_loss *= valid_mask

        p_loss = p_loss.sum() / max(valid_mask.sum(), 1e-8)
        q_loss = q_loss.sum() / max(valid_mask.sum(), 1e-8)
    else:
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def dice_loss_fn(input, target):
    """

    :param input: logits (N, C, H, W)
    :param target: one-hot encodings (N, C, H, W)
    :return:
    """
    eps = 1e-5
    input_norm = torch.softmax(input, dim=1)
    input_norm = input_norm.contiguous().view(input_norm.shape[0], input_norm.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1)
    up_part = 2 * torch.sum(input_norm * target, dim=2)  # N×C
    down_part = torch.sum(input_norm, dim=2) + torch.sum(target, dim=2) + eps  # N×C
    # dice_loss = 1. - torch.mean(torch.div(up_part, down_part))  # setting the objective as 1. may be bad, as there exist empty cases.
    dice = up_part / down_part
    return - torch.mean(dice)

def multi_label_soft_margin_loss(input, target):
    """

    :param input: (N, C)
    :param target: (N, C)
    :return:
    """
    return F.multilabel_soft_margin_loss(input, target)


# # TODO Need proof
# def focal_loss(input, target, gamma=0.5):
#     """
#     -(1-p_t)**gamma log(p_t), where p_t is the predicted probability of ground truth
#     and gamma adjusts weights on well-classified and uncertain classes, useful when
#     overwhelming gradients from background examples dilutes few gradients from hard foreground objects.
#
#     :param input: logits (N, C, H, W)
#     :param target: one-hot encodings (N, C, H, W)
#     :param gamma:
#     :return:
#     """
#     prob = F.softmax(input, 1)
#     prob_t = (prob * target).sum(1, keepdim=True)
#     focal_loss = (- (1-prob_t)**gamma * prob_t.log()).mean()
#     return loss_focal
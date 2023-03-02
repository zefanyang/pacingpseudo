#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 11/2/2021 5:12 PM
# @Author: yzf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet
from losses.losses import *
from .aux_path_memory import *

class ConsistencyRegulr(nn.Module):
    """Module combines queries and attention mechanism"""
    def __init__(self, kwargs_unet, kwargs_aux_path=None, args_parser=None):
        super(ConsistencyRegulr, self).__init__()
        self.kwargs_unet = kwargs_unet
        self.kwargs_aux_path = kwargs_aux_path
        self.args = args_parser

        self.backbone = UNet(**kwargs_unet)
        self.aux_path = AuxPath(**kwargs_aux_path)

    def forward(self, names_to_data, mode=None, step=None):
        assert mode in ['train', 'val', None]
        net_outputs = {}

        # Partial cross entropy loss
        end_points = self.backbone(names_to_data['image'])
        logits_weak = end_points['segmentation/logits']
        scb_target = torch.argmax(names_to_data['scribble'], dim=1).long()
        loss_pce = partial_cross_entropy_loss(logits_weak, scb_target, ignore_index=self.args.ignored_index)
        net_outputs.update({
            'segmentation/logits': logits_weak,
            'loss_pce': loss_pce,
        })

        # Entropy minimization loss
        valid_mask = names_to_data.get('valid_mask')
        if mode == 'train' and self.args.do_loss_ent:
            loss_ent = entropy_minimization_loss(logits_weak, valid_mask=valid_mask)
            net_outputs.update({
                'loss_ent': loss_ent,
            })

        # Consistency regularization
        if mode == 'train' and self.args.do_decoder_consistency:
            end_points_strong = self.backbone(names_to_data['image_strong'])
            logits_strong = end_points_strong['segmentation/logits']
            prob_strong = torch.softmax(logits_strong, 1)
            prob_weak_cr = torch.softmax(logits_weak, 1)

            if self.args.detach_weak_cr:
                prob_weak_cr = prob_weak_cr.detach()

            if self.args.loss_cr_variants == 'ce_loss':
                loss_cr = soft_label_cross_entropy_loss(input=logits_strong, target=prob_weak_cr, valid_mask=valid_mask)
            elif self.args.loss_cr_variants == 'l1_loss':
                loss_cr = l1_loss(input=prob_strong, target=prob_weak_cr, valid_mask=valid_mask)
            elif self.args.loss_cr_variants == 'l2_loss':
                loss_cr = l2_loss(input=prob_strong, target=prob_weak_cr, valid_mask=valid_mask)
            elif self.args.loss_cr_variants == 'kl_loss':
                loss_cr = kl_loss(input=logits_strong, target=logits_weak, valid_mask=valid_mask)
            else:
                raise ValueError('The loss is not implemented.')

            net_outputs.update({
                'loss_cr': loss_cr,
                'segmentation/logits_strong': logits_strong,
            })

        # Auxiliary loss
        if mode == 'train' and self.args.do_aux_path:
            aux_outputs = self.aux_path(
                end_points,
                names_to_data['scribble'],
                step,
            )

            # Auxiliary segmentation
            loss_aux_cls = partial_cross_entropy_loss(
                aux_outputs['logits_aux_cls'],
                aux_outputs['aux_targets'],
                self.args.ignored_index,
            )

            net_outputs.update({
                'logits_aux_cls': aux_outputs['logits_aux_cls'],
                'loss_aux_cls': loss_aux_cls,
            })

            # Memory
            if self.args.do_memory:
                loss_memory = cross_entropy_loss(
                    aux_outputs['logits_memory'].squeeze(-1).squeeze(-1),
                    aux_outputs['memory_target'],
                )

                net_outputs.update({
                    'loss_memory': loss_memory
                })
        return net_outputs
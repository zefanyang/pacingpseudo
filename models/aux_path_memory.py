#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 11/17/2021 9:51 AM
# @Author: yzf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxPath(nn.Module):
    """Memory bank that is updated by local embeddings"""
    def __init__(self, **kwargs):
        super(AuxPath, self).__init__()
        # Auxiliary path params
        self.num_classes = kwargs['num_classes']
        self.feat_stage = kwargs['feat_stage']
        self.feat_ch = kwargs['feat_ch']
        self.hid_ch = kwargs['hid_ch']
        self.aux_drop_prob = kwargs['aux_drop_prob']

        # Projection layer
        self.layer_bottleneck = nn.Sequential(
            nn.Dropout2d(self.aux_drop_prob),
            nn.Conv2d(sum(self.feat_ch), self.hid_ch, 3, 1, 1),
            nn.BatchNorm2d(self.hid_ch),
            nn.LeakyReLU(1e-2),
        )

        # Classification layer
        self.fc_cls = nn.Sequential(
            nn.Dropout2d(self.aux_drop_prob),
            nn.Conv2d(self.hid_ch, self.num_classes, 1, bias=False),
        )

        # Memory bank params
        self.do_memory = kwargs['do_memory']
        self.max_step = kwargs['max_step']
        self.momentum = kwargs['update_momentum']
        self.ensemble_mode = kwargs['ensemble_mode']
        self.memory_bank = nn.Parameter(
            torch.zeros((self.num_classes, self.hid_ch, 1, 1), dtype=torch.float32),
            requires_grad=False
        )
        self.memory_target = torch.arange(self.num_classes, dtype=torch.long).cuda()

    def forward(self, end_points, scribble, step):
        names_to_outputs = dict()
        # Auxiliary classification
        feat = torch.cat([end_points.get(stg) for stg in self.feat_stage], dim=1)
        aux_features = self.layer_bottleneck(feat)
        logits_aux_cls = self.fc_cls(aux_features)
        logits_aux_cls = F.interpolate(logits_aux_cls, size=scribble.shape[-2:], mode='bilinear', align_corners=True,)
        names_to_outputs.update({
            'logits_aux_cls': logits_aux_cls,
            'aux_targets': scribble.argmax(1).long(),
        })

        # Memory bank
        if self.do_memory:
            self.memory_update(aux_features, scribble, step)
            logits_memory = self.fc_cls(self.memory_bank)
            names_to_outputs.update({
                'logits_memory': logits_memory,
                'memory_target': self.memory_target,
            })
        return names_to_outputs

    @torch.no_grad()
    def memory_update(self, aux_features, scribble, step):
        # permute: bs, num_channels, h, w -> bs, h, w, num_channels
        # flatten: bs, h, w, num_channels -> bs, h*w, num_channels
        # unsqueeze: bs, h*w, num_channels, 1, 1
        _, _, h, w = scribble.shape
        scribble = scribble.permute(0, 2, 3, 1).flatten(1, 2).unsqueeze(-1).unsqueeze(-1)
        aux_features = F.interpolate(aux_features, size=(h, w), mode='bilinear', align_corners=True)
        aux_features = aux_features.permute(0, 2, 3, 1).flatten(1, 2).unsqueeze(-1).unsqueeze(-1)

        # mf_samp: h*w, num_channels, 1, 1
        # scb_samp: h*w, num_classes+1, 1, 1
        for mf_samp, scb_samp in zip(aux_features, scribble):
            for cls_idx in range(self.num_classes):
                # -> h*w, 1, 1
                mask = scb_samp[:, cls_idx] == 1
                if not mask.sum():
                    continue
                # [h*w, num_channels, 1, 1]([h*w]) -> N_cls, num_channels, 1, 1
                embd_cls = mf_samp[mask.squeeze()]
                # -> 1, num_channels, 1, 1
                memory_cls = self.memory_bank[cls_idx].unsqueeze(0)

                update = True
                if (memory_cls == 0).sum() == self.hid_ch:
                    # -> num_channels, 1, 1
                    memory_cls_update = embd_cls.mean(0)
                    update = False

                if update:
                    if self.ensemble_mode == 'mean':
                        # N_cls, num_channels, 1, 1 -> num_channels, 1, 1
                        memory_cls_update = embd_cls.mean(0)
                    elif self.ensemble_mode == 'cosine_similarity':
                        # L2 norm
                        # -> N_cls, num_channels, 1, 1
                        embd_cls /= (embd_cls.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
                        # -> 1, num_channels, 1, 1
                        memory_cls /= (memory_cls.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
                        # [N_cls, num_channels, 1, 1] * [1, num_channels, 1, 1] -> N_cls, 1, 1, 1
                        cosine_sim = (embd_cls * memory_cls).sum(1, keepdim=True)
                        weights = (1 - cosine_sim) / ((1 - cosine_sim).sum() + 1e-8)  # Each pixel has a small weight if N_cls is large.
                        # -> num_channels, 1, 1
                        memory_cls_update = (embd_cls * weights).sum(0)

                    m = _ramp_up_mo(step, self.max_step, self.momentum)
                    memory_cls_update = (1 - m) * memory_cls.squeeze(0) + m * memory_cls_update  # The weight for memory ramps up.
                self.memory_bank.data[cls_idx].copy_(memory_cls_update)
            return

def _ramp_up_mo(step, max_step, base_mo=0.9, gamma=0.9):
    """Ramp down momentum from 0.9"""
    return (1-step/max_step)**gamma * base_mo

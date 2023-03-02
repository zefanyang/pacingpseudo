#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/21/2021 8:46 PM
# @Author: yzf
"""Inference interface for ACDC, CHAOS, and lvsc"""
import os
import sys
import argparse
import logging
import random
import time
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
from datasets.augmentations import MeanStdNorm
from datasets.acdc.acdc_dataset import ACDCTwoStream
from datasets.chaos.chaos_dataset import CHAOSTwoStream
from datasets.lvsc.lvsc_dataset import LVSCTwoStream
from losses.losses import *
from utils.utils import *
from medpy import metric
from models.unet import UNet
import matplotlib.pyplot as plt

torch.manual_seed(1)

parser = argparse.ArgumentParser()
# Environment
parser.add_argument('--gpu', type=str, default='1')

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--root', type=str, default='./outputs',
                    help='root directory')

parser.add_argument('--session', type=str, default='Inference',
                    help='session name')

parser.add_argument('--fold', type=int, required=True,
                    help='the fold index')

parser.add_argument('--checkpoint_file', type=str, required=True,
                    help='checkpoint file')

parser.add_argument('--best_ckp', action='store_true', default=False,
                    help='use the best checkpoint')

# Dataset
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc', 'chaost1', 'chaost2', 'lvsc'],
                    help='dataset name')

parser.add_argument('--spacing', type=dict,
                    default={'acdc': (1.51, 1.51),
                             'chaost1': (1.62, 1.62),
                             'chaost2': (1.62, 1.62),
                             'lvsc': (1.48, 1.48)},
                    help='y and x spacing (mm) of pixels')

parser.add_argument('--num_classes', type=dict,
                    default={'acdc': 4,
                             'chaost1': 5,
                             'chaost2': 5,
                             'lvsc': 2},
                    help='number of classes (including background and not including ignored class)')

parser.add_argument('--num_workers', type=int, default=4,
                    help='number of processes to load data')

parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')

# Backbone params
parser.add_argument('--input_ch', type=int, default=1,
                    help='number of network input channel(s)')

parser.add_argument('--init_ch', type=int, default=32, choices=[32],
                    help='number of channels')

parser.add_argument('--max_ch', type=int, default=512, choices=[512, 1024, 728],
                    help='maximum number of channels')

parser.add_argument('--output_stride', type=int, default=8, choices=[32, 16, 8],
                    help='the stride of encoder output')

parser.add_argument('--is_stride_conv', type=bool, default=False,
                    help='whether to use stride conv or maxpool')

parser.add_argument('--is_trans_conv', type=bool, default=False,
                    help='whether to use trans conv or upsample')

parser.add_argument('--elab_end_points', type=bool, default=False,
                    help='whether to elaborate end points')

def main_interface(args):
    num_classes = args.num_classes.get(args.dataset)
    spacing = args.spacing.get(args.dataset)
    logging.info(f'Number of classes: {num_classes}')
    logging.info(f'Spacing: {spacing}')

    # Define model
    model = UNet(
        input_ch=args.input_ch,
        init_ch=args.init_ch,
        max_ch=args.max_ch,
        num_classes=num_classes,
        output_stride=args.output_stride,
        is_stride_conv=args.is_stride_conv,
        is_trans_conv=args.is_trans_conv,
        elab_end_points=args.elab_end_points,
    ).cuda()
    model_path, _, model_name = model.__module__.rpartition('.')
    shutil.copy(os.path.join(model_path, model_name+'.py'), os.path.join(args.child, model_name+'.py'))

    # Prepare dataset
    if args.dataset == 'chaost1' or args.dataset == 'chaost2':
        dataset_class = CHAOSTwoStream
    elif args.dataset == 'acdc':
        dataset_class = ACDCTwoStream
    elif args.dataset == 'lvsc':
        dataset_class = LVSCTwoStream

    test_dataset = dataset_class(args.test_ls,
                                 num_classes,
                                 base_transforms=[MeanStdNorm()],
                                 strong_transforms=None,
                                 do_strong=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False)
    logging.info('Length {}'.format(len(test_dataloader)))

    # Load parameters
    state_dict = torch.load(args.checkpoint_file)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict_backbone = OrderedDict()
        for key, value in state_dict.items():
            if 'backbone' in key:
                state_dict_backbone.update({key.partition('.')[-1]: value})
        model.load_state_dict(state_dict_backbone)
    model.eval()

    # Inference
    dicearr, hd95arr = [], []
    meter_dice = [AvgMeter() for _ in range(num_classes)]
    meter_hd95 = [AvgMeter() for _ in range(num_classes)]
    for idx, batch in tqdm(enumerate(test_dataloader)):
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.cuda()
        # Forward
        with torch.no_grad():
            output = model(batch.get('image'))

        # Compute Dice and Hausdorff distance
        logits = output.get('segmentation/logits')
        prob = logits.softmax(1)
        pred_hard = prob.argmax(1).cpu().numpy()
        label = batch.get('label').argmax(1).cpu().numpy()
        dicelog = _compute_dice(pred_hard[0], label[0], num_classes)
        hd95log = _compute_95hd(pred_hard[0], label[0], num_classes, spacing)

        # Compute Dice of this fold for validation
        for cls in range(num_classes):
            v = dicelog[cls]
            d = hd95log[cls]
            if not np.isnan(v):
                meter_dice[cls].update(v)
            if not np.isnan(d):
                meter_hd95[cls].update(d)

        # Log Dice and HD95 of each patient
        dicearr.append(dicelog)
        hd95arr.append(hd95log)

    dicearr = np.array(dicearr, dtype=np.float32)
    hd95arr = np.array(hd95arr, dtype=np.float32)

    np.savez(os.path.join(args.child, 'eval_data'), dicearr=dicearr, hd95arr=hd95arr)

    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('Number of clases: {}'.format(num_classes))
    foldavgdice = np.mean([meter_dice[_].avg for _ in range(1, num_classes)])
    foldavghd95 = np.mean([meter_hd95[_].avg for _ in range(1, num_classes)])

    logging.info('Fold {}, overall Dice: {:.4f}, overall HD95: {:.2f}'.format(args.fold, foldavgdice, foldavghd95))
    logging.info('Shape of the Dice array: {}'.format(dicearr.shape))
    logging.info('Shape of the HD95 array: {}'.format(hd95arr.shape))

def _compute_dice(pred_hard, label, num_classes):
    """
    Dice = 2 * sum(p & q) / sum(p + q)
    :param pred_hard: (h, w)
    :param label: (h, w)
    :param num_classes: int
    :return dicelog: (num_classes,)
    """
    dicelog = []
    for cls in range(num_classes):
        input = pred_hard == cls
        target = label == cls
        # input and target is empty
        if not np.any(input) and not np.any(target):
            dice = np.nan
        else:
            numerator = 2 * (input * target).sum()
            denomiator = input.sum() + target.sum()
            dice = numerator / max(denomiator, 1e-8)
        dicelog.append(dice)
    return dicelog

def _compute_95hd(pred_hard, label, num_classes, spacing):
    """
    95HD = (95P_q(min_p D(q, p)), 95P_p(min_q D(p, q)))/2
    :param pred_hard: (h, w)
    :param label: (h, w)
    :param num_classes: int
    :param spacing: int
    :return hd95log: (num_classes,)
    """
    hd95log = []
    for cls in range(num_classes):
        input = pred_hard == cls
        target = label == cls
        # input or target is empty or full.
        if not np.any(input) or not np.any(target) or np.all(input) or np.all(target):
            hd95 = np.nan
        else:
            hd95 = metric.hd95(input, target, spacing, 1)
        hd95log.append(hd95)
    return hd95log

def _compute_hd(pred_hard, label, num_classes, spacing):
    """
    HD = (max(min_p D(q, p)), max(min_q D(p, q)))/2
    :param pred_hard: (h, w)
    :param label: (h, w)
    :param num_classes: int
    :param spacing: int
    :return hdlog: (num_classes,)
    """
    hdlog = []
    for cls in range(num_classes):
        input = pred_hard == cls
        target = label == cls
        # input or target is empty or full.
        if not np.any(input) or not np.any(target) or np.all(input) or np.all(target):
            hd = np.nan
        else:
            hd = metric.hd(input, target, spacing, 1)
        hdlog.append(hd)
    return hdlog

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # deterministically select an algorithm

    # Make sure the alignment of the checkpoint and test_fold
    assert f'fold{args.fold}' in args.checkpoint_file

    # Make run directory
    args.child = os.path.join(args.root,
                              args.session,
                              args.dataset,
                              f'{os.path.basename(args.checkpoint_file)}')
    os.makedirs(args.child, exist_ok=True)

    # Get the lastest checkpoint
    if args.best_ckp:
        args.checkpoint_file = os.path.join(args.checkpoint_file, 'ckps/best_ckp.pth')
        if not os.path.isfile(args.checkpoint_file):
            a = args.checkpoint_file.rstrip('ckps/best_ckp.pth')
            args.checkpoint_file = os.path.join(a, 'best_ckp.pth')
    else:
        if args.dataset in ['acdc', 'chaost1', 'chaost2']:
            args.checkpoint_file = os.path.join(args.checkpoint_file, 'ckps/ckp_399.pth')
        elif args.dataset == 'lvsc':
            args.checkpoint_file = os.path.join(args.checkpoint_file, 'ckps/ckp_39.pth')
    # shutil.copy(args.checkpoint_file, args.child)
    shutil.copy(sys.argv[0], os.path.join(args.child, sys.argv[0].split('/')[-1]))

    # Logger configuration
    logging.basicConfig(filename=args.child + "/log.txt", level=logging.INFO, filemode='w',
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    args_info = ''.join(f'{k}={v}\n' for k, v in args._get_kwargs())
    logging.info(args_info)

    # Parse txt
    if args.dataset == 'acdc':
        with open(f'./data/{args.dataset}/train_test_split/five_fold_split/test_fold{args.fold}.txt', 'r') as f:
            test_ls = f.readlines()
        args.test_ls = [(f'./data/{args.dataset}/' + p).rstrip('\n') for p in test_ls]

    elif args.dataset == 'chaost1':
        with open(f'./data/chaos/train_test_split/five_fold_split/t1/test_fold{args.fold}.txt', 'r') as f:
            test_ls = f.readlines()
        args.test_ls = [(f'./data/chaos/' + p).rstrip('\n') for p in test_ls]

    elif args.dataset == 'chaost2':
        with open(f'./data/chaos/train_test_split/five_fold_split/t2/test_fold{args.fold}.txt', 'r') as f:
            test_ls = f.readlines()
        args.test_ls = [(f'./data/chaos/' + p).rstrip('\n') for p in test_ls]

    elif args.dataset == 'lvsc':
        with open(f'./data/lvsc/train_test_split/five_fold_split/test_fold{args.fold}.txt', 'r') as f:
            test_ls = f.readlines()
        args.test_ls = [(f'./data/lvsc/' + p).rstrip('\n') for p in test_ls]

    main_interface(args)

if __name__ == '__main__':
    main()
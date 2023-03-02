#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/20/2021 5:23 PM
# @Author: yzf
import os
import sys
import argparse
import logging
import random
import time
import shutil
import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.augmentations import MeanStdNorm
from datasets.chaos.chaos_dataset import CHAOSTwoStream
from losses.losses import partial_cross_entropy_loss, dice_loss_fn
from utils.utils import AvgMeter, poly_lr_decay
from utils.metrics import compute_dice
from models.unet import UNet

parser = argparse.ArgumentParser()
# Session
parser.add_argument('--gpu', type=str, default='1')

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--dataset', type=str, default='chaos',
                    help='dataset name')

parser.add_argument('--root', type=str, default='./outputs/chaos')

parser.add_argument('--session', type=str, default='Upperbound')

parser.add_argument('--tag', type=str, required=True,
                    help='experiment name')

# Dataset
parser.add_argument('--fold', type=int, default=1, choices=[0, 1, 2, 3, 4],
                    help='fold index to perform cross-validation')

parser.add_argument('--modality', type=str, default='t1', choices=['t1', 't2'],
                    help='modality of MRI images')

parser.add_argument('--num_classes', type=int, default=5,
                    help='number of classes (including background and not including ignored class)')

parser.add_argument('--num_workers', type=int, default=4,
                    help='number of processes to load data')

parser.add_argument('--augmentation_configs', type=str, default='datasets.chaos.chaos_aug_configs',
                    help='augmentation configuration module')

parser.add_argument('--augmentations', type=str, default='TransformsColor', choices=['TransformsColor'],
                    help='specify augmentation sequence')

# Backbone
parser.add_argument('--input_ch', type=int, default=1,
                    help='number of network input channel(s)')

parser.add_argument('--init_ch', type=int, default=32, choices=[32],
                    help='number of channels')

parser.add_argument('--max_ch', type=int, default=512,
                    help='maximum number of channels')

parser.add_argument('--output_stride', type=int, default=8, choices=[32, 16, 8],
                    help='the stride of encoder output')

parser.add_argument('--is_stride_conv', type=bool, default=False,
                    help='whether to use stride conv or maxpool')

parser.add_argument('--is_trans_conv', type=bool, default=False,
                    help='whether to use trans conv or upsample')

parser.add_argument('--elab_end_points', type=bool, default=True,
                    help='whether to elaborate end points')

## Optimizer
parser.add_argument('--loss_dice', action='store_true', default=True,
                    help='whether to use Dice loss')

parser.add_argument('--ignored_index', type=int, default=5,
                    help='the value indexing ignored regions')

parser.add_argument('--epoch', type=int, default=400, choices=[200, 400, 600],
                    help='number of epoch')

parser.add_argument('--batch_size', type=int, default=12, choices=[8, 12, 16, 24],
                    help='bacth size')

parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'],)

parser.add_argument('--momentum', type=float, default=0.9,
                    help='the momentum value of SGD optimizer')

parser.add_argument('--lr', type=float, default=0.0001, choices=[0.0001, 0.001, 0.01],
                    help='base learning rate')

parser.add_argument('--lr_decay', type=str, default='poly', choices=['linear', 'poly', 'cosine'],
                    help='learning rate decay policy')

parser.add_argument('--wd', type=float, default=0.0003,
                    help='weight decay')

parser.add_argument('--ckp_interval', type=int, default=10000,
                    help='interval of saving checkpoints')

def train_interface(args):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(os.path.join(args.child, 'tb_summary'))
    tb_writer = SummaryWriter(log_dir=os.path.join(args.child, 'tb_summary'))
    best_avg, best_epoch, best_avg_class = 0, 0, []

    model = UNet(
        input_ch=args.input_ch,
        init_ch=args.init_ch,
        max_ch=args.max_ch,
        num_classes=args.num_classes,
        output_stride=args.output_stride,
        is_stride_conv=args.is_stride_conv,
        is_trans_conv=args.is_trans_conv,
        elab_end_points=args.elab_end_points,
    ).cuda()
    logging.info(model)
    model_path, _, model_name = model.__module__.rpartition('.')
    shutil.copy(os.path.join(model_path, model_name+'.py'), os.path.join(args.child, model_name+'.py'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    transforms = getattr(importlib.import_module(args.augmentation_configs), args.augmentations)()
    train_dataset = CHAOSTwoStream(args.train_ls,
                                  args.num_classes,
                                  base_transforms=transforms.base_transforms,
                                  strong_transforms=None,
                                  do_strong=False)  # only weak transforms
    val_dataset = CHAOSTwoStream(args.val_ls,
                                args.num_classes,
                                base_transforms=[MeanStdNorm()],
                                strong_transforms=None,
                                do_strong=False)  # only mean-std norm
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    valdice = np.zeros(args.epoch)
    for curr_epoch in range(args.epoch):
        epoch_tic = time.time()
        meter_loss_ce = AvgMeter()
        meter_loss_dice = AvgMeter()
        optimizer, new_lr = poly_lr_decay(optimizer, curr_epoch, args.epoch, args.lr)
        for idx, batch in enumerate(train_dataloader):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.cuda()
            # Forward
            end_points = model(batch.get('image'))
            # Compute loss
            logits = end_points.get('segmentation/logits')
            target = torch.argmax(batch.get('label'), dim=1).long()
            loss_ce = partial_cross_entropy_loss(logits, target, args.ignored_index)
            loss = loss_ce  # NOTE: loss and loss_ce share memory. Changing loss also changes loss_ce.
            meter_loss_ce.update(loss_ce.item(), n=batch.get('image').shape[0])
            if args.loss_dice:
                loss_dice = dice_loss_fn(logits, batch.get('label'))
                loss += loss_dice
                meter_loss_dice.update(loss_dice.item(), n=batch.get('image').shape[0])
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_toc = time.time()
        logging.info("epoch: {:03d}, lr: {:.6f}, loss_ce: {:.6f}, loss_dice: {:.6f}, {:.2f} s/epoch".format(
            curr_epoch, new_lr, meter_loss_ce.avg, meter_loss_dice.avg, epoch_toc-epoch_tic))

        tb_writer.add_scalar('losses/loss_ce_train', meter_loss_ce.avg, curr_epoch)
        tb_writer.add_scalar('losses/loss_dice_train', meter_loss_dice.avg, curr_epoch)
        tb_writer.add_scalar('lr/current_lr', new_lr, curr_epoch)

        model.eval()
        epoch_tic = time.time()
        meter_loss_dice_val = AvgMeter()
        meter_loss_ce_val = AvgMeter()
        meter_dsc = [AvgMeter() for _ in range(args.num_classes)]
        for idx, batch in enumerate(val_dataloader):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.cuda()
            # Forward
            with torch.no_grad():
                end_points_val = model(batch.get('image'))

            # Compute loss
            logits_val = end_points_val.get('segmentation/logits')
            target_val = torch.argmax(batch.get('label'), dim=1).long()
            loss_ce_val = partial_cross_entropy_loss(logits_val, target_val, args.ignored_index)
            loss_dice_val = dice_loss_fn(logits_val, batch.get('label'))

            meter_loss_dice_val.update(loss_dice_val.item(), n=batch.get('image').shape[0])
            meter_loss_ce_val.update(loss_ce_val.item(), n=batch.get('image').shape[0])

            # Compute Dice
            prob_val = torch.softmax(logits_val, dim=1)
            for n in range(batch.get('image').shape[0]):
                dice_ls = compute_dice(prob_val.cpu().numpy()[n], batch.get('label').cpu().numpy()[n])  # compute class Dice values per sample
                for cls, dice in enumerate(dice_ls):  # c and d are the class index and Dice value
                    if not np.isnan(dice):
                        meter_dsc[cls].update(dice)

        epoch_toc = time.time()
        avg_all = np.mean([meter_dsc[_].avg for _ in range(1, args.num_classes)])
        logging.info("val: {:03d}, loss_ce: {:.6f}, loss_dice: {:.6f}, {:.2f} s/epoch".format(
            curr_epoch, meter_loss_ce_val.avg, meter_loss_dice_val.avg, epoch_toc-epoch_tic))
        logging.info("[BG: {:.4f}, Liver: {:.4f}, R-Kidney: {:.4f}, L-Kidney: {:.4f}, Spleen: {:.4f}, All: {:.4f}]".format(
            meter_dsc[0].avg, meter_dsc[1].avg, meter_dsc[2].avg, meter_dsc[3].avg, meter_dsc[4].avg, avg_all))

        # Log overall Dice every epoch
        valdice[curr_epoch] = avg_all

        # Save checkpoints every 10 epochs
        if curr_epoch+1 % args.ckp_interval == 0 or curr_epoch+1 == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.child, 'ckps', 'ckp_{:d}.pth'.format(curr_epoch)))

        if avg_all > best_avg:
            best_epoch = curr_epoch
            best_avg = avg_all
            best_avg_class = [meter_dsc[_].avg for _ in range(1, args.num_classes)]
            torch.save(model.state_dict(), args.child+'/best_ckp.pth')

        tb_writer.add_scalar('losses/loss_ce_val', meter_loss_ce_val.avg, curr_epoch)
        tb_writer.add_scalar('losses/loss_dice_val', meter_loss_dice_val.avg, curr_epoch)
        tb_writer.add_scalar('DSC/BG', meter_dsc[0].avg, curr_epoch)
        tb_writer.add_scalar('DSC/Liver', meter_dsc[1].avg, curr_epoch)
        tb_writer.add_scalar('DSC/R-Kidney', meter_dsc[2].avg, curr_epoch)
        tb_writer.add_scalar('DSC/L-Kidney', meter_dsc[3].avg, curr_epoch)
        tb_writer.add_scalar('DSC/Spleen', meter_dsc[4].avg, curr_epoch)
        tb_writer.add_scalar('DSC/All', avg_all, curr_epoch)
        tb_writer.add_scalar('DSC/Best', best_avg, curr_epoch)

    logging.info("The best at epoch: {:d}, Liver: {:.4f}, R-Kidney: {:.4f}, L-Kidney: {:.4f}, Spleen: {:.4f},"
                 " All: {:.4f}".format(best_epoch, *best_avg_class, best_avg))
    np.savez(os.path.join(args.child, 'valdice'), valdice=valdice)
    tb_writer.close()

def train_main():
    # Environment and random seed
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # deterministically select an algorithm

    # Make run directory
    args.child = os.path.join(os.path.join(args.root, args.modality), args.session,
                              f'{args.session}-{time.strftime("%H-%M-%S-%m%d")}-fold{args.fold}-{args.tag}')
    os.makedirs(args.child, exist_ok=False)
    os.makedirs(os.path.join(args.child, 'ckps'), exist_ok=True)
    shutil.copy(sys.argv[0], os.path.join(args.child, sys.argv[0].split('/')[-1]))

    # Logger configuration
    logging.basicConfig(filename=args.child + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    args_info = ''.join(f'{k}={v}\n' for k, v in args._get_kwargs())
    logging.info(args_info)

    # Parse txt
    with open(f'./data/{args.dataset}/train_test_split/five_fold_split/{args.modality}/train_fold{args.fold}.txt',
              'r') as f:
        train_ls = f.readlines()
    with open(f'./data/{args.dataset}/train_test_split/five_fold_split/{args.modality}/test_fold{args.fold}.txt',
              'r') as f:
        val_ls = f.readlines()

    args.train_ls = [(f'./data/{args.dataset}/' + p).rstrip('\n') for p in train_ls]
    args.val_ls = [(f'./data/{args.dataset}/' + p).rstrip('\n') for p in val_ls]

    train_interface(args)

if __name__ == '__main__':
    train_main()
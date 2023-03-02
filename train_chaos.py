#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 12/8/2021 10:53 PM
# @Author: yzf
import os
import sys
import argparse
import logging
import random
import time
import shutil
import importlib
import numpy as np
from torch.utils.data import DataLoader
from datasets.augmentations import MeanStdNorm
from datasets.chaos.chaos_dataset import CHAOSTwoStream
from losses.losses import *
from utils.utils import *
from utils.metrics import compute_dice
from models.consistency_reglur_memory import ConsistencyRegulr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
## Baseline options
## Session
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--dataset', type=str, default='chaos',
                    help='dataset name')

parser.add_argument('--root', type=str, default='./outputs/chaos',
                    help='root directory')

parser.add_argument('--session', type=str, default='Control',
                    choices=['Control', 'Experiment'],
                    help='session name')

parser.add_argument('--tag', type=str, required=True,
                    help='experiment name')

## Dataset
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

parser.add_argument('--augmentations', type=str, default='TransformsColor',
                    choices=['TransformsColor', 'TransformsColorBlur', 'TransformsColorMixup', 'TransformsColorLow'],
                    help='specify augmentation sequence')

## Network
# Backbone
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

parser.add_argument('--elab_end_points', type=bool, default=True,
                    help='whether to elaborate end points')

## Optimizer
parser.add_argument('--ignored_index', type=int, default=5,
                    help='the value indexing ignored regions')

parser.add_argument('--epoch', type=int, default=400, choices=[200, 400, 600],
                    help='number of epoch')

parser.add_argument('--batch_size', type=int, default=12, choices=[8, 12, 16, 24],
                    help='bacth size')

parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'momentum'],
                    help='the optimizer')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='the momentum value of SGD optimizer')

parser.add_argument('--lr', type=float, default=0.0001, choices=[0.0001, 0.0003, 0.0005],
                    help='base learning rate')

parser.add_argument('--lr_decay', type=str, default='poly', choices=['linear', 'poly', 'cosine'],
                    help='learning rate decay policy')

parser.add_argument('--wd', type=float, default=0.0003, choices=[0.0003, 0.0001, 0.0005],
                    help='weight decay')

parser.add_argument('--ckp_interval', type=int, default=10000,
                    help='interval of saving checkpoints')

## Baseline plus options
## Entropy minimization
parser.add_argument('--do_loss_ent', action='store_true', default=False,
                    help='whether to use entropy minimization loss')

parser.add_argument('--loss_ent_weight', type=float, default=1.,
                    help='weight of entropy minimization loss')

parser.add_argument('--ramp_up_loss_ent', action='store_true', default=True,
                    help='whether to ramp up entropy minimization loss')

parser.add_argument('--ramp_up_scale', type=float, default=8., choices=[5., 8., 10.],
                    help='exponential scale of ramping up')

## Consistency options
parser.add_argument('--do_decoder_consistency', action='store_true', default=False,
                    help='whether to impose consistency regularization on decoder outputs')

parser.add_argument('--ramp_up_loss_cr', action='store_true', default=True,
                    help='whether to ramp up consistency regularization loss')

parser.add_argument('--detach_weak_cr', action='store_true', default=False,
                    help='whether to detach the weak probability')

parser.add_argument('--loss_cr_variants', type=str, default='ce_loss', choices=['ce_loss', 'l1_loss', 'l2_loss', 'kl_loss'],
                    help='specify variants of consistency regularization loss')

parser.add_argument('--strength', type=float, default=1., choices=[0.125, 0.25, 0.5, 1.],
                    help='the strength of color distortion')

parser.add_argument('--loss_cr_weight', type=float, default=1.,
                    help='weight of consistency regularization loss')

## Auxiliary path
parser.add_argument('--do_aux_path', action='store_true', default=False,
                    help='whether to adopt auxiliary path')

parser.add_argument('--feat_stage', type=list, default=['encoder/stage6', 'encoder/stage5'],
                    choices=[['encoder/stage6'], ['encoder/stage6', 'encoder/stage5'], ['encoder/stage6', 'encoder/stage5', 'encoder/stage4']],
                    help='feature from which stage to perform memory storage')

parser.add_argument('--feat_ch', type=list, default=[512, 512],
                    choices=[[512], [512, 512], [512, 512, 256]],
                    help='number of channels of the encoder output')

parser.add_argument('--loss_aux_weight', type=float, default=0.01, choices=[1., 0.01, 0.001],
                    help='weight of auxiliary loss')

parser.add_argument('--hid_ch', type=int, default=64, choices=[256, 128, 64],
                    help='number of channels of features in memory bank')

parser.add_argument('--aux_drop_prob', type=float, default=0., choices=[0., 0.5, 0.8],
                    help='dropout probability')

## Memory bank options
parser.add_argument('--do_memory', action='store_true', default=False,
                    help='whether to do memory bank')

parser.add_argument('--loss_memory_weight', type=float, default=1, choices=[1., 0.01],
                    help='weight of memory loss')

parser.add_argument('--update_momentum', type=float, default=0.9,
                    help='memory update momentum')

parser.add_argument('--ensemble_mode', type=str, default='cosine_similarity', choices=['cosine_similarity', 'mean'],
                    help='method to ensemble pixel features')

def train_interface(args):
    # This line makes specifying cuda devices ineffective.
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(os.path.join(args.child, 'tb_summary'))
    tb_writer = SummaryWriter(log_dir=os.path.join(args.child, 'tb_summary'))
    best_avg, best_epoch, best_avg_class = 0, 0, []

    model = ConsistencyRegulr(
        # Backbone kwargs
        kwargs_unet=dict(
            input_ch=args.input_ch,
            init_ch=args.init_ch,
            max_ch=args.max_ch,
            num_classes=args.num_classes,
            output_stride=args.output_stride,
            is_stride_conv=args.is_stride_conv,
            is_trans_conv=args.is_trans_conv,
            elab_end_points=args.elab_end_points,
            ),
        # Auxiliary path kwargs
        kwargs_aux_path=dict(
            num_classes=args.num_classes,
            feat_stage=args.feat_stage,
            feat_ch=args.feat_ch,
            hid_ch=args.hid_ch,
            aux_drop_prob=args.aux_drop_prob,
            do_memory=args.do_memory,
            max_step=args.epoch,
            update_momentum=args.update_momentum,
            ensemble_mode=args.ensemble_mode,
        ),
        args_parser=args
        ).cuda()
    logging.info(model)
    model_path, _, model_name = model.__module__.rpartition('.')
    shutil.copy(os.path.join(model_path, model_name+'.py'), os.path.join(args.child, model_name+'.py'))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        raise ValueError('Unimplemented optimizer')

    # Prepare dataset
    transforms = getattr(importlib.import_module(args.augmentation_configs), args.augmentations)(args.strength)
    train_dataset = CHAOSTwoStream(args.train_ls,
                                  args.num_classes,
                                  base_transforms=transforms.base_transforms,
                                  strong_transforms=transforms.strong_transforms,
                                  do_strong=args.do_decoder_consistency)
    val_dataset = CHAOSTwoStream(args.val_ls,
                                args.num_classes,
                                base_transforms=[MeanStdNorm()],
                                strong_transforms=None,
                                do_strong=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    num_iterations = args.epoch * (len(train_dataset) // args.batch_size)

    # Training
    valdice = np.zeros(args.epoch)
    for curr_epoch in range(args.epoch):
        epoch_tic = time.time()
        meter_loss_pce = AvgMeter()
        meter_loss_ent = AvgMeter()
        meter_loss_cr = AvgMeter()
        meter_loss_aux_cls = AvgMeter()
        meter_loss_memory = AvgMeter()

        # Learning rate decay strategy
        # new_lr can be got by optimizer.param_groups[0]['lr']
        if args.lr_decay == 'poly':
            optimizer, new_lr = poly_lr_decay(optimizer, curr_epoch, args.epoch, args.lr)
        elif args.lr_decay == 'cosine':
            optimizer, new_lr = cosine_lr_decay(optimizer, curr_epoch, args.epoch, args.lr)
        elif args.lr_decay == 'linear':
            optimizer, new_lr = linear_lr_decay(optimizer, curr_epoch, args.epoch, args.lr)
        else:
            raise ValueError('Unimplemented learning rate decay policy.')

        ## Training
        for idx, batch in enumerate(train_dataloader):
            del batch['label']
            if args.do_decoder_consistency:
                del batch['label_strong']
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.cuda()

            # Forward
            net_outputs = model(batch, mode='train', step=curr_epoch)
            loss_pce = net_outputs['loss_pce']
            loss = loss_pce
            meter_loss_pce.update(loss_pce.item(), n=batch['image'].shape[0])

            if args.do_loss_ent:
                loss_ent = net_outputs['loss_ent']
                if args.ramp_up_loss_ent:
                    w = gaussian_ramp_up(t=curr_epoch, base_value=args.loss_ent_weight, scale=args.ramp_up_scale)
                    loss_ent = loss_ent * w
                loss += loss_ent
                meter_loss_ent.update(loss_ent.item(), n=batch['image'].shape[0])

            if args.do_decoder_consistency:
                loss_cr = net_outputs['loss_cr']
                if args.ramp_up_loss_cr:
                    w = gaussian_ramp_up(t=curr_epoch, base_value=args.loss_cr_weight, scale=args.ramp_up_scale)
                    loss_cr = loss_cr * w
                loss += loss_cr
                meter_loss_cr.update(loss_cr.item(), n=batch['image'].shape[0])

            # The memory classification could regularize training with improved performance.
            if args.do_aux_path:
                loss_aux_cls = net_outputs['loss_aux_cls']
                w = args.loss_aux_weight
                # if args.ramp_up_loss_aux:
                #     w = gaussian_ramp_up(t=curr_epoch, base_value=args.loss_aux_weight, scale=args.ramp_up_scale)
                loss_aux_cls *= w
                loss += loss_aux_cls
                meter_loss_aux_cls.update(loss_aux_cls.item(), n=batch['image'].shape[0])

                if args.do_memory:
                    loss_memory = net_outputs['loss_memory']
                    w = args.loss_memory_weight
                    # if args.ramp_up_loss_aux:
                    #     w = gaussian_ramp_up(t=curr_epoch, base_value=args.loss_memory_weight, scale=args.ramp_up_scale)
                    loss_memory *= w
                    loss += loss_memory
                    meter_loss_memory.update(loss_memory.item())

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_toc = time.time()
        # After one epoch, log information
        logging.info("epoch: {:03d}, lr: {:.6f}, loss_pce: {:.6f}, loss_ent: {:.6f}, loss_cr: {:.6f}, loss_aux_cls: {:.6f}, loss_memory: {:.6f}, {:.2f} s/epoch"
                     .format(curr_epoch, new_lr, meter_loss_pce.avg, meter_loss_ent.avg, meter_loss_cr.avg, meter_loss_aux_cls.avg, meter_loss_memory.avg, epoch_toc-epoch_tic))

        # Tensorboard monitoring
        # Weak image
        summary_image = batch['image'].detach().cpu()[0, 0]
        fig = plt.figure(); plt.subplot(); plt.imshow(summary_image, 'gray')
        tb_writer.add_figure('predictions/image', fig, curr_epoch)

        # Scribble image
        summary_scribble = batch['scribble'].detach().cpu().argmax(1)[0]
        fig = plt.figure(); plt.subplot(); plt.imshow(summary_scribble, interpolation='nearest')
        tb_writer.add_figure('predictions/scribble', fig, curr_epoch)

        # Weak probability
        logits_weak = net_outputs.get('segmentation/logits')
        prediction_weak = logits_weak.detach().cpu().softmax(1).argmax(1)[0]
        fig = plt.figure(); plt.subplot(); plt.imshow(prediction_weak)
        tb_writer.add_figure('predictions/prediction_decoder_weak', fig, curr_epoch)
        # Histogram of probability weak maximum
        prob_weak_max = logits_weak.detach().cpu().softmax(1).max(1)[0]
        tb_writer.add_histogram('histogram/prob_weak_max', prob_weak_max, curr_epoch)

        logits_strong = net_outputs.get('segmentation/logits_strong')
        if logits_strong is not None:
            # Strong image
            summary_image = batch['image_strong'].detach().cpu()[0, 0]
            fig = plt.figure(); plt.subplot(); plt.imshow(summary_image, 'gray')
            tb_writer.add_figure('predicitons/image_strong', fig, curr_epoch)
            # Strong prediction
            prediction_strong = logits_strong.detach().cpu().softmax(1).argmax(1)[0]
            fig = plt.figure(); plt.subplot(); plt.imshow(prediction_strong)
            tb_writer.add_figure('predictions/prediction_decoder_strong', fig, curr_epoch)
            # Histogram of probability strong maximum
            prob_strong_max = logits_strong.detach().cpu().softmax(1).max(1)[0]
            tb_writer.add_histogram('histogram/prob_strong_max', prob_weak_max, curr_epoch)

        logits_aux_cls = net_outputs.get('logits_aux_cls')
        if logits_aux_cls is not None:
            # Auxiliary prediction
            prediction_aux = logits_aux_cls.detach().cpu().softmax(1).argmax(1)[0]
            fig = plt.figure(); plt.subplot(); plt.imshow(prediction_aux)
            tb_writer.add_figure('predictions/prediction_auxiliary_segmentation', fig, curr_epoch)

        tb_writer.add_scalar('losses/loss_pce_train', meter_loss_pce.avg, curr_epoch)
        tb_writer.add_scalar('losses/loss_cr', meter_loss_cr.avg, curr_epoch)
        tb_writer.add_scalar('losses/loss_ent', meter_loss_ent.avg, curr_epoch)
        tb_writer.add_scalar('losses/loss_aux_cls', meter_loss_aux_cls.avg, curr_epoch)
        tb_writer.add_scalar('losses/loss_memory', meter_loss_memory.avg, curr_epoch)
        tb_writer.add_scalar('lr/current_lr', new_lr, curr_epoch)

        ## Validation
        model.eval()
        meter_loss_pce_val = AvgMeter()
        meter_dsc = [AvgMeter() for _ in range(args.num_classes)]
        tic = time.time()
        for idx, batch in enumerate(val_dataloader):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.cuda()
            # Forward
            with torch.no_grad():
                net_outputs = model(batch, mode='val')
            # Compute PCE
            loss_pce_val = net_outputs['loss_pce']
            meter_loss_pce_val.update(loss_pce_val.item(), n=batch['image'].shape[0])
            # Compute class Dice metrics
            logits_seg = net_outputs['segmentation/logits']
            value_sm = torch.softmax(logits_seg, dim=1)
            for n in range(batch['image'].shape[0]):
                dice_ls = compute_dice(value_sm.cpu().numpy()[n], batch['label'].cpu().numpy()[n])  # compute class Dice values per sample
                for cls, dice in enumerate(dice_ls):
                    if not np.isnan(dice):
                        meter_dsc[cls].update(dice)
        toc = time.time()

        # After one epoch, report information
        avg_all = np.mean([meter_dsc[_].avg for _ in range(1, args.num_classes)])  # should exclude background
        logging.info("val: {:03d}, loss_pce: {:.6f}, time: {:.2f} s/epoch".format(
            curr_epoch, meter_loss_pce_val.avg, toc-tic))
        logging.info("[BG: {:.4f}, Liver: {:.4f}, R-Kidney: {:.4f}, L-Kidney: {:.4f}, Spleen: {:.4f}, All: {:.4f}]".format(
            meter_dsc[0].avg, meter_dsc[1].avg, meter_dsc[2].avg, meter_dsc[3].avg, meter_dsc[4].avg, avg_all))

        # Log overall Dice every epoch
        valdice[curr_epoch] = avg_all

        # Save checkpoints every 10 epochs
        if curr_epoch+1 % args.ckp_interval == 0 or curr_epoch+1 == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.child, 'ckps', 'ckp_{:d}.pth'.format(curr_epoch)))

        # Log best checkpoint
        if avg_all > best_avg:
            best_epoch = curr_epoch
            best_avg = avg_all
            best_avg_class = [meter_dsc[_].avg for _ in range(1, args.num_classes)]
            torch.save(model.state_dict(), args.child+'/best_ckp.pth')

        ## Tensorboard summary
        tb_writer.add_scalar('losses/loss_pce_val', meter_loss_pce_val.avg, curr_epoch)
        tb_writer.add_scalar('DSC/BG', meter_dsc[0].avg, curr_epoch)
        tb_writer.add_scalar('DSC/Liver', meter_dsc[1].avg, curr_epoch)
        tb_writer.add_scalar('DSC/R-Kidney', meter_dsc[2].avg, curr_epoch)
        tb_writer.add_scalar('DSC/L-Kidney', meter_dsc[3].avg, curr_epoch)
        tb_writer.add_scalar('DSC/Spleen', meter_dsc[4].avg, curr_epoch)
        tb_writer.add_scalar('DSC/All', avg_all, curr_epoch)
        tb_writer.add_scalar('DSC/Best', best_avg, curr_epoch)

    # Log final checkpoint
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
    with open(f'./data/{args.dataset}/train_test_split/five_fold_split/{args.modality}/train_fold{args.fold}.txt', 'r') as f:
        train_ls = f.readlines()
    with open(f'./data/{args.dataset}/train_test_split/five_fold_split/{args.modality}/test_fold{args.fold}.txt', 'r') as f:
        val_ls = f.readlines()

    args.train_ls = [(f'./data/{args.dataset}/' + p).rstrip('\n') for p in train_ls]
    args.val_ls = [(f'./data/{args.dataset}/' + p).rstrip('\n') for p in val_ls]

    train_interface(args)

if __name__ == '__main__':
    train_main()
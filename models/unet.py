#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 9/20/2021 3:13 PM
# @Author: yzf
import math
import numpy as np
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self,
                 input_ch=1,
                 init_ch=32,
                 max_ch=512,
                 num_classes=4,
                 output_stride=32,
                 is_stride_conv=False,
                 is_trans_conv=False,
                 elab_end_points=False,
                 ):
        super(UNet, self).__init__()
        self.elab_end_points = elab_end_points
        self.end_points = dict()

        assert is_trans_conv == is_stride_conv, "Only combo of stride_conv and trans_conv or maxpool and upsample is allowed."
        # output channels of encoder stages
        ch_ls = [min(max_ch, 2**k*init_ch) for k in range(6)]
        self.enc_block1 = EncBlock(input_ch, ch_ls[0], do_subsamp=False, is_stride_conv=is_stride_conv)
        self.enc_block2 = EncBlock(ch_ls[0], ch_ls[1], do_subsamp=True, is_stride_conv=is_stride_conv)
        self.enc_block3 = EncBlock(ch_ls[1], ch_ls[2], do_subsamp=True, is_stride_conv=is_stride_conv)
        self.enc_block4 = EncBlock(ch_ls[2], ch_ls[3], do_subsamp=True, is_stride_conv=is_stride_conv)

        assert output_stride in [8, 16, 32]
        if output_stride == 32:
            self.enc_block5 = EncBlock(ch_ls[3], ch_ls[4], do_subsamp=True, is_stride_conv=is_stride_conv)
            self.enc_block6 = EncBlock(ch_ls[4], ch_ls[5], do_subsamp=True, is_stride_conv=is_stride_conv)
            self.dec_block5 = DecBlock(ch_ls[5], ch_ls[4], ch_ls[4], 2, 2, is_trans_conv=is_trans_conv)
            self.dec_block4 = DecBlock(ch_ls[4], ch_ls[3], ch_ls[3], 2, 2, is_trans_conv=is_trans_conv)

        elif output_stride == 16:
            # Following the "Fully Convolutional Instance-aware Semantic Segmentation" paper,
            # we discard the maxpool and adopt atrous convolution in the stage 5.
            # This increases the feature resolution, while maintaining the receptive field.
            self.enc_block5 = EncBlock(ch_ls[3], ch_ls[4], do_subsamp=True, is_stride_conv=is_stride_conv)
            self.enc_block6 = EncBlock(ch_ls[4], ch_ls[5], do_subsamp=False, is_stride_conv=is_stride_conv, dilation=2)
            self.dec_block5 = DecBlock(ch_ls[5], ch_ls[4], ch_ls[4], 1, 1, is_trans_conv=is_trans_conv)
            self.dec_block4 = DecBlock(ch_ls[4], ch_ls[3], ch_ls[3], 2, 2, is_trans_conv=is_trans_conv)

        elif output_stride == 8:
            self.enc_block5 = EncBlock(ch_ls[3], ch_ls[4], do_subsamp=False, is_stride_conv=is_stride_conv, dilation=2)
            self.enc_block6 = EncBlock(ch_ls[4], ch_ls[5], do_subsamp=False, is_stride_conv=is_stride_conv, dilation=4)
            self.dec_block5 = DecBlock(ch_ls[5], ch_ls[4], ch_ls[4], 1, 1, is_trans_conv=is_trans_conv)
            self.dec_block4 = DecBlock(ch_ls[4], ch_ls[3], ch_ls[3], 1, 1, is_trans_conv=is_trans_conv)


        self.dec_block3 = DecBlock(ch_ls[3], ch_ls[2], ch_ls[2], is_trans_conv=is_trans_conv)
        self.dec_block2 = DecBlock(ch_ls[2], ch_ls[1], ch_ls[1], is_trans_conv=is_trans_conv)
        self.dec_block1 = DecBlock(ch_ls[1], ch_ls[0], ch_ls[0], is_trans_conv=is_trans_conv)

        self.final_conv = nn.Conv2d(ch_ls[0], num_classes, 1, 1)

    def forward(self, x):
        enc1 = self.enc_block1(x)
        enc2 = self.enc_block2(enc1)
        enc3 = self.enc_block3(enc2)
        enc4 = self.enc_block4(enc3)
        enc5 = self.enc_block5(enc4)
        enc6 = self.enc_block6(enc5)

        dec5 = self.dec_block5(enc6, enc5)
        dec4 = self.dec_block4(dec5, enc4)
        dec3 = self.dec_block3(dec4, enc3)
        dec2 = self.dec_block2(dec3, enc2)
        dec1 = self.dec_block1(dec2, enc1)
        logits = self.final_conv(dec1)

        if not self.elab_end_points:
            self.end_points.update({
                "segmentation/logits": logits,
            })
        else:
            self.end_points.update({
                "encoder/stage1": enc1,
                "encoder/stage2": enc2,
                "encoder/stage3": enc3,
                "encoder/stage4": enc4,
                "encoder/stage5": enc5,
                "encoder/stage6": enc6,

                "decoder/stage5": dec5,
                "decoder/stage4": dec4,
                "decoder/stage3": dec3,
                "decoder/stage2": dec2,
                "decoder/stage1": dec1,

                "segmentation/logits": logits,
            })
        return self.end_points

class EncBlock(nn.Module):
    """Encoder block"""
    def __init__(self, in_ch, out_ch,
                 do_subsamp=True,
                 is_stride_conv=False,
                 dilation=1,):
        super().__init__()
        self.pooling = None
        if do_subsamp and not is_stride_conv:
            self.pooling = nn.MaxPool2d(2, 2)
            self.conv_block = DoubleConv(in_ch, out_ch,
                                         ks1=3, stride1=1, padding1=dilation, dilation1=dilation,
                                         ks2=3, stride2=1, padding2=dilation, dilation2=dilation)
        elif do_subsamp and is_stride_conv:
            self.conv_block = DoubleConv(in_ch, out_ch,
                                         ks1=3, stride1=2, padding1=dilation, dilation1=dilation,  # stride1=2
                                         ks2=3, stride2=1, padding2=dilation, dilation2=dilation)
        else:
            # Do two dilated convolution
            self.conv_block = DoubleConv(in_ch, out_ch,
                                         ks1=3, stride1=1, padding1=dilation, dilation1=dilation,
                                         ks2=3, stride2=1, padding2=dilation, dilation2=dilation)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.conv_block(x)
        return x

class DecBlock(nn.Module):
    """Decoder block"""
    def __init__(self,
                 lower_ch,
                 skip_ch,
                 out_ch,
                 trans_ks=2,
                 trans_stride=2,
                 is_trans_conv=False):
        super().__init__()
        if is_trans_conv:
            # The transposed convolution adjust lower_ch to skip_ch, following nnUNet.
            self.up_samp = nn.ConvTranspose2d(lower_ch, skip_ch, trans_ks, trans_stride, bias=False)
            self.conv_block = DoubleConv(2 * skip_ch, out_ch)
        else:
            self.up_samp = nn.Upsample(scale_factor=trans_stride, mode='bilinear', align_corners=True)
            self.conv_block = DoubleConv(lower_ch+skip_ch, skip_ch)


    def forward(self, x, skip):
        if self.up_samp is not None:
            x = self.up_samp(x)
        x = self.conv_block(torch.cat((x, skip), 1))
        return x

class DoubleConv(nn.Module):
    """Convolutional block"""
    def __init__(self, in_ch, out_ch,
                 ks1=3, stride1=1, padding1=1, dilation1=1,
                 ks2=3, stride2=1, padding2=1, dilation2=1):
        super().__init__()
        # A severe bug ...
        # This causes unstable learning and performance decay.
        # self.conv1 = nn.Conv2d(in_ch, out_ch, ks1, stride1, padding1, dilation1)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, ks2, stride2, padding2, dilation2)
        self.conv_layer1 = ConvLayer(in_ch, out_ch,
                                     ks1, stride1, padding1, dilation1,
                                     norm_op=nn.BatchNorm2d,
                                     nonlin_op=nn.LeakyReLU, negative_slop=1e-2,
                                     )
        self.conv_layer2 = ConvLayer(out_ch, out_ch,
                                     ks2, stride2, padding2, dilation2,
                                     norm_op=nn.BatchNorm2d,
                                     nonlin_op=nn.LeakyReLU, negative_slop=1e-2,
                                     )

    def forward(self, x):
        return self.conv_layer2(self.conv_layer1(x))

class ConvLayer(nn.Module):
    """Convolutional layer"""
    def __init__(self, in_ch, out_ch,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 norm_op=nn.BatchNorm2d,
                 nonlin_op=nn.LeakyReLU, negative_slop=1e-2,):
        super(ConvLayer, self).__init__()
        # Conv2d's weight is in default initialized by init.kaiming_uniform_ and
        # bias by init.uniform_. Thus, doing initialization is not necessary.
        # self.apply fn can be used for initialization.
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.norm_op = norm_op(out_ch)
        self.nonlin_op = nonlin_op(negative_slop)

    def forward(self, x):
        return self.nonlin_op(self.norm_op(self.conv(x)))

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # cfgs = dict(input_ch=1, init_ch=16, no_classes=4,
    #             elab_end_points=True, is_stride_conv=False, is_trans_conv=False)
    x = torch.randn((1, 1, 224, 224)).cuda()
    # model = UNet(**cfgs).cuda()
    model = UNet(is_stride_conv=False, is_trans_conv=False, output_stride=32).cuda()
    print(model)
    end_points = model(x)
    print('Output shape {}'.format(end_points['segmentation/logits'].shape))
    pass




## The following programs are not maintained since 11/2/2021.
# class UNet(nn.Module):
#     def __init__(self, input_ch=1, init_ch=16, no_classes=4, output_stride=32,
#                  elab_end_points=False, is_stride_conv=False, is_trans_conv=False, is_dropout=False):
#         """
#
#         :param input_ch: number of input image channels
#         :param init_ch: number of initial channels
#         :param no_classes: number of classes
#         :param output_stride: output stride of encoder. Set a output stride as such each of output shape dimensions is smaller than 10.
#         :param elab_end_points: whether to elaborate blocks' end points
#         :param is_stride_conv: whether to use stride convolution in encoder block
#         :param is_trans_conv: whether to use transposed convolution in decoder block
#         """
#         super(UNet, self).__init__()
#         self.elab_end_points = elab_end_points
#         self.encoder_net = UNetEncoder(input_ch, init_ch, output_stride)
#         self.decoder_net = UNetDecoder(init_ch, no_classes, output_stride)
#
#     def forward(self, x):
#         encoder_end_points = self.encoder_net(x)
#         decoder_end_points = self.decoder_net(encoder_end_points)
#         if self.elab_end_points:
#             return encoder_end_points, decoder_end_points
#         else:
#             return decoder_end_points[-1]
#
# class UNetEncoder(nn.Module):
#     """UNet encoder"""
#     def __init__(self, input_ch=1, init_ch=16, output_stride=32, is_stride_conv=False):
#         super(UNetEncoder, self).__init__()
#         self.input_ch = input_ch
#         self.init_ch = init_ch
#         self.output_stride = 32
#         self.enc_depth, self.block_out_ch = self.compute_out_chns()
#
#         block_ls = []
#         for dph in range(self.enc_depth):
#             is_pool = False if dph == 0 else True
#             in_ch = self.block_out_ch[dph]
#             out_ch = self.block_out_ch[dph+1]
#             block_ls.append(EncBlock(in_ch, out_ch , is_pool, is_stride_conv))
#         self.encoder_block = nn.Sequential(*block_ls)
#
#     def compute_out_chns(self):
#         """We double the number of output channels in each encoder stage (except stage-one)"""
#         enc_depth = np.log2(self.output_stride) + 1  # no. max-pooling + 1
#         assert enc_depth % 1 == 0
#         enc_depth = int(enc_depth)
#         block_out_ch = [self.input_ch] + [2**d * self.init_ch for d in range(enc_depth)]  # 2**0 ~ 2**5
#         return enc_depth, block_out_ch
#
#     def forward(self, x):
#         encoder_end_points = []
#         for idx, e_blk in enumerate(self.encoder_block):
#             x = e_blk(x)
#             encoder_end_points.append(x)
#         return encoder_end_points
#
#
# class UNetDecoder(nn.Module):
#     """UNet decoder"""
#     def __init__(self, init_ch=16, no_classes=4, output_stride=32, is_trans_conv=False):
#         super(UNetDecoder, self).__init__()
#         self.init_ch = init_ch
#         self.no_classes = no_classes
#         self.output_stride = output_stride
#         self.dec_depth, self.block_in_ch = self.compute_in_chns()
#
#         block_ls = []
#         for dph in range(self.dec_depth):
#             in_ch = self.block_in_ch[dph]+self.block_in_ch[dph+1]
#             out_ch = self.block_in_ch[dph+1]
#             block_ls.append(DecBlock(in_ch, out_ch, is_trans_conv))
#         self.decoder_block = nn.Sequential(*block_ls)
#         self.final_conv = nn.Conv2d(self.init_ch, self.no_classes, 1, 1)
#
#     def compute_in_chns(self):
#         """We halve the number of input channels in each decoder stage"""
#         dec_depth = np.log2(self.output_stride)
#         assert dec_depth % 1 == 0
#         dec_depth = int(dec_depth)
#         block_in_ch = [2**d * self.init_ch for d in range(dec_depth, 0, -1)] + [self.init_ch] # 2**5 ~ 2**1
#         return dec_depth, block_in_ch
#
#     def forward(self, enc_end_points):
#         skips = enc_end_points[:-1][::-1]  # skip features in reverse order
#         x = enc_end_points[-1]  # decoder input
#         dec_end_points = []
#         for idx, d_blk in enumerate(self.decoder_block):
#             x = d_blk(x, skips[idx])
#             dec_end_points.append(x)
#         logits = self.final_conv(x)
#         dec_end_points.append(logits)
#         return dec_end_points
#
#
# class EncBlock(nn.Module):
#     """Encoder block"""
#     def __init__(self, in_ch, out_ch, is_pool=True, is_stride_conv=False):
#         super().__init__()
#         self.pooling = None
#         if is_pool:
#             if not is_stride_conv:
#                 self.pooling = nn.MaxPool2d(2, 2)
#             else:
#                 self.pooling = nn.Conv2d(in_ch, in_ch, 2, 2)
#         self.conv_block = DoubleConv(in_ch, out_ch)
#
#     def forward(self, x):
#         if self.pooling is not None:
#             x = self.pooling(x)
#         x = self.conv_block(x)
#         return x
#
# class DecBlock(nn.Module):
#     """Decoder block"""
#     def __init__(self, in_ch, out_ch, is_trans_conv=False):
#         super().__init__()
#         self.up_samp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         if is_trans_conv:
#             self.up_samp = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
#         self.conv_block = DoubleConv(in_ch, out_ch)
#
#     def forward(self, x, skip):
#         x = self.up_samp(x)
#         x = self.conv_block(torch.cat((x, skip), 1))
#         return x
#
# class DoubleConv(nn.Module):
#     """Convolutional block"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv_blk = nn.Sequential(
#         ConvLayer(in_ch, out_ch),
#         ConvLayer(out_ch, out_ch)
#         )
#
#     def forward(self, x):
#         return self.conv_blk(x)
#
# class ConvLayer(nn.Module):
#     """Convolutional layer"""
#     def __init__(self, in_ch, out_ch,
#                  conv_kwargs=None,
#                  dropout_p=0.5, is_dropout=False,
#                  norm_op=nn.BatchNorm2d,
#                  nonlin_op=nn.LeakyReLU, negative_slop=1e-2, ):
#         super(ConvLayer, self).__init__()
#         if conv_kwargs is None:
#             conv_kwargs = dict(kernel_size=3, stride=1, padding=1)
#         self.conv = nn.Conv2d(in_ch, out_ch, **conv_kwargs)
#         self.dropout_op = None
#         if is_dropout:
#             self.dropout_op = nn.Dropout2d(p=dropout_p)
#         self.norm_op = norm_op(out_ch)
#         self.nonlin_op = nonlin_op(negative_slop)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.dropout_op is not None:
#             x = self.dropout_op(x)
#         return self.nonlin_op(self.norm_op(x))
#
#
# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#     cfgs = dict(input_ch=1, init_ch=16, no_classes=4, output_stride=32,
#                 elab_end_points=False, is_stride_conv=False, is_trans_conv=False)
#     x = torch.randn((1, 1, 224, 224)).cuda()
#     model = UNet(**cfgs).cuda()
#     print(model)
#     dec_end = model(x)
#     print('Output shape {}'.format(dec_end.shape))
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class UnifiedAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(UnifiedAttention, self).__init__()
        # 通道注意力部分
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        # 空间注意力部分
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_pool = self.global_pool(x)
        x_channel = self.fc1(avg_pool)
        x_channel = self.fc2(x_channel)
        channel_attention = self.sigmoid(x_channel)

        # 空间注意力
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        max_pool_spatial, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        spatial_attention = self.conv(combined)
        spatial_attention = self.sigmoid(spatial_attention)

        # 综合注意力结果
        attention = channel_attention * spatial_attention
        return x * attention


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class AMSTHead(BaseDecodeHead):
    """
    """
    def __init__(self, feature_strides, **kwargs):
        super(AMSTHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.ca_sa_c4 = UnifiedAttention(c4_in_channels)
        self.ca_sa_c3 = UnifiedAttention(c3_in_channels)
        self.ca_sa_c2 = UnifiedAttention(c2_in_channels)
        self.ca_sa_c1 = UnifiedAttention(c1_in_channels)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.fuse_c4_c3 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=3, padding=1)
        self.fuse_c3_c2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=3, padding=1)
        self.fuse_c2_c1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=3, padding=1)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        c4 = self.ca_sa_c4(c4)
        c3 = self.ca_sa_c3(c3)
        c2 = self.ca_sa_c2(c2)
        c1 = self.ca_sa_c1(c1)

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        #_c4 = resize(_c4, size=c3.size()[2:],mode='bilinear',align_corners=False)
        _c4 = resize(_c4, size=c3.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = torch.cat([_c4, _c3], dim=1)
        _c3 = self.fuse_c4_c3(_c3)
        _c3 = resize(_c3, size=c2.size()[2:],mode='bilinear',align_corners=False)
        #_c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = torch.cat([_c3, _c2], dim=1)
        _c2 = self.fuse_c3_c2(_c2)
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        #_c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = torch.cat([_c2, _c1], dim=1)
        _c = self.fuse_c2_c1(_c)
        #_c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
import sys
import torch.fft
import math

import traceback

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
import torch.utils.checkpoint as checkpoint


if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())
        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

        print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

    print('Using PyTorch large kernel dw conv impl')



class Linear_kernel(nn.Module):
    def __init__(self, rank, Module_list_a, Module_list_b):
        super().__init__()
        self.rank = rank
        self.Module_list_a = Module_list_a
        self.Module_list_b = Module_list_b

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(self.rank):
            output = output + torch.mul(self.Module_list_a[i](x), self.Module_list_b[i](x))
        
        return output

class RBF_kernel(nn.Module):
    def __init__(self, in_channel, rank ,Module_list_a, Module_list_b, sigma = 1):
        super().__init__()
        self.sigma = sigma
        self.rank = rank
        self.Module_list_a = Module_list_a
        self.Module_list_b = Module_list_b
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        exp = torch.zeros_like(x)
        for i in range(self.rank):
            exp = exp + torch.pow((self.act(self.Module_list_a[i](x))-self.act(self.Module_list_b[i](x))), 2)/ (2* (self.sigma ** 2))
        exp = self.bn(exp)
        output = torch.exp(-exp)


        return output

class RBF_kernel_in(nn.Module):
    def __init__(self, in_channel, rank ,Module_list_a, Module_list_b, sigma = 1):
        super().__init__()
        self.in_pro_a = nn.Conv2d(in_channel, in_channel, 1)
        self.in_pro_b = nn.Conv2d(in_channel, in_channel, 1)
        self.sigma = sigma
        self.rank = rank
        self.Module_list_a = Module_list_a
        self.Module_list_b = Module_list_b
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        x_a = self.in_pro_a(x)
        x_b = self.in_pro_b(x)
        exp = torch.zeros_like(x)
        for i in range(self.rank):
            exp = exp + torch.pow((self.act(self.Module_list_a[i](x_a))-self.act(self.Module_list_b[i](x_b))), 2)/ (2* (self.sigma ** 2))
        exp = self.bn(exp)
        output = torch.exp(-exp)


        return output


class Kernel_Conv(nn.Module):
    def __init__(self, in_channel, rank = 7, kernel_size = 7, bias = True, kernel = 'linear'):
        super(Kernel_Conv, self).__init__()
        self.dwconv_a = nn.ModuleList(
            [nn.Conv2d(in_channel,in_channel,7,1,'same',groups = in_channel) for i in range(rank)]
        )
        self.dwconv_b = nn.ModuleList(
            [nn.Conv2d(in_channel,in_channel,7,1,'same',groups = in_channel) for i in range(rank)]
        )
        
        if (kernel == 'linear'):
            self.kernel = Linear_kernel(rank, self.dwconv_a, self.dwconv_b)
        elif(kernel == 'RBF'):
            self.kernel = RBF_kernel(in_channel, rank, self.dwconv_a, self.dwconv_b, sigma = 1)
        elif(kernel == 'RBF_in'):
            self.kernel = RBF_kernel_in(in_channel, rank, self.dwconv_a, self.dwconv_b, sigma = 1)

        self.conv_linear_term = get_dwconv(dim = in_channel, kernel= kernel_size, bias = bias)
        
    def forward(self, x):
        x_1 = self.conv_linear_term(x)
        x = self.kernel(x)
        x = x + x_1

        return x

class QuadraBlock1_dw(nn.Module):
    def __init__(self, dim, kernel, drop_path=0., layer_scale_init_value=1e-6, QuadraConv=Kernel_Conv):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.QuadraConv = QuadraConv(dim,kernel = kernel) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.QuadraConv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


@BACKBONES.register_module()
class QuadraNet(nn.Module):
    r""" HorNet
        A PyTorch impl of : `HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions`

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 QuadraConv = Kernel_Conv, block=QuadraBlock1_dw, out_indices=[0, 1, 2, 3],
                 pretrained=None,
                 use_checkpoint=False,
                 kernel = 'linear',
                 ):
        super().__init__()

        self.out_indices = out_indices
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint

        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 


        if not isinstance(QuadraConv, list):
            QuadraConv = [QuadraConv, QuadraConv, QuadraConv, QuadraConv]
        else:
            QuadraConv = QuadraConv
            assert len(QuadraConv) == 4

        if isinstance(QuadraConv[0], str):
            print('[GConvNet]: convert str gconv to func')
            QuadraConv = [eval(g) for g in QuadraConv]

        if isinstance(block, str):
            block = eval(block)

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], kernel= kernel ,drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, QuadraConv=QuadraConv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            raise NotImplementedError()
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint_sequential(self.stages[i], len(self.stages[i]), x)
            else:
                x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
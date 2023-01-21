# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch.nn as nn

from co3d_3d.src.models.mink.modules.common import conv, get_nonlinearity, get_norm


class BasicBlockBase(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        norm_type="BN",
        nonlinearity_type="MinkowskiReLU",
        bn_momentum=0.1,
        D=3,
        conv_mode=0,
    ):
        super(BasicBlockBase, self).__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            D=D,
            conv_mode=conv_mode,
        )
        self.norm1 = get_norm(norm_type, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            D=D,
            conv_mode=conv_mode,
        )
        self.norm2 = get_norm(norm_type, planes, D, bn_momentum=bn_momentum)
        self.downsample = downsample
        self.nonlinearity = get_nonlinearity(nonlinearity_type)()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlinearity(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlinearity(out)

        return out


class BasicBlock(BasicBlockBase):
    pass


class BottleneckBase(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        norm_type="BN",
        nonlinearity_type="ReLU",
        bn_momentum=0.1,
        D=3,
    ):
        super(BottleneckBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(norm_type, planes, D, bn_momentum=bn_momentum)

        self.conv2 = conv(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, D=D
        )
        self.norm2 = get_norm(norm_type, planes, D, bn_momentum=bn_momentum)

        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(
            norm_type, planes * self.expansion, D, bn_momentum=bn_momentum
        )

        self.downsample = downsample
        self.nonlinearity = get_nonlinearity(nonlinearity_type)()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlinearity(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.nonlinearity(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlinearity(out)

        return out


class Bottleneck(BottleneckBase):
    pass


def get_block(
    norm_type,
    inplanes,
    planes,
    stride=1,
    dilation=1,
    downsample=None,
    bn_momentum=0.1,
    D=3,
):

    return BasicBlock(
        inplanes,
        planes,
        stride,
        dilation,
        downsample,
        norm_type,
        nonlinearity_type="MinkowskiReLU",
        bn_momentum=bn_momentum,
        D=D,
    )

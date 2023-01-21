# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import MinkowskiEngine as ME
import torch
import torch.nn as nn

from co3d_3d.src.models.mink.base_model import MinkowskiBaseModel
from co3d_3d.src.models.mink.modules.common import conv, get_norm
from co3d_3d.src.models.mink.modules.resnet_block import BasicBlock, Bottleneck


class GlobalAvgPool(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, tensor):
        y = self.global_avg_pool(tensor)
        return y


class ResNetBase(MinkowskiBaseModel):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    NORM_TYPE = "BN"

    def __init__(self, in_channel, out_channel, D=3):
        MinkowskiBaseModel.__init__(self, D)

        self.D = D

        self.network_initialization(in_channel, out_channel, D=D)
        self.weight_initialization()

    def network_initialization(
        self, in_channels, out_channels, nonlinearity="ReLU", bn_momentum=0.1, D=3
    ):
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(3, 1),
            stride=1,
            D=D,
        )

        self.bn1 = get_norm(
            self.NORM_TYPE, self.inplanes, D=self.D, bn_momentum=bn_momentum
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiSumPooling(
            kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), dimension=D
        )

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
        )
        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
        )
        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
        )
        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
        )

        self.glob_avg = GlobalAvgPool()

        self.final = conv(
            self.PLANES[3] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            D=D,
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type="BN",
        nonlinearity_type="MinkowskiReLU",
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D,
                ),
                get_norm(
                    norm_type,
                    planes * block.expansion,
                    D=self.D,
                    bn_momentum=bn_momentum,
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                nonlinearity_type=nonlinearity_type,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    nonlinearity_type=nonlinearity_type,
                    D=self.D,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.sparse()
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.glob_avg(out)
        out = self.final(out)
        return out.F


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)

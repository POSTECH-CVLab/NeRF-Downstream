# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import unittest

import gin
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
import torch
import torch.nn as nn

from co3d_3d.src.models.mink.modules.common import (
    conv,
    conv_tr,
    get_nonlinearity,
    get_norm,
)
from co3d_3d.src.models.mink.modules.encoding import MinkowskiPositionalEncoding
from co3d_3d.src.models.mink.modules.resnet_block import BasicBlock, Bottleneck


@gin.configurable
class Res16UNet(nn.Module):
    INSSEG = False
    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self,
        in_channel,
        out_channel,
        PLANES=(32, 48, 64, 96, 96, 96, 64, 64),
        DILATIONS=(1, 1, 1, 1, 1, 1, 1, 1),
        LAYERS=(2, 2, 2, 2, 2, 2, 2, 2),
        BLOCK=BasicBlock,
        NORM_TYPE="BN",
        nonlinearity="MinkowskiReLU",
        bn_momentum=0.1,
        D=3,
        sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        nn.Module.__init__(self)
        self.nonlinearity = nonlinearity
        self.bn_momentum = bn_momentum
        self.D = D

        self.BLOCK = BLOCK
        self.PLANES = PLANES
        self.DILATIONS = DILATIONS
        self.LAYERS = LAYERS
        self.NORM_TYPE = NORM_TYPE

        self.network_initialization(
            in_channel, out_channel, nonlinearity, bn_momentum, D, sparse_mode
        )
        self.weight_initialization()

    def network_initialization(
        self,
        in_channel,
        out_channel,
        nonlinearity="MinkowskiReLU",
        bn_momentum=0.1,
        D=3,
        sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        # Setup net_metadata

        # Output of the first conv concated to conv6
        self.conv0p1s1 = nn.Sequential(
            conv(
                in_channel,
                self.PLANES[0],
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
            conv(
                self.PLANES[0],
                self.PLANES[0],
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )

        self.conv1p1s2 = nn.Sequential(
            conv(
                self.PLANES[0],
                self.PLANES[0],
                kernel_size=2,
                stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )

        self.inplanes = self.PLANES[0]
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[1],
        )

        self.conv2p2s2 = nn.Sequential(
            conv(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[2],
        )

        self.conv3p4s2 = nn.Sequential(
            conv(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[3],
        )

        self.conv4p8s2 = nn.Sequential(
            conv(
                self.inplanes,
                self.inplanes,
                kernel_size=2,
                stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[4],
        )
        self.convtr4p16s2 = nn.Sequential(
            conv_tr(
                self.inplanes,
                self.PLANES[4],
                kernel_size=2,
                upsample_stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[5],
        )
        self.convtr5p8s2 = nn.Sequential(
            conv_tr(
                self.inplanes,
                self.PLANES[5],
                kernel_size=2,
                upsample_stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[6],
        )

        self.convtr6p4s2 = nn.Sequential(
            conv_tr(
                self.inplanes,
                self.PLANES[6],
                kernel_size=2,
                upsample_stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[7],
        )
        self.convtr7p2s2 = nn.Sequential(
            conv_tr(
                self.inplanes,
                self.PLANES[7],
                kernel_size=2,
                upsample_stride=2,
                dilation=1,
                bias=False,
                D=D,
                conv_mode=sparse_mode[0],
            ),
            get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity)(),
        )

        self.inplanes = self.PLANES[7] + self.PLANES[0]
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            norm_type=self.NORM_TYPE,
            nonlinearity_type=nonlinearity,
            bn_momentum=bn_momentum,
            conv_mode=sparse_mode[8],
        )

        self.final = conv(
            self.PLANES[7],
            out_channel,
            kernel_size=1,
            stride=1,
            bias=True,
            D=D,
            conv_mode=sparse_mode[0],
        )

        if self.INSSEG:
            self.offset_block = nn.Sequential(
                conv(
                    self.inplanes,
                    self.inplanes,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    D=D,
                ),
                get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum),
                get_nonlinearity(self.nonlinearity)(),
                conv(self.inplanes, 3, kernel_size=1, stride=1, bias=True, D=D),
            )

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type="BN",
        nonlinearity_type="ReLU",
        bn_momentum=0.1,
        conv_mode=0,
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
                    conv_mode=conv_mode,
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
                norm_type=norm_type,
                nonlinearity_type=nonlinearity_type,
                D=self.D,
                conv_mode=conv_mode,
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
                    norm_type=norm_type,
                    nonlinearity_type=nonlinearity_type,
                    D=self.D,
                    conv_mode=conv_mode,
                )
            )

        return nn.Sequential(*layers)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: ME.TensorField):
        out = x.sparse()
        out_p1 = self.conv0p1s1(out)

        out = self.conv1p1s2(out_p1)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = me.cat(out, out_p1)
        out = self.block8(out)

        if self.INSSEG:
            offsets = self.offset_block(out)
            out = self.final(out)
            return offsets.slice(x).F, out.slice(x).F

        else:
            out = self.final(out)
            return out.slice(x).F


class Res16UNet14(Res16UNet):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

    def __init__(
        self, in_channel, out_channel, sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    ):
        Res16UNet.__init__(
            self,
            in_channel,
            out_channel,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
            BLOCK=self.BLOCK,
            sparse_mode=sparse_mode,
        )


class Res16UNet18(Res16UNet):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

    def __init__(
        self, in_channel, out_channel, sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    ):
        Res16UNet.__init__(
            self,
            in_channel,
            out_channel,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
            BLOCK=self.BLOCK,
            sparse_mode=sparse_mode,
        )


class Res16UNet34(Res16UNet):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

    def __init__(
        self, in_channel, out_channel, sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    ):
        Res16UNet.__init__(
            self,
            in_channel,
            out_channel,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
            BLOCK=self.BLOCK,
            sparse_mode=sparse_mode,
        )


class Res16UNet50(Res16UNet):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

    def __init__(
        self, in_channel, out_channel, sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    ):
        Res16UNet.__init__(
            self,
            in_channel,
            out_channel,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
            BLOCK=self.BLOCK,
            sparse_mode=sparse_mode,
        )


class Res16UNet101(Res16UNet):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)

    def __init__(
        self, in_channel, out_channel, sparse_mode=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    ):
        Res16UNet.__init__(
            self,
            in_channel,
            out_channel,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,
            BLOCK=self.BLOCK,
            sparse_mode=sparse_mode,
        )


class Res16UNet14A(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
    LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18C(Res16UNet18):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Res16UNet18D(Res16UNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Res16UNet14AIns(Res16UNet14A):
    INSSEG = True


class Res16UNet14BIns(Res16UNet14B):
    INSSEG = True


class Res16UNet18AIns(Res16UNet18A):
    INSSEG = True


class Res16UNet18BIns(Res16UNet18B):
    INSSEG = True


class Res16UNet34CIns(Res16UNet34C):
    INSSEG = True


@gin.configurable
class EncodedRes16UNet(Res16UNet):

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self,
        in_channel,
        out_channel,
        ENC_PLANES=(32, 32),
        DEC_PLANES=(48, 48),
        D=3,
    ):
        Res16UNet.__init__(
            self,
            in_channel=ENC_PLANES[-1],
            out_channel=out_channel,
            D=D,
        )

        self.ENC_PLANES = ENC_PLANES
        self.DEC_PLANES = DEC_PLANES
        self.init_mlp(in_channel, out_channel)

    def init_mlp(self, in_channel, out_channel):
        ENC_PLANES = self.ENC_PLANES
        DEC_PLANES = self.DEC_PLANES

        self.encoding = MinkowskiPositionalEncoding(
            in_channel,
        )

        self.enc_mlp = nn.Sequential(
            self.get_mlp_block(self.encoding.out_channels, ENC_PLANES[0]),
            *[
                self.get_mlp_block(ENC_PLANES[i], ENC_PLANES[i + 1])
                for i in range(len(ENC_PLANES) - 1)
            ],
        )

        dec_in_channel = self.block8[-1].conv2.out_channels + ENC_PLANES[-1]
        self.dec_mlp = nn.Sequential(
            self.get_mlp_block(dec_in_channel, DEC_PLANES[0]),
            *[
                self.get_mlp_block(DEC_PLANES[i], DEC_PLANES[i + 1])
                for i in range(len(DEC_PLANES) - 1)
            ],
        )

        self.final = ME.MinkowskiLinear(DEC_PLANES[-1], out_channel, bias=True)

    def get_mlp_block(self, in_chanel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_chanel, out_channel, bias=True),
            get_nonlinearity(self.nonlinearity)(),
        )

    def forward(self, x: ME.TensorField):
        with torch.no_grad():
            enc_x = self.encoding(x)
        enc_x = self.enc_mlp(enc_x)
        out = enc_x.sparse(
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        )

        out_p1 = self.conv0p1s1(out)

        out = self.conv1p1s2(out_p1)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = me.cat(out, out_p1)
        out = self.block8(out)

        out_x = out.slice(enc_x)
        out_x = self.dec_mlp(ME.cat(enc_x, out_x))
        return self.final(out_x).F


class EncodedRes16UNet2(EncodedRes16UNet):

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self,
        in_channel,
        out_channel,
        D=3,
    ):
        EncodedRes16UNet.__init__(
            self,
            in_channel,
            out_channel,
            D=D,
        )

    def init_mlp(self, in_channel, out_channel):
        ENC_PLANES = self.ENC_PLANES
        DEC_PLANES = self.DEC_PLANES

        self.encoding = MinkowskiPositionalEncoding(in_channel)

        self.enc_mlp = nn.Sequential(
            self.get_mlp_block(self.encoding.out_channels, ENC_PLANES[0]),
            *[
                self.get_mlp_block(ENC_PLANES[i], ENC_PLANES[i + 1])
                for i in range(len(ENC_PLANES) - 1)
            ],
        )

        dec_in_channel = self.encoding.out_channels + self.block8[-1].conv2.out_channels
        self.dec_mlp = nn.Sequential(
            self.get_mlp_block(dec_in_channel, DEC_PLANES[0]),
            *[
                self.get_mlp_block(DEC_PLANES[i], DEC_PLANES[i + 1])
                for i in range(len(DEC_PLANES) - 1)
            ],
        )

        self.final = ME.MinkowskiLinear(DEC_PLANES[-1], out_channel, bias=True)

    def forward(self, x: ME.TensorField):
        with torch.no_grad():
            enc_x = self.encoding(x)
        enc_mlp = self.enc_mlp(enc_x)
        out = enc_mlp.sparse(
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        )

        out_p1 = self.conv0p1s1(out)

        out = self.conv1p1s2(out_p1)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = me.cat(out, out_p1)
        out_x = self.block8(out).slice(enc_mlp)

        out_x = self.dec_mlp(ME.cat(enc_x, out_x))
        return self.final(out_x).F


class ResUNetTestCase(unittest.TestCase):
    def test(self):
        unet = Res16UNet(in_channel=3, out_channel=20, D=3)
        logging.info(unet)

    def test_enc(self):
        in_channel, N = 6, 1000
        unet = EncodedRes16UNet(in_channel=in_channel, out_channel=20, D=3)
        logging.info(unet)
        input = ME.TensorField(
            coordinates=torch.rand(N, 4) / 0.01, features=torch.rand(N, in_channel)
        )
        unet(input)

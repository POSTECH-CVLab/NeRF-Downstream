# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
from enum import Enum
from typing import Union

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch.nn as nn

from co3d_3d.src.models.mink.modules.layernorm import MinkowskiLayerNorm
from co3d_3d.src.models.mink.modules.powernorm import MinkowskiPowerNorm
from co3d_3d.src.models.mink.modules.sparse_conv import (
    SparseConvMode,
    WeightSparseConvolution,
    WeightSparseConvolutionTranspose,
)


def get_norm(norm_type: str, n_channels: int, D: int, bn_momentum: float = 0.1):
    if norm_type == "BN":
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == "IN":
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == "LN":
        return MinkowskiLayerNorm(n_channels)
    elif norm_type == "PN":
        return MinkowskiPowerNorm(n_channels)
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


NONLINEARITIES = [
    ME.MinkowskiReLU,
    ME.MinkowskiPReLU,
    ME.MinkowskiLeakyReLU,
    ME.MinkowskiELU,
    ME.MinkowskiCELU,
    ME.MinkowskiSELU,
    ME.MinkowskiGELU,
]

NONLINEARITIES_dict = {i: n for i, n in enumerate(NONLINEARITIES)}
for n in NONLINEARITIES:
    NONLINEARITIES_dict[n.__name__] = n


def get_nonlinearity(nonlinearity_type: Union[str, int]):
    return NONLINEARITIES_dict[nonlinearity_type]


def get_nonlinearity_fn(
    nonlinearity_type: str, input: ME.SparseTensor, *args, **kwargs
):
    if nonlinearity_type == "MinkowskiReLU":
        return MEF.relu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiLeakyReLU":
        return MEF.leaky_relu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiPReLU":
        return MEF.prelu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiCELU":
        return MEF.celu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiSELU":
        return MEF.selu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiGELU":
        return MEF.gelu(input, *args, **kwargs)
    else:
        raise ValueError(f"Norm type: {nonlinearity_type} not supported")


def conv(
    in_planes,
    out_planes,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    D=-1,
    conv_mode=SparseConvMode.DENSE,
):
    assert D > 0, "Dimension must be a positive integer"
    conv_mode = (
        conv_mode
        if isinstance(conv_mode, SparseConvMode)
        else SparseConvMode(conv_mode)
    )
    if conv_mode == SparseConvMode.NATIVE and hasattr(
        ME, "MinkowskiWeightSparseConvolution"
    ):
        return ME.MinkowskiWeightSparseConvolution(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=D,
        )
    elif conv_mode in [
        SparseConvMode.SPARSE,
        SparseConvMode.ZAXIS,
        SparseConvMode.SKIP,
    ]:
        return WeightSparseConvolution(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=D,
            sparse_mode=conv_mode,
        )
    else:
        return ME.MinkowskiConvolution(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=D,
        )


def conv_tr(
    in_planes,
    out_planes,
    kernel_size,
    upsample_stride=1,
    dilation=1,
    bias=False,
    D=-1,
    conv_mode=SparseConvMode.DENSE,
):
    assert D > 0, "Dimension must be a positive integer"
    conv_mode = (
        conv_mode
        if isinstance(conv_mode, SparseConvMode)
        else SparseConvMode(conv_mode)
    )
    if conv_mode == SparseConvMode.NATIVE and hasattr(
        ME, "MinkowskiWeightSparseConvolutionTranspose"
    ):
        return ME.MinkowskiWeightSparseConvolutionTranspose(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=upsample_stride,
            dilation=dilation,
            bias=bias,
            dimension=D,
        )
    elif conv_mode in [
        SparseConvMode.SPARSE,
        SparseConvMode.ZAXIS,
        SparseConvMode.SKIP,
    ]:
        return WeightSparseConvolutionTranspose(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=upsample_stride,
            dilation=dilation,
            bias=bias,
            dimension=D,
            sparse_mode=conv_mode,
        )
    else:
        return ME.MinkowskiConvolutionTranspose(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=upsample_stride,
            dilation=dilation,
            bias=bias,
            dimension=D,
        )


def conv_norm_non(
    in_planes,
    out_planes,
    kernel_size,
    stride=1,
    dilation=1,
    norm_type="BN",
    nonlinearity_type="MinkowskiLeakyReLU",
    D=3,
):
    return nn.Sequential(
        conv(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            D=D,
        ),
        get_norm(norm_type, out_planes, D=D),
        get_nonlinearity(nonlinearity_type)(),
    )

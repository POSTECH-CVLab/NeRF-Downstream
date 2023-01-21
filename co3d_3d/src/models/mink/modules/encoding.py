# Copyright 2020 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging
import unittest
from typing import Union

import gin
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn


@gin.configurable
class PositionEmbedder(nn.Module):
    def __init__(
        self,
        channel,
        max_frequency,
        num_frequencies,
        scale=1,
        include_channel=False,
    ):
        nn.Module.__init__(self)
        self.max_frequency = max_frequency
        self.num_frequencies = num_frequencies
        self.freqs = 2.0 ** torch.linspace(0.0, max_frequency, steps=num_frequencies)
        out_channel = 0

        embed_fns = []
        if include_channel:
            embed_fns.append(lambda x: x)
            out_channel += channel

        for freq in self.freqs:
            for f in [torch.sin, torch.cos]:
                embed_fns.append(lambda x: f(x * freq))
                out_channel += channel

        self.scale = scale
        self.embed_fns = embed_fns
        self.out_channel = out_channel

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(max_freq={self.max_frequency} num_frequencies={self.num_frequencies} scale={self.scale})"
        )

    @torch.no_grad()
    def forward(self, x):
        x *= self.scale
        return torch.cat([fn(x) for fn in self.embed_fns], -1)


@gin.configurable
class MinkowskiPositionalEncoding(ME.MinkowskiModuleBase):
    def __init__(
        self,
        in_channel,
        num_encoding_functions=4,
        include_original_channel_range=None,
        min_resolution=None,
    ):
        r"""
        :attr:`min_resolution` (None or float): Minimum resolution. If None,
        the frequencies will be set to [2^0, ..., 2^{num_encoding_functions - 1}].
        """
        ME.MinkowskiModuleBase.__init__(self)

        # num_eoncoding_fn must be divisible by the num_groups
        # assert (
        #     num_encoding_functions % num_groups == 0
        # ), "Invalid configurations, num_groups:{num_groups} % num_encoding_functions:{num_encoding_functions} != 0"

        # Min resolution = min (Ts / 2) = min (1 / 2 / Fs) = 0.5 * 1 / max (Fs)
        self.num_encoding_functions = num_encoding_functions
        if num_encoding_functions < 1:
            self.shape = (in_channel, in_channel)
            return

        # if min_resolution is
        if min_resolution is not None:
            assert isinstance(
                min_resolution, (int, float)
            ), f"{min_resolution} must be an int or float"
            max_freq_exponent = np.log2(0.5 / min_resolution)
            frequencies = 2.0 ** (
                torch.linspace(
                    max_freq_exponent - num_encoding_functions - 1,
                    max_freq_exponent,
                    num_encoding_functions,
                )
            )
        else:
            frequencies = 2.0 ** torch.arange(num_encoding_functions)
        self.include_original_channel_range = include_original_channel_range
        include_channels = (
            0
            if include_original_channel_range is None
            else (include_original_channel_range[1] - include_original_channel_range[0])
        )
        self.out_enc_channel = in_channel * 2 * num_encoding_functions
        out_channel = self.out_enc_channel + include_channels
        logging.info(
            f"{self.__class__.__name__}: freqs: {frequencies}, {in_channel}, {out_channel}"
        )
        cols = torch.arange(in_channel * 2 * num_encoding_functions)
        rows = (
            torch.arange(in_channel)
            .unsqueeze(1)
            .repeat([1, 2 * num_encoding_functions])
            .view(-1)
        )
        coo = torch.stack([rows, cols])
        frequencies = frequencies.repeat(2 * in_channel)

        # self.mat = nn.Parameter(
        #     torch.sparse_coo_tensor(
        #         coo,
        #         frequencies,
        #         (in_channel, out_channel),
        #         requires_grad=False,
        #     ),
        #     requires_grad=False,
        # ).t()

        self.coo = nn.Parameter(coo, requires_grad=False)
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.shape = (in_channel, out_channel)
        self.offset = nn.Parameter(
            (np.pi / 2)
            * torch.arange(2)
            .unsqueeze(1)
            .repeat([1, in_channel * num_encoding_functions])
            .view(1, -1),
            requires_grad=False,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channel={self.shape[0]}, out_channel={self.shape[1]}, frequencies={self.frequencies})"

    @property
    def out_channels(self):
        return self.shape[1]

    def forward(self, input: Union[ME.SparseTensor, ME.TensorField]):
        if self.num_encoding_functions < 1:
            return input

        if not hasattr(self, "mat"):
            self.mat = nn.Parameter(
                torch.sparse_coo_tensor(
                    self.coo,
                    self.frequencies,
                    (self.shape[0], self.out_enc_channel),
                    requires_grad=False,
                    device=input.device,
                    dtype=input.dtype,
                ),
                requires_grad=False,
            ).t()

        F = (self.mat @ input.F.T).T + self.offset
        F = torch.sin(F)
        if self.include_original_channel_range is not None:
            F = torch.cat(
                (
                    F,
                    input.F[
                        :,
                        self.include_original_channel_range[
                            0
                        ] : self.include_original_channel_range[1],
                    ],
                ),
                dim=1,
            )

        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                F,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                F,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )


class PositionalEncodingTestCase(unittest.TestCase):
    def test(self):
        encoding = MinkowskiPositionalEncoding(3, 6, 0.01)
        coo = torch.IntTensor([[0, 0, 0], [0, 0, 1]])
        val = torch.rand(len(coo), 3)
        sinput = ME.SparseTensor(coordinates=coo, features=val)
        print(encoding(sinput))

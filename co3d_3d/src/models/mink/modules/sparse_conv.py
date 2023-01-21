import logging
import math
from enum import Enum
from typing import List, Optional, Union

import torch
from MinkowskiEngine.MinkowskiCommon import MinkowskiModuleBase
from MinkowskiEngine.MinkowskiCoordinateManager import CoordinateManager
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
from MinkowskiEngine.MinkowskiSparseTensor import CoordinateMapKey, SparseTensor
from MinkowskiEngine.sparse_matrix_functions import spmm
from MinkowskiEngineBackend._C import ConvolutionMode, CoordinateMapKey, RegionType
from torch.autograd import Function
from torch.nn import Parameter

from co3d_3d.src.utils import Timer


class SparseConvMode(Enum):
    DENSE = 0
    SPARSE = 1
    ZAXIS = 2
    NATIVE = 3
    SKIP = 4
    SPARSE_DENSE = 5


# A*B @ B*C
# A * B * C
def auto_spmm(sparse_mat, dense_mat, skip=False):
    # strided tensor
    ops = 0
    if sparse_mat.layout == torch.strided:
        if skip:
            output = torch.zeros((sparse_mat.shape[0], dense_mat.shape[1]))
        else:
            output = sparse_mat.matmul(dense_mat)
        ops = int(sparse_mat.shape[0] * dense_mat.numel())
    # coo tensor
    elif sparse_mat.layout == torch.sparse_coo:
        indices = sparse_mat.indices()
        values = sparse_mat.values()
        output = spmm(
            indices[0, :], indices[1, :], values, sparse_mat.size(), dense_mat, True
        )
    # csr tensor
    elif sparse_mat.layout == torch.sparse_csr:
        assert not sparse_mat.is_cuda and not dense_mat.is_cuda
        if skip:
            output = torch.ones((sparse_mat.shape[0], dense_mat.shape[1]))
        else:
            output = sparse_mat.matmul(dense_mat)
        ops = int(sparse_mat._nnz() * dense_mat.shape[1])
    return output, ops


class WeightSparseConvolutionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        kernel_weights: Union[torch.Tensor, List[torch.Tensor]],
        kernel_generator: KernelGenerator,
        convolution_mode: ConvolutionMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        sparse_mode: SparseConvMode = SparseConvMode.SPARSE,
        valid_kernel: Optional[List[int]] = None,
    ):
        timer = Timer()
        timer.tic()

        cm = coordinate_manager
        in_key = in_coordinate_map_key
        out_key = out_coordinate_map_key
        logging.debug(f"input preparation: {timer.toc(False):.3e}")

        timer.tic()
        N_out = cm.size(out_key)
        C_out = (
            kernel_weights.shape[-1]
            if isinstance(kernel_weights, torch.Tensor)
            else kernel_weights[0].shape[0]
        )
        out_F = input_features.new(N_out, C_out).zero_()
        logging.debug(f"output preparation: {timer.toc(False):.3e}")

        timer.tic()
        kernel_map = cm.kernel_map(
            in_key,
            out_key,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_size,
            kernel_generator.kernel_dilation,
        )
        logging.debug(f"kernel map retrieval: {timer.toc(False):.3e}")

        timer.tic()
        input_features = input_features.t().contiguous()
        logging.debug(f"input_feature transpose: {timer.toc(False):.3e}")

        timer_1, timer_2, timer_3, timer_4, timer_5 = (
            Timer(),
            Timer(),
            Timer(),
            Timer(),
            Timer(),
        )
        timer.tic()
        ops = 0
        # if sparse_mode == SparseConvMode.ZAXIS:
        #     for k in valid_kernel:
        #         in_out = kernel_map[k].long().to(input_features.device)
        #         input_F = input_features[:, in_out[0]]
        #         output_F, ops_i = auto_spmm(kernel_weights[k], input_F)
        #         ops += ops_i
        #         output_F = output_F.t()
        #         out_F[in_out[1]] += output_F
        # else:
        # for k, in_out in kernel_map.items():
        for k in valid_kernel:
            timer_1.tic()
            in_out = kernel_map[k].long().to(input_features.device)
            timer_1.toc()

            timer_2.tic()
            input_F = input_features[:, in_out[0]]
            timer_2.toc()

            timer_3.tic()
            output_F, ops_i = auto_spmm(
                kernel_weights[k], input_F, skip=sparse_mode == SparseConvMode.SKIP
            )
            ops += ops_i
            timer_3.toc()

            timer_4.tic()
            output_F = output_F.t()
            timer_4.toc()

            timer_5.tic()
            out_F[in_out[1]] += output_F
            timer_5.toc()

        logging.debug(f"for loop: {timer.toc(False):.3e}")
        logging.debug(f">>>> in_out.long.to(device)    : {timer_1.sum:.3e}")
        logging.debug(f">>>> input_features[in_out[0]] : {timer_2.sum:.3e}")
        logging.debug(f">>>> spmm                      : {timer_3.sum:.3e}")
        logging.debug(f">>>> output_F.t()              : {timer_4.sum:.3e}")
        logging.debug(f">>>> out_F += output_F         : {timer_5.sum:.3e}")
        return out_F, ops

    @staticmethod
    def backward(ctx, grad_out_feat: torch.Tensor):
        # do nothing
        return ()


class WeightSparseConvolutionTransposeFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        kernel_weights: torch.Tensor,
        kernel_generator: KernelGenerator,
        convolution_mode: ConvolutionMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        sparse_mode: SparseConvMode = SparseConvMode.SPARSE,
        valid_kernel: Optional[List[int]] = None,
    ):
        if out_coordinate_map_key is None:
            out_coordinate_map_key = CoordinateMapKey(
                in_coordinate_map_key.get_coordinate_size()
            )
        timer = Timer()
        timer.tic()

        cm = coordinate_manager
        in_key = in_coordinate_map_key
        out_key = out_coordinate_map_key
        logging.debug(f"input preparation: {timer.toc(False):.3e}")

        timer.tic()
        N_out = cm.size(out_key)
        C_out = (
            kernel_weights.shape[-1]
            if isinstance(kernel_weights, torch.Tensor)
            else kernel_weights[0].shape[0]
        )
        out_F = input_features.new(N_out, C_out).zero_()
        logging.debug(f"output preparation: {timer.toc(False):.3e}")

        timer.tic()
        kernel_map = cm.kernel_map(
            in_key,
            out_key,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_size,
            kernel_generator.kernel_dilation,
            is_transpose=True,
        )
        logging.debug(f"kernel map retrieval: {timer.toc(False):.3e}")

        timer.tic()
        input_features = input_features.t().contiguous()
        logging.debug(f"input_feature transpose: {timer.toc(False):.3e}")

        timer_1, timer_2, timer_3, timer_4, timer_5 = (
            Timer(),
            Timer(),
            Timer(),
            Timer(),
            Timer(),
        )
        timer.tic()
        ops = 0
        # if sparse_mode == SparseConvMode.ZAXIS:
        #     for k in valid_kernel:
        #         in_out = kernel_map[k].long().to(input_features.device)
        #         input_F = input_features[:, in_out[0]]
        #         output_F, ops_i = auto_spmm(kernel_weights[k], input_F)
        #         ops += ops_i
        #         output_F = output_F.t()
        #         out_F[in_out[1]] += output_F
        # else:
        #     for k, in_out in kernel_map.items():
        for k in valid_kernel:
            timer_1.tic()
            in_out = kernel_map[k].long().to(input_features.device)
            timer_1.toc()

            timer_2.tic()
            input_F = input_features[:, in_out[0]]
            timer_2.toc()

            timer_3.tic()
            output_F, ops_i = auto_spmm(
                kernel_weights[k], input_F, skip=sparse_mode == SparseConvMode.SKIP
            )
            ops += ops_i
            timer_3.toc()

            timer_4.tic()
            output_F = output_F.t()
            timer_4.toc()

            timer_5.tic()
            out_F[in_out[1]] += output_F
            timer_5.toc()
        logging.debug(f"for loop: {timer.toc(False):.3e}")
        logging.debug(f">>>> in_out.long.to(device)    : {timer_1.sum:.3e}")
        logging.debug(f">>>> input_features[in_out[0]] : {timer_2.sum:.3e}")
        logging.debug(f">>>> spmm                      : {timer_3.sum:.3e}")
        logging.debug(f">>>> output_F.t()              : {timer_4.sum:.3e}")
        logging.debug(f">>>> out_F += output_F         : {timer_5.sum:.3e}")
        return out_F, ops

    @staticmethod
    def backward(ctx, grad_out_feat: torch.Tensor):
        # do nothing
        return ()


class WeightSparseConvolutionBase(MinkowskiModuleBase):

    __slots__ = (
        "in_channels",
        "out_channels",
        "is_transpose",
        "kernel_generator",
        "dimension",
        "use_mm",
        "kernel",
        "bias",
        "conv",
    )

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        is_transpose=False,  # only the base class has this argument
        expand_coordinates=False,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=-1,
        sparse_mode: SparseConvMode = SparseConvMode.SPARSE,
    ):
        super(WeightSparseConvolutionBase, self).__init__()
        assert (
            dimension > 0
        ), f"Invalid dimension. Please provide a valid dimension argument. dimension={dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                expand_coordinates=expand_coordinates,
                dimension=dimension,
            )
        else:
            kernel_generator.expand_coordinates = expand_coordinates

        self.sparse_mode = sparse_mode
        self.is_transpose = is_transpose
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel_volume is 1

        Tensor = torch.FloatTensor
        if (
            self.kernel_generator.kernel_volume == 1
            and self.kernel_generator.requires_strided_coordinates
        ):
            kernel_shape = (self.in_channels, self.out_channels)
            self.use_mm = True
        else:
            kernel_shape = (
                self.kernel_generator.kernel_volume,
                self.in_channels,
                self.out_channels,
            )
        self.kernel = Parameter(Tensor(*kernel_shape))
        self.bias = Parameter(Tensor(1, out_channels)) if bias else None
        self.convolution_mode = convolution_mode
        self.conv = (
            WeightSparseConvolutionTransposeFunction()
            if is_transpose
            else WeightSparseConvolutionFunction()
        )
        self.sparse_kernel = []
        self.valid_kernel = []

    def sparsify(self, layout="csr"):
        if not self.use_mm:
            density = (self.kernel != 0).sum(
                dim=(self.kernel.ndim - 2, self.kernel.ndim - 1)
            ) / (self.kernel.shape[-1] * self.kernel.shape[-2])
            valid_kernel = []
            for i in range(self.kernel.shape[0]):
                sum_w = (self.kernel[i] != 0).sum().item()
                if sum_w == 0:
                    sparse_kernel = self.kernel[i].t().contiguous()
                elif layout == "csr":
                    sp = self.kernel[i].t()._to_sparse_csr()
                    crow_indices = sp.crow_indices().int()
                    col_indices = sp.col_indices().int()
                    values = sp.values()
                    sparse_kernel = torch._sparse_csr_tensor(
                        crow_indices,
                        col_indices,
                        values,
                        size=sp.shape,
                        dtype=sp.dtype,
                    )
                elif layout == "coo":
                    sparse_kernel = self.kernel[i].t().to_sparse().coalesce()
                elif layout == "strided":
                    sparse_kernel = self.kernel[i].t().contiguous()
                if sum_w != 0:
                    valid_kernel.append(i)
                self.sparse_kernel.append(sparse_kernel)
            # if sparse_mode is SparseConvMode.ZAXIS, more aggresive pruning is applied.
            # kernel besides the z-axis is totally removed.
            self.valid_kernel = valid_kernel
            if self.sparse_mode == SparseConvMode.ZAXIS:
                self.valid_kernel = [4, 13, 22]

    def forward(
        self,
        input: SparseTensor,
        coordinates: Union[torch.Tensor, CoordinateMapKey, SparseTensor] = None,
    ):
        # assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        assert self.use_mm or len(self.sparse_kernel) > 0

        ops = 0
        if self.use_mm:
            # If the kernel_size == 1, the convolution is simply a matrix multiplication
            out_coordinate_map_key = input.coordinate_map_key
            outfeat = input.F.mm(self.kernel)
            ops = int(outfeat.numel() * input.F.shape[1])
        else:
            if self.is_transpose:
                in_key = input.coordinate_map_key
                out_coordinate_map_key = CoordinateMapKey(in_key.get_coordinate_size())
                out_tensor_stride = [int(s / self.stride) for s in input.tensor_stride]
                out_coordinate_map_key.set_key(out_tensor_stride, "")
            else:
                out_coordinate_map_key = input._manager.stride(
                    input.coordinate_map_key, self.kernel_generator.kernel_stride
                )
            outfeat, ops = self.conv.apply(
                input.F,
                self.sparse_kernel,
                self.kernel_generator,
                self.convolution_mode,
                input.coordinate_map_key,
                out_coordinate_map_key,
                input._manager,
                self.sparse_mode,
                self.valid_kernel,
            )
        if self.bias is not None:
            outfeat += self.bias
            ops += int(self.bias.numel() * outfeat.shape[0])
        self._flops = ops
        return SparseTensor(
            outfeat,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input._manager,
        )

    def reset_parameters(self, is_transpose=False):
        with torch.no_grad():
            n = (
                self.out_channels if is_transpose else self.in_channels
            ) * self.kernel_generator.kernel_volume
            stdv = 1.0 / math.sqrt(n)
            self.kernel.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = "(in={}, out={}, ".format(
            self.in_channels,
            self.out_channels,
        )
        if self.kernel_generator.region_type in [RegionType.CUSTOM]:
            s += "region_type={}, kernel_volume={}, ".format(
                self.kernel_generator.region_type, self.kernel_generator.kernel_volume
            )
        else:
            s += "kernel_size={}, ".format(self.kernel_generator.kernel_size)
        s += "stride={}, dilation={})".format(
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_dilation,
        )
        return self.__class__.__name__ + s


class WeightSparseConvolution(WeightSparseConvolutionBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        expand_coordinates=False,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=None,
        sparse_mode=1,
    ):
        r"""convolution on a sparse tensor
        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.
            :attr:`out_channels` (int): the number of output channels in the
            output tensor.
            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.
            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.
            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.
            :attr:`bias` (bool, optional): if True, the convolution layer
            has a bias.
            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.
            :attr:`expand_coordinates` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by default.
            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.
        """
        WeightSparseConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            kernel_generator,
            is_transpose=False,
            expand_coordinates=expand_coordinates,
            convolution_mode=convolution_mode,
            dimension=dimension,
            sparse_mode=sparse_mode,
        )
        self.reset_parameters()


class WeightSparseConvolutionTranspose(WeightSparseConvolutionBase):
    r"""A generalized sparse transposed convolution or deconvolution layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        expand_coordinates=False,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=None,
        sparse_mode=1,
    ):
        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension,
            )

        WeightSparseConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            kernel_generator,
            is_transpose=True,
            expand_coordinates=expand_coordinates,
            convolution_mode=convolution_mode,
            dimension=dimension,
            sparse_mode=sparse_mode,
        )
        self.reset_parameters(True)


if __name__ == "__main__":
    # benchmark WeightSparseConvolution
    import argparse
    import cProfile
    import itertools

    import MinkowskiEngine as ME
    import numpy as np
    import pytorch_lightning as pl
    from tqdm import trange

    from co3d_3d.src.data.scannet import ScannetDataset
    from co3d_3d.src.utils import AverageMeter, Timer

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="benchmark")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    print(args)

    # set logger
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    # fix seed
    pl.seed_everything(0)

    def run_benchmark(in_channel, out_channel, sparsity, device):
        print(
            f"in_channels: {in_channel}, out_channels: {out_channel}, sparsity: {sparsity}"
        )

        # prepare sparse convolution weight
        dense_conv = ME.MinkowskiConvolution(
            in_channel, out_channel, kernel_size=3, dimension=3
        )
        num_weights = dense_conv.kernel.data.numel()
        idx = np.random.choice(
            num_weights, int((1 - sparsity) * num_weights), replace=False
        )
        dense_conv.kernel.data.view(-1)[idx] = 0.0

        # make equivalent WeightSparseConvolution layer
        sparse_conv = WeightSparseConvolution(
            in_channel, out_channel, kernel_size=3, dimension=3
        )
        sparse_conv.kernel.data = dense_conv.kernel.data.clone()
        sparse_conv.sparsify()

        # to device
        torch_device = torch.device(device)
        dense_conv = dense_conv.to(torch_device)

        # prepare input data
        dataset = ScannetDataset(phase="val")

        # meters
        input_meter = AverageMeter()
        sparse_timer, dense_timer = Timer(), Timer()

        # benchmark sparse conv
        for i in trange(100):
            coords = dataset[i]["coordinates"]
            feats = torch.rand(coords.shape[0], in_channel).float()
            coords, feats = ME.utils.sparse_collate(
                [coords], [feats], dtype=torch.float32
            )
            sinput_cpu = ME.TensorField(coordinates=coords, features=feats).sparse()
            input_meter.update(sinput_cpu.shape[0])

            sparse_timer.tic()
            sparse_output = sparse_conv(sinput_cpu)
            sparse_timer.toc()

        # benchmark dense conv
        # for i in trange(100):
        #     coords = dataset[i]["coordinates"]
        #     feats = torch.rand(coords.shape[0], in_channel).float()
        #     coords, feats = ME.utils.sparse_collate(
        #         [coords], [feats], dtype=torch.float32
        #     )
        #     sinput_device = ME.TensorField(
        #         coordinates=coords.to(torch_device), features=feats.to(torch_device)
        #     ).sparse()

        #     if device == "cuda":
        #         torch.cuda.synchronize()
        #     dense_timer.tic()
        #     dense_output = dense_conv(sinput_device)
        #     if device == "cuda":
        #         torch.cuda.synchronize()
        #     dense_timer.toc()

        # save results
        # if not os.path.isdir(args.out_dir):
        #     os.makedirs(args.out_dir)
        # np.savez(
        #     f"{args.out_dir}/device{device}_sparsity{sparsity:.2f}_in{in_channel}_out{out_channel}.npz",
        #     num_points=input_meter.history,
        #     sparse_time=sparse_timer.history,
        #     dense_time=dense_timer.history,
        # )
        # print(
        #     f"avg dense_time: {dense_timer.avg:.4f}, sparse_time: {sparse_timer.avg:.4f}"
        # )

    run_benchmark(32, 64, 0.1, args.device)
    # for in_channel, sparsity in itertools.product(
    #     [32, 64, 128, 256], [0.1, 0.05, 0.01]
    # ):
    #     run_benchmark(in_channel, 32, sparsity, args.device)

    # for out_channel, sparsity in itertools.product(
    #     [32, 64, 128, 256], [0.1, 0.05, 0.01]
    # ):
    #     run_benchmark(32, out_channel, sparsity, args.device)

import MinkowskiEngine as ME
import torch
import torch.nn.utils.prune as torch_prune

from co3d_3d.src.models.mink.modules.sparse_conv import (
    WeightSparseConvolution,
    WeightSparseConvolutionTranspose,
)


def count_parameters(model):
    result = {
        "total": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "pruned": float(
            sum(
                (1 - p.int()).sum()
                for n, p in model.named_buffers()
                if "kernel_mask" in n or "weight_mask" in n
            )
        ),
    }
    return result


def count_flops(model):
    flops = 0
    for name, module in model.named_modules():
        if hasattr(module, "_flops"):
            # print(name, module._flops)
            flops += module._flops
    return flops / pow(10, 9)


def get_parameters_to_prune(model):
    parameters_to_prune = list()
    prunable_layers = [
        ME.MinkowskiConvolution,
        ME.MinkowskiConvolutionTranspose,
        WeightSparseConvolution,
        WeightSparseConvolutionTranspose,
    ]
    if hasattr(ME, "MinkowskiWeightSparseConvolution") and hasattr(
        ME, "MinkowskiWeightSparseConvolutionTranspose"
    ):
        prunable_layers.extend(
            [
                ME.MinkowskiWeightSparseConvolution,
                ME.MinkowskiWeightSparseConvolutionTranspose,
            ]
        )
    prunable_layers = tuple(prunable_layers)
    for name, module in model.named_modules():
        if isinstance(module, prunable_layers):
            parameters_to_prune.append((module, "kernel"))
        elif isinstance(module, ME.MinkowskiLinear):
            parameters_to_prune.append((module.linear, "weight"))
    return parameters_to_prune


def register_prune_buffers(parameters_to_prune):
    for module, name in parameters_to_prune:
        torch_prune.identity(module, name)


def masks_and_parameters(parameters_to_prune):
    for module, name in parameters_to_prune:
        assert hasattr(
            module, f"{name}_mask"
        ), f"module {module} has no attribute with name {name}_mask"
        assert hasattr(
            module, name
        ), f"module {module} has no attribute with name {name}"

        mask = getattr(module, f"{name}_mask")
        param = getattr(module, f"{name}_orig")
        yield mask, param

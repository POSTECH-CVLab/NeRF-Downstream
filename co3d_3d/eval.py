#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time

import gin
import numpy as np
import torch
import torch.nn.utils.prune as torch_prune
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from train import get_model

from co3d_3d.src.data.data_module import DataModule
from co3d_3d.src.modules import get_training_module
from co3d_3d.src.utils.prune import count_parameters, get_parameters_to_prune


@gin.configurable
def evaluate(
    save_path,
    load_path,
    ignore_label: int = -100,
    training_module: str = "SegmentationTraining",
    log_every_n_steps: int = 5,
    device="cuda",
    convert_powernorm=False,
    layout="csr",
    tag="default",
    visualize=False,
    replace=False,
    profile=False,
    val_phase="val",
):
    if save_path is not None and not os.path.exists(save_path):
        logging.info("Creating folder " + save_path)
        os.system("mkdir -p " + save_path)

    json_path = os.path.join(save_path, f"{tag}.json")
    if not replace and os.path.isfile(json_path):
        print("====== skip existing experiment =====")
        return

    model = get_model().eval()
    ckpt = torch.load(load_path)

    keys = list(ckpt["state_dict"].keys())
    is_pruned = np.any(["_mask" in k for k in keys])
    if is_pruned:
        logging.info("received checkpoint of pruned network")
        parameters_to_prune = get_parameters_to_prune(model)
        # apply identity pruning to enable loading checkpoint with "_mask", "_orig" parameter
        for module, name in parameters_to_prune:
            torch_prune.identity(module, name)

    data_module = DataModule(val_batch_size=1, val_phase=val_phase)
    pl_module = get_training_module(training_module)(
        model,
        export_path=save_path,
        datamodule=data_module,
        save_pred=visualize,
        save_pred_path=os.path.join(save_path, "figure", tag),
        ignore_label=ignore_label,
    ).to(device)
    pl_module.load_state_dict(ckpt["state_dict"])

    num_params = count_parameters(pl_module.model)
    num_nonzero = num_params["total"] - num_params["pruned"]
    print(
        f"num_params, total={num_params['total']}, net={num_nonzero}, ratio={num_nonzero/num_params['total']*100:.2f}"
    )

    if is_pruned:
        # make pruned state permanent
        for module, name in parameters_to_prune:
            torch_prune.remove(module, name)

        # sparsify model
        for name, module in pl_module.model.named_modules():
            if hasattr(module, "sparsify"):
                module.sparsify(layout)

    trainer = Trainer(
        default_root_dir=save_path,
        gpus=1 if device == "cuda" else None,
        accelerator=None,
        log_every_n_steps=log_every_n_steps,
        resume_from_checkpoint=save_path,
        logger=[],
        profiler=AdvancedProfiler(dirpath=save_path, filename=tag) if profile else None,
    )
    ts = time.time()
    val_results = trainer.validate(pl_module, datamodule=data_module)
    elapsed = time.time() - ts
    logging.info(
        f"elapsed time: {elapsed} s, iter time: {elapsed / len(data_module.val_dataloader())}"
    )
    with open(json_path, "w") as f:
        json_dump = json.dumps(val_results, indent=4)
        f.write(json_dump)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument("--training_module", type=str, default="SegmentationTraining")
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save results",
        default=None,
    )
    parser.add_argument(
        "--load_path",
        type=str,
        help="path to learned weights and configuration files",
        default=None,
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--convert_powernorm", action="store_true")
    parser.add_argument("--sparsify", action="store_true")
    parser.add_argument("--sparse_mode", type=str, default="0,0,0,0,0,0,0,0,0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--layout", type=str, choices=["csr", "coo", "strided"], default="csr"
    )
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()
    ginbs = []

    if args.ginb:
        ginbs.extend(args.ginb)

    if args.sparsify:
        print(args.sparse_mode)
        sparse_mode = [int(n) for n in args.sparse_mode.split(",")]
        assert len(sparse_mode) == 9, "sparse mode should have length 8."
        assert (
            0 not in sparse_mode
        ), f"sparse_mode shoud be positive value when sparsify flag is on."
    else:
        sparse_mode = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ginbs.append(f"get_model.sparse={sparse_mode}")

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")
    gin.parse_config_files_and_bindings(args.ginc, ginbs)

    dataset_name = gin.query_parameter("get_dataset.dataset_name")
    ignore_label = gin.query_parameter(f"{dataset_name}.ignore_label")
    layout = args.layout
    if args.device != "cpu":
        layout = "coo"

    if args.tag is None:
        tag = args.load_path.split("/")[-2]
        tag = f"{tag}-{args.device}-{args.sparsify}"
    else:
        tag = args.tag
    logging.info(tag)

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.dirname(args.load_path)
    evaluate(
        save_path=save_path,
        load_path=args.load_path,
        ignore_label=ignore_label,
        training_module=args.training_module,
        device=args.device,
        convert_powernorm=args.convert_powernorm,
        layout=layout,
        tag=tag,
        visualize=args.visualize,
        replace=args.replace,
        profile=args.profile,
    )

#!/usr/bin/env python3
"""
Trainer script for src.pl_modules.supervised_learning. Example run command: python train.py --ginc configs/cnn.gin.
"""
import argparse
import json
import logging
import os
import sys
from typing import Optional

import gin
import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_info

from co3d_3d.src.data.data_module import DataModule
from co3d_3d.src.loggers import get_logger
from co3d_3d.src.models import get_model
from co3d_3d.src.modules import get_training_module
from co3d_3d.src.profilers import SumProfiler

logger = logging.getLogger(__name__)


def setup_logger(exp_name, debug):
    from imp import reload

    reload(logging)

    CUDA_TAG = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    EXP_TAG = exp_name
    ch = logging.StreamHandler(sys.stdout)
    logger_config = dict(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"{CUDA_TAG}:[{EXP_TAG}] %(asctime)s %(message)s",
        handlers=[ch],
        datefmt="[%X]",
    )
    logging.basicConfig(**logger_config)


@gin.configurable
def train(
    save_path: str,
    gpus: int,
    run_name: str,
    run_name_postfix: str,
    project_name: str,
    max_steps: int,
    max_epochs: int,
    warmup_steps: int = -1,
    model=None,
    training_module: str = "SegmentationTraining",
    optimizer_name: str = "SGD",
    scheduler_name: str = "PolyLR",
    scheduler_interval: str = "step",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    val_batch_size: int = 6,
    prune_batch_size: int = 8,
    train_num_workers: int = 4,
    val_num_workers: int = 2,
    collate_func_name: str = "collate_mink",
    val_every_n_steps: int = 1000,
    log_every_n_steps: int = 10,
    reset_profiler_every_n_steps: int = 1000,
    progressbar_refresh_rate: int = 1,
    loggers: list = ["csv"],
    resume_training: bool = False,
    checkpoint_path: str = None,
    load_weights: bool = False,
    load_optimizers: bool = False,
    transfer_self_supervised: bool = False,
    use_sync_batchnorm: bool = False,
    use_sync_grad: bool = False,
    ignore_label: int = -100,
    train_phase="train",
    val_phase="val",
    test_phase="test",
    monitor_metric: str = "val/mIoU",
    evaluate: bool = False,
    void_weight: Optional[float] = None,
    debug: bool = False,
):
    r"""
    resume_training: resume the training from the same training parameters such as max steps, last learning rate, last global step.
    load_weights: train with a new set of training parameters, but with loaded weights
    load_optimizers: train with a new set of training parameters, but with loaded optimizers
    """
    if not os.path.exists(save_path):
        logging.info("Creating folder " + save_path)
        os.system("mkdir -p " + save_path)

    # create model
    if model is None:
        model = get_model()
    if use_sync_batchnorm and gpus > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    # setup run name
    if run_name is None or "default" in run_name.lower() or run_name == "":
        run_name = f"b{batch_size}x{gpus}-{model.__class__.__name__}"
    logging.info(f"== run name: {run_name}")
    logging.info(f"== run name postfix: {run_name_postfix}")
    if run_name_postfix is not None:
        run_name += "-" + run_name_postfix

    # Create module and pass to training
    data_module = DataModule(
        train_phase=train_phase,
        val_phase=val_phase,
        test_phase=test_phase,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        prune_batch_size=prune_batch_size,
        train_num_workers=train_num_workers,
        val_num_workers=val_num_workers,
        collate_func_name=collate_func_name,
    )

    pl_module = get_training_module(training_module)(
        model=model,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        ignore_label=ignore_label,
        log_every_n_steps=log_every_n_steps,
        reset_profiler_every_n_steps=reset_profiler_every_n_steps,
        load_weights=load_weights,
        load_optimizers=load_optimizers,
        transfer_self_supervised=transfer_self_supervised,
        checkpoint_path=checkpoint_path,
        debug=debug,
        use_sync_grad=use_sync_grad,
        datamodule=data_module,
        scheduler_interval=scheduler_interval,
        void_weight=void_weight,
    )

    # setup logger
    loggers = [
        get_logger(l, save_path=save_path, run_name=run_name, project_name=project_name)
        for l in loggers
    ]

    # Create dynamically callbacks
    callback_modules = [
        TQDMProgressBar(refresh_rate=progressbar_refresh_rate),
        ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            save_last=True,
            monitor=monitor_metric,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # profiler
    profiler = SumProfiler()

    # Training
    trainer = pl.Trainer(
        default_root_dir=save_path,
        max_epochs=max_epochs if max_epochs > 0 else 10000000,
        max_steps=max_steps + (warmup_steps if warmup_steps > 0 else 0),
        logger=loggers,
        devices=gpus,
        accelerator="gpu",
        callbacks=callback_modules,
        log_every_n_steps=log_every_n_steps,
        resume_from_checkpoint=checkpoint_path if resume_training else None,
        strategy=DDPPlugin(find_unused_parameters=False),
        profiler=profiler,
    )
    trainer.fit(pl_module, data_module)
    rank_zero_info(profiler.summary())

    # Final evaluation
    if evaluate:
        (results,) = trainer.test(datamodule=data_module, ckpt_path="best")
        logger.info(results)
        with open(os.path.join(save_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


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
    parser.add_argument(
        "--save_path",
        type=str,
        help="path for logging",
        default="experiments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run_name_postfix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="num_gpus",
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # logger
    run_name = args.run_name if args.run_name is not None else "default"
    if args.run_name_postfix is not None:
        run_name = f"{run_name}-{args.run_name_postfix}"
    run_name += f"_{args.seed}"
    setup_logger(run_name, args.debug)

    logging.info(f"Found {torch.cuda.device_count()} GPUs")
    ginbs = [f"train.gpus={args.gpus}"]
    if args.ginb:
        ginbs.extend(args.ginb)
    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    # Seed
    pl.seed_everything(args.seed)
    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    train(
        save_path=args.save_path,
        resume_training=args.resume,
        run_name=args.run_name,
        run_name_postfix=args.run_name_postfix,
    )

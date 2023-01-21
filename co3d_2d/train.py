
import argparse

import torch
import logging
import os
import sys
import gin
import pytorch_lightning as pl

from typing import Optional
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger

from co3d_2d.src.modules.classification import LitModel
from co3d_2d.src.data.loader import DataModule

ch = logging.StreamHandler(sys.stdout)

logging.getLogger("lightning").setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split(".")[0] + " %(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


@gin.configurable()
def run(
    ckpt_path: str,
    resume_training: bool,
    seed: int,
    run_name: Optional[str] = None,
    num_gpus: int = 2, 
    log_every_n_steps: int = 100,
    max_epochs: int = 1000,
    check_val_every_n_epoch: int = 10,
    precision: int = 16,
    progressbar_refresh_rate: int = 20,
    run_train: bool = True,
    run_eval: bool = True,
): 
    run_name += f"_{str(seed)}"
    os.system("mkdir -p co3d_2d/logs")
    os.system(f"mkdir -p co3d_2d/logs/{run_name}")

    data_module = DataModule()
    if not resume_training:
        model = LitModel()
    else:
        model = LitModel.load_from_checkpoint(ckpt_path)

    kwargs = {}

    if run_train:    
        wandb_logger = WandbLogger(
            name=run_name,
            project="co3d-downstream-2d"
        )
    else:
        wandb_logger = WandbLogger(
            name=run_name,
            project="co3d-downstream-2d-test"
        )
    kwargs["logger"] = wandb_logger
    
    model_checkpoint = ModelCheckpoint(
        dirpath=f"co3d_2d/logs/{run_name}",
        monitor="val/acc",
        save_top_k=1,
        save_last=True,
        mode="max",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)
    callbacks = [model_checkpoint, lr_monitor, tqdm_progrss]

    kwargs["log_every_n_steps"] = log_every_n_steps
    kwargs["max_epochs"] = max_epochs
    kwargs["gpus"] = num_gpus
    kwargs["accelerator"] = "ddp" # "gpu"
    kwargs["check_val_every_n_epoch"] = check_val_every_n_epoch
    kwargs["precision"] = precision
    kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(**kwargs)

    if num_gpus > 1:
        kwargs["replace_sampler_ddp"] = True
        kwargs["sync_batchnorm"] = True
        kwargs["strategy"] = "ddp_find_unused_parameters_false"

    if run_train:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    
    if run_eval:
        trainer.test(model, data_module, ckpt_path="best" if ckpt_path is None else ckpt_path)
    if num_gpus > 1:
        torch.distributed.destroy_process_group()

    # if trainer.global_rank == 0:
    #     (
    #         kwargs["gpus"],
    #         kwargs["strategy"],
    #         kwargs["sync_batchnorm"],
    #         kwargs["callbacks"],
    #     ) = (1, None, False, [model_checkpoint])
    #     kwargs.pop("callbacks")
    #     trainer = pl.Trainer(**kwargs)
    #     trainer.test(model, data_module)


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
        "--ckpt_path",
        type=str,
        default=None,
        help="path to load the ckpt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=333
    )

    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    # Seed
    pl.seed_everything(args.seed)
    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    run(
        ckpt_path=args.ckpt_path,
        resume_training=args.resume,
        seed=args.seed,
    )

import logging
from typing import Optional, OrderedDict

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core import LightningModule

from co3d_3d.src.modules.optim import get_optimizer, get_scheduler


class BaseModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str = "SGD",
        scheduler_name: str = "PolyLR",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_steps: int = -1,
        ignore_label: int = -100,
        void_weight: Optional[float] = None,
        log_every_n_steps: int = 10,
        reset_profiler_every_n_steps: int = 1000,
        load_weights: bool = False,
        load_optimizers: bool = False,
        transfer_self_supervised: bool = False,
        checkpoint_path: str = None,
        export_path: str = None,
        debug: bool = False,
        use_sync_grad: bool = False,
        datamodule: Optional[LightningDataModule] = None,
        scheduler_interval: str = "step",
        save_pred: bool = False,
        save_pred_path: Optional[str] = None,
    ):
        super().__init__()
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        if load_weights or load_optimizers:
            assert checkpoint_path is not None

    def log_dict_with_step(self, dict_data) -> None:
        dict_data["global_step"] = self.global_step
        self.log_dict(dict_data)

    def profiler_time(self, key):
        return (
            self.trainer.profiler.recorded_durations[key]
            / self.trainer.profiler.call_counts[key]
        )

    def forward(self, x):
        return self.model(x)

    def convert_self_supervised_checkpoint(self, state_dict):
        logging.info(f"converting self supervised checkpoint")
        keys = list(state_dict.keys())
        # remove final linear layer and predictor layers
        keys = list(filter(lambda x: "predictor" not in x and "final" not in x, keys))
        new_state_dict = OrderedDict()
        for k in keys:
            new_key = k.replace("model.encoder", "model")
            new_state_dict[new_key] = state_dict[k]
        return new_state_dict

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.optimizer_name,
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path)
            if self.load_weights:
                if self.transfer_self_supervised:
                    state_dict = self.convert_self_supervised_checkpoint(
                        checkpoint["state_dict"]
                    )
                    missing_keys, unexpected_keys = self.load_state_dict(
                        state_dict, strict=False
                    )
                    logging.warn(
                        f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                    )
                else:
                    self.load_state_dict(checkpoint["state_dict"])
                if self.trainer.is_global_zero:
                    print(f"Loading model weights from {self.checkpoint_path}")
            if self.load_optimizers:
                if self.trainer.is_global_zero:
                    print(f"Loading optimizer parameters from {self.checkpoint_path}")
                optimizer_states = checkpoint["optimizer_states"]
                for optimizer, opt_state in zip(
                    self.trainer.optimizers, optimizer_states
                ):
                    # Remote the last learning rate and start from the current learning rate
                    opt_state["lr"] = self.lr
                    optimizer.load_state_dict(opt_state)

        opts = dict(optimizer=optimizer)
        if self.scheduler_name.lower() != "none":
            scheduler = get_scheduler(self.scheduler_name, optimizer, self.warmup_steps)
            opts["lr_scheduler"] = dict(
                scheduler=scheduler,
                interval=self.scheduler_interval,
            )
        if self.scheduler_name.lower() != "none":
            assert "lr_scheduler" in opts.keys()
        return opts

    def on_pretrain_routine_start(self):
        if self.trainer.is_global_zero:
            print(self.model)
            print("Optimizers:", self.optimizers())
            print("Schedulers:", self.trainer.lr_schedulers)

    def training_step(self, *args):
        raise NotImplementedError()

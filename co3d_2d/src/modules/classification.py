import gin
import torch
import logging
import numpy as np
import gc

from torch.optim import SGD
import pytorch_lightning as pl

from typing import Optional

from co3d_2d.src.model.models import *

resnet_list = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "resnet152", "resnext50_32x4d", "resnext101_32x8d",
    "wide_resnet50_2", "wide_resnet101_2"
]

vit_list = [
    "vit_small_patch16_224", "vit_base_patch16_224", 
    "vit_large_patch16_224", "deit3_small_patch16_224",
    "deit3_base_patch16_224", "deit3_large_patch16_224"
]

def select_model(model_name):

    if model_name is None:
        raise NameError(f"Oops?")

    if model_name in resnet_list:
        return ResNetBased(model_name)

    elif model_name in vit_list:
        return ViTBased(model_name)

    else:
        raise NameError(f"Unknown model name : {model_name}")

logger = logging.getLogger("lightning")

@gin.configurable()
class LitModel(pl.LightningModule):

    def __init__(
        self, 
        model_name: Optional[str] = None,
        lr: float = 0.1, 
        weight_decay: float = 1e-4,
    ): 
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitModel, self).__init__()
        

    def setup(self, stage):
        self.model = select_model(self.model_name)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.005)
        self.num_training_steps = self._num_training_steps()

    def configure_optimizers(self):

        optimizer = SGD(self.parameters(), self.lr, momentum=0.9)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        
        labels = batch["labels"]
        imgs = batch["images"]
        prediction = self.model(imgs)
        celoss = self.loss(prediction, labels)

        argmax = torch.argmax(prediction, axis=1)
        acc = (argmax == labels).float().mean() * 100

        wdloss = 0
        for (name, param) in self.named_parameters():
            if (
                "conv" in name or "fc" in name or "downsample" in name    
            ) and "weight" in name:
                wdloss += self.weight_decay * param.norm()

        self.log("train/celoss", celoss.item(), prog_bar=True, on_step=True)
        self.log("train/wdloss", wdloss.item(), prog_bar=True, on_step=True)
        self.log("train/acc", acc.item(), prog_bar=True, on_step=True)

        return celoss + wdloss

    def on_epoch_end(self) -> None:
        torch.cuda.empty_cache() 
        gc.collect()
        return super().on_epoch_end()

    def evaluation_step(self, batch, batch_idx):
        labels = batch["labels"]
        imgs = batch["images"]
        preds = self.model(imgs)
        return {"preds": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx)

    def on_eval_epoch_end(self, outputs, prefix):
        
        preds = torch.cat([output["preds"] for output in outputs])
        labels = torch.cat([output["labels"] for output in outputs])
        argmax = torch.argmax(preds, axis=1)
        acc = (argmax == labels).float().mean() * 100
        celoss = self.loss(preds, labels)
        self.log(f"{prefix}/acc", acc.item(), on_epoch=True)
        self.log(f"{prefix}/loss", celoss.item(), on_epoch=True)

        logger.log(logging.INFO, f"{prefix} Acc: {acc.item()}, {prefix} Loss: {celoss.item()}") 

    def validation_epoch_end(self, outputs):
        self.on_eval_epoch_end(outputs, "val")
        return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        self.on_eval_epoch_end(outputs, "test")
        return super().test_epoch_end(outputs)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        peak = int(self.num_training_steps * 0.1)

        if step <= peak:
            lr = self.lr * (step / peak)
        else:
            lr = self.lr * np.cos((step - peak) / (self.num_training_steps - peak) * np.pi / 2)
        
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step(closure=optimizer_closure)

    def _num_training_steps(self) -> int:
        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )
        num_devices = max(1, self.trainer.num_devices)
        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

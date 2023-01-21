import json
import logging
import os

import gin
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning.utilities.distributed as pl_dist
import torch
import torch.nn as nn
import torch.nn.functional as F

from co3d_3d.src.data.scannet import CLASS_LABELS
from co3d_3d.src.metrics import IoUMeter
from co3d_3d.src.modules.lightning_module_base import BaseModule
from co3d_3d.src.modules.optim import get_learning_rate
from co3d_3d.src.utils import (
    IoUAccumulator,
    Timer,
    fast_hist,
    per_class_iu,
    precision_at_one,
)
from co3d_3d.src.utils.prune import count_flops, count_parameters


class SegLoss(nn.Module):
    def __init__(self, ignore_index, num_labels=None, void_weight=None):
        super(SegLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = torch.ones(num_labels)
        if void_weight is not None and void_weight > 0:
            self.weight[-1] = void_weight

    def forward(self, output, batch):
        device = output.device
        labels = batch["labels"].long().to(device)
        loss = F.cross_entropy(
            output,
            labels,
            weight=self.weight.to(device),
            ignore_index=self.ignore_index,
        )
        return loss


class SegmentationTraining(BaseModule):
    """
    Module defining a supervised learning system.
    """

    def setup(self, stage=None):
        self.best_miou = -1
        num_labels = gin.query_parameter("get_model.out_channel")
        void_label = gin.query_parameter("PlenoxelScannetDataset.void_label")
        self.criterion = SegLoss(
            ignore_index=self.ignore_label,
            num_labels=num_labels,
            void_weight=self.void_weight,
        )
        self.iou_meter = IoUMeter(
            num_labels, ignore_label=self.ignore_label, void_label=void_label
        )
        self.iou_meter.reset()

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        tfield = ME.TensorField(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        logits = self(tfield)

        # Outputs
        loss = self.criterion(logits, batch)

        if self.global_step % self.log_every_n_steps == 0 and self.global_step > 0:
            loss_float = loss.detach().cpu().item()
            if not np.isfinite(loss_float):
                raise ValueError(f"Invalid loss: {loss_float}")
            labels = batch["labels"].long()
            log_results = self._eval_metrics(
                logits,
                labels,
                num_labels=logits.shape[1],
                ignore_label=self.ignore_label,
            )
            out_results = dict()
            for k, v in log_results.items():
                out_results[f"train/{k}"] = v
            out_results["train/loss"] = loss_float
            out_results["train/lr"] = get_learning_rate(self.optimizers())
            out_results["train/data_time"] = self.profiler_time("get_train_batch")
            out_results["train/iter_time"] = self.profiler_time("run_training_batch")
            out_results["train/ignore_ratio"] = (
                (labels == self.ignore_label).sum() / labels.shape[0] * 100
            ).item()

            self.log_dict_with_step(out_results)

        if (
            self.global_step % self.reset_profiler_every_n_steps == 0
            and self.global_step > 0
        ):
            self.trainer.profiler.reset()

        num_points = torch.tensor(
            batch["coordinates"].shape[0], device=batch["coordinates"].device
        )
        result = dict(loss=loss, num_points=num_points)
        return result

    def training_step_end(self, batch_parts):
        if self.use_sync_grad:
            gathered_num_points = pl_dist.gather_all_tensors(batch_parts["num_points"])
            batch_parts["loss"] *= (
                batch_parts["num_points"]
                / sum(gathered_num_points)
                * len(gathered_num_points)
            )
        return batch_parts

    def validation_step(self, batch, batch_idx):
        # if self.save_pred and batch_idx % 9 != 1:
        #     return
        timer = Timer()
        torch.cuda.empty_cache()
        tfield = ME.TensorField(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        timer.tic()
        logits = self(tfield)
        timer.toc()
        flops = count_flops(self.model)
        labels = batch["labels"].long()
        loss = self.criterion(logits, batch)
        results = self._eval_metrics(
            logits, labels, num_labels=logits.shape[1], ignore_label=self.ignore_label
        )
        results["loss"] = loss.cpu().item()
        results["iter_time"] = timer.diff
        results["flops"] = flops
        self.iou_meter(logits.argmax(1), labels)
        if self.save_pred and batch_idx % 2 == 0:
            assert self.save_pred_path is not None
            if not os.path.exists(self.save_pred_path):
                os.makedirs(self.save_pred_path, exist_ok=True)
            inst_id = batch["metadata"][0]["file"]
            filename = os.path.join(self.save_pred_path, f"{inst_id}.pth")
            torch.save(
                dict(
                    coordinates=batch["coordinates"],
                    logits=logits,
                    dists=batch["dists"],
                    labels=labels,
                ),
                filename,
            )
            logging.info(f"saved {filename}")
        return results

    def validation_epoch_end(self, results):
        assert len(results) > 0
        out_results = dict()
        for k in ["OA", "loss", "iter_time", "flops"]:
            out_results[f"val/{k}"] = np.stack([r[k] for r in results]).mean(0)
        miou, ious, mAcc, accs = self.iou_meter.compute()
        print("")
        logging.info("==== per_class ious")
        argsort = np.argsort(CLASS_LABELS)
        class_labels = np.array(CLASS_LABELS)[argsort]
        ious = ious[argsort] * 100
        accs = accs[argsort] * 100
        iou_list = ious.cpu().tolist()
        print(" & ".join(class_labels))
        print(" & ".join([f"{i:.1f}" for i in iou_list]))
        logging.info("==== per-class acc")
        acc_list = accs.cpu().tolist()
        print(" & ".join(class_labels))
        print(" & ".join([f"{i:.1f}" for i in acc_list]))

        print(f"miou: {miou.item()}")
        print(f"macc: {mAcc.item()}")
        print("")
        self.iou_meter.reset()
        result_dict = dict(
            labels=class_labels.tolist(),
            iou=[*iou_list, miou.item()],
            acc=[*acc_list, mAcc.item()],
        )
        with open(
            os.path.join(self.trainer.default_root_dir, "eval_results.json"), "w"
        ) as f:
            json.dump(result_dict, f)

        out_results["val/mIoU"] = miou * 100
        out_results["val/mAcc"] = mAcc * 100
        if out_results["val/mIoU"] > self.best_miou:
            self.best_miou = out_results["val/mIoU"]
        out_results["val/best_mIoU"] = self.best_miou
        num_params = count_parameters(self.model)
        for k, v in num_params.items():
            out_results[f"val/{k}_params"] = v
        self.log_dict_with_step(out_results)

    def test_step(self, batch, batch_idx):
        tfield = ME.TensorField(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        pred = self(tfield).argmax(1)
        labels = batch["labels"].long()
        self.iou_meter(pred, labels)
        metadata = batch["metadata"]
        if self.export_path is not None:
            self.pred_path = self.test_dataset.save_prediction(
                pred, self.export_path, metadata[0]
            )
            # postfix["export_path"] = pred_path.split('/')[-1]

    @staticmethod
    def _eval_metrics(output, target, num_labels, ignore_label=255):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            pred = output.argmax(1)
            acc = precision_at_one(pred, target, ignore_label)
            hist = fast_hist(pred, target, num_labels)
            ious = per_class_iu(hist) * 100
            results = {"OA": acc, "mIoU": ious.mean()}
            return results


class ExceptionSafeSegmentationTraining(SegmentationTraining):
    """
    Module defining a supervised learning system.
    """

    def setup(self, stage=None):
        self.automatic_optimization = False
        self.fail_count = 0

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.best_miou = -1

        num_labels = gin.query_parameter("get_model.out_channel")
        self.iou_accumulator = IoUAccumulator(
            num_labels, ignore_label=self.ignore_label
        )
        self.iou_accumulator.reset()

    def scheduler_step(self):
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, (list, tuple)):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        opt = self.optimizers()
        opt.zero_grad()

        # This does not work with multi gpu.
        # Requires Pytorch lightning >= 1.3.8
        try:
            tfield = ME.TensorField(
                coordinates=batch["coordinates"], features=batch["features"]
            )
            logits = self(tfield)
            labels = batch["labels"].long()

            # Outputs
            loss = self.criterion(logits, labels)
            self.manual_backward(loss)
            opt.step()
            self.scheduler_step()
        except RuntimeError as e:
            self.fail_count += 1
            print(
                f"Failed with {e}. Failure rate: {float(self.fail_count) / (self.global_step + 1)}"
            )
            self.scheduler_step()  # regardless of the failure status, update
            torch.cuda.synchronize()
            return

        if self.global_step % self.log_every_n_steps == 0 and self.global_step > 0:
            loss_float = loss.detach().cpu().item()
            if not np.isfinite(loss_float):
                raise ValueError(f"Invalid loss: {loss_float}")
            log_results = self._eval_metrics(
                logits,
                labels,
                num_labels=logits.shape[1],
            )
            out_results = dict()
            for k, v in log_results.items():
                out_results[f"train/{k}"] = v
            out_results["train/loss"] = loss_float
            out_results["train/lr"] = get_learning_rate(self.optimizers())
            out_results["train/data_time"] = self.profiler_time("get_train_batch")
            out_results["train/iter_time"] = self.profiler_time("run_training_batch")

            self.log_dict_with_step(out_results)

        if (
            self.global_step % self.reset_profiler_every_n_steps == 0
            and self.global_step > 0
        ):
            self.trainer.profiler.reset()

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        try:
            tfield = ME.TensorField(
                coordinates=batch["coordinates"], features=batch["features"]
            )
            logits = self(tfield)
            labels = batch["labels"].long()
            loss = self.criterion(logits, labels)
        except RuntimeError as e:
            print(f"Validation step failed with {e}.")
            torch.cuda.synchronize()
            return {"valid": False, "loss": float("inf")}

        results = self._eval_metrics(logits, labels, num_labels=logits.shape[1])
        results["valid"] = True
        results["loss"] = loss.cpu().item()
        self.iou_accumulator.accumulate(logits.argmax(1), labels)
        return results

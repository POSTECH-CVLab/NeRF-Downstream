import gin
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from co3d_3d.src.modules.lightning_module_base import BaseModule
from co3d_3d.src.utils.prune import count_parameters
from torchmetrics import Accuracy


class ClassificationTraining(BaseModule):
    def setup(self, stage=None):
        num_classes = gin.query_parameter("get_model.out_channel")
        self.acc1_meter = Accuracy(num_classes=num_classes)
        self.acc5_meter = Accuracy(num_classes=num_classes, top_k=5)
        self.acc1_meter.reset()
        self.acc5_meter.reset()

    def forward(self, batch):
        binput = self.model.process_input(batch)
        output = self.model(binput)
        return output

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        # tfield = ME.TensorField(
        #     coordinates=batch["coordinates"], features=batch["features"]
        # )
        # output = self(tfield)
        output = self(batch)

        labels = batch["labels"].long()
        loss_val = F.cross_entropy(output, labels)
        acc1, acc5 = self.__accuracy(output, labels, topk=(1, 5))

        if self.global_step % self.log_every_n_steps == 0 and self.global_step > 0:
            loss_float = loss_val.detach().cpu().item()
            if not np.isfinite(loss_float):
                raise ValueError(f"Invalid loss: {loss_float}")

            self.log_dict_with_step(
                {
                    "train/data_time": self.profiler_time("get_train_batch"),
                    "train/iter_time": self.profiler_time("run_training_batch"),
                    "train/loss": loss_float,
                    "train/acc1": acc1,
                    "train/acc5": acc5,
                }
            )
        return loss_val

    def validation_step(self, batch, batch_idx):
        # tfield = ME.TensorField(
        #     coordinates=batch["coordinates"], features=batch["features"]
        # )
        # logits = self(tfield)
        logits = self(batch)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels.long())
        self.acc1_meter(logits, labels)
        self.acc5_meter(logits, labels)
        return {"loss": loss.cpu()}

    def validation_epoch_end(self, results):
        assert len(results) > 0
        out_results = dict()
        loss = np.array([r["loss"] for r in results]).mean()
        acc1 = self.acc1_meter.compute()
        acc5 = self.acc5_meter.compute()
        out_results["val/acc1"] = acc1
        out_results["val/acc5"] = acc5
        out_results["val/loss"] = loss
        num_params = count_parameters(self.model)
        for k, v in num_params.items():
            out_results[f"val/{k}_params"] = v
        self.log_dict_with_step(out_results)
        self.acc1_meter.reset()
        self.acc5_meter.reset()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    @torch.no_grad()
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).cpu().item())
        return res

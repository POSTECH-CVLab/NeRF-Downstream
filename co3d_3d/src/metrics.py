import torch
from torchmetrics import Metric


class IoUMeter(Metric):
    def __init__(self, num_classes, ignore_label, void_label=None):
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.void_label = void_label

        self.add_state(
            "total_seen", default=torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_correct", default=torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_positive",
            default=torch.zeros(self.num_classes),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        valid = targets != self.ignore_label
        preds = preds[valid]
        targets = targets[valid]

        for i in range(self.num_classes):
            self.total_seen[i] += (targets == i).sum()
            self.total_correct[i] += (
                torch.logical_and(targets == i, preds == targets).float().sum()
            )
            self.total_positive[i] += (preds == i).sum()

    def compute(self):
        ious = torch.zeros_like(self.total_correct)
        accs = torch.zeros_like(self.total_correct)
        for i in range(self.num_classes):
            if not self.total_seen[i] == 0:
                cur_iou = self.total_correct[i] / (
                    self.total_seen[i] + self.total_positive[i] - self.total_correct[i]
                )
                cur_acc = self.total_correct[i] / self.total_seen[i]
                ious[i] = cur_iou
                accs[i] = cur_acc
        if self.void_label is not None:
            miou = torch.mean(ious[:-1])
            mAcc = torch.mean(accs[:-1])
        else:
            miou = torch.mean(ious)
            mAcc = torch.mean(accs)
        return miou, ious, mAcc, accs

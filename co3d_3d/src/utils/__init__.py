import time
from typing import Tuple

import numpy as np
import torch

EPS = 1e-10


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec


def is_number_tryexcept(s):
    """Returns True is string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_arg(arg):
    arg = arg[2:]  # remove --
    parsed_arg = arg.split("=")
    value = parsed_arg[1]
    if "," not in value and not is_number_tryexcept(value):
        return f"{parsed_arg[0]}='{parsed_arg[1]}'"
    else:
        return arg


def list2str(input_list: list):
    return ",".join([str(x) for x in input_list])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
        self.max = 0
        self.min = np.inf
        self.history = []

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            n = val.size
            val = val.mean()
        elif isinstance(val, torch.Tensor):
            n = val.nelement()
            val = val.mean().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val
        self.history.append(val)


class Timer(AverageMeter):
    """A simple timer."""

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.update(self.diff)
        if average:
            return self.avg
        else:
            return self.diff


def precision_at_one(pred, target, ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return float("nan")


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    if isinstance(k, torch.Tensor):
        count = (n * label[k] + pred[k]).cpu().numpy()
    else:
        count = n * label[k].astype(int) + pred[k]
    return np.bincount(count, minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + EPS)


class HistogramAccumulator:
    def __init__(self, num_class, ignore_label=-100):
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.num_class, self.num_class))

    def accumulate(self, outputs, targets) -> None:
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        valid = targets != self.ignore_label
        outputs = outputs[valid]
        targets = targets[valid]
        self.accumulate_hist(fast_hist(outputs, targets, self.num_class))

    def accumulate_hist(self, hist) -> None:
        self.hist += hist

    def report(self) -> Tuple:
        ious = per_class_iu(self.hist)
        return np.mean(ious), ious


class IoUAccumulator:
    def __init__(self, num_classes: int, ignore_label: int):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def accumulate(self, outputs, targets) -> None:
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        valid = targets != self.ignore_label
        outputs = outputs[valid]
        targets = targets[valid]
        for i in range(self.num_classes):
            self.total_seen[i] += np.sum(targets == i)
            self.total_correct[i] += np.sum((targets == i) & (outputs == targets))
            self.total_positive[i] += np.sum(outputs == i)

    def report(self) -> Tuple:
        for i in range(self.num_classes):
            self.total_seen[i] = np.sum(self.total_seen[i])
            self.total_correct[i] = np.sum(self.total_correct[i])
            self.total_positive[i] = np.sum(self.total_positive[i])

        ious = []
        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (
                    self.total_seen[i] + self.total_positive[i] - self.total_correct[i]
                )
                ious.append(cur_iou)

        miou = np.mean(ious)
        return miou, ious

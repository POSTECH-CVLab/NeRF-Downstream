import logging
import math
import pickle
import unittest

import gin
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


@gin.configurable
class SGD(optim.SGD):
    pass


@gin.configurable
class ASGD(optim.ASGD):
    pass


@gin.configurable
class Adam(optim.Adam):
    pass


@gin.configurable
class AdamW(optim.AdamW):
    pass


@gin.configurable
class Adagrad(optim.Adagrad):
    pass


@gin.configurable
class Adadelta(optim.Adadelta):
    pass


@gin.configurable
class Adamax(optim.Adamax):
    pass


@gin.configurable
class RMSprop(optim.RMSprop):
    pass


@gin.configurable
class Rprop(optim.Rprop):
    pass


GINNED_OPTIMIZERS = [SGD, ASGD, Adam, AdamW, Adagrad, Adadelta, Adamax, RMSprop, Rprop]


def get_optimizer(optimizer_name, parameters, lr, weight_decay):
    NAME2OPT = {opt.__name__: opt for opt in GINNED_OPTIMIZERS}
    if optimizer_name not in NAME2OPT:
        raise ValueError(f"optimizer {optimizer_name} not recognized in {NAME2OPT}.")
    else:
        return NAME2OPT[optimizer_name](
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )


@gin.configurable
class StepLR(lr_scheduler.StepLR):
    pass


@gin.configurable
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones=[20000, 40000],
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        lr_scheduler.MultiStepLR.__init__(
            self, optimizer, milestones, gamma, last_epoch, verbose
        )


@gin.configurable
class ExponentialLR(lr_scheduler.ExponentialLR):
    r"""
    10% every 20k iterations:
    np.exp( np.log(0.1) / 20000) == 0.9998848773724686
    """

    def __init__(self, optimizer, gamma=0.99):
        lr_scheduler.ExponentialLR.__init__(self, optimizer, gamma)


@gin.configurable
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, eta_min=0, last_epoch=-1, verbose=False):
        interval = gin.query_parameter("train.scheduler_interval")
        T_max = (
            gin.query_parameter("train.max_steps")
            if interval == "step"
            else gin.query_parameter("train.max_epochs")
        )
        lr_scheduler.CosineAnnealingLR.__init__(
            self,
            optimizer,
            T_max,
            eta_min,
            last_epoch,
            verbose=verbose,
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def __repr__(self):
        return f"{self.__class__.__name__}(T_max={self.T_max}, eta_min={self.eta_min})"


class ExpFunctor:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, cycle_num):
        return self.gamma ** cycle_num


class CosineFunctor:
    def __init__(self, max_cycle, eta_min=0):
        self.max_cycle = max_cycle
        self.eta_min = eta_min

    def __call__(self, cycle_num):
        return (1 + math.cos(cycle_num / self.max_cycle * math.pi)) / 2


@gin.configurable
class CyclicLR(lr_scheduler.CyclicLR):
    def __init__(
        self, optimizer, base_lr, step_size_up=2000, mode="trianglular", gamma=1
    ):
        scale_fn = None
        scale_mode = "cycle"
        if mode == "exp_range":
            mode = None
            scale_fn = ExpFunctor(gamma)
        elif mode == "cosine":
            mode = None
            T_max = gin.query_parameter("train.max_steps")
            cycle_num = T_max / (2 * step_size_up)
            scale_fn = CosineFunctor(cycle_num)
        elif mode in ["triangular", "triangular2"]:
            pass
        else:
            raise ValueError(f"Invalid mode:{mode}")

        max_lr = gin.query_parameter("train.lr")
        lr_scheduler.CyclicLR.__init__(
            self,
            optimizer,
            base_lr,
            max_lr,
            step_size_up=step_size_up,
            mode=mode,
            scale_fn=scale_fn,
            scale_mode=scale_mode,
            gamma=gamma,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(max_lr={self.max_lrs}, base_lr={self.base_lrs}, total_size={self.total_size}, mode={self.mode}, gamma={self.gamma})"


class PolyFunctor:
    def __init__(self, max_steps, poly_exp):
        self.max_steps = max_steps
        self.poly_exp = poly_exp

    def __call__(self, step):
        return (1 - step / (self.max_steps + 1)) ** self.poly_exp


@gin.configurable
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, poly_exp, verbose=False):
        max_steps = gin.query_parameter("train.max_steps")
        lr_scheduler.LambdaLR.__init__(
            self,
            optimizer,
            PolyFunctor(max_steps, poly_exp),
            verbose=verbose,
        )
        self.max_steps = max_steps
        self.poly_exp = poly_exp

    def __repr__(self):
        return f"{self.__class__.__name__}(max_steps={self.max_steps}, poly_exp={self.poly_exp})"


@gin.configurable
class SquaredLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, max_iter, last_step=-1):
        super(SquaredLR, self).__init__(
            optimizer, lambda s: (1 - s / (max_iter + 1)) ** 2, last_step
        )
        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"


GINNED_SCHEDULERS = [
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
    PolyLR,
    SquaredLR,
]

NAME2SCHEDULER = {sch.__name__: sch for sch in GINNED_SCHEDULERS}


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

    MIT License

    Copyright (c) 2019 Ildoo Kim
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater than or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.total_epoch})"


def get_scheduler(scheduler_name, optimizer, warmup_steps):
    if scheduler_name not in NAME2SCHEDULER:
        raise ValueError(
            f"optimizer {scheduler_name} not recognized in {NAME2SCHEDULER}."
        )
    else:
        logging.info(f"Initializing scheduler: {scheduler_name}")
    scheduler = NAME2SCHEDULER[scheduler_name](optimizer)

    if warmup_steps is not None and warmup_steps > 0:
        scheduler = GradualWarmupScheduler(
            optimizer, 1.0, warmup_steps, after_scheduler=scheduler
        )
        logging.info(f"Using WarmupScheduler: {scheduler}")
    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

import logging
import os
import time

# import comet_ml
import gin

# from comet_ml import ExistingExperiment as CometExistingExperiment
# from comet_ml import Experiment as CometExperiment
# from comet_ml import OfflineExperiment as CometOfflineExperiment
from pytorch_lightning.loggers import (
    CometLogger,
    CSVLogger,
    TestTubeLogger,
    WandbLogger,
    TensorBoardLogger,
    NeptuneLogger
)
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from wandb.wandb_run import Run

import wandb

MAX_RETRY = 100


@gin.configurable
def logged(hyper_params):
    pass


# https://github.com/wandb/client/issues/1409
class RetryingWandbLogger(WandbLogger):
    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            print("Initializing wandb")
            hparams = dict()
            print("Logging:", gin.query_parameter("logged.hyper_params"))
            for i in gin.query_parameter("logged.hyper_params"):
                try:
                    hparams[i] = gin.query_parameter(i)
                except Exception as e:
                    print(e)
            print(f"Logging configs in wandb: {hparams}")

            for i in range(MAX_RETRY):
                try:
                    self._experiment = wandb.init(
                        config=hparams,
                        **self._wandb_init,
                    )
                    break
                except (
                    TimeoutError,
                    ConnectionError,
                    wandb.errors.UsageError,
                    wandb.errors.CommError,
                ) as e:
                    print(f"Error {e}. Retrying in 5 sec")
                    time.sleep(5)

            # save checkpoints in wandb dir to upload on W&B servers
            if self._log_model:
                self._save_dir = self._experiment.dir
        return self._experiment

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # offset future training logged on same W&B run
        if self._experiment is not None:
            self._step_offset = self._experiment.step

            # upload all checkpoints from saving dir
            if self._log_model:
                wandb.save(os.path.join(self.save_dir, "*.ckpt"))

    # @property
    # def version(self):
    #     print(self._experiment)
    #     print(self._id)
    #     print(self._name)
    #     _version = self._experiment.id if self._experiment else self._id
    #     return _version + "_" + self._name


def get_logger(
    logger_name: str, save_path: str, run_name: str, project_name: str = "default"
):
    if logger_name == "testtube":
        logging.info(f"TestTubeLogger {save_path}, {project_name}:{run_name}")
        return TestTubeLogger(
            save_dir=save_path,
            name=project_name + "_testtube",
            debug=False,
            create_git_tag=False,
        )
    elif logger_name == "wandb":
        logging.info(f"WandbLogger {save_path}, {project_name}:{run_name}")
        return RetryingWandbLogger(
            name=run_name,
            project=project_name,
            log_model=True,
            save_dir=save_path,
        )
    elif logger_name == "csv":
        logging.info(f"CSVLogger {save_path}, {project_name}:{run_name}")
        return CSVLogger(save_path, name=run_name)
    elif logger_name == "tensorboard":
        logging.info(f"TBD Logger {save_path}, {project_name}:{run_name}")
        return TensorBoardLogger(save_path, name=run_name)
    elif logger_name == "neptune": 
        logging.info(f"Neptune Logger {save_path}, {project_name}:{run_name}")
        return NeptuneLogger(
            name=run_name,
            project="jeongyw12382/co3d-downstream-3d",
        )
    else:
        raise ValueError(f"Invalid logger: {logger_name}")

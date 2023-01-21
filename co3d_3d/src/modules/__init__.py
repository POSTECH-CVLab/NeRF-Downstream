from pytorch_lightning import LightningModule

from .classification_training import ClassificationTraining
from .segmentation_training import SegmentationTraining

modules = [
    SegmentationTraining,
    ClassificationTraining,
]
modules_dict = {m.__name__: m for m in modules}


def get_training_module(module_name: str) -> LightningModule:
    assert (
        module_name in modules_dict.keys()
    ), f"{module_name} not in {modules_dict.keys()}"
    return modules_dict[module_name]

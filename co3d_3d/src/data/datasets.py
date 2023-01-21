"""
Datasets available for use in the project.
"""
import logging

import gin
from torch.utils.data import Dataset

from .co3d import Co3D10pDataset, Co3DDataset
from .modelnet40 import ModelNet40H5Dataset
from .scannet import PlenoxelScannetDataset, ScannetDataset
from .semantic_kitti import SemanticKITTIDataset
from .stanford import StanfordDataset

logger = logging.getLogger(__name__)

DATASETS = {var.__name__: var for var in globals() if isinstance(var, Dataset)}


@gin.configurable
def get_dataset(dataset_name: str):
    return globals()[dataset_name]

import glob
import logging
import os.path as osp
import unittest

import gin
import MinkowskiEngine as ME
import numpy as np
from torch.utils.data import DataLoader, Dataset

import co3d_3d.src.data.transforms as transforms
from co3d_3d.src.data.scannet import ScannetDataset
from co3d_3d.src.data.utils import load_ply

S3DIS_COLOR_MAP = {
    0: [50.0, 50.0, 50.0],  # clutter
    1: [255.0, 255.0, 0.0],  # beam
    2: [200.0, 200.0, 200.0],  # board
    3: [10.0, 200.0, 100.0],  # bookcase
    4: [0.0, 255.0, 0.0],  # ceiling
    5: [255, 0, 0],  # chair
    6: [255.0, 0.0, 255.0],  # column
    7: [200.0, 200.0, 100.0],  # door
    8: [0.0, 0.0, 255.0],  # floor
    9: [200.0, 100.0, 100.0],  # sofa
    10: [170, 120, 200],  # table
    11: [0.0, 255.0, 255.0],  # wall
    12: [100.0, 100.0, 255.0],  # window
}

CLASS_LABELS = (
    "clutter",
    "beam",
    "board",
    "bookcase",
    "ceiling",
    "chair",
    "column",
    "door",
    "floor",
    "sofa",
    "table",
    "wall",
    "window",
)
VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13)
CLASS_LABELS_INSTANCE = (
    "clutter",
    "beam",
    "board",
    "bookcase",
    "chair",
    "column",
    "door",
    "sofa",
    "table",
    "window",
)
VALID_CLASS_IDS_INSTANCE = (0, 1, 2, 3, 5, 6, 7, 9, 11, 13)


@gin.configurable
class StanfordDataset(ScannetDataset):
    NUM_LABELS = 14
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IGNORE_LABELS_INSTANCE = tuple(
        set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE)
    )
    DATA_PATH_FILE = {
        "train": "stanford_train.txt",
        "val": "stanford_val.txt",
        "test": "stanford_test.txt",
    }
    CLASS_LABELS = CLASS_LABELS
    CLASS_LABELS_INSTANCE = CLASS_LABELS_INSTANCE
    VALID_CLASS_IDS = VALID_CLASS_IDS
    VALID_CLASS_IDS_INSTANCE = VALID_CLASS_IDS_INSTANCE

    def __init__(
        self,
        phase: str,
        data_root: str = "datasets/stanford",
        downsample_voxel_size=0.015,  # in meter
        voxel_size=0.03,
        train_transformations=[
            "ChromaticTranslation",
            "ChromaticJitter",
            "CoordinateDropout",
            "RandomHorizontalFlip",
            "RandomRotation",
            "NormalizeColor",
        ],
        eval_transformations=[
            "NormalizeColor",
        ],
        ignore_label=-100,
        features=["colors"],
    ):
        r"""
        downsample_voxel_size: voxel size used to downsample the point cloud
        """
        ScannetDataset.__init__(
            self,
            phase,
            data_root,
            downsample_voxel_size,
            voxel_size,
            train_transformations,
            eval_transformations,
            ignore_label,
            features,
        )


class StanfordTestCase(unittest.TestCase):
    def test_read(self):
        dataset = StanfordDataset(
            "train", "datasets/stanford", train_transformations=[]
        )
        print(len(dataset))
        print(dataset[0])

    def test_transformations(self):
        dataset = StanfordDataset(
            "train",
            "datasets/stanford",
            train_transformations=[
                "ChromaticTranslation",
                "ChromaticJitter",
                "CoordinateDropout",
                "RandomHorizontalFlip",
                "RandomRotation",
                "NormalizeColor",
            ],
        )
        print(len(dataset))
        print(dataset[0])

    def test_gin(self):
        gin.parse_config_file("./configs/stanford.gin")
        dataset = StanfordDataset("train", "datasets/stanford")
        print(dataset[0])

    def test_loader(self):
        from co3d_3d.src.data.utils import collate_fn

        dataset = StanfordDataset("train", "datasets/stanford")
        data_loader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=4,
            shuffle=True,
        )

        # Start from index 1
        iter = data_loader.__iter__()
        for i in range(100):
            data = iter.next()
            print(data["features"])
            stensor = ME.TensorField(
                coordinates=data["coordinates"], features=data["features"]
            )
            # pcd = make_pcd(coords, feats)
            # o3d.visualization.draw_geometries([pcd])

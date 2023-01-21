import glob
import os
import subprocess
import unittest

import gin
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import co3d_3d.src.data.transforms as transforms


def download_modelnet40_dataset(path):
    if not os.path.exists(os.path.join(path)):
        print("Downloading the 2k downsampled ModelNet40 dataset...")
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
            ]
        )
        subprocess.run(["unzip", "modelnet40_ply_hdf5_2048.zip"])


@gin.configurable
class ModelNet40H5Dataset(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "modelnet40h5",
        train_transformations=["CoordinateUniformTranslation"],
        eval_transformations=[],
        num_points=2048,
        voxel_size=0.05,
        download=False,
    ):
        Dataset.__init__(self)
        download_modelnet40_dataset(data_root)
        phase = "test" if phase in ["val", "test"] else "train"
        self.data, self.label = self.load_data(data_root, phase)
        transformations = (
            train_transformations if phase == "train" else eval_transformations
        )
        self.transformations = (
            transforms.Compose([transforms.__dict__[t]() for t in transformations])
            if len(transformations) > 0
            else None
        )
        self.phase = phase
        self.voxel_size = voxel_size
        self.num_points = num_points

    def load_data(self, data_root, phase):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, "ply_data_%s*.h5" % phase))
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int64"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transformations is not None:
            xyz, _, _ = self.transformations(xyz, None, None)

        label = self.label[i]
        xyz = xyz.astype(np.float32)
        return {
            "coordinates": xyz / self.voxel_size,
            "features": xyz,
            "labels": label,  # labels rather than label for collation
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase}, length={len(self)}, transform={self.transformations})"


class TestModelNet40H5(unittest.TestCase):
    def setUp(self):
        self.dataset = ModelNet40H5Dataset(
            "train", "./datasets/modelnet40_ply_hdf5_2048", voxel_size=0.05
        )

    def test(self):
        print(self.dataset)
        print(len(self.dataset))
        print(self.dataset[0])


if __name__ == "__main__":
    dataset = ModelNet40H5Dataset(
        "train", "./datasets/modelnet40_ply_hdf5_2048", [], [], -1
    )
    import pdb

    pdb.set_trace()
    pass

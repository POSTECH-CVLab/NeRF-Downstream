# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import os
import pickle
import unittest
from typing import List, Optional, Union

import gin
import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import co3d_3d.src.data.transforms as transforms
from co3d_3d.src.data.utils import load_ply

CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)
VALID_CLASS_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)  # total 20
CLASS_LABELS_INSTANCE = (
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)
VALID_CLASS_IDS_INSTANCE = (
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)  # total 18
TEST_FULL_PLY_PATH = "test/%s_vh_clean_2.ply"
SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


@gin.configurable
class ScannetDataset(Dataset):

    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IGNORE_LABELS_INSTANCE = tuple(
        set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE)
    )
    DATA_PATH_FILE = {
        "train": "scannetv2_train.txt",
        "val": "scannetv2_val.txt",
        "test": "scannetv2_test.txt",
    }
    CLASS_LABELS = CLASS_LABELS
    CLASS_LABELS_INSTANCE = CLASS_LABELS_INSTANCE
    VALID_CLASS_IDS = VALID_CLASS_IDS
    VALID_CLASS_IDS_INSTANCE = VALID_CLASS_IDS_INSTANCE

    def __init__(
        self,
        phase: str,
        data_root: str = "datasets/scannet",
        downsample_voxel_size=None,  # in meter
        voxel_size=0.02,
        train_transformations=[
            "ChromaticTranslation",
            "ChromaticJitter",
            "CoordinateDropout",
            "RandomHorizontalFlip",
            "RandomAffine",
            "RandomTranslation",
            "NormalizeColor",
        ],
        eval_transformations=[
            "NormalizeColor",
        ],
        ignore_label=-100,
        features=["colors"],  # ["colors", "xyzs"]
    ):
        Dataset.__init__(self)
        self.ignore_label = ignore_label
        self.data_root = data_root
        self.phase = phase
        transformations = (
            train_transformations if phase == "train" else eval_transformations
        )
        self.transformations = transforms.Compose(
            [transforms.__dict__[t]() for t in transformations]
        )
        self.pc_files = []
        with open(os.path.join(self.data_root, self.DATA_PATH_FILE[phase]), "r") as f:
            self.pc_files.extend([l.rstrip("\n") for l in f.readlines()])

        if downsample_voxel_size is None:
            downsample_voxel_size = voxel_size / 2
        self.downsample_voxel_size = downsample_voxel_size
        self.voxel_size = voxel_size

        # map labels not evaluated to ignore_label
        label_map, n_used = dict(), 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                label_map[l] = ignore_label
            else:
                label_map[l] = n_used
                n_used += 1
        label_map[ignore_label] = ignore_label
        self.label_map = label_map
        self.features = features

        logging.info(
            f"{self.__class__.__name__}(phase={phase}, total size={len(self.pc_files)}, downsample_voxel_size={downsample_voxel_size}, voxel_size={voxel_size})"
        )

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, i: int):
        xyzs, colors, labels, instances = load_ply(
            os.path.join(self.data_root, self.pc_files[i]),
            load_label=True,
            load_instance=True,
        )

        if self.downsample_voxel_size > 0:
            pre_len = len(xyzs)
            _, colors, labels, row_inds = ME.utils.sparse_quantize(
                np.ascontiguousarray(xyzs),
                colors,
                labels=labels,
                quantization_size=self.downsample_voxel_size,
                return_index=True,
                ignore_label=self.ignore_label,
            )
            # Maintain the continuous coordinates
            xyzs = xyzs[row_inds] / self.voxel_size
            instances = instances[row_inds]
            logging.debug(f"Downsampled point cloud index {i} from {pre_len} to {xyzs}")
        else:
            xyzs /= self.voxel_size

        # if self.IGNORE_LABELS_INSTANCE is not None:
        #     condition = labels == self.ignore_label
        #     instances[condition] = -1
        #     for ignore_id in self.IGNORE_LABELS_INSTANCE:
        #         condition = labels == ignore_id
        #         instances[condition] = -1

        xyzs, colors, labels = self.transformations(xyzs, colors, labels)
        # instances_info = self.get_instance_info(xyzs, instances)
        # centers = instances_info["center"]
        # instance_ids = instances_info["ids"]
        if self.IGNORE_LABELS is not None:
            labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

        features = []
        for f in self.features:
            features.append(eval(f))
        return {
            "coordinates": xyzs.astype(np.float32),
            "features": np.concatenate(features, axis=1).astype(np.float32),
            "labels": labels,
            "colors": colors,
            # "instance_centers": centers,
            # "instance_ids": instance_ids,
            "dataset": "scannet",
        }

    def get_instance_info(self, xyz, instance_ids):
        """
        :param xyz: (n, 3)
        :param instance_ids: (n), int, (1~nInst, -1)
        :return: instance_num, dict
        """
        centers = (
            np.ones((xyz.shape[0], 3), dtype=np.float32) * -1
        )  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz, occ, num_instances)
        occupancy = {}  # (nInst), int
        bbox = {}
        unique_ids = np.unique(instance_ids)
        for id_ in unique_ids:
            if id_ == -1:
                continue

            mask = instance_ids == id_
            xyz_ = xyz[mask]
            bbox_min = xyz_.min(0)
            bbox_max = xyz_.max(0)
            center = xyz_.mean(0)

            centers[mask] = center
            occupancy[id_] = mask.sum()
            bbox[id_] = np.concatenate([bbox_min, bbox_max])

        return {
            "ids": instance_ids,
            "center": centers,
            "occupancy": occupancy,
            "bbox": bbox,
        }


class ScannetTestCase(unittest.TestCase):
    def test_read(self):
        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            train_transformations=[
                "NormalizeColor",
            ],
        )
        print(len(dataset))
        print(dataset[0])

    def test_no_downsample(self):
        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            downsample_voxel_size=0,
            train_transformations=[
                "NormalizeColor",
            ],
        )
        print(len(dataset))
        print(dataset[0])

    def test_features(self):
        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            train_transformations=[
                "NormalizeColor",
            ],
            features=["colors"],
        )
        print(len(dataset))
        self.assertTrue(dataset[0]["features"].shape[1] == 3)

        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            train_transformations=[],
            features=["colors", "xyzs"],
        )
        print(len(dataset))
        self.assertTrue(dataset[0]["features"].shape[1] == 6)

    def test_augmentation(self):
        import open3d as o3d

        from co3d_3d.src.data.utils import collate_fn, create_o3d_pointcloud

        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            train_transformations=[
                "CoordinateDropout",
                "ChromaticTranslation",
                "ChromaticJitter",
                "ChromaticAutoContrast",
                "RandomHorizontalFlip",
                "CoordinateJitter",
                "RandomAffine",
                "RegionDropout",
                "RandomTranslation",
                "ElasticDistortion",
                "NormalizeColor",
            ],
        )
        print(len(dataset))
        for i in range(len(dataset)):
            data_dict = dataset[i]
            pcd = create_o3d_pointcloud(
                data_dict["coordinates"], (data_dict["features"] + 0.5)
            )
            o3d.visualization.draw_geometries([pcd])

    def test_perlin_noise(self):
        import open3d as o3d

        from co3d_3d.src.data.utils import collate_fn, create_o3d_pointcloud

        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            train_transformations=[
                "PerlinNoise",
                "NormalizeColor",
            ],
        )
        print(len(dataset))

        for i in range(len(dataset)):
            data_dict = dataset[i]
            pcd = create_o3d_pointcloud(
                data_dict["coordinates"], (data_dict["features"] + 0.5)
            )
            o3d.visualization.draw_geometries([pcd])

    def test_loader(self):
        import open3d as o3d

        from co3d_3d.src.data.utils import collate_fn, create_o3d_pointcloud

        dataset = ScannetDataset(
            "train",
            "datasets/scannet",
            train_transformations=[
                "CoordinateDropout",
                "ChromaticTranslation",
                "ChromaticJitter",
                "ChromaticAutoContrast",
                "RandomHorizontalFlip",
                "RandomAffine",
                "RegionDropout",
                "RandomTranslation",
                "ElasticDistortion",
                "NormalizeColor",
            ],
        )
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
            stensor = ME.TensorField(
                coordinates=data["coordinates"], features=data["features"]
            )
            coordinates, features = stensor.decomposed_coordinates_and_features
            for i, (xyzs, colors) in enumerate(zip(coordinates, features)):
                pcd = create_o3d_pointcloud(xyzs, (colors + 0.5))
                o3d.visualization.draw_geometries([pcd])


@gin.configurable()
class PlenoxelScannetDataset(Dataset):
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IGNORE_LABELS_INSTANCE = tuple(
        set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE)
    )
    DATA_PATH_FILE = {
        "train": "scannet_256_train.txt",
        "val": "scannet_256_val.txt",
        "test": "scannet_256_val.txt",
    }
    CLASS_LABELS = CLASS_LABELS
    VALID_CLASS_IDS = VALID_CLASS_IDS

    def __init__(
        self,
        phase: str,
        data_root: str = "co3d_3d/datasets/co3d",
        train_transformations=[],
        eval_transformations=[],
        downsample_mode=1,
        downsample_stride=2,
        voxel_size: float = 0.02,
        num_points: int = -1,
        features: List[str] = ["sh"],
        ignore_label: int = -100,
        void_label: Optional[int] = None,
        valid_thres: float = 0.05,
        ignore_thres: Optional[float] = None,
    ) -> None:
        Dataset.__init__(self)
        phase = "test" if phase in ["val", "test"] else "train"
        transformations = (
            train_transformations if phase == "train" else eval_transformations
        )
        self.transformations = (
            transforms.Compose([transforms.__dict__[t]() for t in transformations])
            if len(transformations) > 0
            else None
        )
        self.phase = phase
        self.data_root = data_root
        self.num_points = num_points
        self.features = features
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.void_label = void_label if void_label is not None else ignore_label
        self.valid_thres = valid_thres
        self.ignore_thres = ignore_thres
        self.downsample_mode = downsample_mode
        self.downsample_stride = downsample_stride

        with open(
            os.path.join(
                os.path.dirname(self.data_root), "split", self.DATA_PATH_FILE[phase]
            ),
            "r",
        ) as f:
            self.files = [l.strip("\n") for l in f.readlines() if not l.startswith("#")]

        if self.downsample_mode == 0:
            self.pool = ME.MinkowskiAvgPooling(
                kernel_size=self.downsample_stride,
                stride=self.downsample_stride,
                dimension=3,
            )

        # map labels not evaluated to ignore_label
        label_map, n_used = dict(), 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                label_map[l] = ignore_label
            else:
                label_map[l] = n_used
                n_used += 1
        label_map[ignore_label] = ignore_label
        if void_label is not None and void_label != ignore_label:
            label_map[void_label] = n_used
        self.label_map = label_map

        with open(
            os.path.join(os.path.dirname(self.data_root), "split", "scene_scales.data"),
            "rb",
        ) as f:
            scene_scales = pickle.load(f)
        self.scene_scales = scene_scales
        logging.info(
            f"{self.__class__.__name__}(phase={phase}, total size={len(self.files)}, num_classes={len(self.CLASS_LABELS)}, downsample stride={self.downsample_stride})"
        )

    def downsample(self, coordinates, features):
        if self.downsample_mode == 0:
            bcoords = ME.utils.batched_coordinates([coordinates])
            stensor = ME.SparseTensor(features=features, coordinates=bcoords)
            output = self.pool(stensor)
            results = (output.C[:, 1:].float() / 2, output.F)
        elif self.downsample_mode == 1:
            sel = (coordinates % self.downsample_stride == 0).all(dim=1)
            # results = (coordinates[sel] / self.downsample_stride, features[sel])
            results = (coordinates[sel], features[sel])
        else:
            raise ValueError(f"Downsample mode {self.downsample_mode} is invalid.")

        logging.debug(
            f"voxel downsample with mode {self.downsample_mode} stride {self.downsample_stride}: from {coordinates.shape[0]} to {results[0].shape[0]}"
        )
        return results

    def load_data(self, inst_id):
        ckpt_path = os.path.join(
            self.data_root, f"plenoxel_torch_{inst_id}", "data.npz"
        )
        ckpt = np.load(ckpt_path)
        links = torch.from_numpy(ckpt["links"])
        density = torch.from_numpy(ckpt["density"])
        sh = ckpt["sh"].astype(np.float32) * ckpt["sh_scale"] + ckpt["sh_min"]
        sh = torch.from_numpy(sh)
        reso = ckpt["reso"]
        labels = torch.from_numpy(ckpt["labels"]).unsqueeze(1)

        dists = torch.from_numpy(ckpt["dists"]).unsqueeze(1)

        is_void = dists > self.valid_thres
        labels[is_void] = self.void_label

        if self.ignore_thres is not None and self.ignore_thres > 0:
            valid = dists < self.ignore_thres
            links = links[valid]
            sh = sh[valid]
            density = density[valid]
            labels = labels[valid]
        return dict(
            links=links, density=density, sh=sh, reso=reso, labels=labels, dists=dists
        )

    def __getitem__(self, index) -> dict:
        inst_id = self.files[index]

        data = self.load_data(inst_id)
        links, density, sh, reso, labels, dists = (
            data["links"],
            data["density"],
            data["sh"],
            data["reso"],
            data["labels"],
            data["dists"],
        )
        coordinates = torch.stack(
            [
                links // (reso[1] * reso[2]),
                links % (reso[1] * reso[2]) // reso[2],
                links % reso[2],
            ],
            1,
        ).float()

        if len(self.features) > 1:
            density /= np.abs(density).max() + 1e-5

        coordinates, dist_density_sh_label = self.downsample(
            coordinates, torch.cat([dists, density, sh, labels], dim=1)
        )
        norm_coordinates = coordinates / reso * 2 - 1.0
        scene_scale = self.scene_scales[inst_id]
        scaled_coordinates = norm_coordinates / scene_scale
        xyzs = scaled_coordinates / self.voxel_size
        labels = dist_density_sh_label[:, -1]
        dist_density_sh = dist_density_sh_label[:, :-1]

        # normalize xyzs to fit in unit sphere
        # xyzs = coordinates - coordinates.mean(dim=1, keepdim=True)
        # max_norm = torch.linalg.norm(xyzs, dim=1).max()
        # xyzs = xyzs / max_norm
        raw_features = torch.cat([xyzs, dist_density_sh], dim=1).float()

        xyzs = xyzs.numpy().astype(np.float32)
        raw_features = raw_features.numpy().astype(np.float32)

        if self.transformations is not None:
            xyzs, raw_features, labels = self.transformations(
                xyzs, raw_features, labels
            )

        dists = raw_features[:, 3:4]
        density = raw_features[:, 4:5]
        sh = raw_features[:, 5:]
        ones = np.ones(density.shape)

        features = []
        for f in self.features:
            features.append(eval(f))
        features = np.concatenate(features, axis=1).astype(np.float32)
        if self.IGNORE_LABELS is not None:
            labels = np.array(
                [self.label_map[x] for x in labels.numpy()], dtype=np.int32
            )

        return {
            "coordinates": xyzs.astype(np.float32),
            "features": features.astype(np.float32),
            "xyzs": xyzs,
            "labels": labels,
            "dists": dists,
            "metadata": {"file": self.files[index]},
        }

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase}, length={len(self)}, transform={self.transformations})"


if __name__ == "__main__":
    dataset = PlenoxelScannetDataset(
        "train",
        "./datasets/scannet_dist",
        downsample_stride=2,
        void_label=-333,
        ignore_thres=0.20,
    )
    dataset_2 = ScannetDataset(
        "val",
        "./datasets/scannet",
        voxel_size=0.02,
    )
    data = dataset[0]
    data_2 = dataset_2[0]
    import pdb

    pdb.set_trace()

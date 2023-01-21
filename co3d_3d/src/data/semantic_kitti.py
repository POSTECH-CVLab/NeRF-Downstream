import logging
import os
import unittest

import gin
import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import co3d_3d.src.data.transforms as transforms

CLASS_LABELS = (
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
)

label_name_mapping = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}


@gin.configurable
class SemanticKITTIDataset(Dataset):

    NUM_LABELS = 19

    def __init__(
        self,
        phase: str,
        data_root: str = "datasets/semantic-kitti/",
        downsample_voxel_size=None,  # in meter
        voxel_size=0.05,
        train_transformations=[
            "CoordinateDropout",
            "RandomHorizontalFlip",
            "RandomAffine",
            "RandomTranslation",
        ],
        eval_transformations=[],
        ignore_label=-100,
        features=["xyzi"],  # ["colors", "xyzs"]
    ):
        Dataset.__init__(self)
        self.data_root = data_root
        self.phase = phase
        self.ignore_label = ignore_label
        self.transformations_list = (
            train_transformations if phase == "train" else eval_transformations
        )
        self.transformations = transforms.Compose(
            [transforms.__dict__[t]() for t in self.transformations_list]
        )
        self.pc_files = []
        # fmt: off
        if phase == "train":
            seqs = ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10")
        elif phase == "trainval":
            seqs = ("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
        elif phase == "val":
            seqs = ("08",)
        elif phase == "test":
            seqs = ("11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21")
        # fmt: on

        for seq in seqs:
            seq_files = sorted(
                os.listdir(
                    os.path.join(self.data_root, "dataset/sequences", seq, "velodyne")
                )
            )
            seq_files = [os.path.join(seq, "velodyne", x) for x in seq_files]
            self.pc_files.extend(seq_files)

        if phase == "small_val":
            self.pc_files = self.pc_files[::10]

        if downsample_voxel_size is None:
            downsample_voxel_size = voxel_size / 2
        self.downsample_voxel_size = downsample_voxel_size
        self.voxel_size = voxel_size
        self.features = features

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        self.label_inv_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace("moving-", "") in CLASS_LABELS:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace("moving-", "")
                    ]
                else:
                    self.label_map[label_id] = self.ignore_label
            elif label_id == 0:
                self.label_map[label_id] = self.ignore_label
            else:
                if label_name_mapping[label_id] in CLASS_LABELS:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[label_name_mapping[label_id]] = cnt
                    self.label_inv_map[cnt] = label_id
                    cnt += 1
                else:
                    self.label_map[label_id] = self.ignore_label

        self.reverse_label_name_mapping = reverse_label_name_mapping
        logging.info(
            f"{self.__class__.__name__}(phase={phase}, total size={len(self.pc_files)}, features={self.features}, downsample_voxel_size={downsample_voxel_size}, voxel_size={voxel_size}, transformations={self.transformations_list})"
        )

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, data_index: int):
        path_split = self.pc_files[data_index].split("/")
        sequence = path_split[0]
        filename = path_split[-1]
        full_path = os.path.join(
            self.data_root, "dataset/sequences", self.pc_files[data_index]
        )
        with open(full_path, "rb") as b:
            xyzi = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        xyzs = xyzi[:, :3]

        label_file = full_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros((xyzi.shape[0])).astype(np.int32)

        labels = self.label_map[all_labels & 0xFFFF].astype(np.int64)
        # inst_labels = (all_labels >> 16).astype(np.int64)  # instance labels

        assert len(labels) == len(xyzs)

        if self.downsample_voxel_size > 0:
            _, xyzi, labels, row_inds = ME.utils.sparse_quantize(
                np.ascontiguousarray(xyzs),
                xyzi,
                labels=labels,
                quantization_size=self.downsample_voxel_size,
                ignore_label=self.ignore_label,
                return_index=True,
            )
            # Maintain the continuous coordinates
            xyzs = xyzs[row_inds]

        xyzs, xyzi, labels, _ = self.transformations(xyzs, xyzi, labels, None)
        xyzi[:, :3] = xyzs[:, :3]
        intensities = xyzi[:, 3, None]

        features = []
        for f in self.features:
            features.append(eval(f))

        # coordinates
        coordinates = xyzs / self.voxel_size
        return {
            "coordinates": coordinates.astype(np.float32),
            "features": np.concatenate(features, axis=1).astype(np.float32),
            "labels": labels,
            "metadata": {
                "file": self.pc_files[data_index],
                "sequence": sequence,
                "filename": filename,
                "data_index": data_index,
            },
            "instance_centers": np.zeros(1),
            "instance_ids": np.zeros(1),
        }

    def save_prediction(self, prediction, save_path, metadata):
        pred_file = os.path.join(
            save_path,
            "sequences",
            metadata["sequence"],
            "predictions",
            metadata["filename"].replace("bin", "label"),
        )
        os.makedirs(os.path.dirname(pred_file), exist_ok=True)
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        orig_label_prediction = self.label_inv_map[prediction]
        orig_label_prediction = orig_label_prediction.astype(np.uint32)
        orig_label_prediction.tofile(pred_file)
        return pred_file


class TestCase(unittest.TestCase):
    def test_read(self):
        dataset = SemanticKITTIDataset(
            "train",
            "datasets/semantic-kitti",
            train_transformations=[],
        )
        print(len(dataset))
        print(dataset[0])

    def test_mean(self):
        dataset = SemanticKITTIDataset(
            "train",
            "datasets/semantic-kitti",
            train_transformations=[],
        )
        print(len(dataset))
        for i in range(100):
            print(
                dataset[i]["coordinates"].mean(0),
                dataset[i]["coordinates"].min(0),
                dataset[i]["coordinates"].max(0),
            )
            print(
                dataset[i]["features"].mean(0),
                dataset[i]["features"].min(0),
                dataset[i]["features"].max(0),
            )

    def test_augmentation(self):
        import open3d as o3d

        from co3d_3d.src.data.utils import create_o3d_pointcloud

        dataset = SemanticKITTIDataset(
            "train",
            "datasets/semantic-kitti",
            downsample_voxel_size=0.05,  # in meter
            train_transformations=[
                "CoordinateDropout",
                "RandomHorizontalFlip",
                "RandomAffine",
                "RandomTranslation",
            ],
        )
        print(len(dataset))
        for i in range(len(dataset)):
            data_dict = dataset[i]
            print(len(data_dict["coordinates"]))
            assert len(data_dict["coordinates"]) == len(
                data_dict["features"]
            ), f'{data_dict["coordinates"].shape} != {data_dict["features"].shape}'
            pcd = create_o3d_pointcloud(data_dict["coordinates"])
            o3d.visualization.draw_geometries([pcd])

    def test_loader(self):
        import open3d as o3d

        from co3d_3d.src.data.utils import collate_fn, create_o3d_pointcloud

        dataset = SemanticKITTIDataset(
            "train",
            "datasets/semantic-kitti",
            train_transformations=[
                "CoordinateDropout",
                "RandomHorizontalFlip",
                "RandomAffine",
                "RegionDropout",
                "RandomTranslation",
                "PerlinNoise",
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

import json
import logging
import os
from typing import List, Optional, OrderedDict

import gin
import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import co3d_3d.src.data.transforms as transforms

CLASSES = [
    "apple",
    "backpack",
    "ball",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "book",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "couch",
    "cup",
    "donut",
    "frisbee",
    "hairdryer",
    "handbag",
    "hotdog",
    "hydrant",
    "keyboard",
    "kite",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "remote",
    "sandwich",
    "skateboard",
    "stopsign",
    "suitcase",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]
CLASSES_IDX = {k: v for (k, v) in zip(CLASSES, range(len(CLASSES)))}


@gin.configurable()
class Co3DDatasetBase(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "co3d_3d/datasets/co3d",
        train_transformations=[],
        eval_transformations=[],
        downsample_mode=1,
        downsample_stride=2,
        num_points: int = -1,
        features: List[str] = ["sh"],
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
        self.downsample_mode = downsample_mode
        self.downsample_stride = downsample_stride

        with open(os.path.join(f"filelist/{phase}.txt"), "r") as f:
            self.files = [l.strip("\n").split()[:2] for l in f.readlines()]

        self.CLASS_LABELS = CLASSES
        self.NUM_CLASSES = len(self.CLASS_LABELS)

        if self.downsample_mode == 0:
            self.pool = ME.MinkowskiAvgPooling(
                kernel_size=self.downsample_stride,
                stride=self.downsample_stride,
                dimension=3,
            )
        logging.info(
            f"{self.__class__.__name__}(phase={phase}, total size={len(self.files)}, num_classes={self.NUM_CLASSES}, downsample stride={self.downsample_stride})"
        )

    def downsample(self, coordinates, features):
        if self.downsample_mode == 0:
            bcoords = ME.utils.batched_coordinates([coordinates])
            stensor = ME.SparseTensor(features=features, coordinates=bcoords)
            output = self.pool(stensor)
            results = (output.C[:, 1:].float(), output.F)
        elif self.downsample_mode == 1:
            sel = (coordinates % self.downsample_stride == 0).all(dim=1)
            results = (coordinates[sel], features[sel])
        else:
            raise ValueError(f"Downsample mode {self.downsample_mode} is invalid.")

        logging.debug(
            f"voxel downsample with mode {self.downsample_mode} stride {self.downsample_stride}: from {coordinates.shape[0]} to {results[0].shape[0]}"
        )
        return results

    def _load_data_torch(self, inst_id):
        ckpt_path = os.path.join(
            self.data_root, f"plenoxel_co3d_{inst_id}", "last.ckpt"
        )
        # conf_path = os.path.join(
        #     self.data_root, f"plenoxel_torch_{inst_id}", "args.txt"
        # )

        # # get resolution
        # assert os.path.exists(conf_path), f"{conf_path} not exists."
        # with open(conf_path, "r") as f:
        #     lines = [l.strip() for l in f.readlines()]
        #     key, value = zip(*[l.split("=") for l in lines])
        #     key = [k.strip() for k in key]
        #     value = [v.strip() for v in value]
        #     conf = dict(zip(key, value))
        # reso_list = json.loads(conf["reso"])

        # load checkpoint
        assert os.path.exists(ckpt_path), f"{ckpt_path} not exists."
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # reso_idx = ckpt["reso_idx"]
        reso = [256, 256, 256]
        links = ckpt["state_dict"]["model.links_idx"]
        density = ckpt["state_dict"]["model.density_data"]
        sh = ckpt["state_dict"]["model.sh_data"]
        sh_min = ckpt["model.sh_data_min"]
        sh_scale = ckpt["model.sh_data_scale"]
        sh = sh.float() * sh_scale + sh_min
        return dict(links=links, density=density, sh=sh, reso=reso)

    def _load_data_numpy(self, inst_id):
        ckpt_path = os.path.join(self.data_root, f"plenoxel_co3d_{inst_id}", "data.npz")
        ckpt = np.load(ckpt_path)
        links = torch.from_numpy(ckpt["links"])
        density = torch.from_numpy(ckpt["density"])
        sh = ckpt["sh"].astype(np.float32) * ckpt["sh_scale"] + ckpt["sh_min"]
        sh = torch.from_numpy(sh)
        reso = [128, 128, 128]
        return dict(links=links, density=density, sh=sh, reso=reso)

    def load_data(self, inst_id):
        scene_path = os.path.join(self.data_root, f"plenoxel_co3d_{inst_id}")
        numpy_file = os.path.join(scene_path, "data.npz")
        torch_file = os.path.join(scene_path, "last.ckpt")
        if os.path.exists(numpy_file):
            return self._load_data_numpy(inst_id)
        elif os.path.exists(torch_file):
            return self._load_data_torch(inst_id)
        else:
            raise ValueError(f"{inst_id} not exist in {self.data_root}")

    def __getitem__(self, index) -> dict:
        label, inst_id = self.files[index]
        label = self.CLASS_LABELS.index(label)

        data = self.load_data(inst_id)
        links, density, sh, reso = (
            data["links"],
            data["density"],
            data["sh"],
            data["reso"],
        )
        coordinates = torch.stack(
            [
                torch.div(links, (reso[1] * reso[2]), rounding_mode="trunc"),
                torch.div(links % (reso[1] * reso[2]), reso[2], rounding_mode="trunc"),
                links % reso[2],
            ],
            1,
        ).float()

        density_sh = torch.cat([density, sh], dim=1)
        # coordinates, density_sh = self.downsample(
        #     coordinates, torch.cat([density, sh], dim=1)
        # )

        # normalize xyzs to fit in unit sphere
        xyzs = coordinates - coordinates.mean(dim=1, keepdim=True)
        max_norm = torch.linalg.norm(xyzs, dim=1).max()
        xyzs = xyzs / max_norm
        raw_features = torch.cat([xyzs, density_sh], dim=1).float()

        if self.transformations is not None:
            coordinates, raw_features, _ = self.transformations(
                coordinates, raw_features, None
            )

        xyzs = raw_features[:, :3]
        density = raw_features[:, 3:4]
        sh = raw_features[:, 4:]
        ones = torch.ones_like(density)

        features = []
        for f in self.features:
            features.append(eval(f))
        features = torch.cat(features, dim=1).float()

        return {
            "coordinates": coordinates,
            "features": features,
            "xyzs": xyzs,
            "labels": np.array([label]),
            # "metadata": {
            #     "file": self.files[index],
            #     # "ckpt_path": ckpt_path,
            #     # "conf_path": conf_path,
            #     "reso": reso,
            # },
        }

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase}, length={len(self)}, transform={self.transformations})"


class Co3D10pDataset(Co3DDatasetBase):
    DATA_PATH_FILE = {
        "train": "co3d_10p_train.txt",
        "trainval": "co3d_10p_trainval.txt",
        "val": "co3d_10p_val.txt",
        "test": "co3d_10p_test.txt",
    }
    CLASS_FILE = "co3d_10p_classes.txt"


class Co3DDataset(Co3DDatasetBase):
    DATA_PATH_FILE = {
        "train": "co3d_train.txt",
        "trainval": "co3d_trainval.txt",
        "val": "co3d_val.txt",
        "test": "co3d_test.txt",
    }
    CLASS_FILE = "co3d_classes.txt"


if __name__ == "__main__":
    dataset = Co3DDataset("train", "./datasets/co3d")
    import pdb

    pdb.set_trace()
    pass

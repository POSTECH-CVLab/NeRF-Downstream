import os
import time
import torch
import tqdm
import numpy as np

import typing

import pytorch_lightning as pl
import gin

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from co3d_2d.src.data.transforms import *
from co3d_2d.src.data.augmix import augment_and_mix

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
class Co3DTrainDataset(Dataset):

    def __init__(
        self, 
        train_transformations: typing.List = [
            "RandomResizedCrop", "ColorJitter", "RandomHorizontalFlip", 
            "ToTensor", "PCALoss", "Normalize"
        ],
    ):
        filelist = f"filelist/train.txt"
        self.files, self.labels, self.num_frames = [], [], []
        self.transforms = transforms.Compose(
			[eval(transform)() for transform in train_transformations]
		)
        with open(filelist) as fp:
            lines = fp.readlines()
            lines = [line.rstrip("/").split() for line in lines]
        for (cls_name, scene_name, frame_num) in tqdm.tqdm(lines, desc=f"Loading Co3D train set"):
            scene_path = os.path.join("co3d_2d/data/co3d", cls_name, scene_name)
            self.files.append(os.path.join(scene_path, "images"))
            self.num_frames.append(frame_num)
            self.labels.append(CLASSES_IDX[cls_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        l = self.labels[idx]
        random_idx = np.random.randint(self.num_frames[idx])
        fname = os.listdir(f"{self.files[idx]}")[random_idx]
        fpath = os.path.join(self.files[idx], fname)
        x = Image.open(fpath)
        x = augment_and_mix(x, self.transforms)
        return {"images": x, "labels": l}

@gin.configurable()
class Co3DEvalDataset(Dataset):

    def __init__(
        self, 
        phase: str,
        eval_transformations: typing.List = ["CenterCrop", "ToTensor", "Normalize"],
    ):
        filelist = f"filelist/{phase}.txt"
        self.files, self.labels, self.num_frames = [], [], []
        self.transforms = transforms.Compose(
			[eval(transform)() for transform in eval_transformations]
		)

        with open(filelist) as fp:
            lines = fp.readlines()
            lines = [line.rstrip("/").split() for line in lines]

        for (cls_name, scene_name, frame_num) in tqdm.tqdm(lines, desc=f"Loading Co3D {phase} set"):
            scene_path = os.path.join("co3d_2d/data/co3d", cls_name, scene_name)
            images_path = os.path.join(scene_path, "images")
            for frame_name in sorted(os.listdir(images_path)):
                self.files.append(os.path.join(images_path, frame_name))
                self.labels.append(CLASSES_IDX[cls_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        l = self.labels[idx]
        x = Image.open(self.files[idx])
        x = self.transforms(x)
        return {"images": x, "labels": l}


@gin.configurable()
class PeRFCeptionCo3DTrainDataset(Dataset):

    def __init__(
        self,
        train_transformations: typing.List = [
            "RandomResizedCrop", "ColorJitter", "RandomHorizontalFlip", 
            "ToTensor", "PCALoss", "Normalize"
        ],
        bkgd_aug: float = 0.0 
    ):
        filelist = f"filelist/train.txt"
        self.files, self.labels, self.num_frames = [], [], []
        self.bkgd_aug = bkgd_aug
        self.transforms = transforms.Compose([eval(transform)() for transform in train_transformations])
        with open(filelist) as fp:
            lines = fp.readlines()
            lines = [line.rstrip("/").split() for line in lines]
        for (cls_name, scene_name, frame_num) in tqdm.tqdm(lines, desc=f"Loading PeRFCeptionCo3D train set"):
            scene_path = os.path.join("co3d_2d/data/perfception", cls_name, scene_name)
            self.files.append(os.path.join(scene_path, "fgbg"))
            self.labels.append(CLASSES_IDX[cls_name])
        if bkgd_aug > 0:
            self.bkgd_aug_fun = BackgroundAug()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        l = self.labels[idx]
        random_idx = np.random.randint(50)
        p = np.random.random()
        fname = os.listdir(f"{self.files[idx]}")[random_idx]
        fpath = os.path.join(self.files[idx], fname)
        x = Image.open(fpath)
        
        if p < self.bkgd_aug:
            bkgd_idx = np.random.randint(len(self.files))
            bkgd_frame_idx = np.random.randint(50)
            bgdirname = self.files[bkgd_idx].replace("fgbg", "bg")
            maskdirname = self.files[idx].replace("fgbg", "mask")
            bgpath = os.path.join(bgdirname, f"image{str(bkgd_frame_idx).zfill(3)}.jpg")
            maskpath = os.path.join(maskdirname, f"mask{fname[5:]}")

            bgimg = Image.open(bgpath)
            maskimg = Image.open(maskpath)
            x = self.bkgd_aug_fun(x, bgimg, maskimg)
			
        x = augment_and_mix(x, self.transforms)
        return {"images": x, "labels": l}

	
@gin.configurable()
class PeRFCeptionCo3DEvalDataset(Dataset):

    def __init__(
        self,
        phase: str,
        eval_transformations: typing.List = ["CenterCrop", "ToTensor", "Normalize"],
    ):
        filelist = f"filelist/{phase}.txt"
        self.files, self.labels, self.num_frames = [], [], []
        self.transforms = transforms.Compose([eval(transform)() for transform in eval_transformations])

        with open(filelist) as fp:
            lines = fp.readlines()
            lines = [line.rstrip("/").split() for line in lines]

        for (cls_name, scene_name, frame_num) in tqdm.tqdm(lines, desc=f"Loading PeRFCeptionCo3D {phase} set"):
            scene_path = os.path.join("co3d_2d/data/perfception", cls_name, scene_name)
            images_path = os.path.join(scene_path, "fgbg")
            for frame_name in sorted(os.listdir(images_path)):
                self.files.append(os.path.join(images_path, frame_name))
                self.labels.append(CLASSES_IDX[cls_name])

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        l = self.labels[idx]
        x = Image.open(self.files[idx])
        x = self.transforms(x)
        return {"images": x, "labels": l}



@gin.configurable()
class DataModule(pl.LightningDataModule):

    def __init__(
        self, 
        num_workers: int = 16,
        batch_size: int = 32,
        chunks: int = 32,
        train_co3d: bool = True,
        eval_co3d: bool = True
    ):
        super(DataModule, self).__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.chunks = chunks
        self.train_co3d = train_co3d
        self.eval_co3d = eval_co3d

    def train_dataloader(self):
        return DataLoader(
            Co3DTrainDataset() if self.train_co3d else PeRFCeptionCo3DTrainDataset(), 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Co3DEvalDataset("val") if self.eval_co3d else PeRFCeptionCo3DEvalDataset("val"),
            batch_size=self.chunks,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            Co3DEvalDataset("test") if self.eval_co3d else PeRFCeptionCo3DEvalDataset("test"),
            batch_size=self.chunks,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
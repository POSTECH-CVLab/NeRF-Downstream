import os

import gin
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info
from co3d_3d.src.data.datasets import get_dataset
from co3d_3d.src.data.utils import collate_mink, collate_pair, collate_pointnet
from torch.utils.data import DataLoader, Subset


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_phase: str = "train",
        val_phase: str = "val",
        test_phase: str = "test",
        prune_phase: str = "train",
        batch_size: int = 12,
        val_batch_size: int = 6,
        test_batch_size: int = 1,
        prune_batch_size: int = 6,
        train_num_workers: int = 4,
        val_num_workers: int = 2,
        test_num_workers: int = 2,
        prune_num_workers: int = 2,
        prune_subset_ratio: float = 0.1,
        collate_func_name: str = "collate_mink",
    ):
        LightningDataModule.__init__(self)

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        if collate_func_name == "collate_pair":
            self.collate_fn = collate_pair
        elif collate_func_name == "collate_mink":
            self.collate_fn = collate_mink
        elif collate_func_name == "collate_pointnet":
            self.collate_fn = collate_pointnet
        else:
            raise ValueError(f"{collate_func_name} is not supported.")

    def get_dataset(self, phase="train"):
        DatasetClass = get_dataset()
        return DatasetClass(phase=phase)

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = self.get_dataset(self.train_phase)

        if not hasattr(self, "__train_dataloader"):
            self.__train_dataloader = DataLoader(
                self.train_dataset,
                num_workers=min(
                    max(int(self.batch_size / int(os.environ.get("WORLD_SIZE", 1))), 2),
                    self.train_num_workers,
                ),
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                pin_memory=False,
                persistent_workers=True,
            )
            rank_zero_info(
                f"Train dataloader: {self.__train_dataloader.__class__.__name__}(num_workers={self.__train_dataloader.num_workers}, batch_size={self.batch_size})"
            )
        return self.__train_dataloader

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = self.get_dataset(self.val_phase)

        if not hasattr(self, "__val_dataloader"):
            self.__val_dataloader = DataLoader(
                self.val_dataset,
                num_workers=min(self.val_batch_size, self.val_num_workers),
                batch_size=self.val_batch_size,
                collate_fn=self.collate_fn,
                pin_memory=False,
                persistent_workers=True,
            )
            rank_zero_info(
                f"Val dataloader: {self.__val_dataloader.__class__.__name__}(num_workers={self.__val_dataloader.num_workers}, batch_size={self.val_batch_size})"
            )
        return self.__val_dataloader

    def test_dataloader(self):
        self.test_dataset = get_dataset()(
            phase=self.test_phase, downsample_voxel_size=0
        )
        dataloader = DataLoader(
            self.test_dataset,
            num_workers=self.test_num_workers,
            batch_size=self.test_batch_size,
            collate_fn=self.collate_fn,
        )
        rank_zero_info(
            f"Test dataloader: {dataloader.__class__.__name__}(num_workers={dataloader.num_workers}, batch_size={self.test_batch_size})"
        )
        return dataloader

    def prune_dataloader(self):
        dataset = self.get_dataset(self.prune_phase)
        length = int(len(dataset) * self.prune_subset_ratio)
        indices = torch.randperm(len(dataset))[:length]
        self.prune_dataset = Subset(dataset, indices)

        if not hasattr(self, "__prune_dataloader"):
            self.__prune_dataloader = DataLoader(
                self.prune_dataset,
                num_workers=self.prune_num_workers,
                batch_size=self.prune_batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                pin_memory=False,
                persistent_workers=True,
            )
            rank_zero_info(
                f"Prune dataloader: {self.__prune_dataloader.__class__.__name__}(num_workers={self.__prune_dataloader.num_workers}, batch_size={self.prune_batch_size})"
            )

        return self.__prune_dataloader

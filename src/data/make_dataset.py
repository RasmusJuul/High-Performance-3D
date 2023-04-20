# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, List, Optional, Union

import PIL
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torchvision import transforms
from src import _PATH_DATA

import itertools
from enum import Enum, IntEnum, auto

import pandas as pd
from tifffile import tifffile
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation


class Label(IntEnum):
    Blowfly = 2,
    CurlyWingedFly = 5,
    Pupae = 9,
    Maggot = 7,
    BuffaloBeetleLarvae = 3,
    Mealworm = 8,
    SoliderFlyLarvae = 10,
    Woodlice = 11,
    BlackCricket = 1,
    Grasshopper = 6,
    BrownCricket = 0,
    BlowflyPupae = 4,

    @staticmethod
    def abbreviation_dict():
        assert len(Label) == 12
        return {
            Label.Blowfly: "BF",
            Label.CurlyWingedFly: "CF",
            Label.Pupae: "PP",
            Label.Maggot: "MA",
            Label.BuffaloBeetleLarvae: "BL",
            Label.Mealworm: "ML",
            Label.SoliderFlyLarvae: "SL",
            Label.Woodlice: "WO",
            Label.BlackCricket: "BC",
            Label.Grasshopper: "GH",
            Label.BrownCricket: "AC",
            Label.BlowflyPupae: "BP"
        }

    @property
    def abbreviation(self):
        return self.abbreviation_dict()[self]

    @staticmethod
    def from_abbreviation(abbreviation: str):
        return next(label for label, label_abbreviation in Label.abbreviation_dict().items() if label_abbreviation == abbreviation.upper())


class SplitType(Enum):
    Train = auto()
    Validation = auto()
    Test = auto()


class BugNIST(torch.utils.data.Dataset):
    def __init__(self, type: SplitType, seed=42, as_rgb=False, transforms=False):

        dataset_path = os.path.join(_PATH_DATA,"raw/Bugs/bugnist_256")

        self.image_paths, self.image_labels = self.dataset_images(dataset_path, type)
        self.image_paths = [os.path.join(dataset_path, path) for path in self.image_paths]
        self.rng = np.random.default_rng(seed=seed)
        # self.as_rgb = as_rgb
        self.transforms = transforms
        self.rot = RandomRotation(180)

    @staticmethod
    def dataset_images(MNInSecT_root: str, type: SplitType) -> tuple[list[str], list[Label]]:
        assert len(SplitType) == 3
        file_name = "train" if type == SplitType.Train else "test" if type == SplitType.Test else "validation"
        files = pd.read_csv(f"{MNInSecT_root}/{file_name}.csv", names=["files"], header=0).files
        labels = [Label.from_abbreviation(abbreviation) for abbreviation in files.map(lambda x: x[:2]).to_list()]
        return files.to_list(), labels

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]
        label = self.image_labels[idx]

        image = tifffile.imread(image_path)

        image = np.expand_dims(image, 0)
        
        X = torch.Tensor(image)
        if self.transforms:
            X = self.rot(X)
        # if self.as_rgb:
        #     size = X.shape
        #     X = X.expand([3, size[1], size[2], size[3]])

        y = label.value
        one_hot_encoded = torch.zeros(self.num_classes())
        one_hot_encoded[y] = 1

        return X, one_hot_encoded

    @staticmethod
    def num_classes() -> int:
        return len(Label)

    @staticmethod
    def label_to_name(label: int) -> str:
        return Label(label).abbreviation

    def get_name_of_image(self, idx: int) -> str:
        return self.image_paths[idx].split("/")[-1].split(".")[0]




class BugNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = _PATH_DATA, batch_size: int = 64, num_workers: int = 0, transforms = True, seed = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.seed = seed

    def setup(self, stage=None):        
        if stage == "test" or stage is None:
            self.bugnist_test = BugNIST(type=SplitType.Test, seed=self.seed, transforms=self.transforms)

        if stage == "fit" or stage is None:
            self.bugnist_train = BugNIST(type=SplitType.Train, seed=self.seed, transforms=self.transforms)

            self.bugnist_val = BugNIST(type=SplitType.Validation, seed=self.seed, transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.bugnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.bugnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.bugnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

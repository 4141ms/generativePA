# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：dataset.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:25 
"""
## utils/dataset.py
# utils/dataset.py
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from monai.transforms import LoadImaged
from monai.data import PersistentDataset
import torch
import pandas as pd
# from config import cfg

from typing import Tuple, Union


def get_datalist(
        ids_path: str,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
                "report": "T1-weighted image of a brain.",
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_dataloader(
        cache_dir: Union[str, Path],
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        num_workers: int = 8,
        model_type: str = "vqvae",
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[158, 222, 158],  # 目标尺寸(D, H, W)
            ),
            # ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    if model_type == "vqvae":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[158, 222, 158],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[158, 222, 158],
                    prob=0.1,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
            ]
        )

    train_dicts = get_datalist(ids_path=training_ids)
    train_ds = PersistentDataset(data=train_dicts, transform=train_transforms, cache_dir=str(cache_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist(ids_path=validation_ids)
    val_ds = PersistentDataset(data=val_dicts, transform=val_transforms, cache_dir=str(cache_dir))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader

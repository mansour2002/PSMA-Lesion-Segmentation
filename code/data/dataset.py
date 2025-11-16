import torch
import pytorch_lightning as pl
import numpy as np
from functools import partial
from monai.data import (
    CacheDataset,
    partition_dataset,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    RandFlipd,
    RandShiftIntensityd,
    ToTensord,
    EnsureChannelFirstd,
    RandScaleIntensityd,
    RandAffined,
    RandCoarseDropoutd
)
from monai.data import DataLoader
import json
import os
from data.smart_sampler import SmartPosNegSampler
from monai.transforms import MapTransform
import numpy as np
import random


class SkipZeroLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, skip_prob=0.5):
        """
        base_dataset: your original MONAI dataset (CacheDataset)
        skip_prob: probability to skip zero-label samples
        """
        self.base_dataset = base_dataset
        self.skip_prob = skip_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample_list = self.base_dataset[idx]
        
        labels = [s.get("label", None) for s in sample_list]

        # Check if label is all zeros
        if all(label.sum() == 0 for label in labels):
            # Skip with probability skip_prob
            if random.random() < self.skip_prob:
                new_idx = random.randint(0, len(self.base_dataset) - 1)
                return self.__getitem__(new_idx)

        return sample_list
    
class RandScalePETd(MapTransform):
    def __init__(self, keys, min_factor=0.5, max_factor=1.7, prob=1.0):
        super().__init__(keys)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if np.random.rand() < self.prob:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            for key in self.keys:
                d[key] = d[key] * factor
        return d
class LesionSegmentationDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        cache_dir: str,
        downsample_ratio: list = [1.0, 1.0, 1.0],
        batch_size: int = 4,
        val_batch_size: int = 1,
        num_workers: int = 8,
        cache_num: int = 24,
        cache_rate: float = 1.0,
        dist: bool = False,
        samples_per_image: int = 1, 
        val_samples_per_image: int = 1, 
        roi_size: list = [96, 96, 96],
        foreground_ratio = 0.5,
        intensity_range_ct: list = [-1000, 1200],  # CT intensity range
        intensity_range_pet: list = [0, 5000],      # PET intensity range
        skip_prob: float = 0.8,
        binarize: bool = True,
        exclude_roi_labels: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.downsample_ratio = downsample_ratio
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.dist = dist
        self.samples_per_image = samples_per_image 
        self.val_samples_per_image = val_samples_per_image 
        self.roi_size = roi_size
        self.foreground_ratio = foreground_ratio
        self.intensity_range_ct = intensity_range_ct
        self.intensity_range_pet = intensity_range_pet
        self.skip_prob = skip_prob
        self.binarize = binarize
        self.exclude_roi_labels = exclude_roi_labels

        # Load dataset JSON
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)

        # Define transforms
        self.train_transforms = Compose([
            LoadImaged(keys=["ct_image", "pt_image", "labels"], image_only=True),
            EnsureChannelFirstd(keys=["ct_image", "pt_image", "labels"]),
            
            ScaleIntensityRanged(
                keys=["ct_image"],
                a_min=self.intensity_range_ct[0], 
                a_max=self.intensity_range_ct[1],
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            ScaleIntensityRanged(
                keys=["pt_image"],
                a_min=self.intensity_range_pet[0], 
                a_max=self.intensity_range_pet[1],
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            # Combine augmentations
            RandFlipd(keys=["ct_image", "pt_image", "labels"], spatial_axis=[0, 1, 2], prob=0.50),
            RandScaleIntensityd(keys=["ct_image", "pt_image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["ct_image", "pt_image"], offsets=0.1, prob=0.5),
            RandAffined(keys=["ct_image", "pt_image", "labels"],
                        rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1),
                        prob=0.5),
            
            RandCoarseDropoutd(
                keys=["pt_image"],
                holes=10,     # random number of dots
                spatial_size=1, # each dot = 1 voxel
                dropout_holes=True,    # replace with random intensity, not 0
                fill_value = (0, 0.12),
                max_holes = 1000,
                prob=0.3
            ),
            
            
            SmartPosNegSampler(
                keys=["ct_image", "pt_image", "labels"],
                roi_size=self.roi_size,
                num_samples=self.samples_per_image,
                foreground_ratio=self.foreground_ratio,
            ),

            
            # Concatenate modalities and prepare final output
            partial(self.concatenate_modalities_and_labels, binarize=self.binarize, exclude_roi_labels=self.exclude_roi_labels),
            ToTensord(keys=["image", "label"]),
        ])

        # Validation transforms
        self.val_transforms = Compose([
            LoadImaged(keys=["ct_image", "pt_image", "labels"], image_only=True),
            EnsureChannelFirstd(keys=["ct_image", "pt_image", "labels"]),

            ScaleIntensityRanged(
                keys=["ct_image"], 
                a_min=self.intensity_range_ct[0], 
                a_max=self.intensity_range_ct[1],
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            ScaleIntensityRanged(
                keys=["pt_image"], 
                a_min=self.intensity_range_pet[0], 
                a_max=self.intensity_range_pet[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),

            SmartPosNegSampler(
                keys=["ct_image", "pt_image", "labels"],
                roi_size=self.roi_size,
                num_samples=self.samples_per_image,
                foreground_ratio=self.foreground_ratio,
            ),
            
            # Concatenate modalities and prepare final output
            partial(self.concatenate_modalities_and_labels, binarize=self.binarize, exclude_roi_labels=self.exclude_roi_labels),
            ToTensord(keys=["image", "label"]),
        ])

    
    @staticmethod
    def concatenate_modalities_and_labels(x, binarize, exclude_roi_labels):
        """Concatenate CT and PET images and labels along the axial (depth) dimension."""
        label = x["labels"]
        label = label.long()
        if exclude_roi_labels:
            label[label==1] = 0
            label[label==5] = 0
            label[label==8] = 0
        if binarize:
            label = (label > 0).long() 
        return {
            "image": torch.cat([
                x["ct_image"],  # [1, 96, 96, 96]
                x["pt_image"]   # [1, 96, 96, 96]
            ], dim=0),  # [2, 96, 96, 96]
            "label": label,
            "case_id": x["case_id"]
        }

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            if self.dist:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                train_partition = partition_dataset(
                    data=self.dataset["training"],
                    num_partitions=world_size,
                    shuffle=True,
                    even_divisible=True,
                )[rank]
                valid_partition = partition_dataset(
                    data=self.dataset["validation"],
                    num_partitions=world_size,
                    shuffle=False,
                    even_divisible=True,
                )[rank]
                print(f"[rank: {rank}] Training samples: {len(train_partition)}")
                print(f"[rank: {rank}] Validation samples: {len(valid_partition)}")
            else:
                train_partition = self.dataset["training"]
                valid_partition = self.dataset["validation"]

            self.train_ds = CacheDataset(
                train_partition,
                cache_num=self.cache_num,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                transform=self.train_transforms,
            )

            self.train_ds = SkipZeroLabelDataset(self.train_ds, skip_prob=self.skip_prob)

            self.valid_ds = CacheDataset(
                valid_partition,
                cache_num=self.cache_num // 4,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                transform=self.val_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(not self.dist),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=2,
        )

    def __getitem__(self, index):
        return self.train_ds[index]

    def __len__(self):
        return len(self.train_ds)



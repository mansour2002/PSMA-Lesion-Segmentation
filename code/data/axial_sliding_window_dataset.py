"""
Axial Sliding Window (ASW) Dataset
Extracts 6 overlapping axial patches at fixed XY center, distributed along Z-axis
XY center is chosen on a random lesion voxel when the case has lesions
Assumes volumes are already resampled to [4.0, 4.0, 3.27] mm spacing with Z (axial) as the last axis
"""
import torch
import pytorch_lightning as pl
import numpy as np
import random
from functools import partial
from monai.data import CacheDataset, Dataset, load_decathlon_datalist, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    RandFlipd,
    RandAffined,
    ToTensord,
    MapTransform,
)
import json


class AxialSlidingWindowSampler(MapTransform):
    """
    Custom transform that extracts 6 overlapping patches along Z-axis
    - All patches share same XY center (vertical column)
    - XY center is chosen to ensure at least 1 patch has a lesion
    - Patches are equally distributed along Z with overlap
    """
    def __init__(self, keys, roi_size=(96, 96, 96), num_patches=6):
        super().__init__(keys)
        self.keys = keys
        self.roi_d, self.roi_h, self.roi_w = roi_size
        self.num_patches = num_patches

    def __call__(self, data):
        # Get dimensions
        # After MONAI loading: shape is [C, X, Y, Z] where Z is axial (last dimension)
        image = data['image']
        labels = data['label']
        C, X, Y, Z = image.shape
        case_id = data.get('case_id', 'Unknown')

        # Find lesion coordinates for smart XY sampling
        # pos_coords shape: (N, 3) corresponding to [X, Y, Z] indices
        pos_coords = torch.nonzero(labels[0] > 0, as_tuple=False)

        # Select random XY center (lesion-aware if possible)
        # We want to fix X and Y position, slide along Z
        if pos_coords.numel() > 0:
            # Choose random lesion voxel and use its X,Y as center
            idx = random.randint(0, pos_coords.shape[0] - 1)
            cx, cy, _ = pos_coords[idx].tolist()  # Extract X, Y coordinates

            # Add random jitter (max ±48 voxels)
            jitter_range = min(self.roi_w // 2, self.roi_h // 2)
            jx = random.randint(-jitter_range, jitter_range)
            jy = random.randint(-jitter_range, jitter_range)
            cx, cy = cx + jx, cy + jy
        else:
            # No lesions: random XY center in valid range
            # Ensure range is valid (min <= max)
            cx = random.randint(self.roi_w // 2, max(self.roi_w // 2 + 1, X - self.roi_w // 2))
            cy = random.randint(self.roi_h // 2, max(self.roi_h // 2 + 1, Y - self.roi_h // 2))

        # Calculate start positions (top-left corner of patch)
        sx_raw = cx - self.roi_w // 2
        sy_raw = cy - self.roi_h // 2

        # Clamp to valid range: [0, X - roi_w] and [0, Y - roi_h]
        sx = max(0, min(sx_raw, X - self.roi_w))
        sy = max(0, min(sy_raw, Y - self.roi_h))

        # Calculate axial (Z) positions for 6 patches with overlap
        # Z is the last dimension (axial direction in medical imaging)
        # Each patch covers: roi_d (96) voxels along Z
        # We need to cover Z voxels total (e.g., 257-527)

        if Z <= self.roi_d:
            # Volume smaller than patch: just one patch at z=0
            z_starts = [0] * self.num_patches
        else:
            # Calculate stride to cover full Z range
            # We want: z_starts[0] = 0, z_starts[-1] = Z - roi_d
            # With num_patches positions evenly spaced
            stride = (Z - self.roi_d) / (self.num_patches - 1)
            z_starts = [int(i * stride) for i in range(self.num_patches)]

            # Verify coverage
            last_patch_end = z_starts[-1] + self.roi_d
            if last_patch_end < Z:
                print(f"⚠️  Warning: {case_id} - Incomplete Z coverage!")
                print(f"   Z size: {Z}, Last patch ends at: {last_patch_end}, Gap: {Z - last_patch_end}")

        # Extract 6 patches at same XY position, sliding along Z
        # Shape: [C, X, Y, Z] → extract [C, sx:sx+96, sy:sy+96, sz:sz+96]
        patches = []
        for i, sz in enumerate(z_starts):
            patch = {
                'image': image[:, sx:sx + self.roi_w, sy:sy + self.roi_h, sz:sz + self.roi_d].clone(),
                'label': labels[:, sx:sx + self.roi_w, sy:sy + self.roi_h, sz:sz + self.roi_d].clone(),
            }
            patches.append(patch)

        # Store metadata for stitching during loss calculation
        data['patches'] = patches  # List of 6 dicts with 'image' and 'label'
        data['z_starts'] = z_starts  # Z positions for stitching
        data['xy_start'] = (sx, sy)  # Fixed XY position
        data['original_shape'] = (C, X, Y, Z)  # For extracting GT region
        data['case_id'] = case_id

        return data


class AxialSlidingWindowDataset(pl.LightningDataModule):
    """
    Dataset for axial sliding window training (V41T01)
    - Loads whole-body volumes with spacing [4.0, 4.0, 3.27]
    - Applies augmentations (flip, affine)
    - Extracts 6 axial patches at lesion-aware XY center
    - Returns patches + metadata for stitching
    """
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        cache_dir: str,
        roi_size: list = [96, 96, 96],
        num_patches: int = 6,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 6,
        cache_num: int = 4,
        cache_rate: float = 1.0,
        dist: bool = False,
        intensity_range_ct: list = [-1000, 1200],
        intensity_range_pet: list = [0, 5000],
        binarize: bool = True,
        exclude_roi_labels: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.roi_size = roi_size
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.dist = dist
        self.intensity_range_ct = intensity_range_ct
        self.intensity_range_pet = intensity_range_pet
        self.binarize = binarize
        self.exclude_roi_labels = exclude_roi_labels

        # Load dataset JSON
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)

    def concatenate_modalities_and_labels(self, data, binarize=True, exclude_roi_labels=True):
        """Concatenate CT and PET, process labels"""
        ct = data['ct_image']
        pt = data['pt_image']
        labels = data['labels']

        # Concatenate along channel dimension: [2, D, H, W]
        image = torch.cat([ct, pt], dim=0)

        # Process labels
        if exclude_roi_labels:
            # Exclude ROI labels 1, 5, 8 (kidneys, bladder)
            labels = torch.where((labels == 1) | (labels == 5) | (labels == 8),
                                 torch.zeros_like(labels), labels)

        if binarize:
            labels = (labels > 0).float()

        data['image'] = image
        data['label'] = labels

        return data

    def setup(self, stage=None):
        # Training transforms
        train_transforms = Compose([
            LoadImaged(keys=["ct_image", "pt_image", "labels"], image_only=True),
            EnsureChannelFirstd(keys=["ct_image", "pt_image", "labels"]),

            # Note: Assuming data is already at [4.0, 4.0, 3.27] spacing from preprocessing
            # If not, add Spacingd here

            # Intensity normalization
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

            # Augmentations (before sampling XY center)
            RandFlipd(
                keys=["ct_image", "pt_image", "labels"],
                spatial_axis=0,
                prob=0.5
            ),
            RandFlipd(
                keys=["ct_image", "pt_image", "labels"],
                spatial_axis=1,
                prob=0.5
            ),
            RandAffined(
                keys=["ct_image", "pt_image", "labels"],
                mode=("bilinear", "bilinear", "nearest"),
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.2, 0.2, 0.2),
                padding_mode="border",
            ),

            # Concatenate modalities and process labels
            partial(self.concatenate_modalities_and_labels,
                   binarize=self.binarize,
                   exclude_roi_labels=self.exclude_roi_labels),

            # Extract 6 axial patches at lesion-aware XY center
            AxialSlidingWindowSampler(
                keys=["image", "label"],
                roi_size=self.roi_size,
                num_patches=self.num_patches
            ),
        ])

        # Validation transforms (no augmentation)
        val_transforms = Compose([
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

            partial(self.concatenate_modalities_and_labels,
                   binarize=self.binarize,
                   exclude_roi_labels=self.exclude_roi_labels),

            AxialSlidingWindowSampler(
                keys=["image", "label"],
                roi_size=self.roi_size,
                num_patches=self.num_patches
            ),
        ])

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = CacheDataset(
                data=self.dataset['training'],
                transform=train_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

            self.val_dataset = CacheDataset(
                data=self.dataset['validation'],
                transform=val_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

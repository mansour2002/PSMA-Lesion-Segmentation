# PSMA PET/CT Lesion Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Deep learning framework for automated lesion segmentation in PSMA PET/CT imaging using UNETR (UNet Transformers) architecture with Dice Loss.


## Overview

This repository provides a clean, standalone implementation for training 3D segmentation models on PSMA PET/CT scans. The framework uses:

- **Architecture**: UNETR (Vision Transformer encoder + CNN decoder) with patch size 8
- **Loss Function**: Dice Loss (1 - Dice coefficient) for handling class imbalance
- **Input**: Dual-modality CT + PET images (channel-concatenated)
- **Output**: Binary lesion segmentation masks
- **Framework**: PyTorch Lightning for reproducible training

## Key Features

- **Smart Patch Sampling**: Balanced foreground/background crop extraction (70% near lesions, 30% random)
- **Data Augmentation**: Comprehensive augmentations (flips, rotations, intensity shifts, coarse dropout)
- **Multi-GPU Support**: Distributed training with DDP (Data Distributed Parallel)
- **Efficient Caching**: MONAI CacheDataset for accelerated data loading
- **Reproducible**: Fixed random seeds and deterministic operations

## Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/mansour2002/PSMA-Lesion-Segmentation.git
cd PSMA-Lesion-Segmentation
conda create -n psma-seg python=3.8 -y && conda activate psma-seg

# 2. Install dependencies
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

# 3. Prepare your dataset (see Dataset Format section)
cp data/dataset_template.json data/dataset.json
# Edit data/dataset.json with your file paths

# 4. Start training
export PYTHONPATH=code
python code/experiments/train_segmentation.py fit --config code/configs/train_config.yaml

# 5. Monitor with TensorBoard
tensorboard --logdir outputs/PSMA_Lesion_Segmentation
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 2x A6000 or V100)
- 48GB+ GPU memory for full training (or reduce batch size)


## Dataset Format

### Directory Structure

```
data/
├── dataset.json          # Dataset split definition
└── cases/
    ├── patient_001/
    │   ├── ct.nii     # CT image
    │   ├── pet.nii    # PET image
    │   └── seg.nii    # Lesion segmentation mask
    ├── patient_002/
    └── ...
```

### JSON Format

Create `dataset.json` with training/validation splits (see `data/dataset_template.json` for a complete example):

```json
{
  "training": [
    {
      "case_id": "patient_001",
      "ct_image": "data/cases/patient_001/ct.nii",
      "pt_image": "data/cases/patient_001/pet.nii",
      "labels": "data/cases/patient_001/seg.nii"
    },
    ...
  ],
  "validation": [
    {
      "case_id": "patient_050",
      "ct_image": "data/cases/patient_050/ct.nii",
      "pt_image": "data/cases/patient_050/pet.nii",
      "labels": "data/cases/patient_050/seg.nii"
    },
    ...
  ]
}
```

### Image Requirements

- **Format**: NIfTI (.nii)
- **CT Intensity**: Hounsfield Units (typical range: -1000 to 1200 HU)
- **PET Intensity**: SUV (Standardized Uptake Value, typical range: 0 to 5000)
- **Segmentation Labels**: Binary (0 = background, 1 = lesion) or multi-class
  - If multi-class, set `binarize: True` in config to convert to binary

## Training

### Basic Training

```bash
# Option 1: Using the training script wrapper
sh train.sh

# Option 2: Direct command
export PYTHONPATH=code
python code/experiments/train_segmentation.py fit \
    --config code/configs/train_config.yaml
```

### Configuration

Edit `code/configs/train_config.yaml` to customize training:

### Multi-GPU Training

```bash
# 2 GPUs with DDP
python code/experiments/train_segmentation.py fit \
    --config code/configs/train_config.yaml \
    --trainer.devices 2 \
    --trainer.strategy ddp
```

### Resume Training

```bash
# Resume from checkpoint
python code/experiments/train_segmentation.py fit \
    --config code/configs/train_config.yaml \
    --ckpt_path outputs/checkpoints/last.ckpt
```

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/PSMA_Lesion_Segmentation

# Open browser to http://localhost:6006
```


#!/bin/bash

# Training script wrapper for PSMA lesion segmentation
# Usage: sh train.sh

# Set PYTHONPATH to code directory
export PYTHONPATH=code

# Run training
python code/experiments/train_segmentation.py fit \
    --config code/configs/train_config.yaml \
    "$@"

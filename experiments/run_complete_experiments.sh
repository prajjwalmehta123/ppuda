#!/bin/bash

# Configuration variables
SAVE_DIR="./saved_models/ta_ghn2_$(date +%Y%m%d)"
WANDB_PROJECT="task-aware-ghn2"
N_WAY=5
K_SHOT=1
N_QUERY=15
EPOCHS=100
BACKBONE="resnet34"
FEATURE_DIM=512
ADAPTATION_DIM=64

# Create directory for saved models
mkdir -p $SAVE_DIR

# Set GHN2 checkpoints - change these paths to match your environment
GHN_CIFAR100="./checkpoints/ghn2_cifar100.pt"
GHN_IMAGENET="./checkpoints/ghn2_imagenet.pt"

# =============================================
# CIFAR-10 Only Experiment
# =============================================
echo "Starting CIFAR-10 Only Experiment..."
python3 train_ta_ghn.py \
    --datasets cifar10 \
    --n-way $N_WAY \
    --k-shot $K_SHOT \
    --n-query $N_QUERY \
    --backbone $BACKBONE \
    --feature-dim $FEATURE_DIM \
    --adaptation-dim $ADAPTATION_DIM \
    --ghn-checkpoint $GHN_CIFAR100 \
    --epochs $EPOCHS \
    --save-dir "${SAVE_DIR}/cifar10" \
    --wandb-project $WANDB_PROJECT \
    --wandb-name "ta-ghn2_cifar10_$(date +%Y%m%d)"

# =============================================
# Omniglot Only Experiment
# =============================================
echo "Starting Omniglot Only Experiment..."
python3 train_ta_ghn.py \
    --datasets omniglot \
    --n-way $N_WAY \
    --k-shot $K_SHOT \
    --n-query $N_QUERY \
    --backbone $BACKBONE \
    --feature-dim $FEATURE_DIM \
    --adaptation-dim $ADAPTATION_DIM \
    --ghn-checkpoint $GHN_IMAGENET \
    --epochs $EPOCHS \
    --save-dir "${SAVE_DIR}/omniglot_only" \
    --wandb-project $WANDB_PROJECT \
    --wandb-name "ta-ghn2_omniglot_$(date +%Y%m%d)"

# =============================================
# Combined Experiment (All Datasets)
# =============================================
echo "Starting Combined Experiment (All Datasets)..."
python3 train_ta_ghn.py \
    --datasets cifar100 cifar10 omniglot \
    --n-way $N_WAY \
    --k-shot $K_SHOT \
    --n-query $N_QUERY \
    --backbone $BACKBONE \
    --feature-dim $FEATURE_DIM \
    --adaptation-dim $ADAPTATION_DIM \
    --ghn-checkpoint $GHN_CIFAR100 \
    --epochs $EPOCHS \
    --save-dir "${SAVE_DIR}/all_datasets" \
    --wandb-project $WANDB_PROJECT \
    --wandb-name "ta-ghn2_all_datasets_$(date +%Y%m%d)"

echo "All experiments completed!"
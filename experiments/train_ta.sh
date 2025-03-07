#!/bin/bash

# Create output directory
mkdir -p ./checkpoints/ta_ghn_cifar100

# Run training
python experiments/train_ta_ghn.py \
  --dataset cifar100 \
  --data_dir ./data \
  --save ./checkpoints/ta_ghn_cifar100 \
  --split train \
  --n_way 5 \
  --k_shot 1 \
  --query_size 15 \
  --meta_batch_size 4 \
  --arch_batch_size 1 \
  --task_embed_dim 128 \
  --backbone resnet18 \
  --steps_per_epoch 100 \
  --val_steps 30 \
  --epochs 50 \
  --lr 0.001 \
  --wd 0.0001 \
  --amp \
  --scheduler cosine \
  --log_interval 10 \
  --num_workers 4 \
  --grad_clip 5.0 \
  --seed 42
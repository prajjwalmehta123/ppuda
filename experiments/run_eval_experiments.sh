#!/bin/bash

python3 experiments/eval-taghn.py C100Base_CIFAR100+Omniglot --ghn-checkpoint ./checkpoints/ghn2_cifar100.pt --cuda --use-wandb

python3 experiments/eval-taghn.py C100Base_CIFAR100+SVHN --ghn-checkpoint ./checkpoints/ghn2_cifar100.pt --cuda --use-wandb

python3 experiments/eval-taghn.py C100+all_datasets --ghn-checkpoint ./checkpoints/ghn2_cifar100.pt --cuda --use-wandb

python3 experiments/eval-taghn.py C100Base_Cifar10+CIFAR100 --ghn-checkpoint ./checkpoints/ghn2_cifar100.pt --cuda --use-wandb


import os
from datetime import datetime

import torch
import random
import numpy as np
import wandb

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, SVHN

from ppuda.task.task_adaptation import TaskAdaptationModule,TaskAdaptiveEncoder
from ppuda.task.task_encoder import TaskEncoder, initialize_with_ghn
from ppuda.utils.data_utils import setup_meta_dataloaders, MetaDataset, get_transform, CombinedMetaDataset
from ppuda.utils.meta_training import train_adaptive_model, evaluate, parse_args

"""
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_encoder = TaskEncoder()
    base_encoder = initialize_with_ghn(
        base_encoder,
        ghn_checkpoint_path="/Users/prajjwalmehta/Desktop/projects/ppuda/checkpoints/ghn2_cifar100.pt",
        device=device
    )
    adaptation_module = TaskAdaptationModule(feature_dim=512, adaptation_dim=64)
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)
    meta_train_loader, meta_val_loader = setup_meta_dataloaders(
        n_way=5, k_shot=1, n_query=15)
    model = train_adaptive_model(
        model,
        meta_train_loader,
        meta_val_loader,
        learning_rate=0.001,
        epochs=50,
        device=device
    )
    model.load_state_dict(torch.load('best_adaptive_model.pth'))
"""


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_name
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ta-ghn2_{'-'.join(args.datasets)}_{timestamp}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    # Create base encoder
    base_encoder = TaskEncoder(backbone=args.backbone)
    base_encoder = initialize_with_ghn(
        base_encoder,
        ghn_checkpoint_path=args.ghn_checkpoint,
        device=args.device
    )
    for name, param in base_encoder.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter: {name}")
        if torch.isinf(param).any():
            print(f"Inf found in parameter: {name}")
        if param.abs().max() > 1e3:
            print(f"Extreme value in {name}: {param.abs().max().item()}")
    # Create adaptation module and full model
    adaptation_module = TaskAdaptationModule(
        feature_dim=args.feature_dim,
        adaptation_dim=args.adaptation_dim
    )
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)
    meta_train_loader, meta_val_loader = setup_meta_dataloaders(
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        target_datasets=args.datasets,
        n_episodes=args.n_episodes
    )

    # Train the model
    model = train_adaptive_model(
        model,
        meta_train_loader,
        meta_val_loader,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=args.device,
        wandb_logging=not args.no_wandb
    )

    # Save the best model
    model_path = os.path.join(args.save_dir, f"model_all_datasets.pth")
    torch.save(model.state_dict(), model_path)

    # Evaluate on each dataset separately
    #evaluate_cross_dataset(model, args)
    if not args.no_wandb:
        wandb.finish()


def evaluate_cross_dataset(model, args):
    device = args.device
    # Create test loaders for each dataset
    datasets = ['cifar100', 'cifar10', 'svhn', 'omniglot']
    test_loaders = {}

    for dataset_name in datasets:
        if dataset_name == 'cifar100':
            test_dataset = CIFAR100(root='./data', train=False, download=True)
        elif dataset_name == 'cifar10':
            test_dataset = CIFAR10(root='./data', train=False, download=True)
        elif dataset_name == 'svhn':
            test_dataset = SVHN(root='./data', split='test', download=True)
        elif dataset_name == 'omniglot':
            from torchvision.datasets import Omniglot
            test_dataset = Omniglot(root='./data', background=False, download=True)

        meta_dataset = MetaDataset(
            test_dataset, n_way=5, k_shot=1, n_query=15,
            n_episodes=600, transform=get_transform(dataset_name)
        )
        test_loaders[dataset_name] = DataLoader(meta_dataset, batch_size=1)

    # Evaluate on each dataset
    print("\nCross-Dataset Evaluation Results:")
    print("=================================")
    for dataset_name, loader in test_loaders.items():
        accuracy = evaluate(model, loader, device)
        print(f"{dataset_name}: {accuracy:.4f}")


# Test the full pipeline on a small subset
if __name__ == "__main__":
    args = parse_args()
    main(args)
import os
import argparse
import torch
import numpy as np
import random
import wandb
from tqdm.auto import tqdm
from datetime import datetime

from ppuda.ghn.nn import GHN
from ppuda.deepnets1m.graph import Graph, GraphBatch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your dataset utilities
from ppuda.utils.data_utils import setup_meta_dataloaders, get_transform


def parse_args():
    parser = argparse.ArgumentParser(description='GHN2 Baseline Evaluation')

    # Dataset arguments
    parser.add_argument('--experiment', type=str, choices=[
        'C100Base_CIFAR100+Omniglot',
        'C100Base_CIFAR100+SVHN',
        'C100+all_datasets',
        'C100Base_Cifar10+CIFAR100'
    ], required=True, help='Experiment combination to run')

    parser.add_argument('--n-way', type=int, default=5, help='N-way classification')
    parser.add_argument('--k-shot', type=int, default=1, help='K-shot learning')
    parser.add_argument('--n-query', type=int, default=15, help='Query examples per class')
    parser.add_argument('--n-episodes', type=int, default=600, help='Number of episodes per dataset')

    # Model arguments
    parser.add_argument('--ghn-checkpoint', type=str, required=True, help='Path to GHN2 checkpoint')

    # System arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')

    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='ghn2-baselines', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--use-wandb', action='store_true', help='Enable wandb logging')

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    return args


def setup_datasets(experiment_name):
    """Configure datasets based on experiment name"""
    if experiment_name == 'C100Base_CIFAR100+Omniglot':
        return ['cifar100', 'omniglot']
    elif experiment_name == 'C100Base_CIFAR100+SVHN':
        return ['cifar100', 'svhn']
    elif experiment_name == 'C100+all_datasets':
        return ['cifar100', 'cifar10', 'svhn', 'omniglot']
    elif experiment_name == 'C100Base_Cifar10+CIFAR100':
        return ['cifar100', 'cifar10']
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")


class GHN2Model(nn.Module):
    """Simple wrapper for architectures with GHN2-predicted parameters"""

    def __init__(self, backbone='resnet18', num_classes=5, device='cpu'):
        super().__init__()
        # Create backbone network
        if backbone == 'resnet18':
            self.network = models.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.network = models.resnet34(pretrained=False)
        elif backbone == 'resnet50':
            self.network = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify for few-shot classification
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

        self.device = device
        self.to(device)

    def forward(self, x):
        return self.network(x)


def initialize_with_ghn2(model, ghn_checkpoint, device='cpu'):
    """Initialize model parameters using GHN2"""
    # Load GHN2
    ghn = GHN.load(ghn_checkpoint, device=device)

    # Create graph representation
    graph = Graph(model)
    graphs = GraphBatch([graph])
    graphs.to_device(device)

    # Predict parameters
    with torch.no_grad():
        ghn(model, graphs)

    return model


def compute_prototypes(support_features, support_labels, n_way):
    """Compute class prototypes from support features"""
    prototypes = []
    for c in range(n_way):
        class_mask = (support_labels == c)
        if class_mask.sum() > 0:
            # Average features to get prototype
            class_features = support_features[class_mask]
            prototypes.append(class_features.mean(0))
        else:
            prototypes.append(torch.zeros_like(support_features[0]))
    return torch.stack(prototypes)


def evaluate_few_shot(model, data_loader, device, n_way):
    """Evaluate GHN2 model on few-shot tasks"""
    model.eval()
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (task_batch, dataset_indices) in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Process each task in the batch
            for i in range(len(task_batch[0])):
                # Get support and query data
                support_imgs = task_batch[0][i].to(device)
                support_labs = task_batch[1][i].to(device)
                query_imgs = task_batch[2][i].to(device)
                query_labs = task_batch[3][i].to(device)

                # Extract features
                support_features = model.network.fc(F.adaptive_avg_pool2d(
                    model.network.avgpool(
                        model.network.layer4(
                            model.network.layer3(
                                model.network.layer2(
                                    model.network.layer1(
                                        model.network.relu(
                                            model.network.bn1(
                                                model.network.conv1(support_imgs)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ), 1).flatten(1))

                query_features = model.network.fc(F.adaptive_avg_pool2d(
                    model.network.avgpool(
                        model.network.layer4(
                            model.network.layer3(
                                model.network.layer2(
                                    model.network.layer1(
                                        model.network.relu(
                                            model.network.bn1(
                                                model.network.conv1(query_imgs)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ), 1).flatten(1))

                # Compute prototypes and classify
                prototypes = compute_prototypes(support_features, support_labs, n_way)

                # Compute distances to prototypes
                dists = torch.cdist(query_features, prototypes)
                logits = -dists  # Negative distance as logits

                # Get predictions
                _, preds = torch.min(dists, dim=1)

                # Calculate accuracy
                acc = (preds == query_labs).float().mean().item()
                total_acc += acc
                count += 1

                # Log progress
                if batch_idx % 10 == 0 and i == 0:
                    print(f"Batch {batch_idx}, Task {i}, Accuracy: {acc:.4f}")

    # Return average accuracy
    return total_acc / max(1, count)


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup datasets based on experiment
    datasets = setup_datasets(args.experiment)
    print(f"Running experiment with datasets: {datasets}")

    # Initialize wandb
    if args.use_wandb:
        run_name = f"GHN2_baseline_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    # Create model and initialize with GHN2
    print("Creating model and initializing with GHN2")
    model = GHN2Model(backbone='resnet18', num_classes=args.n_way, device=args.device)
    model = initialize_with_ghn2(model, args.ghn_checkpoint, device=args.device)

    # Setup meta-dataloaders
    print("Setting up meta-dataloaders")
    meta_train_loader, meta_val_loader = setup_meta_dataloaders(
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        target_datasets=datasets,
        n_episodes=args.n_episodes
    )

    # Evaluate on validation set
    print("Evaluating on validation set")
    val_acc = evaluate_few_shot(model, meta_val_loader, args.device, args.n_way)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Log results
    if args.use_wandb:
        wandb.log({"val_acc": val_acc})

    # Save results to file
    result_file = os.path.join(args.output_dir, f"ghn2_baseline_{args.experiment}.txt")
    with open(result_file, 'w') as f:
        f.write(f"GHN2 Baseline Results for {args.experiment}\n")
        f.write(f"Datasets: {datasets}\n")
        f.write(f"N-way: {args.n_way}\n")
        f.write(f"K-shot: {args.k_shot}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")

    print(f"Results saved to {result_file}")

    # Clean up wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
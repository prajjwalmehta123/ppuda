import torch
import random
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, SVHN

from ppuda.task.task_adaptation import TaskAdaptationModule,TaskAdaptiveEncoder
from ppuda.task.task_encoder import TaskEncoder, initialize_with_ghn
from ppuda.utils.data_utils import setup_meta_dataloaders, MetaDataset, get_transform, CombinedMetaDataset
from ppuda.utils.meta_training import train_adaptive_model, evaluate


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


def evaluate_cross_dataset(model, device):
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


def evaluate_by_dataset(model, data_loaders, device="cuda"):
    """Evaluate model performance on each dataset separately"""
    model.eval()
    results = {}

    for dataset_name, loader in data_loaders.items():
        correct = 0
        total = 0

        with torch.no_grad():
            for support_imgs, support_labs, query_imgs, query_labs in loader:
                # Process episode
                support_imgs = support_imgs.squeeze(0).to(device)
                support_labs = support_labs.squeeze(0).to(device)
                query_imgs = query_imgs.squeeze(0).to(device)
                query_labs = query_labs.squeeze(0).to(device)

                # Forward pass
                logits = model(support_imgs, support_labs, query_imgs,
                               n_way=support_labs.max().item() + 1)

                # Calculate accuracy
                pred = logits.argmax(dim=1)
                correct += (pred == query_labs).sum().item()
                total += query_labs.size(0)

        results[dataset_name] = correct / total

    return results

# Test the full pipeline on a small subset
if __name__ == "__main__":
    test_phases = [
        ['cifar100'],
        ['cifar100', 'cifar10'],  # similar datasets
        ['cifar100', 'cifar10', 'svhn']  # add moderate difference
    ]
    for phase, datasets in enumerate(test_phases):
        print(f"\n=== Phase {phase + 1}: Testing with {datasets} ===\n")
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        epochs = 5
        train_loader,val_loader = setup_meta_dataloaders(target_datasets=datasets,n_episodes=20)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_encoder = TaskEncoder()
        base_encoder = initialize_with_ghn(
            base_encoder,
            ghn_checkpoint_path="./checkpoints/ghn2_cifar100.pt",
            device=device
        )
        adaptation_module = TaskAdaptationModule(feature_dim=512, adaptation_dim=64)
        model = TaskAdaptiveEncoder(base_encoder, adaptation_module)
        model = train_adaptive_model(model, train_loader, val_loader,epochs=epochs, device=device)
        eval_loaders = {}
        for dataset_name in datasets:
            # Create single-dataset loader for evaluation
            if dataset_name == 'cifar100':
                test_dataset = CIFAR100(root='./data', train=False, download=True)
            elif dataset_name == 'cifar10':
                test_dataset = CIFAR10(root='./data', train=False, download=True)
            elif dataset_name == 'svhn':
                test_dataset = SVHN(root='./data', split='train', download=True)

            meta_dataset = MetaDataset(
                test_dataset, n_way=5, k_shot=1, n_query=15,
                n_episodes=50, transform=get_transform(dataset_name)
            )
            eval_loaders[dataset_name] = DataLoader(meta_dataset, batch_size=1)
            results = evaluate_by_dataset(model, eval_loaders, device)
            print("\nResults after training:")
            for dataset_name, acc in results.items():
                print(f"  {dataset_name}: {acc:.4f}")
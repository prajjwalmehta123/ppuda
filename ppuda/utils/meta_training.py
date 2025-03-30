import argparse

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Task Adaptive Meta-Learning with GHN2')

    # Dataset arguments
    parser.add_argument('--datasets', nargs='+', default=['cifar100', 'cifar10', 'svhn'],
                        help='List of datasets to use (cifar100 is always included)')
    parser.add_argument('--n-way', type=int, default=5,
                        help='N-way classification (default: 5)')
    parser.add_argument('--k-shot', type=int, default=1,
                        help='K-shot learning (default: 1)')
    parser.add_argument('--n-query', type=int, default=15,
                        help='Number of query examples per class (default: 15)')
    parser.add_argument('--n-episodes', type=int, default=600,
                        help='Number of episodes per dataset (default: 600)')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture (default: resnet34)')
    parser.add_argument('--feature-dim', type=int, default=512,
                        help='Feature dimension (default: 512)')
    parser.add_argument('--adaptation-dim', type=int, default=64,
                        help='Task adaptation dimension (default: 64)')
    parser.add_argument('--ghn-checkpoint', type=str,
                        default='./checkpoints/ghn2_cifar100.pt',
                        help='Path to GHN2 checkpoint')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Meta-batch size (default: 4)')
    parser.add_argument('--eval-episodes', type=int, default=50,
                        help='Number of episodes for evaluation (default: 50)')

    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--save-dir', type=str, default='./saved_models',
                        help='Directory to save models (default: ./saved_models)')

    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='task-adaptive-ghn2',
                        help='Wandb project name (default: task-adaptive-ghn2)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity name (default: None)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')

    # Experiment mode
    parser.add_argument('--mode', type=str, default='phased',
                        choices=['phased', 'single'],
                        help='Training mode: phased (incremental datasets) or single (all at once)')

    args = parser.parse_args()

    # Set CUDA availability
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Always include cifar100 in datasets
    if 'cifar100' not in args.datasets:
        args.datasets = ['cifar100'] + args.datasets

    return args


import torch
import torch.nn.functional as F
import wandb


def train_adaptive_model(model, meta_train_loader, meta_val_loader,
                         learning_rate=0.001, epochs=50, device="cuda",
                         wandb_logging=False):
    # Move model to device
    model = model.to(device)

    # Freeze base encoder, train only adaptation layers
    for param in model.base_encoder.parameters():
        param.requires_grad = False

    # Set adaptation layers to training mode
    for param in model.adaptation_module.parameters():
        param.requires_grad = True

    # Optimization setup
    optimizer = torch.optim.Adam(model.adaptation_module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        tasks_processed = 0

        for task_batch in meta_train_loader:
            meta_loss = 0
            correct = 0
            total = 0

            support_imgs, support_labs, query_imgs, query_labs = task_batch
            support_imgs = support_imgs.to(device)
            support_labs = support_labs.to(device)
            query_imgs = query_imgs.to(device)
            query_labs = query_labs.to(device)

            # Forward pass
            logits = model(support_imgs, support_labs, query_imgs, n_way=support_labs.max().item() + 1)

            # Calculate loss
            loss = F.cross_entropy(logits, query_labs)
            meta_loss += loss

            # Track accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == query_labs).sum().item()
            total += query_labs.size(0)

            # Update weights
            optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # Gradient clipping
            optimizer.step()

            # Track metrics
            train_loss += meta_loss.item()
            train_acc += correct / total
            tasks_processed += 1

        # Calculate average training metrics
        avg_train_loss = train_loss / tasks_processed
        avg_train_acc = train_acc / tasks_processed

        # Validation
        model.eval()
        val_acc = evaluate(model, meta_val_loader, device)
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_adaptive_model.pth')

        # Log metrics
        if wandb_logging:
            wandb.log({
                "train/loss": avg_train_loss,
                "train/acc": avg_train_acc,
                "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch
            })

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, "
              f"Train Acc {avg_train_acc:.4f}, Val Acc {val_acc:.4f}")

    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

    # Load best model for return
    model.load_state_dict(torch.load('best_adaptive_model.pth'))

    return model


def evaluate(model, data_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for support_imgs, support_labs, query_imgs, query_labs in data_loader:
            # Move data to device
            support_imgs = support_imgs.to(device)
            support_labs = support_labs.to(device)
            query_imgs = query_imgs.to(device)
            query_labs = query_labs.to(device)

            # Forward pass
            logits = model(support_imgs, support_labs, query_imgs, n_way=support_labs.max().item() + 1)

            # Calculate accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == query_labs).sum().item()
            total += query_labs.size(0)

    return correct / total


if __name__ == '__main__':
    from ppuda.task.task_adaptation import TaskAdaptiveEncoder, TaskAdaptationModule
    from torch.utils.data import DataLoader
    from ppuda.task.task_encoder import TaskEncoder, initialize_with_ghn

    # Create and test the full model
    base_encoder = TaskEncoder()
    base_encoder = initialize_with_ghn(
        base_encoder,
        ghn_checkpoint_path="/Users/prajjwalmehta/Desktop/projects/ppuda/checkpoints/ghn2_cifar100.pt",
        device='cpu',
    )

    adaptation_module = TaskAdaptationModule(feature_dim=512, adaptation_dim=64)
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)

    # Before doing full training, test the training procedure on a small batch
    from torch.utils.data import DataLoader


    # Create simple mock data loaders
    class MockMetaDataset:
        def __init__(self, n_episodes=100):
            self.n_episodes = n_episodes

        def __len__(self):
            return self.n_episodes

        def __getitem__(self, idx):
            n_way, k_shot = 5, 1
            support_images = torch.randn(n_way * k_shot, 3, 84, 84)
            support_labels = torch.arange(n_way).repeat_interleave(k_shot)
            query_images = torch.randn(n_way * 5, 3, 84, 84)
            query_labels = torch.arange(n_way).repeat_interleave(5)
            return support_images, support_labels, query_images, query_labels


    meta_train_dataset = MockMetaDataset(n_episodes=10)
    meta_val_dataset = MockMetaDataset(n_episodes=5)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=1)
    meta_val_loader = DataLoader(meta_val_dataset, batch_size=1)

    # Test training for 2 epochs
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)
    train_adaptive_model(model, meta_train_loader, meta_val_loader, epochs=2,device='cpu')
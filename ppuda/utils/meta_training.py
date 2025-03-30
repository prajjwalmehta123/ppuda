import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm.auto import tqdm


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
    parser.add_argument('--n-episodes', type=int, default=500,
                        help='Number of episodes per dataset (default: 500)')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture (default: resnet34)')
    parser.add_argument('--feature-dim', type=int, default=512,
                        help='Feature dimension (default: 512)')
    parser.add_argument('--adaptation-dim', type=int, default=64,
                        help='Task adaptation dimension (default: 64)')
    parser.add_argument('--ghn-checkpoint', type=str,
                        default='./checkpoints/ghn2_imagenet.pt',
                        help='Path to GHN2 checkpoint')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Meta-batch size (default: 4)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of episodes for evaluation (default: 100)')

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

    args = parser.parse_args()

    # Set CUDA availability
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Always include cifar100 in datasets
    if 'cifar100' not in args.datasets:
        args.datasets = ['cifar100'] + args.datasets

    return args

def lr_lambda(epoch):
    if epoch < 10:
        return epoch / 10
    else:
        return 0.5 * (1 + math.cos(math.pi * (epoch - 10) / (epoch - 10)))

def train_adaptive_model(model, meta_train_loader, meta_val_loader,
                         learning_rate=0.001, epochs=50, device="cuda",
                         wandb_logging=False):
    # Move model to device
    model = model.to(device)

    # Freeze base encoder, train only adaptation layers
    for param in model.base_encoder.parameters():
        param.requires_grad = False

    # Optimization setup
    optimizer = torch.optim.Adam(
        model.adaptation_module.parameters(),
        lr=learning_rate,  # Lower learning rate
        weight_decay=1e-4  # Add weight decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_acc = 0
    best_epoch = 0
    eval_frequency = 5
    patience = 15
    patience_counter = 0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        train_acc = 0
        tasks_processed = 0
        train_pbar = tqdm(meta_train_loader, desc="Training", leave=False)

        for task_batch, dataset_indices in train_pbar:
            batch_size = len(dataset_indices)
            all_logits = []
            all_query_labs = []
            for i in range(batch_size):
                support_imgs = task_batch[0][i].to(device)
                support_labs = task_batch[1][i].to(device)
                query_imgs = task_batch[2][i].to(device)
                query_labs = task_batch[3][i].to(device)

                logits = model(support_imgs, support_labs, query_imgs,
                               n_way=support_labs.max().item() + 1)

                all_logits.append(logits)
                all_query_labs.append(query_labs)

            optimizer.zero_grad()
            meta_loss = 0
            correct = 0
            total = 0

            for logits, query_labs in zip(all_logits, all_query_labs):
                # Calculate loss
                loss = F.cross_entropy(logits, query_labs)
                meta_loss += loss / batch_size  # Average across tasks

                # Track accuracy
                pred = logits.argmax(dim=1)
                correct += (pred == query_labs).sum().item()
                total += query_labs.size(0)

            for name, param in model.adaptation_module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"NaN or Inf gradient detected in {name}!")
                        param.grad.zero_()

            # Backprop and update
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # Track metrics
            train_loss += meta_loss.item()
            train_acc += correct / total
            tasks_processed += 1
            train_pbar.set_postfix_str(
                f"loss: {meta_loss.item():.4f} | acc: {correct / total:.2%} | lr: {scheduler.get_last_lr()[0]:.6f}"
            )
        avg_train_loss = train_loss / tasks_processed
        avg_train_acc = train_acc / tasks_processed
            # Validation
        model.eval()
        val_acc = evaluate(model, meta_val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss={avg_train_loss:.4f}, "
                  f"Train Acc={avg_train_acc:.4f}, "
                  f"Val Acc={val_acc:.4f}")

            # Save best model
            # Log metrics
        if wandb_logging:
            wandb.log({
                "train/loss": avg_train_loss,
                "train/acc": avg_train_acc,
                "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch
            })

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), './experiments/best_taskaware_model.pth')
        else:
            patience_counter +=1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

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
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            # Check if the batch is from CombinedMetaDataset or regular MetaDataset
            if isinstance(batch, list) and len(batch) == 2:
                task_batch, _ = batch  # Unpack and ignore dataset indices

            else:
                support_imgs, support_labs, query_imgs, query_labs = batch

            # Move data to device
            for i in range(len(task_batch[0])):
                support_imgs = task_batch[0][i].to(device)
                support_labs = task_batch[1][i].to(device)
                query_imgs = task_batch[2][i].to(device)
                query_labs = task_batch[3][i].to(device)

                # Forward pass
                logits = model(support_imgs, support_labs, query_imgs,
                               n_way=support_labs.max().item() + 1)

                # Calculate accuracy
                pred = logits.argmax(dim=1)
                correct += (pred == query_labs).sum().item()
                total += query_labs.size(0)
            progress_bar.set_postfix({'acc': f"{correct / total:.4f}"})

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
"""
Train a Hybrid GHN-based Meta-Learning model for few-shot learning.
"""
import argparse
import os
from datetime import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.models as models
import wandb

from ppuda.deepnets1m.architecture import robust_network_adaptation
from ppuda.ghn.task_aware_ghn import TaskAwareGHN
from ppuda.task.task_sampler import TaskSampler, ClassSubset

def get_dataset(dataset_name, data_dir, is_train=True):
    """Get dataset for few-shot learning."""
    if dataset_name == 'cifar100':
        # CIFAR-100 dataset
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )

        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        dataset = datasets.CIFAR100(
            root=data_dir,
            train=is_train,
            download=True,
            transform=transform
        )

    elif dataset_name == 'miniimagenet':
        # Add MiniImageNet support here if needed
        raise NotImplementedError("MiniImageNet dataset not implemented yet")

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset

def collate_task_batch(tasks, dataset):
    """Collate a batch of tasks into properly structured tensors."""
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for task_idx, (support_idx, query_idx, classes) in enumerate(tasks):
        # Create mapping from original class to task-specific class index
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # Process support set
        for idx in support_idx:
            img, label = dataset[idx]
            support_images.append(img)
            # Map original label to task-specific class index
            support_labels.append(class_to_idx[label])

        # Process query set
        for idx in query_idx:
            img, label = dataset[idx]
            query_images.append(img)
            # Map original label to task-specific class index
            query_labels.append(class_to_idx[label])

    # Stack tensors
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels

def compute_accuracy(logits, targets):
    """Compute classification accuracy."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float().sum()
    return correct.item() / targets.size(0)


def compute_meta_loss(model, task_support_images, task_support_labels,
                      task_query_images, task_query_labels, arch_embedding):
    """Compute loss with auxiliary components."""
    # Get task embedding
    task_embedding = model.encode_task(task_support_images, task_support_labels)

    # Main classification loss
    query_logits = model(task_query_images, arch_embedding, task_embedding)
    classification_loss = F.cross_entropy(query_logits, task_query_labels)

    # Add auxiliary classification loss on support set
    support_logits = model(task_support_images, arch_embedding, task_embedding)
    support_loss = F.cross_entropy(support_logits, task_support_labels)

    # Total loss (weighted combination)
    total_loss = classification_loss + 0.3 * support_loss

    return total_loss, classification_loss.item(), compute_accuracy(query_logits, task_query_labels)

def meta_training(model, train_dataset, train_sampler, networks, optimizer, device, args):
    """
    Train with a variety of architectures to improve generalization.

    Args:
        model: TaskAwareGHN model
        train_dataset: Training dataset
        train_sampler: Task sampler
        networks: List of different network architectures
        optimizer: Optimizer
        device: Device to use
        args: Training arguments

    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.train()
    train_losses = []
    train_accs = []

    train_iter = tqdm(range(args.steps_per_epoch), desc=f"Training")
    for step in train_iter:
        # Sample tasks
        tasks = train_sampler.sample_batch(args.meta_batch_size)
        support_images, support_labels, query_images, query_labels = collate_task_batch(tasks, train_dataset)

        # Move data to device
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        # Forward pass and compute loss
        optimizer.zero_grad()

        # Batch losses and accs
        batch_losses = []
        batch_accs = []

        for i in range(args.meta_batch_size):
            # Get task-specific data
            start_idx = i * args.n_way * args.k_shot
            end_idx = (i + 1) * args.n_way * args.k_shot
            task_support_images = support_images[start_idx:end_idx]
            task_support_labels = support_labels[start_idx:end_idx]

            start_idx = i * args.n_way * args.query_size
            end_idx = (i + 1) * args.n_way * args.query_size
            task_query_images = query_images[start_idx:end_idx]
            task_query_labels = query_labels[start_idx:end_idx]

            # Sample a random network architecture
            network_idx = np.random.randint(len(networks))
            network = networks[network_idx].to(device)

            # Use the simple architecture encoder for stability
            arch_embedding = model.encode_architecture_simple(network)

            # Compute enhanced loss with auxiliary components
            loss, _, acc = compute_meta_loss(
                model, task_support_images, task_support_labels,
                task_query_images, task_query_labels, arch_embedding
            )

            batch_losses.append(loss)
            batch_accs.append(acc)

        # Average loss across tasks
        loss = torch.mean(torch.stack(batch_losses))

        # Apply gradient stabilization
        success = train_with_gradient_stabilization(model, optimizer, loss, args)

        if success:
            train_losses.append(loss.item())
            train_accs.append(np.mean(batch_accs))

            # Update tqdm progress bar
            train_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{np.mean(batch_accs):.4f}"
            })

    return np.mean(train_losses), np.mean(train_accs)

def evaluate_arch_aware(model, val_dataset, val_sampler, device, args):
    """
    Evaluate architecture-aware model.

    Args:
        model: TaskAwareGHN model
        val_dataset: Validation dataset
        val_sampler: Task sampler
        device: Device to use
        args: Evaluation arguments

    Returns:
        Tuple of (average loss, average accuracy)
    """
    model.eval()
    val_losses = []
    val_accs = []

    with torch.no_grad():
        val_iter = tqdm(range(args.val_steps), desc=f"Validation")
        for step in val_iter:
            # Sample tasks
            tasks = val_sampler.sample_batch(args.meta_batch_size)
            support_images, support_labels, query_images, query_labels = collate_task_batch(tasks, val_dataset)

            # Move data to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            batch_losses = []
            batch_accs = []

            for i in range(args.meta_batch_size):
                # Get task-specific data
                start_idx = i * args.n_way * args.k_shot
                end_idx = (i + 1) * args.n_way * args.k_shot
                task_support_images = support_images[start_idx:end_idx]
                task_support_labels = support_labels[start_idx:end_idx]

                start_idx = i * args.n_way * args.query_size
                end_idx = (i + 1) * args.n_way * args.query_size
                task_query_images = query_images[start_idx:end_idx]
                task_query_labels = query_labels[start_idx:end_idx]

                # Encode architecture
                arch_embedding = model.encode_architecture_simple(model.backbone)

                # Compute loss and accuracy
                _, loss, acc = compute_meta_loss(
                    model, task_support_images, task_support_labels,
                    task_query_images, task_query_labels, arch_embedding
                )

                batch_losses.append(loss)
                batch_accs.append(acc)

            val_loss_step = np.mean(batch_losses)
            val_acc_step = np.mean(batch_accs)
            val_losses.append(val_loss_step)
            val_accs.append(val_acc_step)

            # Update tqdm progress bar
            val_iter.set_postfix({
                'loss': f"{val_loss_step:.4f}",
                'acc': f"{val_acc_step:.4f}"
            })

    return np.mean(val_losses), np.mean(val_accs)

def create_network_family():
    """Create a diverse family of networks adapted for CIFAR-100."""
    networks = []

    # ResNet variants with different depths
    resnet18 = robust_network_adaptation(models.resnet18(pretrained=True))
    networks.append(resnet18)

    resnet34 = robust_network_adaptation(models.resnet34(pretrained=True))
    networks.append(resnet34)

    # Create a mini resnet with fewer parameters
    mini_resnet = robust_network_adaptation(models.resnet18(pretrained=True))
    mini_resnet.layer4 = nn.Identity()  # Remove last layer
    networks.append(mini_resnet)

    # DenseNet variant
    densenet = robust_network_adaptation(models.densenet121(pretrained=True))
    networks.append(densenet)

    # MobileNet variant
    mobilenet = robust_network_adaptation(models.mobilenet_v2(pretrained=True))
    networks.append(mobilenet)

    # Add a shallower version of mobilenet
    shallow_mobilenet = robust_network_adaptation(models.mobilenet_v2(pretrained=True))
    shallow_mobilenet.features = nn.Sequential(*list(shallow_mobilenet.features)[:10])
    networks.append(shallow_mobilenet)

    return networks

def evaluate_cross_domain(model, test_dataset, test_sampler, networks, device, args):
    """
    Evaluate cross-domain generalization.
    """
    model.eval()
    results = {}

    for net_idx, network in enumerate(networks):
        net_name = type(network).__name__
        test_losses = []
        test_accs = []

        with torch.no_grad():
            test_iter = tqdm(range(args.test_steps), desc=f"Testing with {net_name}")
            for step in test_iter:
                # Sample tasks
                tasks = test_sampler.sample_batch(args.meta_batch_size)
                support_images, support_labels, query_images, query_labels = collate_task_batch(tasks, test_dataset)

                # Move data to device
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                batch_losses = []
                batch_accs = []

                for i in range(args.meta_batch_size):
                    # Get task-specific data
                    start_idx = i * args.n_way * args.k_shot
                    end_idx = (i + 1) * args.n_way * args.k_shot
                    task_support_images = support_images[start_idx:end_idx]
                    task_support_labels = support_labels[start_idx:end_idx]

                    start_idx = i * args.n_way * args.query_size
                    end_idx = (i + 1) * args.n_way * args.query_size
                    task_query_images = query_images[start_idx:end_idx]
                    task_query_labels = query_labels[start_idx:end_idx]

                    # Use the simple architecture encoder for stability
                    network = network.to(device)
                    arch_embedding = model.encode_architecture_simple(network)

                    # Set the current architecture as the backbone temporarily
                    original_backbone = model.backbone
                    model.backbone = network

                    # Encode task from support set
                    task_embedding = model.encode_task(task_support_images, task_support_labels)

                    # Restore original backbone
                    model.backbone = original_backbone

                    # Get predictions on query set
                    query_logits = model(task_query_images, arch_embedding, task_embedding)
                    loss = F.cross_entropy(query_logits, task_query_labels)

                    # Calculate accuracy
                    acc = compute_accuracy(query_logits, task_query_labels)

                    batch_losses.append(loss.item())
                    batch_accs.append(acc)

                test_loss_step = np.mean(batch_losses)
                test_acc_step = np.mean(batch_accs)
                test_losses.append(test_loss_step)
                test_accs.append(test_acc_step)

                # Update tqdm progress bar
                test_iter.set_postfix({
                    'loss': f"{test_loss_step:.4f}",
                    'acc': f"{test_acc_step:.4f}"
                })
        results[net_name] = {
            'loss': np.mean(test_losses),
            'accuracy': np.mean(test_accs)
        }
        print(
            f"Results with {net_name}: Loss = {results[net_name]['loss']:.4f}, Accuracy = {results[net_name]['accuracy']:.4f}")
    return results


def train_with_gradient_stabilization(model, optimizer, loss, args):
    """Apply gradient stabilization techniques during backpropagation."""
    # Calculate gradient norm before clipping for monitoring
    optimizer.zero_grad()
    loss.backward()

    # Check for NaN or Inf gradients
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                valid_gradients = False
                print(f"Warning: NaN or Inf gradients in {name}")
                break

    if valid_gradients:
        # Calculate gradient norm by parameter group for monitoring
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_key = name.split('.')[0]  # Group by top-level module
                if param_key not in grad_norms:
                    grad_norms[param_key] = 0
                grad_norms[param_key] += param.grad.norm().item()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Apply weight decay outside of optimizer
        for name, param in model.named_parameters():
            if param.grad is not None and 'bias' not in name and 'layer_norm' not in name:
                param.grad.add_(param * args.wd)

        # Step optimizer
        optimizer.step()

        return True, grad_norms

    # Skip this batch if gradients are invalid
    print("Skipping batch due to invalid gradients")
    optimizer.zero_grad()
    return False, {}


def adjust_network_for_cifar(network):
    """Adapt ImageNet pretrained networks for CIFAR-100 images (32x32)"""
    # Replace first convolutional layer with smaller kernel for ResNets
    if hasattr(network, 'conv1'):
        in_channels = network.conv1.in_channels
        out_channels = network.conv1.out_channels
        network.conv1 = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False)

    # Remove maxpool which reduces spatial dimensions too much for CIFAR images
    if hasattr(network, 'maxpool'):
        network.maxpool = nn.Identity()

    # For DenseNet
    if hasattr(network, 'features') and isinstance(network.features[0], nn.Conv2d):
        # Replace first conv
        in_channels = network.features[0].in_channels
        out_channels = network.features[0].out_channels
        network.features[0] = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=False)
        # Remove pooling if it exists
        for i, module in enumerate(network.features):
            if isinstance(module, nn.MaxPool2d):
                network.features[i] = nn.Identity()
                break

    # For MobileNetV2
    if hasattr(network, 'features'):
        # MobileNetV2 has a different structure - need to handle it separately
        # Find the first Conv2d layer in the features
        for i, module in enumerate(network.features):
            if hasattr(module, 'conv') and hasattr(module.conv, 'stride'):
                # For InvertedResidual blocks
                module.conv.stride = (1, 1)
                break
            elif isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                # Direct Conv2d
                network.features[i].stride = (1, 1)
                break

    return network


def setup_progressive_training(model, epoch, total_epochs):
    """Configure model for progressive training based on current epoch."""
    phase = 1
    if epoch < total_epochs // 3:
        # Phase 1: Train only task encoder and classifier
        phase = 1
        for name, param in model.named_parameters():  # Changed from model.parameters()
            param.requires_grad = 'task_encoder' in name or 'classifier' in name
    elif epoch < 2 * total_epochs // 3:
        # Phase 2: Add parameter generator
        phase = 2
        for name, param in model.named_parameters():  # Changed from model.parameters()
            param.requires_grad = True
    else:
        # Phase 3: Train everything
        phase = 3
        for name, param in model.named_parameters():  # Changed from model.parameters()
            param.requires_grad = True
    return phase


def initialize_monitoring(args):
    """Initialize monitoring tools for tracking training progress."""
    monitoring = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'grad_norms': [],
        'lr_history': [],
        'phase_history': [],
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'current_epoch': 0
    }

    # Create a log directory for this run
    log_dir = os.path.join(args.save, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create a log file
    log_file = os.path.join(log_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Args: {args}\n\n")

    monitoring['log_file'] = log_file

    return monitoring

def log_training_stats(monitoring, epoch, train_loss, train_acc, val_loss, val_acc, phase, lr):
    """Log training statistics to file and update monitoring."""
    monitoring['current_epoch'] = epoch
    monitoring['train_losses'].append(train_loss)
    monitoring['train_accs'].append(train_acc)
    monitoring['val_losses'].append(val_loss)
    monitoring['val_accs'].append(val_acc)
    monitoring['phase_history'].append(phase)
    monitoring['lr_history'].append(lr)

    # Update best model info
    if val_acc > monitoring['best_val_acc']:
        monitoring['best_val_acc'] = val_acc
        monitoring['best_epoch'] = epoch

    # Log to file
    with open(monitoring['log_file'], 'a') as f:
        f.write(f"\nEpoch {epoch} (Phase {phase}) - LR: {lr:.6f}\n")
        f.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
        f.write(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        if val_acc == monitoring['best_val_acc']:
            f.write(f"New best model with validation accuracy {val_acc:.4f}\n")

    return monitoring


def main():
    """Main training function for architecture-aware GHN."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Architecture-Aware GHN Meta-Learning Model')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset name (cifar100, miniimagenet)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Dataset for cross-domain evaluation (cub, omniglot)')

    # Model arguments
    parser.add_argument('--arch_embed_dim', type=int, default=128,
                        help='Architecture embedding dimension')
    parser.add_argument('--task_embed_dim', type=int, default=128,
                        help='Task embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--ve_cutoff', type=int, default=50,
                        help='Maximum shortest path length for virtual edges')

    # Task arguments
    parser.add_argument('--n_way', type=int, default=5,
                        help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='K-shot learning')
    parser.add_argument('--query_size', type=int, default=15,
                        help='Number of query examples per class')

    # Training arguments
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--meta_batch_size', type=int, default=4,
                        help='Number of tasks per batch')
    parser.add_argument('--arch_batch_size', type=int, default=2,
                        help='Number of architectures to use per step')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='Scheduler type (cosine, multistep)')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Steps per epoch')
    parser.add_argument('--val_steps', type=int, default=30,
                        help='Validation steps')
    parser.add_argument('--test_steps', type=int, default=50,
                        help='Testing steps')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Logging and saving arguments
    parser.add_argument('--save', type=str, default='./checkpoints/arch_aware_ghn',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='arch-aware-ghn',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create save directory if it doesn't exist
    os.makedirs(args.save, exist_ok=True)

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"arch_aware_ghn_{args.n_way}way_{args.k_shot}shot"
        )

    # Set device
    device = torch.device(args.device)

    # Load dataset
    dataset = get_dataset(args.dataset, args.data_dir, is_train=True)

    # Split dataset for few-shot learning
    if args.dataset == 'cifar100':
        train_classes = list(range(80))
        val_classes = list(range(80, 90))
        test_classes = list(range(90, 100))

        train_dataset = ClassSubset(dataset, train_classes)
        val_dataset = ClassSubset(dataset, val_classes)
        test_dataset = ClassSubset(dataset, test_classes)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Create task samplers
    train_sampler = TaskSampler(
        train_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_size=args.query_size,
        seed=args.seed
    )

    val_sampler = TaskSampler(
        val_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_size=args.query_size,
        seed=args.seed + 1  # Different seed for validation
    )

    # Create network family for architecture variety
    networks = create_network_family()

    # Create model
    model = TaskAwareGHN(
        arch_embed_dim=args.arch_embed_dim,
        task_embed_dim=args.task_embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_way,
        device=device,
        ve_cutoff=args.ve_cutoff
    ).to(device)

    model.backbone = robust_network_adaptation(model.backbone)

    # Pre-cache architecture embeddings for all networks
    for network in [model.backbone] + networks:
        with torch.no_grad():
            _ = model.encode_architecture_simple(network)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.999)
    )
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * args.steps_per_epoch,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.epochs // 5, args.epochs * 2 // 5, args.epochs * 3 // 5, args.epochs * 4 // 5],
            gamma=0.5
        )
    # Print training info
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training Architecture-Aware GHN with {num_trainable_params} trainable parameters!")
    print(f"Training on {args.n_way}-way {args.k_shot}-shot tasks")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        phase = setup_progressive_training(model, epoch, args.epochs)
        print(f"\nEpoch {epoch + 1}/{args.epochs} - Phase {phase} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Training with architecture variety
        train_loss, train_acc = meta_training(model, train_dataset, train_sampler, networks, optimizer, device, args)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate_arch_aware(
            model, val_dataset, val_sampler, device, args
        )
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "train/epoch_accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0]
            })

        # Save checkpoint
        checkpoint_path = os.path.join(args.save, f"model_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save, "model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_path)
            print(f"Saved best model with validation accuracy {best_val_acc:.4f}")

            if args.use_wandb:
                wandb.run.summary["best_val_accuracy"] = best_val_acc
                wandb.run.summary["best_epoch"] = epoch + 1


    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    """
    if args.test_dataset:
        print(f"\nEvaluating cross-domain generalization on {args.test_dataset}...")
        test_dataset = get_dataset(args.test_dataset, args.data_dir, is_train=False)

        test_sampler = TaskSampler(
            test_dataset,
            n_way=args.n_way,
            k_shot=args.k_shot,
            query_size=args.query_size,
            seed=args.seed + 2
        )
        cross_domain_results = evaluate_cross_domain(
            model, test_dataset, test_sampler, networks, device, args
        )
        if args.use_wandb:
            for net_name, result in cross_domain_results.items():
                wandb.run.summary[f"cross_domain_{args.test_dataset}_{net_name}_accuracy"] = result['accuracy']
        """
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

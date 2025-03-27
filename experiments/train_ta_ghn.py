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
from torch.cuda.amp import autocast, GradScaler
import wandb

from ppuda.deepnets1m.architecture import robust_network_adaptation
from ppuda.ghn.task_aware_ghn import TaskAwareGHN
from ppuda.task.task_sampler import TaskSampler, ClassSubset
from ppuda.ghn.nn import GHN
from ppuda.utils.network_utils import initialize_from_ghn2

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

def augment_support_set(support_images, support_labels, n_way, k_shot):

    augment_transforms = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ]
    transform = transforms.Compose(augment_transforms)

    # Create augmented versions
    augmented_support = []
    augmented_labels = []

    # Process each example in the support set
    for i in range(len(support_images)):
        # Keep original
        augmented_support.append(support_images[i])
        augmented_labels.append(support_labels[i])

        # Add 5 augmented versions
        img = support_images[i].cpu()
        for _ in range(5):
            aug_img = transform(img)
            augmented_support.append(aug_img.to(support_images.device))
            augmented_labels.append(support_labels[i])

    return torch.stack(augmented_support), torch.tensor(augmented_labels,
                                                        device=support_labels.device)

def meta_training(model, train_dataset, train_sampler, networks, optimizer, scheduler, device, args):
    model.train()
    train_losses = []
    train_accs = []

    # Create gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available() and hasattr(args, 'mixed_precision') and args.mixed_precision)

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
            task_support_images = get_task_slice(support_images, i, args.n_way, args.k_shot)
            task_support_labels = get_task_slice(support_labels, i, args.n_way, args.k_shot)
            task_query_images = get_task_slice(query_images, i, args.n_way, args.query_size)
            task_query_labels = get_task_slice(query_labels, i, args.n_way, args.query_size)

            if args.k_shot == 1:
                task_support_images, task_support_labels = augment_support_set(
                    task_support_images, task_support_labels, args.n_way, args.k_shot
                )

            # Sample a random network architecture
            network_idx = np.random.randint(len(networks))
            network = networks[network_idx].to(device)

            # Use GHN2-based architecture encoder if available
            if hasattr(model, 'encode_architecture_ghn2'):
                arch_embedding = model.encode_architecture_ghn2(network)
                #print('Used GHN2 Embedding',arch_embedding)
            else:
                arch_embedding = model.encode_architecture_simple(network)

            # Save original adaptation parameters for later restoration
            adaptation_params = {}
            for name, param in model.named_parameters():
                if 'adapt_layer' in name or 'classifier' in name:
                    adaptation_params[name] = param.data.clone()

            # Inner loop adaptation - simulate fine-tuning
            adaptation_steps = 8 if args.k_shot == 1 else 3

            # Get task embedding with mixed precision
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                task_embedding = model.encode_task(task_support_images, task_support_labels)

            temperature = 1.0
            if args.k_shot == 1:
                temperature = 0.8  # Lower temperature for more careful adaptation

            inner_lr_initial = 0.005  # Lower starting LR
            inner_lr_factor = 0.85  # Decay factor

            # Inner loop optimization with mixed precision
            for ad_step in range(adaptation_steps):
                inner_lr = inner_lr_initial * (inner_lr_factor ** ad_step)

                # Forward pass on support set
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    support_logits = model(task_support_images, arch_embedding, task_embedding, temperature)
                    inner_loss = F.cross_entropy(support_logits, task_support_labels)

                # Backward pass
                if scaler.is_enabled():
                    scaler.scale(inner_loss).backward(retain_graph=True)
                    scaler.unscale_(optimizer)  # Unscale gradients for clipping
                else:
                    inner_loss.backward(retain_graph=True)

                # Manually update adaptation layers with gradient clipping
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if ('adapt_layer' in name or 'classifier' in name) and param.grad is not None:
                            # Apply parameter normalization based on type
                            if 'weight' in name and 'conv' in name:
                                param_type = 'conv_weight'
                            elif 'weight' in name and ('bn' in name or 'layer_norm' in name):
                                param_type = 'bn_weight'
                            elif 'bias' in name:
                                param_type = 'bias'
                            elif 'weight' in name:
                                param_type = 'fc_weight'
                            else:
                                param_type = 'other'

                            # Clip gradient
                            grad_norm = param.grad.norm()
                            if grad_norm > 2.0:
                                param.grad = param.grad * (2.0 / grad_norm)

                            # Update parameter
                            param.data = param.data - inner_lr * param.grad

                            # Apply normalization
                            if hasattr(model, 'normalize_parameters'):
                                param.data = model.normalize_parameters(param.data, param_type)

                            # Clear gradient
                            param.grad = None

            # Evaluate on query set after adaptation with mixed precision
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                query_logits = model(task_query_images, arch_embedding, task_embedding, temperature)
                outer_loss = F.cross_entropy(query_logits, task_query_labels)

                # Add domain alignment loss for 1-shot tasks
                if args.k_shot == 1 and hasattr(model, 'extract_multi_scale_features'):
                    query_features = model.extract_multi_scale_features(task_query_images)[0]
                    support_features = model.extract_multi_scale_features(task_support_images)[0]

                    query_features = F.normalize(query_features, dim=1)
                    support_features = F.normalize(support_features, dim=1)

                    domain_loss = F.mse_loss(
                        query_features.mean(0),
                        support_features.mean(0)
                    ) * 0.1  # Scale factor

                    # Add to outer loss
                    outer_loss = outer_loss + domain_loss

            batch_losses.append(outer_loss)

            # Calculate accuracy
            acc = compute_accuracy(query_logits, task_query_labels)
            batch_accs.append(acc)

            # Restore original parameters for next task
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in adaptation_params:
                        param.data = adaptation_params[name]

        # Average loss across tasks
        loss = torch.mean(torch.stack(batch_losses))

        # Apply gradient updates with mixed precision and stabilization
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            success = train_with_gradient_stabilization(model, optimizer, None, args)
            if success:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            success = train_with_gradient_stabilization(model, optimizer, None, args)
            if success:
                optimizer.step()

        if success:
            train_losses.append(loss.item())
            train_accs.append(np.mean(batch_accs))

            # Update tqdm progress bar
            train_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{np.mean(batch_accs):.4f}"
            })

            if scheduler is not None:
                scheduler.step()

    return np.mean(train_losses), np.mean(train_accs)


def evaluate_arch_aware(model, val_dataset, val_sampler, networks, device, args):
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
                # Get task-specific data using get_task_slice function
                task_support_images = get_task_slice(support_images, i, args.n_way, args.k_shot)
                task_support_labels = get_task_slice(support_labels, i, args.n_way, args.k_shot)
                task_query_images = get_task_slice(query_images, i, args.n_way, args.query_size)
                task_query_labels = get_task_slice(query_labels, i, args.n_way, args.query_size)

                # Sample a random network architecture
                network_idx = np.random.randint(len(networks))
                network = networks[network_idx].to(device)

                # Try to use GHN2 encoding if available, otherwise fallback to simple encoding
                if hasattr(model, 'ghn2') and model.ghn2 is not None:
                    try:
                        arch_embedding = model.encode_architecture_ghn2(network)
                    except Exception as e:
                        print(f"Warning: GHN2 encoding failed: {e}. Falling back to simple encoding.")
                        arch_embedding = model.encode_architecture_simple(network)
                else:
                    arch_embedding = model.encode_architecture_simple(network)

                # Get task embedding
                task_embedding = model.encode_task(task_support_images, task_support_labels)

                # Apply temperature scaling for 1-shot tasks
                temperature = 0.8 if args.k_shot == 1 else 1.0

                # Forward pass directly on query images
                query_logits = model(task_query_images, arch_embedding, task_embedding, temperature)

                # Compute loss and accuracy
                loss = F.cross_entropy(query_logits, task_query_labels)
                acc = compute_accuracy(query_logits, task_query_labels)

                batch_losses.append(loss.item())
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

def train_with_gradient_stabilization(model, optimizer, loss, args):
    """Apply gradient stabilization without stepping optimizer."""
    # Check for NaN gradients
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                valid_gradients = False
                print(f"Warning: NaN or Inf gradients in {name}")
                break

    if valid_gradients:
        # Apply separate gradient clipping for different components
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters() if 'task_encoder' in n],
            args.grad_clip
        )
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters() if 'adapt_layer' in n],
            args.grad_clip * 0.5  # More aggressive clipping for adaptation modules
        )
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters() if 'param_generator' in n],
            args.grad_clip * 0.7  # Moderate clipping for parameter generator
        )

        return True
    else:
        return False

def setup_progressive_training(model, epoch, total_epochs, k_shot):
    """Configure model for progressive training with 1-shot specific handling."""
    # Adapt phase boundaries based on shot count
    phase1_end = total_epochs // 3
    phase2_end = 2 * total_epochs // 3

    phase = 1
    if epoch < phase1_end:
        # Phase 1: Train only task encoder and classifier
        phase = 1
        for name, param in model.named_parameters():
            param.requires_grad = 'task_encoder' in name or 'classifier' in name
    elif epoch < phase2_end:
        # Phase 2: Add parameter generator
        phase = 2
        for name, param in model.named_parameters():
            param.requires_grad = True
            # Freeze backbone for 1-shot to prevent overfitting
            if k_shot == 1 and 'backbone' in name and 'layer4' not in name:
                param.requires_grad = False
    else:
        # Phase 3: Train everything
        phase = 3
        for name, param in model.named_parameters():
            param.requires_grad = True

    return phase

def get_task_slice(tensor, task_idx, n_way, examples_per_class):
    """Get task-specific slice from batched tensor."""
    start_idx = task_idx * n_way * examples_per_class
    end_idx = (task_idx + 1) * n_way * examples_per_class
    return tensor[start_idx:end_idx]

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
    pretrained_ghn2_state_dict = torch.load('./checkpoints/ghn2_cifar100.pt', map_location='cpu')
    # Create GHN2 instance if it's just a state dict
    if isinstance(pretrained_ghn2_state_dict, dict) and 'state_dict' in pretrained_ghn2_state_dict:
        ghn2 = GHN(**pretrained_ghn2_state_dict['config'])
        ghn2.load_state_dict(pretrained_ghn2_state_dict['state_dict'])
        ghn2.eval()
        pretrained_ghn2 = ghn2

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"arch_aware_ghn_{args.n_way}way_{args.k_shot}shot"
        )
    torch.autograd.set_detect_anomaly(True)
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
        seed=args.seed
    )

    # Create network family for architecture variety
    networks = create_network_family()
    scaler = torch.amp.GradScaler(args.device)
    # Create model
    model = TaskAwareGHN(
        arch_embed_dim=args.arch_embed_dim,
        task_embed_dim=args.task_embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_way,
        device=device,
        ve_cutoff=args.ve_cutoff,
        ghn2=pretrained_ghn2
    ).to(device)

    model = initialize_from_ghn2(model, pretrained_ghn2)

    model.backbone = robust_network_adaptation(model.backbone)


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
        phase = setup_progressive_training(model, epoch, args.epochs,args.k_shot)
        print(f"\nEpoch {epoch + 1}/{args.epochs} - Phase {phase} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Training with architecture variety
        train_loss, train_acc = meta_training(model, train_dataset, train_sampler, networks, optimizer, scheduler, device, args)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate_arch_aware(
            model, val_dataset, val_sampler, networks, device, args
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
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

"""
Trains Task-Aware Graph HyperNetwork for few-shot learning.

Example:
    # To train TA-GHN on CIFAR-100:
    sh experiments/train_ta.sh
"""
import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import wandb
from tqdm import tqdm

from ppuda.config import init_config
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.net import Network
from ppuda.utils import capacity
from ppuda.ghn.task_aware_ghn import TaskAwareGHN, create_ta_ghn
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
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset

def get_optimizer(model, lr, weight_decay):
    """Create optimizer that only updates task-specific components."""
    # Only train task encoder and classification layer projection
    task_params = []
    for name, param in model.named_parameters():
        # Only train task encoder and related components
        if 'task_encoder' in name and param.requires_grad:
            task_params.append(param)

    print(f"Training {len(task_params)} task-specific parameters")
    return torch.optim.Adam(task_params, lr=lr, weight_decay=weight_decay)

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

    # Stack tensors - make sure these are proper tensors, not tuples
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

def main():
    """Main training function."""
    # Initialize configuration
    parser = argparse.ArgumentParser(description='Train Task-Aware GHN')
    parser.add_argument('--n_way', type=int, default=5, help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1, help='K-shot learning')
    parser.add_argument('--query_size', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--task_embed_dim', type=int, default=128, help='Task embedding dimension')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone for task encoder')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Number of steps per epoch')
    parser.add_argument('--val_steps', type=int, default=50, help='Number of validation steps')
    parser.add_argument('--arch_batch_size', type=int, default=1, help='Number of architectures per task')
    parser.add_argument('--wandb_project', type=str, default='ta-ghn', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity/username')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode with fewer steps')

    args = init_config(mode='train_ghn', parser=parser)

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.n_way}way_{args.k_shot}shot_{args.backbone}"
        )
        # Log gradients and model parameters
        wandb.watch_called = False

    # Debug mode for quick testing
    if args.debug_mode:
        print("Debug mode enabled: using reduced training steps")
        args.steps_per_epoch = min(5, args.steps_per_epoch)
        args.val_steps = min(3, args.val_steps)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar100':
        dataset = get_dataset(args.dataset, args.data_dir, is_train=True)

        # Split dataset for few-shot learning
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

    # Create graph loader for architectures
    is_imagenet = args.dataset == 'imagenet'
    graphs_queue = DeepNets1M.loader(
        args.arch_batch_size,
        split=args.split,
        nets_dir=args.data_dir,
        virtual_edges=args.virtual_edges,
        num_nets=args.num_nets,
        large_images=is_imagenet
    )

    # Create or load Task-Aware GHN
    start_epoch = 0
    if args.ckpt is not None:
        # Load from checkpoint
        model = TaskAwareGHN.load(
            args.ckpt,
            debug_level=args.debug,
            device=args.device,
            verbose=True
        )
        state_dict = torch.load(args.ckpt, map_location=args.device)
        if 'epoch' in state_dict:
            start_epoch = state_dict['epoch'] + 1
    else:
        # Create new model
        model = create_ta_ghn(
            dataset=args.dataset,
            phase=1,
            task_embed_dim=args.task_embed_dim,
            backbone=args.backbone
        )
        model = model.to(args.device)

    # Watch model with wandb (track gradients and parameters)
    if args.use_wandb and not wandb.watch_called:
        wandb.watch(model, log="all", log_freq=args.log_interval)
        wandb.watch_called = True

    # Create optimizer - only train task encoder parameters
    optimizer = get_optimizer(model, args.lr, args.wd)

    # Load optimizer state if available
    if args.ckpt is not None and 'optimizer' in state_dict:
        try:
            optimizer.load_state_dict(state_dict['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    # Create scheduler
    if args.scheduler == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=args.lr_steps,
            gamma=args.gamma
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr / 100
        )

    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    print(f"Starting training Task-Aware GHN with {capacity(model)[1]} parameters!")
    print(f"Training on {args.n_way}-way {args.k_shot}-shot tasks")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} - LR: {scheduler.get_last_lr()[0]:.6f}")
        # Training
        model.train()
        train_losses = []
        train_accs = []

        # Use tqdm for progress tracking
        train_iter = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1} (Train)")
        for step in train_iter:
            # Sample tasks
            tasks = train_sampler.sample_batch(args.meta_batch_size)
            tasks_data = collate_task_batch(tasks, train_dataset)
            support_images, support_labels, query_images, query_labels = tasks_data

            # Get next architecture batch
            graphs = next(graphs_queue)

            # Create networks
            nets_torch = []
            for nets_args in graphs.net_args:
                net = Network(is_imagenet_input=is_imagenet, num_classes=args.n_way, light=True, **nets_args)
                nets_torch.append(net)

            # Forward/backward pass
            optimizer.zero_grad()
            batch_losses = []
            batch_accs = []

            for net in nets_torch:
                support_data = (support_images.to(args.device), support_labels.to(args.device))
                query_data = (query_images.to(args.device), query_labels.to(args.device))

                # Task-conditioned parameter prediction
                wrapped_net = model(
                    net,
                    support_data=(support_data, support_labels.to(args.device)),
                    graphs=graphs.to_device(args.device)
                )

                # Evaluate on query set
                query_logits = wrapped_net(query_images.to(args.device))
                if isinstance(query_logits, tuple):
                    query_logits = query_logits[0]
                loss = F.cross_entropy(query_logits, query_labels.to(args.device))

                # Calculate accuracy
                acc = compute_accuracy(query_logits, query_labels.to(args.device))

                batch_losses.append(loss)
                batch_accs.append(acc)

            # Average loss across architectures
            loss = torch.mean(torch.stack(batch_losses))
            acc = np.mean(batch_accs)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc)

            # Update tqdm progress bar
            train_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.4f}"
            })

            # Log metrics to wandb
            if args.use_wandb and step % args.log_interval == 0:
                wandb.log({
                    "train/step": epoch * args.steps_per_epoch + step,
                    "train/loss": loss.item(),
                    "train/accuracy": acc,
                    "train/lr": scheduler.get_last_lr()[0]
                })

        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            val_iter = tqdm(range(args.val_steps), desc=f"Epoch {epoch+1} (Val)")
            for step in val_iter:
                # Sample validation tasks
                tasks = val_sampler.sample_batch(args.meta_batch_size)
                tasks_data = collate_task_batch(tasks, val_dataset)
                support_images, support_labels, query_images, query_labels = tasks_data

                # Get next architecture batch
                graphs = next(graphs_queue)

                # Create networks
                nets_torch = []
                for nets_args in graphs.net_args:
                    net = Network(
                        is_imagenet_input=is_imagenet,
                        num_classes=args.n_way,
                        light=True,
                        **nets_args
                    )
                    nets_torch.append(net)

                batch_losses = []
                batch_accs = []

                for net in nets_torch:
                    # Task-conditioned parameter prediction
                    wrapped_net = model(
                        net,
                        support_data=(support_images.to(args.device), support_labels.to(args.device)),
                        graphs=graphs.to_device(args.device)
                    )

                    # Evaluate on query set
                    query_output = wrapped_net(query_images.to(args.device))
                    query_logits = query_output[0] if isinstance(query_output, tuple) else query_output
                    loss = F.cross_entropy(query_logits, query_labels.to(args.device))

                    # Calculate accuracy
                    acc = compute_accuracy(query_logits, query_labels.to(args.device))

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

        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Log epoch metrics to wandb
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
        if args.save:
            checkpoint_path = os.path.join(args.save, f"ta_ghn_epoch_{epoch + 1}.pt")
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': {
                    'max_shape': model.max_shape,
                    'num_classes': model.num_classes,
                    'task_embed_dim': model.task_embed_dim,
                    'hypernet': 'gatedgnn',
                    'decoder': 'conv',
                    'weight_norm': model.weight_norm,
                    've': model.ve,
                    'layernorm': model.layernorm,
                    'hid': 32,
                    'phase': 1  # We're using phase 1
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(args.save, "ta_ghn_best.pt")
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': {
                        'max_shape': model.max_shape,
                        'num_classes': model.num_classes,
                        'task_embed_dim': model.task_embed_dim,
                        'hypernet': 'gatedgnn',
                        'decoder': 'conv',
                        'weight_norm': model.weight_norm,
                        've': model.ve,
                        'layernorm': model.layernorm,
                        'hid': 32,
                        'phase': 1
                    }
                }, best_path)
                print(f"Saved best model with validation accuracy {best_val_acc:.4f}")

                # Log best model to wandb
                if args.use_wandb:
                    wandb.run.summary["best_val_accuracy"] = best_val_acc
                    wandb.run.summary["best_epoch"] = epoch + 1

        # Update scheduler
        scheduler.step()

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
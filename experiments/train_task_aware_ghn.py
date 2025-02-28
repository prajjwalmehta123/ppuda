"""
Trains Task-Aware Graph HyperNetwork for few-shot learning.

Example:
    # To train TA-GHN on CIFAR-100:
    python experiments/train_ta_ghn.py -m 4 -n 5 -k 1 --phase 1 --name ta_ghn_phase1_cifar100
"""

import os
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ppuda.config import init_config
from ppuda.ghn.nn import GHN
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.net import Network
from ppuda.utils import capacity

from ppuda.ghn.task_aware_ghn import TaskAwareGHN, create_ta_ghn
from ppuda.task.task_sampler import TaskSampler, ClassSubset


def get_dataset(dataset_name, data_dir, is_train=True):
    """
    Get dataset for few-shot learning.

    Args:
        dataset_name: Dataset name ('cifar100', 'miniimagenet', etc.)
        data_dir: Directory containing datasets
        is_train: Whether to get training set

    Returns:
        PyTorch dataset
    """
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

    if dataset_name == 'imagenet':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

        # Point to your ImageNet directory
        dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'imagenet', 'train' if is_train else 'val'),
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset


class TAGHNTrainer:
    """Trainer for Task-Aware GHN."""

    def __init__(self, model, optimizer, device, meta_batch_size, arch_batch_size,
                 support_weight=0.3, query_weight=0.7, grad_clip=5.0, amp=False):
        """
        Initialize trainer.

        Args:
            model: TaskAwareGHN model
            optimizer: PyTorch optimizer
            device: Device to train on
            meta_batch_size: Number of tasks per batch
            arch_batch_size: Number of architectures per task
            support_weight: Weight for support set loss
            query_weight: Weight for query set loss
            grad_clip: Gradient clipping value
            amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.meta_batch_size = meta_batch_size
        self.arch_batch_size = arch_batch_size
        self.support_weight = support_weight
        self.query_weight = query_weight
        self.grad_clip = grad_clip
        self.amp = amp

        if amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.reset_metrics()

    def reset_metrics(self):
        """Reset training metrics."""
        self.metrics = {
            'loss': 0.0,
            'support_acc': 0.0,
            'query_acc': 0.0,
            'count': 0
        }

    def compute_accuracy(self, logits, targets):
        """Compute classification accuracy."""
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).float().sum()
        return correct.item() / targets.size(0)

    def train_on_task(self, task, graphs_queue, is_imagenet=False):
        """
        Train on a single task.

        Args:
            task: Task data (support_images, support_labels, query_images, query_labels)
            graphs_queue: Queue of architecture graphs
            is_imagenet: Whether training on ImageNet

        Returns:
            Loss and accuracy metrics
        """
        support_images, support_labels, query_images, query_labels = task

        # Move data to device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        # Get architecture batch
        graphs = next(graphs_queue)

        # Create networks
        nets_torch = []
        for nets_args in graphs.net_args:
            net = Network(is_imagenet_input=is_imagenet,
                          num_classes=len(torch.unique(support_labels)),
                          light=True,
                          **nets_args)
            nets_torch.append(net)

        losses = []
        support_accs = []
        query_accs = []

        with torch.cuda.amp.autocast(enabled=self.amp):
            # Predict parameters for all architectures with task conditioning
            for net in nets_torch:
                # Predict parameters using support set
                self.model(net, support_data=(support_images, support_labels),
                           graphs=graphs if isinstance(self.device, (list, tuple))
                           else graphs.to_device(self.device))

                # Compute loss and accuracy on support set
                support_logits = net(support_images.to(self.device))
                support_loss = F.cross_entropy(support_logits, support_labels)
                support_acc = self.compute_accuracy(support_logits, support_labels)

                # Compute loss and accuracy on query set
                query_logits = net(query_images.to(self.device))
                query_loss = F.cross_entropy(query_logits, query_labels)
                query_acc = self.compute_accuracy(query_logits, query_labels)

                # Combine losses
                loss = self.support_weight * support_loss + self.query_weight * query_loss

                losses.append(loss)
                support_accs.append(support_acc)
                query_accs.append(query_acc)

        # Compute mean loss across architectures
        mean_loss = torch.mean(torch.stack(losses))
        mean_support_acc = np.mean(support_accs)
        mean_query_acc = np.mean(query_accs)

        return mean_loss, mean_support_acc, mean_query_acc

    def train_step(self, tasks, graphs_queue, is_imagenet=False):
        """
        Train on a batch of tasks.

        Args:
            tasks: List of tasks
            graphs_queue: Queue of architecture graphs
            is_imagenet: Whether training on ImageNet

        Returns:
            Loss and accuracy metrics
        """
        self.optimizer.zero_grad()

        batch_losses = []
        batch_support_accs = []
        batch_query_accs = []

        for task in tasks:
            loss, support_acc, query_acc = self.train_on_task(task, graphs_queue, is_imagenet)
            batch_losses.append(loss)
            batch_support_accs.append(support_acc)
            batch_query_accs.append(query_acc)

        # Compute mean loss across tasks
        mean_loss = torch.mean(torch.stack(batch_losses))
        mean_support_acc = np.mean(batch_support_accs)
        mean_query_acc = np.mean(batch_query_accs)

        # Update metrics
        self.metrics['loss'] += mean_loss.item()
        self.metrics['support_acc'] += mean_support_acc
        self.metrics['query_acc'] += mean_query_acc
        self.metrics['count'] += 1

        # Backward pass and optimization
        if self.amp:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return mean_loss.item(), mean_support_acc, mean_query_acc

    def get_metrics(self):
        """Get average metrics."""
        if self.metrics['count'] == 0:
            return 0.0, 0.0, 0.0

        return (
            self.metrics['loss'] / self.metrics['count'],
            self.metrics['support_acc'] / self.metrics['count'],
            self.metrics['query_acc'] / self.metrics['count']
        )


def collate_task_batch(tasks):
    """
    Collate a batch of tasks into tensors.

    Args:
        tasks: List of (support_indices, query_indices, classes) tuples

    Returns:
        Tensor batch of (support_images, support_labels, query_images, query_labels)
    """
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for i, (support_idx, query_idx, classes) in enumerate(tasks):
        for img, label in [dataset[idx] for idx in support_idx]:
            support_images.append(img)
            # Use task index as the label offset
            support_labels.append(label)

        for img, label in [dataset[idx] for idx in query_idx]:
            query_images.append(img)
            query_labels.append(label)

    # Stack tensors
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels


def main():
    """Main training function."""
    # Initialize configuration
    args = init_config(mode='train_ta_ghn')

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
        seed=args.seed
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

        # Get config from checkpoint
        state_dict = torch.load(args.ckpt, map_location=args.device)
        if 'epoch' in state_dict:
            start_epoch = state_dict['epoch'] + 1
    else:
        # Create new model
        model = create_ta_ghn(
            dataset=args.dataset,
            phase=args.phase,
            task_embed_dim=args.task_embed_dim,
            backbone=args.backbone
        )
        model = model.to(args.device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

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

    # Create trainer
    trainer = TAGHNTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        meta_batch_size=args.meta_batch_size,
        arch_batch_size=args.arch_batch_size,
        support_weight=args.support_weight,
        query_weight=args.query_weight,
        grad_clip=args.grad_clip,
        amp=args.amp
    )

    print(f"Starting training Task-Aware GHN (Phase {args.phase}) with {capacity(model)[1]} parameters!")
    print(f"Training on {args.n_way}-way {args.k_shot}-shot tasks")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Training
        model.train()
        trainer.reset_metrics()

        for step in range(args.steps_per_epoch):
            # Sample tasks
            tasks = train_sampler.sample_batch(args.meta_batch_size)
            tasks = collate_task_batch(tasks)

            # Train on tasks
            loss, support_acc, query_acc = trainer.train_step(tasks, graphs_queue, is_imagenet)

            if step % args.log_interval == 0:
                print(f"Step {step}/{args.steps_per_epoch} - "
                      f"Loss: {loss:.4f}, Support Acc: {support_acc:.4f}, Query Acc: {query_acc:.4f}")

        # Get training metrics
        train_loss, train_support_acc, train_query_acc = trainer.get_metrics()
        print(f"Train - Loss: {train_loss:.4f}, Support Acc: {train_support_acc:.4f}, "
              f"Query Acc: {train_query_acc:.4f}")

        # Validation
        model.eval()
        trainer.reset_metrics()

        with torch.no_grad():
            for step in range(args.val_steps):
                # Sample validation tasks
                tasks = val_sampler.sample_batch(args.meta_batch_size)
                tasks = collate_task_batch(tasks)

                # Validate on tasks
                loss, support_acc, query_acc = trainer.train_step(tasks, graphs_queue, is_imagenet)

        # Get validation metrics
        val_loss, val_support_acc, val_query_acc = trainer.get_metrics()
        print(f"Validation - Loss: {val_loss:.4f}, Support Acc: {val_support_acc:.4f}, "
              f"Query Acc: {val_query_acc:.4f}")

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
                    'phase': args.phase
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Save best model
            if val_query_acc > best_val_acc:
                best_val_acc = val_query_acc
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
                        'phase': args.phase
                    }
                }, best_path)
                print(f"Saved best model with validation query accuracy {best_val_acc:.4f}")

        # Update scheduler
        scheduler.step()

    print("Training completed!")
    print(f"Best validation query accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
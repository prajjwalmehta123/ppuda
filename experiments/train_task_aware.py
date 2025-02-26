"""
Trains Task-Aware Graph HyperNetwork (TA-GHN) on meta-learning tasks.

Example:

    # To train TA-GHN on CIFAR-100 with 5-way 1-shot tasks:
    python train_taghn.py -m 8 --n-way 5 --k-shot 1 --task-embed-dim 128 --name taghn-5way-1shot

"""

import os
import torch
import argparse
import wandb
from torch.optim.lr_scheduler import MultiStepLR

from ppuda.ghn.taskawareghn import TaskAwareGHN
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.net import Network
from ppuda.task.TaskAwareDatasetManager import setup_task_aware_training
from ppuda.utils.taskawaretrainer import TaskAwareTrainer, init_wandb
from ppuda.utils import capacity, set_seed
from eval_taghn import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train Task-Aware Graph HyperNetwork')

    # Dataset options
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--train-classes', type=int, default=80,
                        help='Number of classes to use for meta-training')
    parser.add_argument('--val-classes', type=int, default=10,
                        help='Number of classes to use for meta-validation')

    # Task options
    parser.add_argument('--n-way', type=int, default=5, help='N-way classification')
    parser.add_argument('--k-shot', type=int, default=1, help='K-shot learning')
    parser.add_argument('--query-size', type=int, default=15, help='Query set size per class')
    parser.add_argument('--task-embed-dim', type=int, default=128, help='Task embedding dimension')

    # Architecture options
    parser.add_argument('--num-nets', type=int, default=1000, help='Number of networks to sample')
    parser.add_argument('-m', '--meta-batch-size', type=int, default=8,
                        help='Meta batch size (number of architectures per iteration)')
    parser.add_argument('--task-batch-size', type=int, default=4,
                        help='Number of tasks to sample per iteration')
    parser.add_argument('-v', '--virtual-edges', type=int, default=50,
                        help='Maximum shortest path length for virtual edges')

    # Model options
    parser.add_argument('--max-shape', type=int, nargs=4, default=[512, 512, 7, 7],
                        help='Maximum shape for predicted parameter tensors')
    parser.add_argument('--hypernet', type=str, default='gatedgnn', choices=['gatedgnn', 'mlp'],
                        help='Type of hypernetwork')
    parser.add_argument('--decoder', type=str, default='conv', choices=['conv', 'mlp'],
                        help='Type of hypernetwork decoder')
    parser.add_argument('--hid', type=int, default=64, help='Hidden dimension for GHN')
    parser.add_argument('--weight-norm', action='store_true', help='Use weight normalization')
    parser.add_argument('--ln', action='store_true', help='Use layer normalization')

    # Training options
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='Gradient clipping norm')
    parser.add_argument('--support-weight', type=float, default=0.3,
                        help='Weight for support set loss')
    parser.add_argument('--query-weight', type=float, default=0.7,
                        help='Weight for query set loss')
    parser.add_argument('--inner-loop-steps', type=int, default=0,
                        help='Number of inner loop adaptation steps')
    parser.add_argument('--inner-lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--lr-steps', type=int, nargs='+', default=[60, 80],
                        help='Epochs at which to reduce learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR reduction factor at schedule steps')

    # System options
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--name', type=str, default='taghn', help='Experiment name')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if not args.no_wandb:
        args.wandb_run = init_wandb(
            config=vars(args),
            project_name="task-aware-ghn",
            name=args.name,
            resume=args.ckpt is not None
        )

    # Setup task generators
    print("Setting up meta-learning datasets...")
    train_generator, val_generator, test_generator = setup_task_aware_training(
        data_dir=args.data_dir,
        train_classes=args.train_classes,
        val_classes=args.val_classes,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_size=args.query_size,
        embedding_dim=args.task_embed_dim
    )

    # Create network architecture loader
    print("Setting up architecture loader...")
    args.graphs_queue = DeepNets1M.loader(
        meta_batch_size=args.meta_batch_size,
        split='train',
        nets_dir=args.data_dir,
        virtual_edges=args.virtual_edges,
        num_nets=args.num_nets
    )

    # Initialize or load model
    start_epoch = 0
    args.best_accuracy = 0.0
    state_dict = None

    if args.ckpt is not None:
        # Load from checkpoint
        print(f"Loading checkpoint from {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location=device)
        config = state_dict['config']
        start_epoch = state_dict['epoch'] + 1
        if 'metrics' in state_dict and 'accuracy' in state_dict['metrics']:
            args.best_accuracy = state_dict['metrics']['accuracy']
    else:
        # Create new configuration
        config = {
            'max_shape': args.max_shape,
            'num_classes': args.n_way,
            'task_embedding_dim': args.task_embed_dim,
            'hypernet': args.hypernet,
            'decoder': args.decoder,
            'weight_norm': args.weight_norm,
            've': args.virtual_edges > 1,
            'layernorm': args.ln,
            'hid': args.hid
        }

    # Initialize Task-Aware GHN
    print("Initializing Task-Aware GHN...")
    ta_ghn = TaskAwareGHN(**config, debug_level=1 if args.debug else 0).to(device)

    if state_dict is not None:
        ta_ghn.load_state_dict(state_dict['state_dict'])
        print(f"Loaded model from epoch {state_dict['epoch']}")

    print(f"Task-Aware GHN has {capacity(ta_ghn)[1]:,} parameters")

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(ta_ghn.parameters(), args.lr, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)

    if state_dict is not None and 'optimizer' in state_dict:
        try:
            optimizer.load_state_dict(state_dict['optimizer'])
            print("Loaded optimizer state")

            # Update scheduler to match epoch
            for _ in range(start_epoch):
                scheduler.step()
        except Exception as e:
            print(f"WARNING: Failed to load optimizer state: {e}")

    # Initialize trainer
    trainer = TaskAwareTrainer(
        optimizer=optimizer,
        num_classes=args.n_way,
        is_imagenet=False,  # We're using CIFAR
        n_batches=args.num_nets // args.meta_batch_size,
        grad_clip=args.grad_clip,
        device=device,
        log_interval=args.log_interval,
        amp=args.amp,
        support_weight=args.support_weight,
        query_weight=args.query_weight,
        inner_loop_steps=args.inner_loop_steps,
        inner_lr=args.inner_lr,
        use_wandb=not args.no_wandb,
        experiment_name=args.name
    )

    print(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Training phase
        trainer.reset()
        ta_ghn.train()

        failed_batches = 0

        for step in range(args.num_nets // args.meta_batch_size):
            try:
                # Sample a batch of tasks
                task = train_generator.sample_task().to(device)

                # Get next batch of architectures
                try:
                    graphs = next(args.graphs_queue)
                except StopIteration:
                    # Reinitialize iterator if needed
                    args.graphs_queue = DeepNets1M.loader(
                        meta_batch_size=args.meta_batch_size,
                        split='train',
                        nets_dir=args.data_dir,
                        virtual_edges=args.virtual_edges,
                        num_nets=args.num_nets
                    )
                    graphs = next(args.graphs_queue)

                graphs = graphs.to_device(device)

                # Create networks from graph batch
                nets_torch = []
                for net_args in graphs.net_args:
                    net = Network(
                        is_imagenet_input=False,  # Using CIFAR
                        num_classes=args.n_way,  # N-way classification
                        light=True,  # Use lightweight version for efficiency
                        **net_args
                    )
                    nets_torch.append(net)

                # Update TA-GHN with task-conditioned parameter prediction
                loss = trainer.update(
                    models=nets_torch,
                    task=task,
                    ghn=ta_ghn,
                    graphs=graphs
                )

                # Log progress
                trainer.log(step=step, epoch=epoch)

                # Clean up to reduce memory usage
                if step % 10 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error: {type(e)}, {e}")
                oom = str(e).find('out of memory') >= 0
                is_nan = str(e).find('NaN') >= 0 or str(e).find('the loss is') >= 0

                if oom or is_nan:
                    if failed_batches > 50:  # Give up after too many failures
                        print(f"Too many failed batches ({failed_batches}). Exiting.")
                        raise

                    if oom:
                        print(f"CUDA out of memory, cleaning cache (attempt #{failed_batches + 1})")
                        ta_ghn.to('cpu')
                        torch.cuda.empty_cache()
                        ta_ghn.to(device)

                    failed_batches += 1
                else:
                    raise

        # Validation phase
        print("\nRunning validation...")
        val_metrics = validate(
            ta_ghn=ta_ghn,
            val_generator=val_generator,
            args=args,
            device=device,
            num_tasks=50  # Number of validation tasks to evaluate
        )

        print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")

        # Save checkpoint
        save_checkpoint(
            ghn=ta_ghn,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            metrics=val_metrics
        )

        # Update learning rate
        scheduler.step()

    print("Training completed!")


def validate(ta_ghn, val_generator, args, device, num_tasks=50):
    """Validate the TaskAware GHN model with proper tuple handling"""
    ta_ghn.eval()

    # Setup architecture loader and metrics
    val_graphs_queue = DeepNets1M.loader(
        meta_batch_size=1,
        split='val',
        nets_dir=args.data_dir,
        virtual_edges=args.virtual_edges,
        num_nets=10
    )
    val_iterator = iter(val_graphs_queue)
    accuracies = []

    with torch.no_grad():
        for _ in range(num_tasks):
            # Sample task and get network
            task = val_generator.sample_task().to(device)
            graphs = next(val_iterator).to_device(device)

            # Create network
            net = Network(
                is_imagenet_input=False,
                num_classes=args.n_way,
                light=True,
                **graphs.net_args[0]
            )

            # Generate parameters
            try:
                net = ta_ghn(net, graphs=graphs, task=task)

                # Process model output
                output = net(task.query_images)

                # Handle tuple output
                if isinstance(output, tuple):
                    output = output[0]  # Get main output

                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                total = task.query_labels.size(0)
                correct = (predicted == task.query_labels).sum().item()
                accuracy = correct / total
                accuracies.append(accuracy)

            except Exception as e:
                print(f"Error evaluating task: {e}")
                continue

    # Calculate average accuracy
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    return {"accuracy": avg_accuracy, "task_accuracies": accuracies}


if __name__ == '__main__':
    main()
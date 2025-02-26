import os
import torch
from tqdm import tqdm
from ppuda.utils import infer
from ppuda.deepnets1m.net import Network


def evaluate(ghn, task_generator, args, num_tasks=100):
    """
    Evaluate Task-Aware GHN on a set of tasks

    Args:
        ghn: Task-Aware GHN model
        task_generator: Generator for evaluation tasks
        args: Configuration arguments
        num_tasks: Number of tasks to evaluate on

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    accuracies = []
    task_accuracies = []

    with torch.no_grad():
        # Sample tasks for evaluation
        tasks = task_generator.sample_batch(num_tasks)

        for task in tqdm(tasks, desc="Evaluating"):
            # Initialize networks for this task
            nets_torch = []
            graphs = next(args.graphs_queue)  # Get next batch of graphs

            for nets_args in graphs.net_args:
                net = Network(
                    is_imagenet_input=args.is_imagenet,
                    num_classes=args.num_classes,
                    light=True,
                    **nets_args
                )
                nets_torch.append(net)

            # Generate parameters conditioned on task
            nets_with_params = ghn(
                nets_torch,
                graphs=graphs,
                task=task,
                predict_class_layers=True,
                bn_train=True
            )

            # Evaluate on query set
            query_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(task.query_images, task.query_labels),
                batch_size=args.test_batch_size
            )

            # Get accuracy on this task
            top1, top5 = infer(
                nets_with_params,
                query_loader,
                verbose=False
            )

            task_accuracies.append(top1)

            # Optional: Evaluate base network performance (without task conditioning)
            if args.evaluate_base:
                base_nets = ghn(
                    nets_torch,
                    graphs=graphs,
                    task=None,  # No task conditioning
                    predict_class_layers=True,
                    bn_train=True
                )
                base_top1, _ = infer(base_nets, query_loader, verbose=False)
                accuracies.append(base_top1)

    # Compute metrics
    metrics = {
        'accuracy': sum(task_accuracies) / len(task_accuracies),
        'task_accuracies': task_accuracies,
    }

    if args.evaluate_base:
        metrics['base_accuracy'] = sum(accuracies) / len(accuracies)

    return metrics


def save_checkpoint(ghn, optimizer, epoch, args, metrics=None):
    """
    Save Task-Aware GHN checkpoint

    Args:
        ghn: Task-Aware GHN model
        optimizer: Optimizer state
        epoch: Current epoch
        args: Training configuration
        metrics: Optional evaluation metrics to save
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Create checkpoint path
    checkpoint_path = os.path.join(
        args.save_dir,
        f'task_aware_ghn_epoch_{epoch}.pt'
    )

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'state_dict': ghn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {
            'max_shape': ghn.max_shape,
            'num_classes': ghn.num_classes,
            'task_embedding_dim': ghn.task_embedding_dim,
            'hypernet': args.hypernet,
            'decoder': args.decoder,
            'weight_norm': args.weight_norm,
            've': args.virtual_edges > 1,
            'layernorm': args.ln,
            'hid': args.hid
        }
    }

    # Add metrics if provided
    if metrics is not None:
        checkpoint['metrics'] = metrics

    # Save to disk
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

    # Save best model if this is the best performance
    if metrics is not None and metrics.get('accuracy', 0) > args.best_accuracy:
        args.best_accuracy = metrics['accuracy']
        best_path = os.path.join(args.save_dir, 'task_aware_ghn_best.pt')
        torch.save(checkpoint, best_path)
        print(f"New best model saved to {best_path}\n")


def load_checkpoint(checkpoint_path, ghn, optimizer=None, device='cuda'):
    """
    Load Task-Aware GHN checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        ghn: Task-Aware GHN model
        optimizer: Optional optimizer to load state
        device: Device to load model on

    Returns:
        epoch: Epoch number of checkpoint
        metrics: Saved metrics if any
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    ghn.load_state_dict(checkpoint['state_dict'])

    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Return checkpoint info
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', None)
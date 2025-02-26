import time
import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import parallel_apply
from ppuda.utils import AvgrageMeter, accuracy
from ppuda.task import Task

import time
import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import parallel_apply
from ppuda.utils import AvgrageMeter, accuracy
from ppuda.task import Task


class TaskAwareTrainer(nn.Module):
    def __init__(self, optimizer, num_classes, is_imagenet, n_batches,
                 grad_clip=5, auxiliary=False, auxiliary_weight=0.4, device='cuda',
                 log_interval=100, amp=False,
                 support_weight=0.3, query_weight=0.7,
                 inner_loop_steps=1, inner_lr=0.01,
                 use_wandb=True, experiment_name=None):
        """
        Enhanced task-aware trainer with dynamic topk accuracy
        """
        super().__init__()
        self.optimizer = optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = criterion
        self.n_batches = n_batches
        self.grad_clip = grad_clip
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.device = device
        self.log_interval = log_interval
        self.amp = amp
        self.num_classes = num_classes

        # Meta-learning specific parameters
        self.support_weight = support_weight
        self.query_weight = query_weight
        self.inner_loop_steps = inner_loop_steps
        self.inner_lr = inner_lr

        # Wandb integration
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name

        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.reset()

    def reset(self):
        self.start = time.time()
        self.step = 0
        self.metrics = {
            'loss': AvgrageMeter(),
            'top1': AvgrageMeter(),
            'support_loss': AvgrageMeter(),
            'query_loss': AvgrageMeter(),
            'adaptation_gain': AvgrageMeter()  # Track improvement from adaptation
        }
        # Add top5 metric only if we have enough classes
        if self.num_classes >= 5:
            self.metrics['top5'] = AvgrageMeter()

    def update(self, models, task: Task, ghn=None, graphs=None):
        """
        Update model parameters using task data with improved meta-learning dynamics
        """
        if not isinstance(models, list):
            models = [models]

        # Move task to device if needed
        if task.support_images.device != self.device:
            task = task.to(self.device)

        self.optimizer.zero_grad()

        # Create topk tuple based on number of classes
        topk = (1,)
        if self.num_classes >= 5:
            topk = (1, 5)
        else:
            # Handle the case with just a few classes
            topk = (1, min(self.num_classes, 2))

        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                         enabled=self.amp):
            # First predict initial parameters with GHN
            if ghn is not None:
                models = ghn(models, graphs=graphs, task=task)

            all_query_logits = []
            total_loss = 0

            for model in models:
                # Step 1: Evaluate initial performance on support set
                model.eval()
                with torch.no_grad():
                    initial_support_out = model(task.support_images)
                    s_out_initial = initial_support_out[0] if isinstance(initial_support_out,
                                                                         tuple) else initial_support_out
                    initial_support_loss = self.criterion(s_out_initial, task.support_labels)

                    initial_query_out = model(task.query_images)
                    q_out_initial = initial_query_out[0] if isinstance(initial_query_out, tuple) else initial_query_out
                    initial_query_accuracy = accuracy(q_out_initial, task.query_labels, topk=(1,))[0].item()

                # Step 2: Perform inner loop adaptation if requested (simulate MAML)
                adapted_params = None
                if self.inner_loop_steps > 0:
                    model.train()  # Enable dropout, etc. during adaptation
                    adapted_params = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            adapted_params[name] = param.clone()

                    # Inner loop adaptation on support set
                    for _ in range(self.inner_loop_steps):
                        support_out = model(task.support_images)
                        s_out = support_out[0] if isinstance(support_out, tuple) else support_out
                        support_loss = self.criterion(s_out, task.support_labels)

                        # Manual gradient computation and update
                        grads = torch.autograd.grad(
                            support_loss,
                            [p for n, p in model.named_parameters() if n in adapted_params],
                            create_graph=True
                        )

                        # Update adapted parameters
                        for (name, param), grad in zip(
                                [(n, p) for n, p in model.named_parameters() if n in adapted_params],
                                grads
                        ):
                            adapted_params[name] = adapted_params[name] - self.inner_lr * grad

                    # Temporarily replace model parameters with adapted ones
                    original_params = {}
                    for name, param in model.named_parameters():
                        if name in adapted_params:
                            original_params[name] = param.data.clone()
                            param.data = adapted_params[name].data.clone()

                # Step 3: Evaluate on support and query sets after adaptation
                model.eval()  # For consistent evaluation
                support_out = model(task.support_images)
                s_out = support_out[0] if isinstance(support_out, tuple) else support_out
                support_loss = self.criterion(s_out, task.support_labels)

                query_out = model(task.query_images)
                q_out = query_out[0] if isinstance(query_out, tuple) else query_out
                query_loss = self.criterion(q_out, task.query_labels)
                adapted_query_accuracy = accuracy(q_out, task.query_labels, topk=(1,))[0].item()

                # Compute adaptation gain (improvement from initial to adapted)
                adaptation_gain = adapted_query_accuracy - initial_query_accuracy

                # Step 4: Compute weighted loss combining support and query objectives
                weighted_loss = self.support_weight * support_loss + self.query_weight * query_loss

                # Add auxiliary loss if needed
                if self.auxiliary and isinstance(support_out, tuple) and isinstance(query_out, tuple):
                    weighted_loss += self.auxiliary_weight * (
                            self.support_weight * self.criterion(support_out[1], task.support_labels) +
                            self.query_weight * self.criterion(query_out[1], task.query_labels)
                    )

                total_loss += weighted_loss
                all_query_logits.append(q_out.detach())

                # Restore original parameters if we did adaptation
                if adapted_params is not None:
                    for name, param in model.named_parameters():
                        if name in original_params:
                            param.data = original_params[name]

        # Average loss across models
        loss = total_loss / len(models)

        if torch.isnan(loss):
            raise RuntimeError('The loss is NaN, unable to proceed')

        # Backward pass and optimization
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        # Gradient clipping
        parameters = []
        for group in self.optimizer.param_groups:
            parameters.extend(group['params'])

        nn.utils.clip_grad_norm_(parameters, self.grad_clip)

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Update metrics
        query_logits = torch.stack(all_query_logits, dim=0)
        query_targets = task.query_labels.reshape(-1, 1).unsqueeze(0).expand(
            query_logits.shape[0], task.query_labels.shape[0], 1).reshape(-1)
        query_logits = query_logits.reshape(-1, query_logits.shape[-1])

        # Calculate accuracy using appropriate topk
        if self.num_classes >= 5:
            prec1, prec5 = accuracy(query_logits, query_targets, topk=topk)
            n = len(query_targets)
            self.metrics['top1'].update(prec1.item(), n)
            self.metrics['top5'].update(prec5.item(), n)
        else:
            prec1 = accuracy(query_logits, query_targets, topk=(1,))[0]
            n = len(query_targets)
            self.metrics['top1'].update(prec1.item(), n)

        # Update other metrics
        self.metrics['loss'].update(loss.item(), n)
        self.metrics['support_loss'].update(support_loss.item(), len(task.support_labels))
        self.metrics['query_loss'].update(query_loss.item(), len(task.query_labels))
        self.metrics['adaptation_gain'].update(adaptation_gain, 1)

        self.step += 1

        return loss

    def log(self, step=None, epoch=None):
        """Log metrics to console and wandb"""
        step_ = self.step if step is None else step

        if step_ % self.log_interval == 0 or step_ >= self.n_batches - 1:
            speed = (time.time() - self.start) / max(1, step_)
            metrics = '\t'.join(['{}={:.2f}'.format(metric, value.avg)
                                 for metric, value in self.metrics.items()])

            print('batch={:04d}/{:04d} \t {} \t {:.2f} sec/batch ({:.1f} min left) '.format(
                step_, self.n_batches, metrics, speed,
                speed * (self.n_batches - step_) / 60), flush=True)

            # Log to wandb
            if self.use_wandb:
                log_dict = {f'train/{k}': v.avg for k, v in self.metrics.items()}

                # Add batch info
                log_dict['train/batch'] = step_
                log_dict['train/speed'] = speed

                # Add epoch if provided
                if epoch is not None:
                    log_dict['train/epoch'] = epoch

                wandb.log(log_dict)

def init_wandb(config, project_name="task-aware-ghn", entity=None, name=None, resume=False, id=None):
    """Initialize wandb with the given configuration"""
    # Create a descriptive run name if none provided
    if name is None:
        name = f"TAGHN-{config['task_embedding_dim']}-{config['meta_batch_size']}"

    # Initialize wandb
    wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=name,
        resume="allow" if resume else False,
        id=id
    )

    return wandb.run
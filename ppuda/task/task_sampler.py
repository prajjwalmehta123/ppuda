import torch
import numpy as np
from torch.utils.data import Subset


class TaskSampler:
    """
    Samples N-way K-shot classification tasks for meta-learning.
    """
    def __init__(self, dataset, n_way=5, k_shot=1, query_size=15, seed=42):
        """
        Initialize task sampler.

        Args:
            dataset: PyTorch dataset with targets attribute
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            query_size: Number of query examples per class
            seed: Random seed
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.seed = seed

        # Create class-to-indices mapping
        self.class_indices = self._build_class_indices()

        # Set random seed
        np.random.seed(seed)

    def _build_class_indices(self):
        """Build a mapping from class index to data indices."""
        targets = self._get_targets()

        # Dictionary to store indices per class
        class_indices = {}

        for idx, target in enumerate(targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)

        return class_indices

    def _get_targets(self):
        """Extract targets from the dataset."""
        if hasattr(self.dataset, 'targets'):
            return self.dataset.targets
        elif hasattr(self.dataset, 'labels'):
            return self.dataset.labels
        elif hasattr(self.dataset, 'targets') and isinstance(self.dataset.targets, list):
            return self.dataset.targets
        else:
            # Try to extract targets by iterating through the dataset
            targets = []
            for _, target in self.dataset:
                targets.append(target)
            return targets

    def sample_task(self):
        """
        Sample a single N-way K-shot task.

        Returns:
            support_indices: Indices for support set
            query_indices: Indices for query set
            classes: List of class indices used in this task
        """
        # Sample N classes
        available_classes = list(self.class_indices.keys())
        classes = np.random.choice(available_classes, self.n_way, replace=False)

        support_indices = []
        query_indices = []

        for cls in classes:
            # Get indices for this class
            cls_indices = self.class_indices[cls]

            # Ensure we have enough samples
            if len(cls_indices) < self.k_shot + self.query_size:
                # If not enough samples, sample with replacement
                support_idx = np.random.choice(cls_indices, self.k_shot, replace=True)
                remaining = np.random.choice(cls_indices, self.query_size, replace=True)
            else:
                # Sample without replacement
                sampled_idx = np.random.choice(cls_indices,
                                               self.k_shot + self.query_size,
                                               replace=False)
                support_idx = sampled_idx[:self.k_shot]
                remaining = sampled_idx[self.k_shot:self.k_shot + self.query_size]

            support_indices.extend(support_idx)
            query_indices.extend(remaining)

        return support_indices, query_indices, classes

    def sample_batch(self, batch_size=4):
        """
        Sample a batch of tasks.

        Args:
            batch_size: Number of tasks to sample

        Returns:
            tasks: List of (support_indices, query_indices, classes) tuples
        """
        tasks = []
        for _ in range(batch_size):
            tasks.append(self.sample_task())

        return tasks


class ClassSubset:
    """
    Subset of a dataset containing only specified classes.
    """

    def __init__(self, dataset, classes):
        """
        Initialize class subset.

        Args:
            dataset: PyTorch dataset with targets attribute
            classes: List of class indices to include
        """
        self.dataset = dataset
        self.classes = set(classes)

        # Get targets
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels
        else:
            raise ValueError("Dataset must have targets or labels attribute")

        # Filter indices based on class
        self.indices = [i for i, t in enumerate(targets) if t in self.classes]

        # Create class mapping
        self.class_mapping = {c: i for i, c in enumerate(sorted(list(self.classes)))}

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        # Map original class to new class index
        target = self.class_mapping[target]
        return img, target

    def __len__(self):
        return len(self.indices)

    @property
    def targets(self):
        """Get targets after remapping."""
        if hasattr(self.dataset, 'targets'):
            original_targets = self.dataset.targets
        else:
            original_targets = self.dataset.labels

        return [self.class_mapping[original_targets[i]] for i in self.indices]
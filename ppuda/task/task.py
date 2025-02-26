import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """Represents a single task for meta-learning"""

    def __init__(self,
                 support_images: torch.Tensor,
                 support_labels: torch.Tensor,
                 query_images: torch.Tensor,
                 query_labels: torch.Tensor,
                 task_embedding: torch.Tensor = None):
        self.support_images = support_images
        self.support_labels = support_labels
        self.query_images = query_images
        self.query_labels = query_labels
        self.task_embedding = task_embedding

    def to(self, device):
        """Move all tensors to specified device"""
        self.support_images = self.support_images.to(device)
        self.support_labels = self.support_labels.to(device)
        self.query_images = self.query_images.to(device)
        self.query_labels = self.query_labels.to(device)
        if self.task_embedding is not None:
            self.task_embedding = self.task_embedding.to(device)
        return self

    def compute_embedding(self, encoder):
        """Compute task embedding using provided encoder"""
        with torch.no_grad():
            self.task_embedding = encoder(self.support_images, self.support_labels)
        return self


class TaskEncoder(nn.Module):
    """Enhanced Task Encoder that uses prototypical class representations"""

    def __init__(self,
                 in_channels: int = 3,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.relation_network = nn.Sequential(
            nn.Linear(256 * 2, hidden_dim),  # Double size to accommodate concatenated pairs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 256, hidden_dim),  # Combined relation + prototype features
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def compute_prototypes(self, support_images, support_labels):
        """Compute class prototypes from support set"""
        features = self.feature_extractor(support_images)  # [N*K, 256, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [N*K, 256]

        # Get unique class labels
        unique_labels = torch.unique(support_labels)
        num_classes = len(unique_labels)

        # Initialize prototypes
        prototypes = torch.zeros(num_classes, features.size(1), device=features.device)

        # Compute mean feature vectors for each class
        for i, label in enumerate(unique_labels):
            class_mask = support_labels == label
            prototypes[i] = features[class_mask].mean(0)

        return prototypes, features

    def compute_relation_features(self, prototypes):
        """Compute relation features between class prototypes"""
        num_classes = prototypes.size(0)
        relation_features = []

        # Process all pairs of prototypes
        for i in range(num_classes):
            for j in range(i + 1, num_classes):  # Only unique pairs
                # Concatenate prototype pairs
                pair = torch.cat([prototypes[i], prototypes[j]], dim=0)
                # Get relation embedding
                relation = self.relation_network(pair)
                relation_features.append(relation)

        # If there are no pairs (single class), return zero tensor
        if len(relation_features) == 0:
            return torch.zeros(1, self.relation_network[-1].out_features, device=prototypes.device)

        # Average all relation features
        relation_features = torch.stack(relation_features).mean(0)
        return relation_features

    def forward(self, support_images, support_labels):
        """
        Generate task embedding that captures class relationships

        Args:
            support_images: [N*K, C, H, W] tensor of support images
            support_labels: [N*K] tensor of support labels

        Returns:
            task_embedding: [embedding_dim] tensor
        """
        # Compute class prototypes
        prototypes, features = self.compute_prototypes(support_images, support_labels)

        # Average prototype for global task representation
        global_prototype = prototypes.mean(0)

        # Compute relation features between prototypes
        relation_features = self.compute_relation_features(prototypes)

        # Combine global prototype with relation features
        combined = torch.cat([global_prototype, relation_features], dim=0)

        # Project to final embedding space
        task_embedding = self.projection(combined)

        return task_embedding


class TaskGenerator:
    """Generates few-shot learning tasks"""

    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 n_way: int = 5,
                 k_shot: int = 1,
                 query_size: int = 15,
                 task_encoder: Optional[TaskEncoder] = None):
        """
        Args:
            dataset: Dataset to sample tasks from
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            query_size: Number of query examples per class
            task_encoder: Optional encoder to generate task embeddings
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.task_encoder = task_encoder

        # Group dataset indices by class
        self.class_indices = self._group_by_class()

    def _group_by_class(self) -> Dict[int, List[int]]:
        """Groups dataset indices by class"""
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def sample_task(self) -> Task:
        """Samples a single few-shot task"""
        # Sample N classes
        classes = np.random.choice(
            list(self.class_indices.keys()),
            size=self.n_way,
            replace=False
        )

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # Sample support and query examples for each class
        for class_idx, class_label in enumerate(classes):
            class_examples = self.class_indices[class_label]

            # Sample K support examples
            support_indices = np.random.choice(
                class_examples,
                size=self.k_shot,
                replace=False
            )

            # Sample query examples
            remaining = list(set(class_examples) - set(support_indices))
            query_indices = np.random.choice(
                remaining,
                size=min(self.query_size, len(remaining)),
                replace=False
            )

            # Get images and labels
            for idx in support_indices:
                image, _ = self.dataset[idx]
                support_images.append(image)
                support_labels.append(class_idx)

            for idx in query_indices:
                image, _ = self.dataset[idx]
                query_images.append(image)
                query_labels.append(class_idx)

        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)

        # Generate task embedding if encoder provided
        task_embedding = None
        if self.task_encoder is not None:
            with torch.no_grad():
                task_embedding = self.task_encoder(support_images, support_labels)

        return Task(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            query_labels=query_labels,
            task_embedding=task_embedding
        )

    def sample_batch(self, batch_size: int) -> List[Task]:
        """Samples a batch of tasks"""
        return [self.sample_task() for _ in range(batch_size)]


# Example usage:
def get_task_generator(
        dataset: torch.utils.data.Dataset,
        n_way: int = 5,
        k_shot: int = 1,
        query_size: int = 15,
        embedding_dim: int = 128,
        use_task_encoder: bool = True
) -> TaskGenerator:
    """Creates a TaskGenerator with optional TaskEncoder"""

    task_encoder = None
    if use_task_encoder:
        task_encoder = TaskEncoder(
            in_channels=dataset[0][0].shape[0],
            embedding_dim=embedding_dim
        )

    return TaskGenerator(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        query_size=query_size,
        task_encoder=task_encoder
    )
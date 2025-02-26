import os
import ssl
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100
from ppuda.vision.transforms import transforms_cifar
from ppuda.task import TaskGenerator, TaskEncoder

# Handle SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


class ClassSubset(Dataset):
    """
    Custom dataset subset that properly maintains class information
    """

    def __init__(self,
                 dataset: Dataset,
                 indices: List[int],
                 targets: List[int],
                 class_mapping: Optional[Dict[int, int]] = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            dataset: Original dataset
            indices: Indices to subset from original dataset
            targets: Class labels for each index
            class_mapping: Optional mapping from original class IDs to new sequential IDs
            transform: Optional transform to apply
        """
        self.dataset = dataset
        self.indices = indices
        self.targets = targets
        self.class_mapping = class_mapping
        self.transform = transform

        # Create attributes needed for meta-learning
        self.classes = sorted(list(set(targets)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        """Get item with remapped class labels if mapping exists"""
        image, label = self.dataset[self.indices[idx]]

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Remap class label if mapping exists
        if self.class_mapping is not None:
            label = self.class_mapping[label]

        return image, label

    def __len__(self):
        return len(self.indices)


class TaskAwareDatasetManager:
    """Manages dataset splitting and task generation for Task-Aware GHN training"""

    def __init__(self,
                 data_dir: str = './data',
                 train_classes: int = 80,
                 val_classes: int = 10,
                 seed: int = 42):
        """
        Args:
            data_dir: Directory for dataset storage
            train_classes: Number of classes for meta-training
            val_classes: Number of classes for meta-validation
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.train_classes = train_classes
        self.val_classes = val_classes
        self.test_classes = 100 - train_classes - val_classes

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.train_transform, self.valid_transform = transforms_cifar()
        os.makedirs(data_dir, exist_ok=True)

        try:
            self.train_dataset = CIFAR100(data_dir, train=True, download=True, transform=None)
            self.test_dataset = CIFAR100(data_dir, train=False, download=True, transform=None)
        except Exception as e:
            print(f"Error loading CIFAR-100 dataset: {str(e)}")
            raise

        # Split classes
        all_classes = list(range(100))
        np.random.shuffle(all_classes)

        self.train_class_ids = all_classes[:train_classes]
        self.val_class_ids = all_classes[train_classes:train_classes + val_classes]
        self.test_class_ids = all_classes[train_classes + val_classes:]

        # Create class mapping for each split
        self.train_class_mapping = {old: new for new, old in enumerate(self.train_class_ids)}
        self.val_class_mapping = {old: new for new, old in enumerate(self.val_class_ids)}
        self.test_class_mapping = {old: new for new, old in enumerate(self.test_class_ids)}

        # Create split datasets
        self.meta_train_dataset = self._filter_classes(
            self.train_dataset,
            self.train_class_ids,
            self.train_class_mapping,
            self.train_transform
        )

        self.meta_val_dataset = self._filter_classes(
            self.train_dataset,
            self.val_class_ids,
            self.val_class_mapping,
            self.valid_transform
        )

        self.meta_test_dataset = self._filter_classes(
            self.test_dataset,
            self.test_class_ids,
            self.test_class_mapping,
            self.valid_transform
        )

    def _filter_classes(self,
                        dataset: Dataset,
                        class_ids: List[int],
                        class_mapping: Dict[int, int],
                        transform: Optional[Callable] = None) -> Dataset:
        """Creates a ClassSubset containing only specified classes"""
        indices = []
        targets = []

        for idx, target in enumerate(dataset.targets):
            if target in class_ids:
                indices.append(idx)
                targets.append(target)

        return ClassSubset(
            dataset=dataset,
            indices=indices,
            targets=targets,
            class_mapping=class_mapping,
            transform=transform
        )

    def get_meta_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Returns meta-train, meta-val, and meta-test datasets"""
        return self.meta_train_dataset, self.meta_val_dataset, self.meta_test_dataset

    def get_class_mappings(self) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """Returns class mappings for each split"""
        return self.train_class_mapping, self.val_class_mapping, self.test_class_mapping

    def get_split_info(self) -> Dict[str, int]:
        """Returns information about the dataset splits"""
        return {
            'total_classes': 100,
            'train_classes': self.train_classes,
            'val_classes': self.val_classes,
            'test_classes': self.test_classes,
            'train_samples': len(self.meta_train_dataset),
            'val_samples': len(self.meta_val_dataset),
            'test_samples': len(self.meta_test_dataset)
        }


def get_image_and_label(dataset, idx):
    """Helper function to get image and label from dataset"""
    image, label = dataset[idx]
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected tensor but got {type(image)}. Make sure transforms are properly applied.")
    return image, label


def setup_task_aware_training(
        data_dir: str = './data',
        train_classes: int = 80,
        val_classes: int = 10,
        n_way: int = 5,
        k_shot: int = 1,
        query_size: int = 15,
        embedding_dim: int = 128
) -> Tuple[TaskGenerator, TaskGenerator, TaskGenerator]:
    """
    Sets up complete training pipeline for Task-Aware GHN

    Args:
        data_dir: Directory for dataset storage
        train_classes: Number of classes for meta-training
        val_classes: Number of classes for meta-validation
        n_way: Number of classes per task
        k_shot: Number of support examples per class
        query_size: Number of query examples per class
        embedding_dim: Dimension of task embeddings

    Returns:
        train_task_generator: Generator for training tasks
        val_task_generator: Generator for validation tasks
        test_task_generator: Generator for test tasks
    """
    # Initialize dataset manager
    dataset_manager = TaskAwareDatasetManager(
        data_dir=data_dir,
        train_classes=train_classes,
        val_classes=val_classes
    )

    # Get datasets
    meta_train_dataset, meta_val_dataset, meta_test_dataset = dataset_manager.get_meta_datasets()

    # Create shared task encoder
    task_encoder = TaskEncoder(
        in_channels=3,  # CIFAR-100 has 3 channels
        embedding_dim=embedding_dim
    )

    # Create task generators
    train_task_generator = TaskGenerator(
        dataset=meta_train_dataset,
        n_way=n_way,
        k_shot=k_shot,
        query_size=query_size,
        task_encoder=task_encoder
    )

    val_task_generator = TaskGenerator(
        dataset=meta_val_dataset,
        n_way=n_way,
        k_shot=k_shot,
        query_size=query_size,
        task_encoder=task_encoder
    )

    test_task_generator = TaskGenerator(
        dataset=meta_test_dataset,
        n_way=n_way,
        k_shot=k_shot,
        query_size=query_size,
        task_encoder=task_encoder
    )

    print("Task-Aware Training Setup Complete:")
    print(f"Meta-Train Classes: {train_classes}")
    print(f"Meta-Val Classes: {val_classes}")
    print(f"Meta-Test Classes: {100 - train_classes - val_classes}")
    print(f"Task Format: {n_way}-way {k_shot}-shot")

    return train_task_generator, val_task_generator, test_task_generator


if __name__ == "__main__":
    train_generator, val_generator, test_generator = setup_task_aware_training(
        data_dir='./data',
        train_classes=80,
        val_classes=10,
        n_way=5,
        k_shot=1,
        query_size=15,
        embedding_dim=128
    )

    train_tasks = train_generator.sample_batch(batch_size=8)
    print('hello')
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, CIFAR100, CIFAR10, SVHN,Omniglot




class MetaDataset(Dataset):
    def __init__(self, dataset, n_way=5, k_shot=1, n_query=15, n_episodes=1000, transform=None):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.transform = transform
        self.is_svhn = isinstance(dataset, SVHN)

        # Group data by class
        self.data_by_class = {}
        for i in range(len(dataset)):
            if self.is_svhn:
                img = dataset.data[i]
                label = dataset.labels[i]
            else:
                img, label = dataset[i]
            if label not in self.data_by_class:
                self.data_by_class[label] = []
            self.data_by_class[label].append((img, i))

        # Keep only classes with enough samples
        min_samples = k_shot + n_query
        self.valid_classes = [c for c in self.data_by_class.keys()
                              if len(self.data_by_class[c]) >= min_samples]

        if len(self.valid_classes) < n_way:
            raise ValueError(f"Not enough classes with {min_samples} samples. "
                             f"Found only {len(self.valid_classes)} valid classes.")

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        # Randomly sample n_way classes
        selected_classes = random.sample(self.valid_classes, self.n_way)

        # Prepare empty tensors
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # Fill tensors with data
        for class_idx, class_label in enumerate(selected_classes):
            class_samples = self.data_by_class[class_label]
            # Randomly sample k_shot + n_query images
            selected_samples = random.sample(class_samples, self.k_shot + self.n_query)

            # Support set
            for i in range(self.k_shot):
                img,orig_idx = selected_samples[i]

                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(class_idx)

            # Query set
            for i in range(self.k_shot, self.k_shot + self.n_query):
                img,orig_idx = selected_samples[i]
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(class_idx)

        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels


def get_transform(dataset_name):
    """Get appropriate transforms for each dataset"""
    if dataset_name in ['cifar10', 'cifar100']:
        # CIFAR datasets are 32x32
        return transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])
    elif dataset_name == 'svhn':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.1980, 0.2010, 0.1970])
        ])
    elif dataset_name == 'omniglot':
        return transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize([0.92206], [0.08426])
        ])
    elif dataset_name == 'miniimagenet':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Default ImageNet stats for other datasets
        return transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def setup_meta_dataloaders(n_way=5, k_shot=1, n_query=15, target_datasets=None,n_episodes=600):
    if target_datasets is None:
        target_datasets = ['cifar10', 'svhn', 'omniglot']
    meta_train_datasets = []
    meta_val_datasets = []

    # CIFAR-100 (source dataset)
    cifar100_train = CIFAR100(root='./data', train=True, download=True)
    cifar100_val = CIFAR100(root='./data', train=False, download=True)
    meta_train_datasets.append(
        MetaDataset(cifar100_train, n_way, k_shot, n_query,
                    n_episodes=n_episodes, transform=get_transform('cifar100'))
    )
    meta_val_datasets.append(
        MetaDataset(cifar100_val, n_way, k_shot, n_query,
                    n_episodes=n_episodes//5, transform=get_transform('cifar100'))
    )
    if 'cifar10' in target_datasets:
        # CIFAR-10 (target dataset 1)
        cifar10_train = CIFAR10(root='./data', train=True, download=True)
        cifar10_val = CIFAR10(root='./data', train=False, download=True)
        meta_train_datasets.append(
            MetaDataset(cifar10_train, n_way, k_shot, n_query,
                        n_episodes=n_episodes, transform=get_transform('cifar10'))
        )
        meta_val_datasets.append(
            MetaDataset(cifar10_val, n_way, k_shot, n_query,
                        n_episodes=n_episodes//5, transform=get_transform('cifar10'))
        )
    if 'svhn' in target_datasets:
        # SVHN (target dataset 2)
        svhn_train = SVHN(root='./data', split='train', download=True)
        svhn_test = SVHN(root='./data', split='test', download=True)
        meta_train_datasets.append(
            MetaDataset(svhn_train, n_way, k_shot, n_query,
                        n_episodes=n_episodes, transform=get_transform('svhn'))
        )
        meta_val_datasets.append(
            MetaDataset(svhn_test, n_way, k_shot, n_query,
                        n_episodes=n_episodes//5, transform=get_transform('svhn'))
        )

    if 'omniglot' in target_datasets:
        # Omniglot (target dataset 3 - very different distribution)
        omniglot_train = Omniglot(root='./data', background=True, download=True)
        omniglot_val = Omniglot(root='./data', background=False, download=True)
        meta_train_datasets.append(
            MetaDataset(omniglot_train, n_way, k_shot, n_query,
                        n_episodes=n_episodes, transform=get_transform('omniglot'))
        )
        meta_val_datasets.append(
            MetaDataset(omniglot_val, n_way, k_shot, n_query,
                        n_episodes=n_episodes//5, transform=get_transform('omniglot'))
        )

    # Create data loaders
    batch_size = 4  # Process 4 episodes per batch
    meta_train_loader = DataLoader(CombinedMetaDataset(meta_train_datasets),
                                   batch_size=batch_size)
    meta_val_loader = DataLoader(CombinedMetaDataset(meta_val_datasets),
                                 batch_size=batch_size)

    return meta_train_loader, meta_val_loader


class CombinedMetaDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.dataset_lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Determine which dataset this index belongs to
        dataset_idx = 0
        while idx >= self.dataset_lengths[dataset_idx]:
            idx -= self.dataset_lengths[dataset_idx]
            dataset_idx += 1

        # Get the episode from the appropriate dataset
        return self.datasets[dataset_idx][idx], dataset_idx


if __name__ == '__main__':
    # Test data loaders
    meta_train_loader, meta_val_loader = setup_meta_dataloaders(n_way=5, k_shot=1)

    # Check a batch from the loader
    for task_batch, dataset_indices in meta_train_loader:
        support_imgs, support_labs, query_imgs, query_labs = task_batch
        print(f"Support images shape: {support_imgs.shape}")
        print(f"Support labels shape: {support_labs.shape}")
        print(f"Query images shape: {query_imgs.shape}")
        print(f"Query labels shape: {query_labs.shape}")
        print(f"Dataset indices: {dataset_indices}")
        break

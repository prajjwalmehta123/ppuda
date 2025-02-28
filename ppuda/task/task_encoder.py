import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TaskEncoder(nn.Module):
    """
    Encodes task data (support set) into a fixed-dimension embedding.
    This encoder processes few-shot learning tasks and produces a task representation.
    """

    def __init__(self, embedding_dim=128, backbone='resnet18'):
        super(TaskEncoder, self).__init__()

        # Feature extractor
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()  # Remove classification layer
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Inter-class relation module
        self.relation_module = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def compute_prototypes(self, features, labels):
        """
        Compute class prototypes by averaging features for each class.

        Args:
            features: Feature vectors [N*K, feature_dim]
            labels: Class labels [N*K]

        Returns:
            prototypes: Average feature vector per class [N, feature_dim]
        """
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        feature_dim = features.shape[1]

        prototypes = torch.zeros(num_classes, feature_dim, device=features.device)

        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            if mask.sum() > 0:  # Ensure there are samples for this class
                prototypes[i] = features[mask].mean(dim=0)

        return prototypes, unique_labels

    def compute_relations(self, prototypes):
        """
        Compute relationships between class prototypes.

        Args:
            prototypes: Class prototypes [N, feature_dim]

        Returns:
            relations: Relation features for the task
        """
        num_classes = prototypes.shape[0]
        if num_classes <= 1:
            return torch.zeros(1, 64, device=prototypes.device)

        relation_features = []

        # Compute pairwise relations
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # Compute relation feature (e.g., difference, product)
                diff = prototypes[i] - prototypes[j]
                relation = self.relation_module(diff)
                relation_features.append(relation)

        # If no relations (single class), return zeros
        if not relation_features:
            return torch.zeros(1, 64, device=prototypes.device)

        relations = torch.stack(relation_features, dim=0)
        return relations.mean(dim=0)  # Average all relations

    def forward(self, support_images, support_labels):
        """
        Encode task data into a task embedding.

        Args:
            support_images: Images from the support set, shape [N*K, C, H, W]
            support_labels: Labels from the support set, shape [N*K]

        Returns:
            task_embedding: Task embedding vector, shape [embedding_dim]
        """
        # Extract features
        features = self.backbone(support_images)  # [N*K, feature_dim]

        # Compute prototypes (average features per class)
        prototypes, unique_labels = self.compute_prototypes(features, support_labels)

        # Compute task representation using prototypes and their relations
        prototype_avg = prototypes.mean(dim=0)
        prototype_relations = self.compute_relations(prototypes)

        # Combine prototype average with relation information
        task_embedding = self.projection(prototype_avg)

        return task_embedding
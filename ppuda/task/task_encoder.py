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
            self.backbone.fc = nn.Identity()
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

    def forward(self,support_data):
        """Extract features and create task embedding from support set."""
        # Extract per-image features
        if isinstance(support_data, tuple) and len(support_data) == 2:
            support_images, support_labels = support_data
        else:
            raise ValueError("support_data must be a tuple of (images, labels)")
        features = self.backbone(support_images)

        # Compute class prototypes
        unique_labels = torch.unique(support_labels)
        class_prototypes = []

        for label in unique_labels:
            mask = (support_labels == label)
            class_prototypes.append(features[mask].mean(dim=0))

        # Average prototype features to get task representation
        if class_prototypes:
            task_features = torch.stack(class_prototypes).mean(dim=0)
        else:
            task_features = features.mean(dim=0)

        # Project to task embedding
        task_embedding = self.projection(task_features)

        return task_embedding
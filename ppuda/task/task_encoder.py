import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


class TaskEncoder(nn.Module):
    """Encodes task data (support set) into a fixed-dimension embedding."""

    def __init__(self, feature_extractor, embedding_dim=128):
        super().__init__()
        self.feature_extractor = feature_extractor

        # Get output dimension of feature extractor
        with torch.no_grad():
            # Save training state and temporarily set to eval mode
            was_training = feature_extractor.training
            feature_extractor.eval()  # This is the key fix

            dummy_input = torch.randn(1, 3, 32, 32)
            features = feature_extractor(dummy_input)
            if isinstance(features, tuple):
                features = features[0]
            feature_dim = features.view(1, -1).size(1)

            # Restore original training mode
            feature_extractor.train(was_training)

        # Projection head to create task embedding
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, support_images, support_labels, n_way):
        """
        Extract task representation from support set.

        Args:
            support_images: Tensor of support images [B*N*K, C, H, W]
            support_labels: Tensor of support labels [B*N*K]
            n_way: Number of classes per task

        Returns:
            Task embedding tensor
        """
        # Extract features from support images
        with torch.no_grad():
            features = self.feature_extractor(support_images)
            if isinstance(features, tuple):
                features = features[0]
            features = features.view(features.size(0), -1)

        # Compute class prototypes
        prototypes = []
        for c in range(n_way):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                prototypes.append(class_features.mean(0))
            else:
                prototypes.append(torch.zeros_like(features[0]))

        task_features = torch.stack(prototypes).mean(0)
        return self.projection(task_features)

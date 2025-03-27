import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TaskEncoder(nn.Module):
    def __init__(self, feature_dim=512, embedding_dim=128):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification layer

        # Multi-scale feature extraction
        self.layer1_proj = nn.Conv2d(64, 128, kernel_size=1)
        self.layer2_proj = nn.Conv2d(128, 128, kernel_size=1)
        self.layer3_proj = nn.Conv2d(256, 128, kernel_size=1)

        # Projection for task embedding
        self.projection = nn.Sequential(
            nn.Linear(3 * 128, 256),  # For multi-scale features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )

    def extract_multi_scale_features(self, support_images):
        """Extract features from multiple network depths for richer representation"""
        x = self.backbone.conv1(support_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Extract features from different layers
        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)

        # Project to uniform dimensionality
        p1 = F.adaptive_avg_pool2d(self.layer1_proj(f1), 1).flatten(1)
        p2 = F.adaptive_avg_pool2d(self.layer2_proj(f2), 1).flatten(1)
        p3 = F.adaptive_avg_pool2d(self.layer3_proj(f3), 1).flatten(1)

        combined = torch.cat([p1, p2, p3], dim=1)

        return combined, f1, f2, f3, f4

    def forward(self, support_images, support_labels, n_way):
        # Extract features from multiple network layers
        features, f1, f2, f3, f4 = self.extract_multi_scale_features(support_images)

        # Compute class prototypes
        prototypes = []
        for c in range(n_way):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                prototypes.append(class_features.mean(0))
            else:
                prototypes.append(torch.zeros_like(features[0]))

        # Compute task representation based on prototypes
        proto_tensor = torch.stack(prototypes)
        task_features = proto_tensor.mean(0)

        return self.projection(task_features)

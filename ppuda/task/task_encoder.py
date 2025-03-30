import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ppuda.ghn.nn import GHN
from ppuda.deepnets1m.graph import Graph, GraphBatch


def initialize_with_ghn(model, ghn_checkpoint_path, device="cuda"):
    # Load pretrained GHN2
    ghn = GHN.load(ghn_checkpoint_path, device=device)
    graph = Graph(model)
    graphs = GraphBatch([graph])
    graphs.to_device(ghn.embed.weight.device)
    with torch.no_grad():
        ghn(model, graphs)

    return model


class TaskEncoder(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=False):
        super().__init__()

        if backbone == 'resnet34':
            self.backbone = models.resnet34(weights= 'DEFAULT')
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights= 'DEFAULT')
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove Classification Layer Layer
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()

        # Projection layers for multi-scale features
        self.layer1_proj = nn.Conv2d(64, 128, kernel_size=1)
        self.layer2_proj = nn.Conv2d(128, 128, kernel_size=1)
        self.layer3_proj = nn.Conv2d(256, 128, kernel_size=1)
        self.layer4_proj = nn.Conv2d(512, 128, kernel_size=1)
        self.out_dim = 512  # 4 layers * 128 features

    def forward(self, x):
        features = []

        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1
        f1 = self.backbone.layer1(x)
        p1 = F.adaptive_avg_pool2d(self.layer1_proj(f1), 1).flatten(1)
        features.append(p1)

        # Layer 2
        f2 = self.backbone.layer2(f1)
        p2 = F.adaptive_avg_pool2d(self.layer2_proj(f2), 1).flatten(1)
        features.append(p2)

        # Layer 3
        f3 = self.backbone.layer3(f2)
        p3 = F.adaptive_avg_pool2d(self.layer3_proj(f3), 1).flatten(1)
        features.append(p3)

        # Layer 4
        f4 = self.backbone.layer4(f3)
        p4 = F.adaptive_avg_pool2d(self.layer4_proj(f4), 1).flatten(1)
        features.append(p4)

        # Concatenate features
        return torch.cat(features, dim=1)

    def get_output_dim(self):
        return self.out_dim


if __name__ == '__main__':
    encoder = TaskEncoder()
    encoder = initialize_with_ghn(
        encoder,
        ghn_checkpoint_path="/Users/prajjwalmehta/Desktop/projects/ppuda/checkpoints/ghn2_cifar100.pt",
        device="cpu"
    )
    for name, param in encoder.named_parameters():
        print(f"{name}: mean={param.mean().item():.5f}, std={param.std().item():.5f}")
    x = torch.randn(2, 3, 84, 84)
    features = encoder(x)
    print(f"Feature shape: {features.shape}")
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

from ppuda.deepnets1m.graph import GraphBatch
from ppuda.deepnets1m.architecture import ArchitectureGraphBuilder, ArchitectureEncoder, JointParameterGenerator
from ppuda.task.task_encoder import TaskEncoder


class TaskAwareGHN(nn.Module):
    """
    Hybrid GHN for few-shot learning.
    Uses both architecture and task information to predict parameters.
    """

    def __init__(self,
                 feature_dim=512,
                 arch_embed_dim=128,
                 task_embed_dim=128,
                 hidden_dim=256,
                 num_classes=5,
                 device='cpu',
                 ve_cutoff=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.arch_embed_dim = arch_embed_dim
        self.task_embed_dim = task_embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.ve_cutoff = ve_cutoff
        self.multi_scale_dim = 128 * 3
        # Feature extractor (frozen ResNet backbone)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification layer
        self.backbone = self.backbone.to(device)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Architecture graph builder
        self.graph_builder = ArchitectureGraphBuilder(ve_cutoff=ve_cutoff)

        # Architecture encoder
        self.arch_encoder = ArchitectureEncoder(
            embedding_dim=arch_embed_dim,
            hidden_dim=hidden_dim // 2,
            ve=True,
            layernorm=True
        ).to(device)

        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(self.multi_scale_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, task_embed_dim)
        ).to(device)

        # Joint parameter generator
        self.param_generator = JointParameterGenerator(
            arch_dim=arch_embed_dim,
            task_dim=task_embed_dim,
            hidden_dim=hidden_dim
        ).to(device)

        # Adaptation modules (parameters will be predicted by GHN)
        self.layer1_proj = nn.Conv2d(64, 128, kernel_size=1).to(device)
        self.layer2_proj = nn.Conv2d(128, 128, kernel_size=1).to(device)
        self.layer3_proj = nn.Conv2d(256, 128, kernel_size=1).to(device)
        self.layer4_proj = nn.Conv2d(512, 128, kernel_size=1).to(device)

        self.adapt_layer1 = AdaptationModule(64).to(device)
        self.adapt_layer2 = AdaptationModule(128).to(device)
        self.adapt_layer3 = AdaptationModule(256).to(device)
        self.adapt_layer4 = AdaptationModule(512).to(device)
        # Final classification layer
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        # Cache for architecture embeddings
        self.arch_embedding_cache = {}
        self.simple_projection = nn.Linear(9, self.arch_embed_dim).to(device)
        nn.init.orthogonal_(self.simple_projection.weight)
        self = self.to(device)

    def encode_architecture_simple(self, network):
        """
        A simplified architecture encoding that doesn't rely on autograd.
        """
        # Generate a fixed embedding based on network characteristics
        network_id = id(network)
        if network_id in self.arch_embedding_cache:
            return self.arch_embedding_cache[network_id]

        # Create a feature vector describing the architecture
        arch_features = []

        # Count layers by type
        layer_counts = {}
        for name, module in network.named_modules():
            layer_type = type(module).__name__
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += 1

        # Check if layer1 exists and is not an Identity
        if hasattr(network, 'layer1') and not isinstance(network.layer1, nn.Identity):
            if hasattr(network.layer1, '__len__'):
                arch_features.append(len(network.layer1))
            else:
                arch_features.append(1)  # Single layer
        else:
            arch_features.append(0)

        # Check if layer2 exists and is not an Identity
        if hasattr(network, 'layer2') and not isinstance(network.layer2, nn.Identity):
            if hasattr(network.layer2, '__len__'):
                arch_features.append(len(network.layer2))
            else:
                arch_features.append(1)  # Single layer
        else:
            arch_features.append(0)

        # Check if layer3 exists and is not an Identity
        if hasattr(network, 'layer3') and not isinstance(network.layer3, nn.Identity):
            if hasattr(network.layer3, '__len__'):
                arch_features.append(len(network.layer3))
            else:
                arch_features.append(1)  # Single layer
        else:
            arch_features.append(0)

        # Check if layer4 exists and is not an Identity
        if hasattr(network, 'layer4') and not isinstance(network.layer4, nn.Identity):
            if hasattr(network.layer4, '__len__'):
                arch_features.append(len(network.layer4))
            else:
                arch_features.append(1)  # Single layer
        else:
            arch_features.append(0)

        # Layer type distribution features
        for layer_type in ['Conv2d', 'BatchNorm2d', 'Linear', 'MaxPool2d', 'AvgPool2d']:
            arch_features.append(layer_counts.get(layer_type, 0))

        # Convert to tensor and normalize
        arch_features = torch.tensor(arch_features, dtype=torch.float32, device=self.device)
        arch_features = arch_features / (arch_features.sum() + 1e-6)

        # Project to the right dimension
        arch_embedding = self.simple_projection(arch_features)

        # Cache the embedding
        self.arch_embedding_cache[network_id] = arch_embedding

        return arch_embedding

    def extract_multi_scale_features(self, support_images):
        """
        Extract multi-scale features from the backbone network.

        Args:
            support_images: Support images tensor

        Returns:
            Multi-scale features tensor
        """
        # Forward pass through initial layers
        x = self.backbone.conv1(support_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Extract features from different layers
        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)

        # Project and pool features to the same dimensionality
        p1 = F.adaptive_avg_pool2d(self.layer1_proj(f1), 1)
        p2 = F.adaptive_avg_pool2d(self.layer2_proj(f2), 1)
        p3 = F.adaptive_avg_pool2d(self.layer3_proj(f3), 1)
        p4 = F.adaptive_avg_pool2d(self.layer4_proj(f4), 1)  # Last layer already has 512 channels

        # Concatenate features from different scales
        combined = torch.cat([
            p1.view(p1.size(0), -1),
            p2.view(p2.size(0), -1),
            p4.view(p4.size(0), -1)  # Using layers 1, 2, and 4 for diverse scales
        ], dim=1)

        return combined, f1, f2, f3, f4

    def encode_task(self, support_images, support_labels):
        """
        Extract task representation from support set.

        Args:
            support_images: Tensor of support images
            support_labels: Tensor of support labels

        Returns:
            Task embedding tensor
        """
        features, _, _, _, _ = self.extract_multi_scale_features(support_images)

        # Compute class prototypes
        prototypes = []
        for c in range(self.num_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                prototypes.append(class_features.mean(0))
            else:
                prototypes.append(torch.zeros_like(features[0]))

        task_features = torch.stack(prototypes).mean(0)

        return self.task_encoder(task_features)

    def set_adaptation_params(self, arch_embedding, task_embedding):
        """
        Use parameter generator to predict parameters for all adaptation modules.

        Args:
            arch_embedding: Architecture embedding tensor
            task_embedding: Task embedding tensor
        """
        # Generate parameters for each adaptation module
        params_layer1 = self.param_generator(arch_embedding, task_embedding, 64)
        params_layer2 = self.param_generator(arch_embedding, task_embedding, 128)
        params_layer3 = self.param_generator(arch_embedding, task_embedding, 256)
        params_layer4 = self.param_generator(arch_embedding, task_embedding, 512)

        # Set parameters for adaptation modules
        self._set_module_params(self.adapt_layer1, params_layer1)
        self._set_module_params(self.adapt_layer2, params_layer2)
        self._set_module_params(self.adapt_layer3, params_layer3)
        self._set_module_params(self.adapt_layer4, params_layer4)

    def _set_module_params(self, module, params_dict):
        """Helper to set parameters for a module."""
        # Create mapping from parameter names to module attributes
        param_mapping = {
            'fc1_weight': (module.fc1, 'weight'),
            'fc1_bias': (module.fc1, 'bias'),
            'fc2_weight': (module.fc2, 'weight'),
            'fc2_bias': (module.fc2, 'bias'),
            'conv_spatial_weight': (module.conv_spatial, 'weight'),
            'conv_spatial_bias': (module.conv_spatial, 'bias')
        }

        for name, param in params_dict.items():
            if name not in param_mapping:
                continue

            target_module, param_name = param_mapping[name]
            original_param = getattr(target_module, param_name)

            # Ensure the parameter has the right shape
            if param.shape != original_param.shape:
                param = param.reshape(original_param.shape)

            # Set the parameter
            setattr(target_module, param_name, nn.Parameter(param))

    def forward(self, query_images, arch_embedding=None, task_embedding=None):
        """
        Process query images with task-specific and architecture-aware adaptation.

        Args:
            query_images: Query images tensor
            arch_embedding: Architecture embedding tensor (optional)
            task_embedding: Task embedding tensor (optional)

        Returns:
            Class logits for query images
        """
        if arch_embedding is not None and task_embedding is not None:
            # Use arch and task embeddings to predict adaptation module parameters
            self.set_adaptation_params(arch_embedding, task_embedding)

        # Apply backbone with adaptation modules
        # First part of ResNet
        x = self.backbone.conv1(query_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 with adaptation
        x = self.backbone.layer1(x)
        x = self.adapt_layer1(x)

        # Layer 2 with adaptation
        x = self.backbone.layer2(x)
        x = self.adapt_layer2(x)

        # Layer 3 with adaptation
        x = self.backbone.layer3(x)
        x = self.adapt_layer3(x)

        # Layer 4 with adaptation
        x = self.backbone.layer4(x)
        x = self.adapt_layer4(x)

        # Global pooling and classification
        x = F.adaptive_avg_pool2d(x, 1)
        features = torch.flatten(x, 1)
        return self.classifier(features)


class AdaptationModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = in_channels // reduction

        # Channel attention path
        self.fc1 = nn.Linear(in_channels, self.mid_channels)
        self.fc2 = nn.Linear(self.mid_channels, in_channels)

        # Spatial attention path
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x, spatial_dims=None):
        # Original input for residual connection
        identity = x

        # Channel attention
        b, c = x.shape[0], x.shape[1]

        # Global average pooling
        y_channel = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # Apply channel attention
        y_channel = F.relu(self.fc1(y_channel))
        y_channel = torch.sigmoid(self.fc2(y_channel))

        # Apply channel attention weights
        x_channel = x * y_channel.view(b, c, 1, 1)

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y_spatial = torch.cat([avg_pool, max_pool], dim=1)
        y_spatial = self.conv_spatial(y_spatial)
        y_spatial = torch.sigmoid(y_spatial)

        # Apply spatial attention
        x_spatial = x * y_spatial

        # Combine attentions with residual connection
        x = identity + x_channel + x_spatial

        # Apply layer normalization (converted to the right shape)
        shape = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        return x
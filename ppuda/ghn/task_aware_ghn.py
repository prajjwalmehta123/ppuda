import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np

from ppuda.deepnets1m.graph import GraphBatch
from ppuda.task.architecture import ArchitectureGraphBuilder, ArchitectureEncoder, JointParameterGenerator
from ppuda.task.task_encoder import TaskEncoder


class ArchitectureAwareGHN(nn.Module):
    """
    Architecture-aware hybrid GHN for few-shot learning.
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
        """
        Args:
            feature_dim: Dimension of backbone features
            arch_embed_dim: Dimension of architecture embeddings
            task_embed_dim: Dimension of task embeddings
            hidden_dim: Dimension of hidden layers
            num_classes: Number of classes (N-way)
            device: Device to use
            ve_cutoff: Maximum shortest path length for virtual edges
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.arch_embed_dim = arch_embed_dim
        self.task_embed_dim = task_embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.ve_cutoff = ve_cutoff

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
        self.task_encoder = TaskEncoder(
            feature_extractor=self.backbone,
            embedding_dim=task_embed_dim
        ).to(device)

        # Joint parameter generator
        self.param_generator = JointParameterGenerator(
            arch_dim=arch_embed_dim,
            task_dim=task_embed_dim,
            hidden_dim=hidden_dim
        ).to(device)

        # Adaptation modules (parameters will be predicted by GHN)
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

        # Network depth features
        if hasattr(network, 'layer1'):
            arch_features.append(len(network.layer1))
        else:
            arch_features.append(0)

        if hasattr(network, 'layer2'):
            arch_features.append(len(network.layer2))
        else:
            arch_features.append(0)

        if hasattr(network, 'layer3'):
            arch_features.append(len(network.layer3))
        else:
            arch_features.append(0)

        if hasattr(network, 'layer4'):
            arch_features.append(len(network.layer4))
        else:
            arch_features.append(0)

        # Layer type distribution features
        for layer_type in ['Conv2d', 'BatchNorm2d', 'Linear', 'MaxPool2d', 'AvgPool2d']:
            arch_features.append(layer_counts.get(layer_type, 0))

        # Convert to tensor and normalize
        arch_features = torch.tensor(arch_features, dtype=torch.float32, device=self.device)
        arch_features = arch_features / (arch_features.sum() + 1e-6)

        # Project to the right dimension
        simple_projection = nn.Linear(len(arch_features), self.arch_embed_dim).to(self.device)
        nn.init.orthogonal_(simple_projection.weight)

        # Generate embedding
        with torch.no_grad():
            arch_embedding = simple_projection(arch_features)

        # Cache the embedding
        self.arch_embedding_cache[network_id] = arch_embedding

        return arch_embedding

    def encode_architecture(self, network):
        """
        Encode network architecture into embedding.
        Uses caching for efficiency.
        """
        # Check if architecture is in cache
        network_id = id(network)
        if network_id in self.arch_embedding_cache:
            return self.arch_embedding_cache[network_id]

        try:
            graph = self.graph_builder.build_graph(network)
            graph_batch = GraphBatch([graph]).to_device(self.device)

            with torch.no_grad():
                arch_embeddings = self.arch_encoder(graph_batch)
                arch_embedding = arch_embeddings[0]

            # Cache the embedding
            self.arch_embedding_cache[network_id] = arch_embedding
            return arch_embedding

        except Exception as e:
            # Return a default embedding as fallback
            return torch.zeros(self.arch_embed_dim, device=self.device)

    def encode_task(self, support_images, support_labels):
        """
        Extract task representation from support set.

        Args:
            support_images: Tensor of support images
            support_labels: Tensor of support labels

        Returns:
            Task embedding tensor
        """
        return self.task_encoder(support_images, support_labels, self.num_classes)

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
        """
        Helper to set parameters for a module.

        Args:
            module: Target module to update
            params_dict: Dictionary of parameter tensors
        """
        for name, param in params_dict.items():
            param_obj = module
            name_parts = name.split('.')

            # Navigate to the correct attribute
            for part in name_parts[:-1]:
                param_obj = getattr(param_obj, part)

            # Get original parameter to ensure correct shape
            original_param = getattr(param_obj, name_parts[-1])

            # Ensure the parameter has the right shape
            if param.shape != original_param.shape:
                param = param.reshape(original_param.shape)

            # Set the parameter
            setattr(param_obj, name_parts[-1], nn.Parameter(param))

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

        # First part of ResNet
        x = self.backbone.conv1(query_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 with adaptation
        x = self.backbone.layer1(x)
        x = self.adapt_layer1(x, spatial_dims=(x.size(2), x.size(3)))

        # Layer 2 with adaptation
        x = self.backbone.layer2(x)
        x = self.adapt_layer2(x, spatial_dims=(x.size(2), x.size(3)))

        # Layer 3 with adaptation
        x = self.backbone.layer3(x)
        x = self.adapt_layer3(x, spatial_dims=(x.size(2), x.size(3)))

        # Layer 4 with adaptation
        x = self.backbone.layer4(x)
        x = self.adapt_layer4(x, spatial_dims=(x.size(2), x.size(3)))

        # Global pooling and classification
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return self.classifier(features)

class AdaptationModule(nn.Module):
    """
    A lightweight adaptation module whose parameters
    will be generated by the GHN.
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = in_channels // reduction

        # These parameters will be predicted by GHN
        self.fc1 = nn.Linear(in_channels, self.mid_channels)
        self.fc2 = nn.Linear(self.mid_channels, in_channels)

    def forward(self, x, spatial_dims=None):
        # Global average pooling
        b, c = x.shape[0], x.shape[1]
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)  # Flatten to [batch_size, channels]

        # Channel attention
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Apply attention weights (reshape to match spatial dimensions)
        return x * y.view(b, c, 1, 1)

class MiniGHN(nn.Module):
    """
    A simplified GHN that predicts parameters for lightweight adaptation modules.
    """

    def __init__(self, task_dim=128, hidden_dim=64):
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim

        # Task feature processing
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU()
        )

        # Parameter generation networks for different channel sizes
        self.decoders = nn.ModuleDict({
            '64': self._create_decoder(64),
            '128': self._create_decoder(128),
            '256': self._create_decoder(256),
            '512': self._create_decoder(512)
        })

    # Replace the _create_decoder method in MiniGHN with this version:
    # Replace the _create_decoder method with this version
    def _create_decoder(self, channels):
        """Create a decoder for generating adaptation module parameters."""
        reduction = 4
        mid_channels = channels // reduction

        class CustomDecoder(nn.Module):
            def __init__(self, in_features, hidden_dim, out_shape):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, np.prod(out_shape))
                )
                self.out_shape = out_shape

            def forward(self, x):
                return self.fc(x).reshape(-1, *self.out_shape)

        decoder = nn.ModuleDict({
            'fc1_weight': CustomDecoder(
                self.hidden_dim, self.hidden_dim * 2, (mid_channels, channels)
            ),
            'fc1_bias': CustomDecoder(
                self.hidden_dim, self.hidden_dim, (mid_channels,)
            ),
            'fc2_weight': CustomDecoder(
                self.hidden_dim, self.hidden_dim * 2, (channels, mid_channels)
            ),
            'fc2_bias': CustomDecoder(
                self.hidden_dim, self.hidden_dim, (channels,)
            )
        })

        return decoder

    def forward(self, task_embedding, channel_size):
        """
        Generate parameters for an adaptation module with the given channel size.

        Args:
            task_embedding: Task-specific embedding
            channel_size: Number of channels (64, 128, 256, or 512)

        Returns:
            Dictionary of parameters for the adaptation module
        """
        # Process task embedding
        h = self.task_encoder(task_embedding)

        # Get the appropriate decoder for the channel size
        decoder = self.decoders[str(channel_size)]

        # Generate parameters
        params = {
            'fc1.weight': decoder['fc1_weight'](h),
            'fc1.bias': decoder['fc1_bias'](h),
            'fc2.weight': decoder['fc2_weight'](h),
            'fc2.bias': decoder['fc2_bias'](h)
        }

        return params

class HybridGHNNetwork(nn.Module):

    def __init__(self, feature_dim=512, task_embed_dim=128, num_classes=5, device='cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.task_embed_dim = task_embed_dim
        self.num_classes = num_classes
        self.device = device

        # Feature extractor (frozen ResNet backbone)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification layer
        self.backbone = self.backbone.to(device)

        # Freeze backbone parameters for stability
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Task encoder (processes support set)
        self.task_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, task_embed_dim)
        ).to(device)

        # Mini GHN for generating adaptation module parameters
        self.ghn = MiniGHN(task_dim=task_embed_dim, hidden_dim=64).to(device)

        # Adaptation modules (parameters will be predicted by GHN)
        self.adapt_layer1 = AdaptationModule(64).to(device)
        self.adapt_layer2 = AdaptationModule(128).to(device)
        self.adapt_layer3 = AdaptationModule(256).to(device)
        self.adapt_layer4 = AdaptationModule(512).to(device)

        # Final classification layer
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

    def encode_task(self, support_images, support_labels):
        """Extract task representation from support set."""
        # Extract features from support images
        with torch.no_grad():
            features = self.backbone(support_images)

        # Compute class prototypes
        prototypes = []
        for c in range(self.num_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                prototypes.append(class_features.mean(0))
            else:
                # Handle case where a class might have no examples
                prototypes.append(torch.zeros(features.size(1), device=features.device))

        if prototypes:
            task_features = torch.stack(prototypes).mean(0)
            return self.task_encoder(task_features)
        else:
            return torch.zeros(self.task_embed_dim, device=features.device)

    def set_adaptation_params(self, task_embedding):
        """Use GHN to predict parameters for all adaptation modules."""
        # Generate parameters for each adaptation module
        params_layer1 = self.ghn(task_embedding, 64)
        params_layer2 = self.ghn(task_embedding, 128)
        params_layer3 = self.ghn(task_embedding, 256)
        params_layer4 = self.ghn(task_embedding, 512)

        # Set parameters for adaptation modules
        self._set_module_params(self.adapt_layer1, params_layer1)
        self._set_module_params(self.adapt_layer2, params_layer2)
        self._set_module_params(self.adapt_layer3, params_layer3)
        self._set_module_params(self.adapt_layer4, params_layer4)

    def _set_module_params(self, module, params_dict):
        """Helper to set parameters for a module."""
        for name, param in params_dict.items():
            param_obj = module
            name_parts = name.split('.')

            # Navigate to the correct attribute
            for part in name_parts[:-1]:
                param_obj = getattr(param_obj, part)

            # Get original parameter to ensure correct shape
            original_param = getattr(param_obj, name_parts[-1])

            # Ensure the parameter has the right shape before setting it
            if param.shape != original_param.shape:
                param = param.reshape(original_param.shape)

            # Set the parameter
            setattr(param_obj, name_parts[-1], nn.Parameter(param))

    def forward(self, query_images, task_embedding=None):
        """Process query images with task-specific adaptation."""
        if task_embedding is not None:
            # Use GHN to predict adaptation module parameters
            self.set_adaptation_params(task_embedding)

        # First part of ResNet
        x = self.backbone.conv1(query_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 with adaptation
        x = self.backbone.layer1(x)
        x = self.adapt_layer1(x, spatial_dims=(x.size(2), x.size(3)))

        # Layer 2 with adaptation
        x = self.backbone.layer2(x)
        x = self.adapt_layer2(x, spatial_dims=(x.size(2), x.size(3)))

        # Layer 3 with adaptation
        x = self.backbone.layer3(x)
        x = self.adapt_layer3(x, spatial_dims=(x.size(2), x.size(3)))

        # Layer 4 with adaptation
        x = self.backbone.layer4(x)
        x = self.adapt_layer4(x, spatial_dims=(x.size(2), x.size(3)))

        # Global pooling and classification
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return self.classifier(features)
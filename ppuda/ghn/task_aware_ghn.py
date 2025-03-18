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
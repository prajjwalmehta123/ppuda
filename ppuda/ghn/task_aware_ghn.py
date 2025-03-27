import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

from ppuda.deepnets1m.architecture import ArchitectureGraphBuilder, ArchitectureEncoder, JointParameterGenerator
from ppuda.task.task_encoder import TaskEncoder
from ppuda.deepnets1m.graph import Graph, GraphBatch


class TaskAwareGHN(nn.Module):
    def __init__(self, feature_dim=512, arch_embed_dim=128, task_embed_dim=128,
                 hidden_dim=256, num_classes=5, device='cpu', ve_cutoff=50, ghn2=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.arch_embed_dim = arch_embed_dim
        self.task_embed_dim = task_embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.ve_cutoff = ve_cutoff
        self.ghn2 = ghn2

        # Feature extractor (frozen ResNet backbone)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification layer
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Task encoder
        self.task_encoder = TaskEncoder(feature_dim, task_embed_dim).to(device)

        # Adaptation modules
        self.adapt_layer1 = AdaptationModule(64, arch_embed_dim=arch_embed_dim).to(device)
        self.adapt_layer2 = AdaptationModule(128, arch_embed_dim=arch_embed_dim).to(device)
        self.adapt_layer3 = AdaptationModule(256, arch_embed_dim=arch_embed_dim).to(device)
        self.adapt_layer4 = AdaptationModule(512, arch_embed_dim=arch_embed_dim).to(device)

        # Parameter generator
        self.param_generator = JointParameterGenerator(
            arch_dim=arch_embed_dim,
            task_dim=task_embed_dim,
            hidden_dim=hidden_dim
        ).to(device)

        # Final classification layer
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        self.simple_projection = nn.Linear(5, self.arch_embed_dim).to(device)

        nn.init.orthogonal_(self.simple_projection.weight)

        # Architecture embedding cache
        self.arch_embedding_cache = {}

    def encode_task(self, support_images, support_labels):
        """Encode task from support set images and labels."""
        return self.task_encoder(support_images, support_labels, self.num_classes)

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


    def encode_architecture_simple(self, network):
        """A simplified architecture encoding that doesn't rely on GHN2"""
        # Check cache first
        network_id = id(network)
        if network_id in self.arch_embedding_cache:
            return self.arch_embedding_cache[network_id]

        # Create a feature vector describing the architecture
        arch_features = [
            # Count layers by type
            sum(1 for _ in network.modules() if isinstance(_, nn.Conv2d)),
            sum(1 for _ in network.modules() if isinstance(_, nn.BatchNorm2d)),
            sum(1 for _ in network.modules() if isinstance(_, nn.Linear)),
            # Depth features
            len(network.modules()) // 10,
            # Network width approximation
            sum(m.out_channels for m in network.modules()
                if isinstance(m, nn.Conv2d)) // 100
        ]

        # Normalize features
        arch_features = torch.tensor(arch_features, dtype=torch.float32, device=self.device)
        arch_features = arch_features / (arch_features.sum() + 1e-6)

        # Project to embedding dimension
        arch_embedding = self.simple_projection(arch_features)

        # Cache and return
        self.arch_embedding_cache[network_id] = arch_embedding
        return arch_embedding

    def encode_architecture_ghn2(self, network):
        """Encode architecture using GHN2's graph neural network."""
        # Check if we have a cached embedding
        network_id = id(network)
        if network_id in self.arch_embedding_cache:
            return self.arch_embedding_cache[network_id]

        model_device = next(self.parameters()).device
        try:
            # Get network device
            for param in network.parameters():
                network_device = param.device
                if network_device != model_device:
                    print(f"Moving network from {network_device} to {model_device}")
                    network = network.to(model_device)
                break
        except Exception as e:
            print(f"Device detection error: {e}")

        if not hasattr(self, 'graph_builder'):
            self.graph_builder = ArchitectureGraphBuilder(ve_cutoff=self.ve_cutoff)
        # Build graph
        graph = self.graph_builder.build_graph(network)

        graph_batch = GraphBatch([graph])
        graph_batch = graph_batch.to_device(model_device)

        # Process through GHN2's encoder
        with torch.no_grad():
            # Get node features using GHN2's embedding layer
            node_features = self.ghn2.embed(graph_batch.node_feat[:, 0])

            # Process through GatedGNN
            node_embeddings = self.ghn2.gnn(node_features, graph_batch.edges, graph_batch.node_feat[:, 1])

            # Apply layer normalization if available
            if hasattr(self.ghn2, 'ln'):
                node_embeddings = self.ghn2.ln(node_embeddings)

            # Pool node embeddings to get graph embedding
            arch_embedding = node_embeddings.mean(dim=0)

            # Project to the right dimensionality if needed
            if hasattr(self, 'arch_projector'):
                arch_embedding = self.arch_projector(arch_embedding)

        # Cache and return
        self.arch_embedding_cache[network_id] = arch_embedding
        return arch_embedding

    def extract_multi_scale_features(self, x):
        """Extract features from multiple layers of the backbone network."""
        # Forward through stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Extract features from different layers
        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)

        # Process through projection layers
        if hasattr(self, 'layer1_proj'):
            p1 = F.adaptive_avg_pool2d(self.layer1_proj(f1), 1).flatten(1)
            p2 = F.adaptive_avg_pool2d(self.layer2_proj(f2), 1).flatten(1)
            p3 = F.adaptive_avg_pool2d(self.layer3_proj(f3), 1).flatten(1)

            # Concatenate for a multi-scale representation
            combined = torch.cat([p1, p2, p3], dim=1)
        else:
            # If projection layers aren't available, use global pooling
            p1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)
            p2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
            p3 = F.adaptive_avg_pool2d(f3, 1).flatten(1)
            p4 = F.adaptive_avg_pool2d(f4, 1).flatten(1)

            combined = torch.cat([p1, p2, p3, p4], dim=1)

        return combined, f1, f2, f3, f4

    def set_adaptation_params(self, arch_embedding, task_embedding, temperature=1.0):
        """Use parameter generator to predict parameters for all adaptation modules."""
        # Generate parameters for each adaptation module
        params_layer1 = self.param_generator(arch_embedding, task_embedding, 64)
        params_layer2 = self.param_generator(arch_embedding, task_embedding, 128)
        params_layer3 = self.param_generator(arch_embedding, task_embedding, 256)
        params_layer4 = self.param_generator(arch_embedding, task_embedding, 512)

        # Optional temperature scaling
        if temperature != 1.0:
            for params in [params_layer1, params_layer2, params_layer3, params_layer4]:
                for k, v in params.items():
                    if 'weight' in k:
                        params[k] = v * temperature

        # Set parameters for each adaptation module
        self._set_module_params(self.adapt_layer1, params_layer1)
        self._set_module_params(self.adapt_layer2, params_layer2)
        self._set_module_params(self.adapt_layer3, params_layer3)
        self._set_module_params(self.adapt_layer4, params_layer4)

    def forward(self, query_images, arch_embedding=None, task_embedding=None, temperature=1.0):
        """Process query images with task-specific and architecture-aware adaptation."""
        if arch_embedding is not None and task_embedding is not None:
            # Use arch and task embeddings to predict adaptation module parameters
            self.set_adaptation_params(arch_embedding, task_embedding, temperature)

        # Apply backbone with adaptation modules
        x = self.backbone.conv1(query_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 with adaptation
        x = self.backbone.layer1(x)
        x = self.adapt_layer1(x, arch_embedding)

        # Layer 2 with adaptation
        x = self.backbone.layer2(x)
        x = self.adapt_layer2(x, arch_embedding)

        # Layer 3 with adaptation
        x = self.backbone.layer3(x)
        x = self.adapt_layer3(x, arch_embedding)

        # Layer 4 with adaptation
        x = self.backbone.layer4(x)
        x = self.adapt_layer4(x, arch_embedding)

        # Global pooling and classification
        x = F.adaptive_avg_pool2d(x, 1)
        features = torch.flatten(x, 1)
        return self.classifier(features)

    def normalize_parameters(self, params, param_type):
        """Normalize parameters based on their type for stable activation distributions."""
        if param_type == 'conv_weight':
            # Fan-in normalization
            fan_in = np.prod(params.shape[1:]) if params.dim() > 1 else 1.0
            return params * (2.0 / fan_in) ** 0.5
        elif param_type == 'bn_weight':
            # BN weights are typically around 1.0
            return 2 * torch.sigmoid(params / 1.0)
        elif param_type == 'ln_weight':
            # LN weights are similar to BN
            return 2 * torch.sigmoid(params / 1.0)
        elif param_type == 'bias':
            # Biases are typically small
            return 0.1 * torch.tanh(params / 0.5)
        elif param_type == 'fc_weight':
            # FC weights use similar normalization as conv weights
            if params.dim() > 1:
                fan_in = params.shape[1]
                return params * (2.0 / fan_in) ** 0.5
            else:
                # Handle the case where the parameter is 1D
                return 0.1 * torch.tanh(params / 0.5)  # Use similar scaling as biases
        else:
            # Default normalization for stability
            return params * 0.1


class AdaptationModule(nn.Module):
    def __init__(self, in_channels, reduction=4,arch_embed_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels // reduction

        # Channel attention path
        self.fc1 = nn.Linear(in_channels, self.mid_channels)
        self.fc2 = nn.Linear(self.mid_channels, in_channels)

        # Spatial attention path
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(in_channels)

        self.arch_projection = nn.Sequential(
            nn.Linear(arch_embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, arch_embedding=None):
        # Original input for residual connection
        identity = x

        # Channel attention
        y_channel = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        y_channel = F.relu(self.fc1(y_channel))
        y_channel = torch.sigmoid(self.fc2(y_channel))

        # Apply conditional scaling if architecture embedding is available
        if arch_embedding is not None:
            # Use architecture information to modulate channel attention
            if arch_embedding.dim() == 1:
                arch_embedding = arch_embedding.unsqueeze(0)
            scale_factors = self.arch_projection(arch_embedding)
            if scale_factors.dim() == 1:
                scale_factors = scale_factors.unsqueeze(0)
            if scale_factors.size(0) == 1 and y_channel.size(0) > 1:
                scale_factors = scale_factors.expand(y_channel.size(0), -1)
            y_channel = y_channel * scale_factors

        # Apply channel attention weights
        x_channel = x * y_channel.view(x.shape[0], -1, 1, 1)

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y_spatial = torch.cat([avg_pool, max_pool], dim=1)
        y_spatial = torch.sigmoid(self.conv_spatial(y_spatial))
        x_spatial = x * y_spatial

        # Combine attention mechanisms with residual connection
        x = identity + x_channel + x_spatial

        # Apply layer normalization (converted to the right shape)
        shape = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        return x
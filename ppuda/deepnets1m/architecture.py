from ppuda.deepnets1m.graph import Graph, GraphBatch
import torch
import numpy as np
import torch.nn as nn
from ppuda.ghn.gatedgnn import GatedGNN
import torch.nn.functional as F
PRIMITIVES_DEEPNETS1M = [
    'max_pool',
    'avg_pool',
    'sep_conv',
    'dil_conv',
    'conv',
    'msa',
    'cse',
    'sum',
    'concat',
    'input',
    'bias',
    'bn',
    'ln',
    'pos_enc',
    'glob_avg',
]

class ArchitectureGraphBuilder:
    """Builds computational graphs from neural networks."""

    def __init__(self, ve_cutoff=50):
        """
        Args:
            ve_cutoff: Maximum shortest path length for virtual edges
        """
        self.ve_cutoff = ve_cutoff
        self.primitives_dict = {op: i for i, op in enumerate(PRIMITIVES_DEEPNETS1M)}

    def build_graph(self, network):
        """
        Converts a neural network to a computational graph.
        """
        # Save states
        was_training = network.training

        # Ensure training mode for graph construction
        network.train()

        device = None
        for param in network.parameters():
            device = param.device
            break
        if device is None:
            # Fallback to CPU if no parameters found
            device = torch.device('cpu')

        # Create a context where gradients are enabled
        with torch.enable_grad():
            # Create a dummy input that requires grad
            dummy_input = torch.randn(2, 3, 32, 32, requires_grad=True,device=device)

            # Get output with grad tracking
            try:
                output = network(dummy_input)

                # Create the graph using this output which should have grad_fn
                graph = Graph(network, ve_cutoff=self.ve_cutoff)
                return graph
            finally:
                # Restore original state
                network.train(was_training)

    def create_graph_batch(self, networks):
        """
        Creates a batch of computational graphs for multiple networks.

        Args:
            networks: List of PyTorch neural networks

        Returns:
            GraphBatch object containing all network graphs
        """
        graphs = [self.build_graph(net) for net in networks]
        return GraphBatch(graphs)


class ArchitectureEncoder(nn.Module):
    """Encodes neural network architectures using graph neural networks."""

    def __init__(self, embedding_dim=128, hidden_dim=64, ve=True, layernorm=True):
        """
        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of GNN hidden states
            ve: Whether to use virtual edges
            layernorm: Whether to use layer normalization
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ve = ve
        self.layernorm = layernorm

        # Initial node embeddings
        self.node_embeddings = nn.Embedding(len(PRIMITIVES_DEEPNETS1M), embedding_dim)

        # GNN for processing the graph
        self.gnn = GatedGNN(in_features=embedding_dim, ve=ve)

        # Optional layer normalization
        if layernorm:
            self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, graph_batch):
        """
        Process a batch of architecture graphs.

        Args:
            graph_batch: GraphBatch object containing architecture graphs

        Returns:
            List of architecture embeddings (one per graph)
        """
        # Get initial node features
        node_features = self.node_embeddings(graph_batch.node_feat[:, 0])

        # Process with GNN
        node_embeddings = self.gnn(node_features, graph_batch.edges, graph_batch.node_feat[:, 1])

        # Apply layer normalization if enabled
        if self.layernorm:
            node_embeddings = self.ln(node_embeddings)

        # Pool node embeddings to get graph embeddings
        graph_embeddings = []
        start_idx = 0

        for n_nodes in graph_batch.n_nodes:
            # Average pooling over node embeddings for each graph
            graph_emb = node_embeddings[start_idx:start_idx + n_nodes].mean(dim=0)
            graph_embeddings.append(graph_emb)
            start_idx += n_nodes

        return graph_embeddings


class JointParameterGenerator(nn.Module):
    def __init__(self, arch_dim=128, task_dim=128, hidden_dim=256):
        super().__init__()
        # Normalization for embeddings
        self.arch_norm = nn.LayerNorm(arch_dim)
        self.task_norm = nn.LayerNorm(task_dim)

        # FiLM-style conditioning
        self.task_to_scale = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, arch_dim),
            nn.Sigmoid()
        )

        self.task_to_bias = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, arch_dim),
            nn.Tanh()
        )

        # Main processor
        self.joint_processor = nn.Sequential(
            nn.Linear(arch_dim + task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Decoders for different channel sizes
        self.decoders = nn.ModuleDict({
            '64': self._create_decoder(64, hidden_dim),
            '128': self._create_decoder(128, hidden_dim),
            '256': self._create_decoder(256, hidden_dim),
            '512': self._create_decoder(512, hidden_dim)
        })


    def _create_decoder(self, channels, hidden_dim):
        """Create decoder for a specific channel size."""
        reduction = 4
        mid_channels = channels // reduction

        return nn.ModuleDict({
            'fc1_weight': self._make_param_decoder(hidden_dim, (mid_channels, channels)),
            'fc1_bias': self._make_param_decoder(hidden_dim, (mid_channels,)),
            'fc2_weight': self._make_param_decoder(hidden_dim, (channels, mid_channels)),
            'fc2_bias': self._make_param_decoder(hidden_dim, (channels,)),
            'conv_spatial_weight': self._make_param_decoder(hidden_dim, (1, 2, 7, 7)),
            'conv_spatial_bias': self._make_param_decoder(hidden_dim, (1,))
        })

    def _make_param_decoder(self, in_dim, out_shape):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, np.prod(out_shape)),
            nn.LayerNorm(np.prod(out_shape))
        )

    def forward(self, arch_embedding, task_embedding, channel_size):
        # Normalize embeddings
        if arch_embedding.dim() == 1:
            arch_embedding = arch_embedding.unsqueeze(0)
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)

        arch_embedding = self.arch_norm(arch_embedding)
        task_embedding = self.task_norm(task_embedding)

        # Apply FiLM-style conditioning
        scale = self.task_to_scale(task_embedding)
        bias = self.task_to_bias(task_embedding)
        conditioned_arch = arch_embedding * scale + bias * 0.1

        # Combine embeddings
        joint_embedding = torch.cat([conditioned_arch, task_embedding], dim=1)

        # Process joint embedding
        processed_embedding = self.joint_processor(joint_embedding)
        processed_embedding = processed_embedding.squeeze(0)

        # Generate parameters using the appropriate decoder
        decoder = self.decoders[str(channel_size)]

        params = {}
        for key, module in decoder.items():
            params[key] = module(processed_embedding)

        return params


def robust_network_adaptation(network):
    """Comprehensive adaptation of ImageNet networks for CIFAR-100."""
    # Save original state
    was_training = network.training
    network.eval()

    # Create a dummy input to trace the network
    x = torch.randn(2, 3, 32, 32)

    # Handle ResNet family
    if hasattr(network, 'conv1'):
        # Replace first conv with smaller kernel and stride
        in_channels = network.conv1.in_channels
        out_channels = network.conv1.out_channels
        network.conv1 = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False)

        # Remove maxpool or replace with smaller pool
        network.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    # Handle DenseNet
    if hasattr(network, 'features'):
        # For DenseNet, the first conv is features[0]
        if isinstance(network.features[0], nn.Conv2d):
            in_channels = network.features[0].in_channels
            out_channels = network.features[0].out_channels
            network.features[0] = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=False)

        # Replace pooling layers
        for i, module in enumerate(network.features):
            if isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                if module.kernel_size > 2:
                    network.features[i] = nn.AvgPool2d(kernel_size=2, stride=module.stride, padding=0)

    # Handle MobileNetV2
    if hasattr(network, 'features') and len(list(network.features)) > 0:
        # Find the first strided convolution and reduce its stride
        for i, block in enumerate(network.features):
            if hasattr(block, 'conv'):
                if hasattr(block.conv, 'stride') and block.conv.stride == (2, 2):
                    # Found the first strided conv, reduce its stride
                    block.conv.stride = (1, 1)
                    break
            elif hasattr(block, 'stride') and block.stride == (2, 2):
                network.features[i].stride = (1, 1)
                break

    # Replace classification head with identity
    if hasattr(network, 'fc'):
        network.fc = nn.Identity()
    elif hasattr(network, 'classifier'):
        if isinstance(network.classifier, nn.Sequential):
            network.classifier[-1] = nn.Identity()
        else:
            network.classifier = nn.Identity()

    # Add multi-scale feature extraction capability
    network.get_multi_scale_features = lambda x: _get_multi_scale_features(network, x)

    # Restore original state
    network.train(was_training)

    return network


def _get_multi_scale_features(network, x):
    """Extract multi-scale features from network."""
    features = []

    # ResNet family
    if hasattr(network, 'layer1'):
        x = network.conv1(x)
        if hasattr(network, 'bn1'):
            x = network.bn1(x)
        x = network.relu(x)
        if hasattr(network, 'maxpool'):
            x = network.maxpool(x)

        f1 = network.layer1(x)
        features.append(F.adaptive_avg_pool2d(f1, 1))

        f2 = network.layer2(f1)
        features.append(F.adaptive_avg_pool2d(f2, 1))

        f3 = network.layer3(f2)
        features.append(F.adaptive_avg_pool2d(f3, 1))

    # DenseNet family
    elif hasattr(network, 'features'):
        # Extract features at 1/4, 1/2, and 3/4 of the network
        layer_count = len(list(network.features))
        checkpoints = [layer_count // 4, layer_count // 2, 3 * layer_count // 4]

        current_feat = x
        for i, module in enumerate(network.features):
            current_feat = module(current_feat)
            if i in checkpoints:
                features.append(F.adaptive_avg_pool2d(current_feat, 1))

    # Return concatenated features
    return torch.cat([f.view(f.size(0), -1) for f in features], dim=1)
from ppuda.deepnets1m.graph import Graph, GraphBatch
import torch
import numpy as np
import torch.nn as nn
from ppuda.ghn.gatedgnn import GatedGNN
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

        # Create a context where gradients are enabled
        with torch.enable_grad():
            # Create a dummy input that requires grad
            dummy_input = torch.randn(2, 3, 32, 32, requires_grad=True)

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
    """Generates parameters based on both architecture and task embeddings."""

    def __init__(self, arch_dim=128, task_dim=128, hidden_dim=256):
        """
        Args:
            arch_dim: Dimension of architecture embeddings
            task_dim: Dimension of task embeddings
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.arch_dim = arch_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim

        # Joint embedding processor
        self.joint_processor = nn.Sequential(
            nn.Linear(arch_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Parameter generators for different channel sizes
        self.decoders = nn.ModuleDict({
            '64': self._create_decoder(64),
            '128': self._create_decoder(128),
            '256': self._create_decoder(256),
            '512': self._create_decoder(512)
        })

    def _create_decoder(self, channels):
        """Create decoder for a specific channel size."""
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
                self.hidden_dim, self.hidden_dim, (mid_channels, channels)
            ),
            'fc1_bias': CustomDecoder(
                self.hidden_dim, self.hidden_dim // 2, (mid_channels,)
            ),
            'fc2_weight': CustomDecoder(
                self.hidden_dim, self.hidden_dim, (channels, mid_channels)
            ),
            'fc2_bias': CustomDecoder(
                self.hidden_dim, self.hidden_dim // 2, (channels,)
            )
        })

        return decoder

    def forward(self, arch_embedding, task_embedding, channel_size):
        """
        Generate parameters based on architecture and task embeddings.

        Args:
            arch_embedding: Architecture embedding tensor [D_arch]
            task_embedding: Task embedding tensor [D_task]
            channel_size: Number of channels (64, 128, 256, or 512)

        Returns:
            Dictionary of generated parameters
        """
        # Combine embeddings
        joint_embedding = torch.cat([arch_embedding, task_embedding], dim=0)

        # Process joint embedding
        processed_embedding = self.joint_processor(joint_embedding)

        # Generate parameters using the appropriate decoder
        decoder = self.decoders[str(channel_size)]

        params = {
            'fc1.weight': decoder['fc1_weight'](processed_embedding),
            'fc1.bias': decoder['fc1_bias'](processed_embedding),
            'fc2.weight': decoder['fc2_weight'](processed_embedding),
            'fc2.bias': decoder['fc2_bias'](processed_embedding)
        }

        return params

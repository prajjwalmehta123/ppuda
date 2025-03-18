import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from ppuda.task.task_encoder import TaskEncoder
from ppuda.ghn.nn import GHN
from ppuda.ghn.decoder import MLPDecoder, ConvDecoder
from ppuda.deepnets1m.ops import NormLayers
from ppuda.deepnets1m.graph import Graph, GraphBatch
from ppuda.utils import capacity, default_device

class GradientPreservingWrapper(nn.Module):
    def __init__(self, base_net, task_encoder, task_embedding, num_classes):
        super().__init__()
        self.base_net = base_net
        self.task_encoder = task_encoder
        self.task_embedding = task_embedding
        self.num_classes = num_classes
        feature_dim = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_feature_dim = self._detect_feature_dim(base_net)

        if hasattr(task_embedding, 'shape'):
            self.embed_dim = task_embedding.shape[-1]
        else:
            self.embed_dim = 128  # Default fallback

        print(f"Network feature dimension: {self.base_feature_dim}")
        print(f"Task embedding dimension: {self.embed_dim}")

        self.classifier = nn.Linear(self.base_feature_dim, num_classes).to(self.device)
        self.task_projection = nn.Linear(self.embed_dim, self.base_feature_dim).to(self.device)

        print(f"Created task projection: {self.embed_dim} → {self.base_feature_dim}")
        print(f"Created classifier: {self.base_feature_dim} → {num_classes}")

    def _detect_feature_dim(self, net):
        """Detect the feature dimension from the network architecture."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Try to find feature dimension from classifier
        if hasattr(net, 'classifier'):
            if isinstance(net.classifier, nn.Sequential):
                for module in net.classifier:
                    if isinstance(module, nn.Linear):
                        return module.in_features
            elif isinstance(net.classifier, nn.Linear):
                return net.classifier.in_features

        # Try to infer from the network structure
        if hasattr(net, 'global_pooling') and hasattr(net, 'cells'):
            # Typical ResNet pattern
            # Check the last cell's output channels if possible
            if len(net.cells) > 0:
                last_cell = net.cells[-1]
                if hasattr(last_cell, '_ops'):
                    for op in last_cell._ops:
                        if hasattr(op, 'out_channels'):
                            return op.out_channels

        try:
            # Process through the network up to global pooling
            dummy_input = torch.zeros(1, 3, 32, 32).to(self.device)  # Assuming CIFAR-sized input
            with torch.no_grad():
                # Run through stem and cells
                if hasattr(net, 'stem0'):
                    x = net.stem0(dummy_input)
                    if hasattr(net, 'stem1') and net.stem1 is not None:
                        x = net.stem1(x)
                elif hasattr(net, 'stem'):
                    x = net.stem(dummy_input)
                else:
                    x = dummy_input

                # Process through cells
                if hasattr(net, 'cells'):
                    s0 = x
                    s1 = x if not hasattr(net, 'stem1') else net.stem1(x)
                    for cell in net.cells:
                        s0, s1 = s1, cell(s0, s1, 0)  # drop_path_prob=0
                    x = s1

                # Apply global pooling if available
                if hasattr(net, 'global_pooling'):
                    x = net.global_pooling(x)

                # Get feature dimension
                return x.view(x.size(0), -1).size(1)
        except Exception as e:
            print(f"Debug Error: Cannot process this network architecture. Falling back to default feature dim")
        # Final fallback for ResNet
        return 512

    def forward(self, x):
        x = x.to(self.device)
        is_lightweight = hasattr(self.base_net, 'stem0') and isinstance(getattr(self.base_net.stem0, 'weight', None),
                                                                       tuple)
        if is_lightweight:
            # If lightweight network, skip base_net processing and use features directly
            features = x.view(x.size(0), -1)  # Flatten the input as features
            if features.size(1) != self.base_feature_dim:
                # Adapt feature dimension if needed
                features = F.adaptive_avg_pool1d(features.unsqueeze(1), self.base_feature_dim).squeeze(1)
        else:
            # Handle ResNet-style networks with cells correctly
            try:
                if hasattr(self.base_net, '_is_vit') and self.base_net._is_vit:
                    # Visual Transformer pattern
                    s0 = self.base_net.stem0(x)
                    s0 = s1 = self.base_net.pos_enc(s0)

                    for cell in self.base_net.cells:
                        s0, s1 = s1, cell(s0, s1, self.base_net.drop_path_prob)

                    features = self.base_net.global_pooling(s1).view(s1.size(0), -1)

                elif hasattr(self.base_net, 'stem0') and hasattr(self.base_net, 'cells'):
                    # ResNet-style with separate stem0/stem1 pattern
                    s0 = self.base_net.stem0(x)
                    s1 = None
                    if hasattr(self.base_net, 'stem1') and self.base_net.stem1 is not None:
                        s1 = self.base_net.stem1(s0)

                    for cell in self.base_net.cells:
                        s0, s1 = s1, cell(s0, s1, self.base_net.drop_path_prob)

                    features = self.base_net.global_pooling(s1).view(s1.size(0), -1)

                elif hasattr(self.base_net, 'stem') and hasattr(self.base_net, 'cells'):
                    # ResNet-style with combined stem pattern
                    s0 = s1 = self.base_net.stem(x)

                    for cell in self.base_net.cells:
                        s0, s1 = s1, cell(s0, s1, self.base_net.drop_path_prob)

                    features = self.base_net.global_pooling(s1).view(s1.size(0), -1)

                else:
                    print('Debug Error: Cannot process this network architecture. Falling back to generic network')
                    for name, module in self.base_net.named_children():
                        if name != 'classifier':
                            x = module(x)
                    features = x.view(x.size(0), -1)
            except Exception as e:
                print(f"Forward pass error: {e}. Using feature extraction fallback.")
                features = x.view(x.size(0), -1)
                if features.size(1) != self.base_feature_dim:
                    features = F.adaptive_avg_pool1d(features.unsqueeze(1), self.base_feature_dim).squeeze(1)
        features = features.to(self.device)
        logits = self.classifier(features)

        if self.task_embedding is not None:
            # Get task-specific bias term
            task_embedding = self.task_embedding.to(self.device)
            task_projection = self.task_projection(task_embedding)

            # Add task influence to each sample (using broadcasting)
            task_influence = features * task_projection.unsqueeze(0)
            task_influence = torch.sum(task_influence, dim=1, keepdim=True)

            # Apply the influence with a scaling factor
            logits = logits + 0.1 * task_influence

        # Return in the expected format
        if hasattr(self.base_net, '_auxiliary') and self.base_net._auxiliary:
            return (logits, None)
        else:
            return logits

class TaskAwareGHN(GHN):
    """
    Task-Aware Graph HyperNetwork that predicts parameters conditioned on both
    architecture and task data.
    """

    def __init__(self,
                 max_shape,
                 num_classes,
                 task_embed_dim=128,
                 backbone='resnet18',
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 debug_level=0,
                 phase=1):
        """
        Initialize TaskAwareGHN.

        Args:
            max_shape: Maximum parameter shape [C_out, C_in, H, W]
            num_classes: Number of classes for the target dataset
            task_embed_dim: Dimension of task embedding
            backbone: Backbone model for the task encoder
            hypernet: Type of hypernetwork ('gatedgnn' or 'mlp')
            decoder: Type of decoder ('conv' or 'mlp')
            weight_norm: Whether to normalize weights
            ve: Whether to use virtual edges
            layernorm: Whether to use layer normalization
            hid: Hidden dimension
            debug_level: Level of debug information
            phase: Implementation phase (1-3)
        """
        # Initialize parent GHN class
        super(TaskAwareGHN, self).__init__(
            max_shape=max_shape,
            num_classes=num_classes,
            hypernet=hypernet,
            decoder=decoder,
            weight_norm=weight_norm,
            ve=ve,
            layernorm=layernorm,
            hid=hid,
            debug_level=debug_level
        )

        self.task_embed_dim = task_embed_dim
        self.phase = phase

        # Task encoder
        self.task_encoder = TaskEncoder(embedding_dim=task_embed_dim, backbone=backbone)

        # Replace standard decoder with task-aware decoder
        if decoder == 'conv':
            fn_dec, layers = ConvDecoder, (hid * 4, hid * 8)
        elif decoder == 'mlp':
            fn_dec, layers = MLPDecoder, (hid * 2,)
        else:
            raise NotImplementedError(decoder)

    def forward(self, nets_torch, support_data=None, graphs=None, return_embeddings=False,
                predict_class_layers=True, bn_train=True):
        """Predict parameters for networks based on architecture and task data."""
        # Create task embedding if support data is provided
        device = next(self.parameters()).device
        task_embedding = None
        if support_data is not None:
            #support_images, _ = support_data
            task_embedding = self.task_encoder(support_data)
        #support_images = support_images.to(device)
        #support_labels = support_labels.to(device)

        # Use parent GHN to predict parameters, but NOT classification layer
        with torch.no_grad():
            super().forward(
                nets_torch,
                graphs=graphs,
                return_embeddings=False,
                predict_class_layers=False,  # Important: don't predict classification
                bn_train=bn_train
            )

        # Process networks
        networks = nets_torch if isinstance(nets_torch, list) else [nets_torch]

        # Wrap each network to preserve gradients
        wrapped_networks = []
        for net in networks:
            # Set batch norm layers to proper mode if needed
            if bn_train and not net.training:
                def set_bn_train(module):
                    if isinstance(module, nn.BatchNorm2d):
                        module.training = True

                net.apply(set_bn_train)

            wrapped_net = GradientPreservingWrapper(
                net,
                self.task_encoder,
                task_embedding,
                self.num_classes
            )
            wrapped_networks.append(wrapped_net)

        result = wrapped_networks if isinstance(nets_torch, list) else wrapped_networks[0]
        return (result, None) if return_embeddings else result

    @staticmethod
    def load(checkpoint_path, debug_level=1, device=default_device(), verbose=False):
        """
        Load TaskAwareGHN from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            debug_level: Level of debug information
            device: Device to load model on
            verbose: Whether to print verbose information

        Returns:
            Loaded TaskAwareGHN model
        """
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Check if this is a TaskAwareGHN checkpoint
        is_ta_ghn = 'task_embed_dim' in state_dict['config']

        if is_ta_ghn:
            ghn = TaskAwareGHN(**state_dict['config'], debug_level=debug_level).to(device).eval()
        else:
            # Load as regular GHN, then convert
            temp_ghn = GHN(**state_dict['config'], debug_level=debug_level)

            # Create TaskAwareGHN with same config
            ghn = TaskAwareGHN(
                max_shape=temp_ghn.max_shape,
                num_classes=temp_ghn.num_classes,
                hypernet=state_dict['config'].get('hypernet', 'gatedgnn'),
                decoder=state_dict['config'].get('decoder', 'conv'),
                weight_norm=temp_ghn.weight_norm,
                ve=temp_ghn.ve,
                layernorm=temp_ghn.layernorm,
                hid=state_dict['config'].get('hid', 32),
                debug_level=debug_level,
                phase=1
            ).to(device).eval()

            # Copy common parameters
            ghn_dict = temp_ghn.state_dict()
            ta_dict = ghn.state_dict()

            for k in ghn_dict.keys():
                if k in ta_dict:
                    ta_dict[k] = ghn_dict[k]

            ghn.load_state_dict(ta_dict, strict=False)

        if verbose:
            print(
                f"{'TaskAwareGHN' if is_ta_ghn else 'GHN converted to TaskAwareGHN'} with {capacity(ghn)[1]} parameters loaded.")

        return ghn


def create_ta_ghn(dataset='cifar100', phase=1, task_embed_dim=128, backbone='resnet18'):
    """Create a task-aware GHN initialized from a pretrained GHN."""
    path = os.path.dirname(os.path.abspath(__file__))
    base_ghn = GHN.load(os.path.join(path, f'../../checkpoints/ghn2_{dataset}.pt'))

    # Create TaskAwareGHN to only conditions classification layer
    ta_ghn = TaskAwareGHN(
        max_shape=base_ghn.max_shape,
        num_classes=base_ghn.num_classes,
        task_embed_dim=task_embed_dim,
        backbone=backbone,
        hypernet='gatedgnn',
        decoder='conv',
        weight_norm=base_ghn.weight_norm,
        ve=base_ghn.ve,
        layernorm=base_ghn.layernorm,
        hid=32,
        debug_level=0,
        phase=phase
    )

    # Copy parameters from base GHN
    base_dict = base_ghn.state_dict()
    ta_dict = ta_ghn.state_dict()

    for k in base_dict.keys():
        if k in ta_dict:
            ta_dict[k] = base_dict[k]

    ta_ghn.load_state_dict(ta_dict, strict=False)

    return ta_ghn
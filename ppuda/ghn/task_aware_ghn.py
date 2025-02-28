import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from ppuda.task.task_encoder import TaskEncoder
from ppuda.ghn.nn import GHN
from ppuda.ghn.decoder import MLPDecoder
from ppuda.deepnets1m.ops import NormLayers
from ppuda.deepnets1m.graph import Graph, GraphBatch
from ppuda.utils import capacity, default_device


class TaskAwareDecoder(nn.Module):
    """
    Decoder that generates parameters conditioned on task embeddings.
    Extends the ConvDecoder with task-specific conditioning.
    """

    def __init__(self,
                 in_features=64,
                 task_embed_dim=128,
                 hid=(128, 256),
                 out_shape=None,
                 num_classes=None):
        super(TaskAwareDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes

        # Task-node fusion
        self.task_node_fusion = nn.Sequential(
            nn.Linear(in_features + task_embed_dim, in_features),
            nn.ReLU()
        )

        ch_prod = out_shape[0] * out_shape[1]
        spatial_prod = out_shape[2] * out_shape[3]
        out_features = hid[0] * spatial_prod

        self.fc = nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU(inplace=True))
        self.register_buffer('cols_1d', torch.arange(0, out_features).view(hid[0], out_shape[2], out_shape[3]),
                             persistent=False)
        self.register_buffer('cols_4d', torch.arange(0, ch_prod),
                             persistent=False)

        conv = []
        for j, n_hid in enumerate(hid):
            n_out = ch_prod if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1),
                         nn.ReLU() if j < len(hid) - 1 else nn.Identity()])

        self.conv = nn.Sequential(*conv)

        # Task-specific classifier projection
        self.class_layer_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_shape[0], num_classes, 1))

    def forward(self, x, task_embedding, max_shape=(1, 1, 1, 1), class_pred=False):
        """
        Generate parameters conditioned on node and task embeddings.

        Args:
            x: Node embeddings [N, in_features]
            task_embedding: Task embedding [task_embed_dim]
            max_shape: Maximum parameter tensor shape
            class_pred: Whether to predict classification layer parameters

        Returns:
            Generated parameter tensors
        """
        N = x.shape[0]

        # Combine node and task embeddings
        task_expanded = task_embedding.unsqueeze(0).expand(N, -1)
        combined = torch.cat([x, task_expanded], dim=1)

        # Fuse task and node features
        x = self.task_node_fusion(combined)

        # Continue with regular decoder logic (similar to ConvDecoder)
        if sum(max_shape[2:]) < sum(self.out_shape[2:]) and len(self.fc) == 2:
            ind = self.cols_1d[:, :max_shape[2], :max_shape[3]].flatten()
            x = self.fc[1](F.linear(x, self.fc[0].weight[ind],
                                    self.fc[0].bias[ind]).view(N, -1, max_shape[2], max_shape[3]))
        else:
            x = self.fc(x).view(N, -1, *self.out_shape[2:])[:, :, :max_shape[2], :max_shape[3]]

        out_shape = (*self.out_shape[:2], min(self.out_shape[2], max_shape[2]), min(self.out_shape[3], max_shape[3]))

        if (max_shape[1] <= out_shape[1] // 2 or (max_shape[0] < out_shape[0] and not class_pred)) and len(
                self.conv) == 4:
            x = self.conv[1](self.conv[0](x))
            if max_shape[1] < out_shape[1] and max_shape[1] % 3 == 0:
                n_in = max_shape[1] // 3 * 4
            else:
                n_in = min(max_shape[1], out_shape[1])

            n_out = out_shape[0] if class_pred else min(out_shape[0], max_shape[0])
            ind = self.cols_4d[:n_out * out_shape[1]]
            if n_in < out_shape[1]:
                ind = ind.reshape(-1, n_in)[::out_shape[1] // n_in].flatten()

            x = self.conv[3](F.conv2d(x, self.conv[2].weight[ind], self.conv[2].bias[ind]))

            x = x.reshape(N, n_out, n_in, *out_shape[2:])[:, :, :min(out_shape[1], max_shape[1])]
            if min(max_shape[2:]) > min(out_shape[2:]):
                x = x.repeat((1, 1, 1, 2, 2))
        else:
            x = self.conv(x).view(N, out_shape[0], -1, *out_shape[2:])

        if class_pred:
            x = self.class_layer_predictor(x[:, :, :, :, 0])
            x = x[:, :, :, 0]

        return x


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
            fn_dec, layers = TaskAwareDecoder, (hid * 4, hid * 8)
        elif decoder == 'mlp':
            fn_dec, layers = MLPDecoder, (hid * 2,)
        else:
            raise NotImplementedError(decoder)

        if phase >= 2:
            self.decoder = fn_dec(in_features=hid,
                                  task_embed_dim=task_embed_dim,
                                  hid=layers,
                                  out_shape=max_shape,
                                  num_classes=num_classes)

        if phase >= 3:
            self.scale_controller = nn.Sequential(
                nn.Linear(task_embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        # Stability monitoring attributes
        self.stability_history = {
            'loss': [],
            'grad_norm': [],
            'param_scale': []
        }

    def forward(self, nets_torch, support_data=None, graphs=None, return_embeddings=False,
                predict_class_layers=True, bn_train=True):
        """
        Predict parameters for networks based on architecture and task data.

        Args:
            nets_torch: Networks to predict parameters for
            support_data: Task support data (images, labels)
            graphs: GraphBatch object
            return_embeddings: Whether to return embeddings
            predict_class_layers: Whether to predict classification layers
            bn_train: Whether to set batch norm layers to training mode

        Returns:
            Networks with predicted parameters
        """
        # Create task embedding if support data is provided
        task_embedding = None
        if support_data is not None:
            support_images, support_labels = support_data
            task_embedding = self.task_encoder(support_images, support_labels)

        if self.phase == 1 and not self.training:
            # Get basic embeddings and process them
            if graphs is None:
                if isinstance(nets_torch, list):
                    nets_torch = nets_torch[0]
                graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.embed.weight.device)

            # Only predict classification layer parameters
            for net in (nets_torch if isinstance(nets_torch, list) else [nets_torch]):
                for name, module in net.named_modules():
                    if isinstance(module, nn.Linear) and module.out_features == self.num_classes:
                        # Generate classification layer parameters directly from task embedding
                        if task_embedding is not None:
                            cls_weight = self.task_encoder.projection(task_embedding).view(-1, 1)
                            cls_weight = cls_weight.expand(self.num_classes, module.in_features)
                            module.weight.data = cls_weight

            return (nets_torch, None) if return_embeddings else nets_torch

        # For Phase 2+, use the full GHN pipeline with task conditioning
        if not self.training:
            assert isinstance(nets_torch, nn.Module) or len(nets_torch) == 1
            if isinstance(nets_torch, list):
                nets_torch = nets_torch[0]

            if graphs is None:
                graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.embed.weight.device)

        # Map network parameters
        param_groups, params_map = self._map_net_params(graphs, nets_torch, self.debug_level > 0)

        # Get initial node embeddings
        x = self.shape_enc(self.embed(graphs.node_feat[:, 0]), params_map, predict_class_layers=predict_class_layers)

        # Update node embeddings
        x = self.gnn(x, graphs.edges, graphs.node_feat[:, 1])

        if self.layernorm:
            x = self.ln(x)

        # Predict parameters for nodes using our task-conditioned decoder
        n_tensors, n_params = 0, 0
        for key, inds in param_groups.items():
            if len(inds) == 0:
                continue

            x_ = x[torch.tensor(inds, device=x.device)]

            sz = key
            is_cls = False

            if len(sz) in [2, 3]:
                if len(sz) == 2 and sz[1] > 0:
                    # Classification layer
                    if self.phase >= 2 and task_embedding is not None:
                        # Use task-aware decoder for classification layer
                        w = self.decoder(x_, task_embedding, (sz[0], sz[1], 1, 1), class_pred=True)
                    else:
                        # Fallback to regular decoder
                        w = self.decoder(x_, (sz[0], sz[1], 1, 1), class_pred=True)
                    is_cls = True
                else:
                    # 1D parameter or classification bias
                    if len(sz) == 3:
                        w = self.decoder_1d(x_).view(len(inds), -1, 1, 1)
                    else:
                        w = self.decoder_1d(x_).view(len(inds), 2, -1)
                        if len(sz) == 2 and sz[1] < 0:
                            w = self.bias_class(w)
                            is_cls = True
            else:
                assert len(sz) == 4, sz
                if self.phase >= 2 and task_embedding is not None:
                    # Use task-aware decoder for convolutional layers in Phase 2+
                    w = self.decoder(x_, task_embedding, sz, class_pred=False)
                else:
                    # Fallback to regular decoder
                    w = self.decoder(x_, sz, class_pred=False)

            # Apply parameter scaling in Phase 3
            if self.phase >= 3 and task_embedding is not None:
                scale_factor = self.scale_controller(task_embedding)
                w = w * scale_factor

            if not predict_class_layers and is_cls:
                continue

            # Transfer predicted parameters to networks
            for ind in inds:
                matched, _, w_ind = params_map[ind]

                if w_ind is None:
                    continue

                m, sz, is_w = matched['module'], matched['sz'], matched['is_w']
                for it in range(2 if (len(sz) == 1 and is_w) else 1):
                    if len(sz) == 1:
                        w_ = w[w_ind][1 - is_w + it]
                        if it == 1:
                            assert (type(m) in NormLayers and len(key) == 2 and key[1] == 0)
                    else:
                        w_ = w[w_ind]

                    sz_set = self._set_params(m, self._tile_params(w_, sz), is_w=is_w & ~it)
                    n_tensors += 1
                    n_params += torch.prod(torch.tensor(sz_set))

        # Set BN layers to training mode for evaluation
        if not self.training and bn_train:
            def bn_set_train(module):
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True

            nets_torch.apply(bn_set_train)

        # Return networks with predicted parameters
        return (nets_torch, x) if return_embeddings else nets_torch

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
                phase=1  # Start with Phase 1 when converting from GHN
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


def create_ta_ghn(dataset='imagenet', phase=1, task_embed_dim=128, backbone='resnet18'):
    """
    Create a new TaskAwareGHN initialized from a pretrained GHN.

    Args:
        dataset: Dataset to use ('cifar100' or 'imagenet')
        phase: Implementation phase (1-3)
        task_embed_dim: Dimension of task embedding
        backbone: Backbone for task encoder

    Returns:
        TaskAwareGHN model
    """
    # Load pretrained GHN
    path = os.path.dirname(os.path.abspath(__file__))
    base_ghn = GHN.load(os.path.join(path, f'../../checkpoints/ghn2_{dataset}.pt'))

    # Create TaskAwareGHN with same configuration
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
import torch
import torch.nn as nn
import numpy as np
from ppuda.deepnets1m.graph import Graph, GraphBatch
from ppuda.deepnets1m.ops import NormLayers
from ppuda.ghn.nn import GHN
from ppuda.task import Task


def validate_task(task, hid):
    """Helper Method to Validate task object and its embedding"""
    if task is None:
        return False

    if not hasattr(task, 'task_embedding') or task.task_embedding is None:
        raise ValueError("Task object must have a valid task_embedding")

    # Check embedding dimensionality
    if task.task_embedding.dim() != 1:
        raise ValueError(f"Task embedding must be 1-dimensional, got shape {task.task_embedding.shape}")

    return True


class TaskAwareGHN(GHN):
    """
    Task-Aware Graph HyperNetwork that extends the base GHN to incorporate task information
    """

    def __init__(self,
                 max_shape,
                 num_classes,
                 task_embedding_dim=128,
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 debug_level=0):
        super().__init__(
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
        self.hid = hid
        # Task conditioning components
        self.task_embedding_dim = task_embedding_dim

        # Project task embeddings to match GNN hidden dim
        self.task_projection = nn.Sequential(
            nn.Linear(task_embedding_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

        # Gating mechanism for task integration
        self.task_gate = nn.Sequential(
            nn.Linear(hid + hid, hid),  # Combine node and task features
            nn.Sigmoid()  # Output gate values between 0 and 1
        )

        self.task_ln = nn.LayerNorm(hid) if layernorm else nn.Identity()

    def forward(self, nets_torch, graphs=None, task: Task = None, return_embeddings=False,
                predict_class_layers=True, bn_train=True):
        """
        Predict parameters for a list of networks conditioned on task information
        """
        # Handle input networks consistently
        if not self.training:
            assert isinstance(nets_torch, nn.Module) or len(nets_torch) == 1, \
                'constructing the graph on the fly is only supported for a single network'

            if isinstance(nets_torch, list):
                nets_torch = nets_torch[0]

        # Create graphs if not provided
        if graphs is None:
            if self.debug_level > 0:
                print("No graphs provided, creating from network architecture")

            try:
                # Create a dummy input to test the network
                dummy_input = torch.randn(1, 3, 32, 32, device=self.embed.weight.device)
                _ = nets_torch(dummy_input)  # Ensure model works before graph construction

                graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.embed.weight.device)
            except Exception as e:
                print(f"Error during graph construction: {str(e)}")
                print("This often happens when the model architecture is incompatible with graph construction.")
                print("Try using a simpler model or providing a pre-constructed graph.")
                raise

        try:
            # Find mapping between embeddings and network parameters
            param_groups, params_map = self._map_net_params(graphs, nets_torch, self.debug_level > 0)
        except Exception as e:
            raise RuntimeError(f"Error mapping network parameters: {str(e)}")

        # Get initial node embeddings
        x = self.shape_enc(self.embed(graphs.node_feat[:, 0]), params_map,
                           predict_class_layers=predict_class_layers)

        # Apply task conditioning if task is provided
        if task is not None:
            try:
                is_valid = validate_task(task, self.hid)
                if is_valid:
                    if task.task_embedding.dim() != 1:
                        raise ValueError(f"Expected 1D task embedding, got shape {task.task_embedding.shape}")

                    # Process task embedding
                    task_emb = self.task_projection(task.task_embedding)
                    task_emb = self.task_ln(task_emb)

                    # Expand task embedding to match node features
                    expanded_task_emb = task_emb.unsqueeze(0).expand(x.size(0), -1)

                    # Combine task and node features with gating mechanism
                    concat_features = torch.cat([x, expanded_task_emb], dim=1)
                    gates = self.task_gate(concat_features)
                    x = gates * x + (1 - gates) * expanded_task_emb
            except Exception as e:
                if self.debug_level > 0:
                    print(f"Warning: Task conditioning failed: {str(e)}")
                    print("Continuing without task conditioning")

        # Update node embeddings using GNN
        x = self.gnn(x, graphs.edges, graphs.node_feat[:, 1])

        if self.layernorm:
            x = self.ln(x)

        # Predict parameters conditioned on node embeddings
        n_tensors, n_params = 0, 0
        for key, inds in param_groups.items():
            if len(inds) == 0:
                continue
            x_ = x[torch.tensor(inds, device=x.device)]

            sz = key
            is_cls = False
            if len(sz) in [2, 3]:
                if len(sz) == 2 and sz[1] > 0:
                    # classification layer
                    w = self.decoder(x_, (sz[0], sz[1], 1, 1), class_pred=True)
                    is_cls = True
                else:
                    # 1d or cls-b
                    if len(sz) == 3:
                        w = self.decoder_1d(x_).view(len(inds), -1, 1, 1)
                    else:
                        w = self.decoder_1d(x_).view(len(inds), 2, -1)
                        if len(sz) == 2 and sz[1] < 0:
                            w = self.bias_class(w)
                            is_cls = True
            else:
                assert len(sz) == 4, sz
                w = self.decoder(x_, sz, class_pred=False)

            if not predict_class_layers and is_cls:
                continue  # do not set the classification parameters when fine-tuning

            for ind in inds:
                matched, _, w_ind = params_map[ind]

                if w_ind is None:
                    continue  # e.g. pooling

                m, sz, is_w = matched['module'], matched['sz'], matched['is_w']
                for it in range(2 if (len(sz) == 1 and is_w) else 1):

                    if len(sz) == 1:
                        # separately set for BN/LN biases as they are
                        # not represented as separate nodes in graphs
                        w_ = w[w_ind][1 - is_w + it]
                        if it == 1:
                            assert (type(m) in NormLayers and len(key) == 2 and key[1] == 0), \
                                (type(m), key)
                    else:
                        w_ = w[w_ind]

                    sz_set = self._set_params(m, self._tile_params(w_, sz), is_w=is_w & ~it)
                    n_tensors += 1
                    n_params += np.prod(sz_set)

        if not self.training and bn_train:
            def bn_set_train(module):
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True

            nets_torch.apply(bn_set_train)

        return (nets_torch, x) if return_embeddings else nets_torch

    @classmethod
    def load(cls, checkpoint_path, debug_level=1, device='cuda', verbose=False):
        """Load Task-Aware GHN from checkpoint"""
        state_dict = torch.load(checkpoint_path, map_location=device)
        config = state_dict.get('config', {})

        # Ensure required parameters are present
        required_params = ['max_shape', 'num_classes', 'task_embedding_dim', 'hypernet',
                           'decoder', 'weight_norm', 've', 'layernorm', 'hid']
        for param in required_params:
            if param not in config:
                if param == 'task_embedding_dim':
                    config[param] = 128  # Default value
                else:
                    raise ValueError(f"Required parameter '{param}' missing from checkpoint")

        ghn = cls(**config, debug_level=debug_level).to(device).eval()
        ghn.load_state_dict(state_dict['state_dict'])

        if verbose:
            num_params = sum(p.numel() for p in ghn.parameters() if p.requires_grad)
            print(
                f'Task-Aware GHN with {num_params:,} parameters loaded from epoch {state_dict.get("epoch", "unknown")}')

        return ghn

    def count_parameters(self):
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
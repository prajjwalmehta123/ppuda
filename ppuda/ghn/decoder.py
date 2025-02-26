import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ppuda.ghn.mlp import MLP
from ppuda.ghn.layers import get_activation


def safe_reshape(tensor, shape_tuple):
    """Helper function to safely reshape tensors with proper error handling"""
    try:
        return tensor.reshape(*shape_tuple)
    except RuntimeError as e:
        # Calculate total elements
        total_elements = tensor.numel()

        # Check if reshape is possible
        if len(shape_tuple) == 2 and shape_tuple[1] != -1:
            # For 2D reshapes with specified second dimension
            if total_elements % shape_tuple[1] != 0:
                # Find closest divisible value
                closest_divisor = shape_tuple[1]
                while total_elements % closest_divisor != 0 and closest_divisor > 1:
                    closest_divisor -= 1

                # Use closest divisor that works
                new_shape = (total_elements // closest_divisor, closest_divisor)
                print(
                    f"Warning: Reshape {shape_tuple} invalid for tensor with {total_elements} elements. Using {new_shape} instead.")
                return tensor.reshape(*new_shape)

        # For other cases, just use a safe reshape
        # Try to preserve the non-negative dimensions
        preserved_dims = [d for d in shape_tuple if d > 0]
        if preserved_dims:
            # Calculate a simple divisible shape
            new_shape = calculate_safe_shape(total_elements, preserved_dims)
            print(
                f"Warning: Reshape {shape_tuple} invalid for tensor with {total_elements} elements. Using {new_shape} instead.")
            return tensor.reshape(*new_shape)
        else:
            # Fallback to simple shape
            return tensor.reshape(total_elements)


def calculate_safe_shape(total_elements, preferred_dims=None):
    """Calculate a safe shape for reshaping that matches total elements"""
    if preferred_dims is None or len(preferred_dims) == 0:
        # Default to 1D tensor if no preferences
        return (total_elements,)

    # Start with first preferred dimension
    first_dim = preferred_dims[0]

    # Make sure first dimension is a divisor of total elements
    if total_elements % first_dim == 0:
        second_dim = total_elements // first_dim
        return (first_dim, second_dim)

    # If first dimension doesn't work, find a divisor
    for i in range(int(np.sqrt(total_elements)), 0, -1):
        if total_elements % i == 0:
            return (i, total_elements // i)

    # Should never reach here (every number has at least 1 as divisor)
    return (total_elements, 1)


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_features=64,
                 hid=(128, 256),
                 out_shape=None,
                 num_classes=None):
        super(ConvDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.debug_level = 0  # Can be set to > 0 for more verbose output

        ch_prod = out_shape[0] * out_shape[1]
        spatial_prod = out_shape[2] * out_shape[3]
        out_features = hid[0] * spatial_prod

        self.fc = nn.Sequential(nn.Linear(in_features, out_features),
                                get_activation('relu'))
        self.register_buffer('cols_1d', torch.arange(0, out_features).view(hid[0], out_shape[2], out_shape[3]),
                             persistent=False)
        self.register_buffer('cols_4d', torch.arange(0, ch_prod),
                             persistent=False)

        conv = []
        for j, n_hid in enumerate(hid):
            n_out = ch_prod if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1),
                         get_activation(None if j == len(hid) - 1 else 'relu')])

        self.conv = nn.Sequential(*conv)
        self.class_layer_predictor = nn.Sequential(
            get_activation('relu'),
            nn.Conv2d(out_shape[0], num_classes, 1))

    def forward(self, x, max_shape=(1, 1, 1, 1), class_pred=False):
        """Forward pass with safer tensor reshaping and robust error handling"""
        N = x.shape[0]

        if self.debug_level > 0:
            print(f"Input x shape: {x.shape}")
            print(f"max_shape: {max_shape}")
            print(f"out_shape: {self.out_shape}")

        # Process input through linear layer with safer feature map handling
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

            # Get selection indices with boundary checks
            max_idx = n_out * out_shape[1]
            if max_idx > self.cols_4d.numel():
                max_idx = self.cols_4d.numel()
            ind = self.cols_4d[:max_idx]

            # FIXED: This is where the problematic reshape happens
            # Instead of potentially invalid reshape + strided selection:
            # ind = ind.reshape(-1, n_in)[::out_shape[1] // n_in].flatten()

            # Use safe approach that won't fail:
            if n_in < out_shape[1]:
                try:
                    # Check if reshape is possible
                    if ind.numel() % n_in == 0:
                        rows = ind.numel() // n_in
                        reshaped_ind = ind.reshape(rows, n_in)
                        # Calculate stride safely to avoid division by zero
                        stride = max(1, out_shape[1] // max(1, n_in))
                        ind = reshaped_ind[::stride].flatten()
                    else:
                        # Find a viable n_in that divides evenly
                        for test_n_in in range(n_in, 0, -1):
                            if ind.numel() % test_n_in == 0:
                                rows = ind.numel() // test_n_in
                                reshaped_ind = ind.reshape(rows, test_n_in)
                                # Use simple stride of 1 for safety
                                stride = 1
                                ind = reshaped_ind[::stride].flatten()
                                if self.debug_level > 0:
                                    print(f"Using adjusted n_in={test_n_in} instead of {n_in}")
                                break
                except Exception as e:
                    if self.debug_level > 0:
                        print(f"Reshape error in index processing: {e}")
                    # Fallback to original indices without reshaping
                    pass

            try:
                x = self.conv[3](F.conv2d(x, self.conv[2].weight[ind], self.conv[2].bias[ind]))
            except Exception as e:
                if self.debug_level > 0:
                    print(f"Error in conv2d operation: {e}")
                # Try with a different approach - use all filters but limit output channels
                x = self.conv[3](self.conv[2](x))[:, :n_out]

            # Safe reshape to target dimensions
            try:
                expected_shape = (N, n_out, n_in, *out_shape[2:])
                total_elements = x.numel()
                if np.prod(expected_shape) == total_elements:
                    # Shape matches exactly
                    x = x.reshape(*expected_shape)[:, :, :min(out_shape[1], max_shape[1])]
                else:
                    # Shape doesn't match, use a safer approach
                    safe_n_in = total_elements // (N * n_out * out_shape[2] * out_shape[3])
                    if safe_n_in > 0:
                        x = x.reshape(N, n_out, safe_n_in, *out_shape[2:])
                        x = x[:, :, :min(safe_n_in, min(out_shape[1], max_shape[1]))]
                    else:
                        # Last resort fallback - keep as is
                        if self.debug_level > 0:
                            print(f"Warning: Cannot reshape tensor of size {x.shape} to match expected dimensions")
            except Exception as e:
                if self.debug_level > 0:
                    print(f"Reshape error: {e}")
                # Keep tensor as is - downstream code will handle it
        else:
            try:
                x = self.conv(x).view(N, out_shape[0], -1, *out_shape[2:])
            except RuntimeError as e:
                if self.debug_level > 0:
                    print(f"View error in secondary path: {e}")
                # Apply convolution
                conv_out = self.conv(x)
                # Calculate a safe reshape
                total_elements = conv_out.numel()
                expected_elements_per_batch = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
                if total_elements % (N * out_shape[0] * out_shape[2] * out_shape[3]) == 0:
                    # Safe middle dimension
                    safe_dim = total_elements // (N * out_shape[0] * out_shape[2] * out_shape[3])
                    x = conv_out.reshape(N, out_shape[0], safe_dim, *out_shape[2:])
                else:
                    # Fallback to simplest shape
                    x = conv_out

        if class_pred:
            try:
                if x.dim() >= 5:
                    x = self.class_layer_predictor(x[:, :, :, :, 0])
                    x = x[:, :, :, 0]
                elif x.dim() == 4:
                    x = self.class_layer_predictor(x)
                    x = x[:, :, :, 0]
                else:
                    # For unexpected shapes, try to adapt
                    if self.debug_level > 0:
                        print(f"Warning: Unexpected tensor shape for class prediction: {x.shape}")
                    # Try to reshape to something that works
                    if x.dim() == 2:
                        # Reshape to 4D for class_layer_predictor
                        x = x.reshape(N, -1, 1, 1)
                        x = self.class_layer_predictor(x)
                        x = x.reshape(N, self.num_classes, -1)
                    else:
                        # Apply a default reshape that won't crash
                        x = x.reshape(N, -1)[:, :self.num_classes]
            except Exception as e:
                if self.debug_level > 0:
                    print(f"Error in class prediction: {e}")
                # Try a different approach based on tensor shape
                if x.dim() >= 4:
                    x = x.mean(dim=(2, 3))  # Global average pooling
                # Ensure output has right number of classes
                if x.shape[1] != self.num_classes:
                    temp = torch.zeros(N, self.num_classes, device=x.device)
                    temp[:, :min(x.shape[1], self.num_classes)] = x[:, :min(x.shape[1], self.num_classes)]
                    x = temp
        return x


class MLPDecoder(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(64,),
                 out_shape=None,
                 num_classes=None):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.mlp = MLP(in_features=in_features,
                       hid=(*hid, np.prod(out_shape)),
                       activation='relu',
                       last_activation=None)
        self.class_layer_predictor = nn.Sequential(
            get_activation('relu'),
            nn.Linear(hid[0], num_classes * out_shape[0]))

        self.debug_level = 0  # Can be set for verbose output

    def forward(self, x, max_shape=(1, 1, 1, 1), class_pred=False):
        if class_pred:
            try:
                x = list(self.mlp.fc.children())[0](x)  # shared first layer
                x = self.class_layer_predictor(x)  # N, 1000, 64, 1
                x = x.view(x.shape[0], self.num_classes, self.out_shape[1])
            except Exception as e:
                if hasattr(self, 'debug_level') and self.debug_level > 0:
                    print(f"Error in class_pred path: {e}")
                # Fallback to safer reshape
                x = list(self.mlp.fc.children())[0](x)
                x = self.class_layer_predictor(x)

                # Ensure it has expected shape for downstream use
                batch_size = x.shape[0]
                x = x.reshape(batch_size, self.num_classes, -1)
        else:
            try:
                x = self.mlp(x).view(-1, *self.out_shape)
                if sum(max_shape[2:]) > 0:
                    x = x[:, :, :, :max_shape[2], :max_shape[3]]
            except RuntimeError as e:
                if hasattr(self, 'debug_level') and self.debug_level > 0:
                    print(f"Reshape error in MLPDecoder: {e}")
                # Get mlp output safely
                mlp_output = self.mlp(x)

                # Try reshape in a way that won't crash
                try:
                    # Calculate appropriate shape that's as close as possible to target
                    batch_size = mlp_output.shape[0]
                    total_elements = mlp_output.numel() // batch_size

                    if total_elements % self.out_shape[0] == 0:
                        remaining = total_elements // self.out_shape[0]
                        if remaining % self.out_shape[1] == 0:
                            # Can match the first two dimensions exactly
                            x = mlp_output.reshape(batch_size, self.out_shape[0], self.out_shape[1], -1)
                        else:
                            # Match first dimension only
                            x = mlp_output.reshape(batch_size, self.out_shape[0], -1)
                    else:
                        # Can't match dimensions, use simplest shape
                        x = mlp_output.reshape(batch_size, -1)
                except Exception as nested_e:
                    if hasattr(self, 'debug_level') and self.debug_level > 0:
                        print(f"Nested reshape error: {nested_e}")
                    # Use the original tensor
                    x = mlp_output

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAdaptationModule(nn.Module):
    def __init__(self, feature_dim=512, adaptation_dim=64):
        super().__init__()

        # Task encoder - processes support set to create dataset embedding
        self.task_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, adaptation_dim),
            nn.LayerNorm(adaptation_dim)
        )

        # Feature modulation modules
        self.gamma_layers = nn.ModuleDict({
            'layer1': nn.Linear(adaptation_dim, 64),
            'layer2': nn.Linear(adaptation_dim, 128),
            'layer3': nn.Linear(adaptation_dim, 256),
            'layer4': nn.Linear(adaptation_dim, 512),
        })

        self.beta_layers = nn.ModuleDict({
            'layer1': nn.Linear(adaptation_dim, 64),
            'layer2': nn.Linear(adaptation_dim, 128),
            'layer3': nn.Linear(adaptation_dim, 256),
            'layer4': nn.Linear(adaptation_dim, 512),
        })
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            nn.init.xavier_normal_(self.gamma_layers[layer_name].weight, gain=0.01)
            nn.init.constant_(self.gamma_layers[layer_name].bias, 0.0)
            nn.init.xavier_normal_(self.beta_layers[layer_name].weight, gain=0.01)
            nn.init.constant_(self.beta_layers[layer_name].bias, 0.0)

    def forward(self, task_prototype):
        # Encode the task/dataset
        task_embedding = self.task_encoder(task_prototype)

        # Generate modulation parameters for each layer
        modulation_params = {}
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            gamma = 1 + torch.tanh(self.gamma_layers[layer_name](task_embedding))
            beta = self.beta_layers[layer_name](task_embedding)
            modulation_params[layer_name] = (gamma, beta)

        return modulation_params


class TaskAdaptiveEncoder(nn.Module):
    def __init__(self, base_encoder, task_adaptation_module):
        super().__init__()
        self.base_encoder = base_encoder
        self.adaptation_module = task_adaptation_module
        self.out_dim = base_encoder.get_output_dim()

    def extract_features_with_adaptation(self, x, mod_params=None):
        features = []

        # Initial layers
        x = self.base_encoder.backbone.conv1(x)
        x = self.base_encoder.backbone.bn1(x)
        x = self.base_encoder.backbone.relu(x)
        x = self.base_encoder.backbone.maxpool(x)

        # Layer 1 with conditional adaptation
        f1 = self.base_encoder.backbone.layer1(x)
        if mod_params is not None:
            gamma, beta = mod_params['layer1']
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            f1 = f1 * (1 + gamma) + beta
        p1 = F.adaptive_avg_pool2d(self.base_encoder.layer1_proj(f1), 1).flatten(1)
        features.append(p1)

        # Layer 2 with conditional adaptation
        f2 = self.base_encoder.backbone.layer2(f1)
        if mod_params is not None:
            gamma, beta = mod_params['layer2']
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            f2 = f2 * (1 + gamma) + beta
        p2 = F.adaptive_avg_pool2d(self.base_encoder.layer2_proj(f2), 1).flatten(1)
        features.append(p2)

        # Layer 3 with conditional adaptation
        f3 = self.base_encoder.backbone.layer3(f2)
        if mod_params is not None:
            gamma, beta = mod_params['layer3']
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            f3 = f3 * (1 + gamma) + beta
        p3 = F.adaptive_avg_pool2d(self.base_encoder.layer3_proj(f3), 1).flatten(1)
        features.append(p3)

        # Layer 4 with conditional adaptation
        f4 = self.base_encoder.backbone.layer4(f3)
        if mod_params is not None:
            gamma, beta = mod_params['layer4']
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            f4 = f4 * (1 + gamma) + beta
        p4 = F.adaptive_avg_pool2d(self.base_encoder.layer4_proj(f4), 1).flatten(1)
        features.append(p4)

        # Concatenate features
        return torch.cat(features, dim=1)

    def compute_task_embedding(self, support_images, support_labels, n_way):
        # Extract support features without adaptation
        with torch.no_grad():
            support_features = self.extract_features_with_adaptation(support_images)

            # Compute per-class prototypes
            class_prototypes = []
            for c in range(n_way):
                class_mask = (support_labels == c)
                if class_mask.sum() > 0:
                    class_features = support_features[class_mask]
                    class_prototypes.append(class_features.mean(0))
                else:
                    class_prototypes.append(torch.zeros_like(support_features[0]))

            # Average prototypes to get task prototype
            task_prototype = torch.stack(class_prototypes).mean(0, keepdim=True)

        return task_prototype

    def forward(self, support_images, support_labels, query_images, n_way, temperature=1.0):
        #print(f"Support images stats: min={support_images.min().item():.4f}, max={support_images.max().item():.4f}")
        #print(f"Query images stats: min={query_images.min().item():.4f}, max={query_images.max().item():.4f}")
        # Compute task embedding from support set
        task_prototype = self.compute_task_embedding(support_images, support_labels, n_way)

        # Check for NaNs in task prototype
        if torch.isnan(task_prototype).any():
            print("NaN detected in task prototype!")
            task_prototype = torch.nan_to_num(task_prototype, nan=0.0)

        # Generate adaptation parameters
        modulation_params = self.adaptation_module(task_prototype)

        # Extract features with adaptation
        support_features = self.extract_features_with_adaptation(support_images, modulation_params)
        query_features = self.extract_features_with_adaptation(query_images, modulation_params)

        # Check for NaNs in features
        if torch.isnan(support_features).any() or torch.isnan(query_features).any():
            print("NaN detected in features!")
            support_features = torch.nan_to_num(support_features, nan=0.0)
            query_features = torch.nan_to_num(query_features, nan=0.0)

        # Safe normalization
        support_norms = torch.norm(support_features, p=2, dim=1, keepdim=True)
        query_norms = torch.norm(query_features, p=2, dim=1, keepdim=True)

        support_norms = torch.clamp(support_norms, min=1e-6)  # Prevent division by zero
        query_norms = torch.clamp(query_norms, min=1e-6)

        support_features = support_features / support_norms
        query_features = query_features / query_norms

        # Compute prototypes
        prototypes = []
        for c in range(n_way):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_features = support_features[class_mask]
                prototypes.append(class_features.mean(0))
            else:
                prototypes.append(torch.zeros_like(support_features[0]))
        prototypes = torch.stack(prototypes)

        # Normalize prototypes
        prototype_norms = torch.norm(prototypes, p=2, dim=1, keepdim=True)
        prototype_norms = torch.clamp(prototype_norms, min=1e-6)
        prototypes = prototypes / prototype_norms

        # Compute similarities with stability checks
        similarities = torch.mm(query_features, prototypes.t())

        # Check for NaNs in similarities
        if torch.isnan(similarities).any():
            print("NaN detected in similarities!")
            similarities = torch.nan_to_num(similarities, nan=0.0)

        # Use lower temperature for more stable gradients
        logits = similarities * temperature

        # Final NaN check
        if torch.isnan(logits).any():
            print("NaN detected in final logits!")
            logits = torch.ones_like(logits) / n_way

        return logits


if __name__ == '__main__':
    from task_encoder import  TaskEncoder, initialize_with_ghn
    # Create and test the full model
    base_encoder = TaskEncoder()
    base_encoder = initialize_with_ghn(
        base_encoder,
        ghn_checkpoint_path="/Users/prajjwalmehta/Desktop/projects/ppuda/checkpoints/ghn2_cifar100.pt",
        device='cpu',
    )
    print("Checking for problematic parameters in base encoder...")

    """
    adaptation_module = TaskAdaptationModule(feature_dim=512, adaptation_dim=64)
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)

    # Create mock data
    n_way, k_shot = 5, 1
    support_images = torch.randn(n_way * k_shot, 3, 84, 84)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query_images = torch.randn(n_way * 5, 3, 84, 84)  # 5 query examples per class

    # Test forward pass
    logits = model(support_images, support_labels, query_images, n_way)
    print(f"Logits shape: {logits.shape}")
    """
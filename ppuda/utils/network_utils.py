import torch
import torch.nn as nn
import torchvision.models as models
from ppuda.deepnets1m.architecture import robust_network_adaptation


def create_network_family():
    """Create a diverse family of networks adapted for CIFAR-100."""
    networks = []

    # ResNet variants with different depths
    resnet18 = robust_network_adaptation(models.resnet18(pretrained=True))
    networks.append(resnet18)

    resnet34 = robust_network_adaptation(models.resnet34(pretrained=True))
    networks.append(resnet34)

    # Create a mini resnet with fewer parameters
    mini_resnet = robust_network_adaptation(models.resnet18(pretrained=True))
    mini_resnet.layer4 = nn.Identity()  # Remove last layer
    networks.append(mini_resnet)

    # DenseNet variant
    densenet = robust_network_adaptation(models.densenet121(pretrained=True))
    networks.append(densenet)

    # MobileNet variant
    mobilenet = robust_network_adaptation(models.mobilenet_v2(pretrained=True))
    networks.append(mobilenet)

    # Add a shallower version of mobilenet
    shallow_mobilenet = robust_network_adaptation(models.mobilenet_v2(pretrained=True))
    shallow_mobilenet.features = nn.Sequential(*list(shallow_mobilenet.features)[:10])
    networks.append(shallow_mobilenet)

    return networks


def initialize_from_ghn2(model, ghn2_model):
    """Initialize some parameters using pretrained GHN2."""
    print("Initializing TA-GHN with pretrained GHN2 model...")

    # First, try to match architectures between GHN2 and TA-GHN
    param_mapping = {}

    # Initialize parameter generator
    if hasattr(ghn2_model, 'decoder') and hasattr(model, 'param_generator'):

        for name, param in ghn2_model.decoder.named_parameters():
            if 'joint_processor' in name or 'mlp' in name:
                param_mapping[f"param_generator.joint_processor.{name.split('.')[-1]}"] = param

    with torch.no_grad():
        for target_name, param in param_mapping.items():
            for name, target_param in model.named_parameters():
                if target_name in name and target_param.shape == param.shape:
                    target_param.copy_(param)

    print("Initialization complete!")
    return model
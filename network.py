import torch
import torch.nn as nn
from typing import Literal

try:
    from monai.networks.nets import resnet as monai_resnet
except ImportError:
    monai_resnet = None


def build_network(
    resnet_type: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18',
    in_channels: int = 1,
    num_classes: int = 1
) -> nn.Module:
    """
    Build a 3D ResNet model with Kaiming He initialization.
    Args:
        resnet_type: Type of ResNet backbone ('resnet18', 'resnet34', etc.).
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        num_classes: Number of output classes/features.
    Returns:
        model: 3D ResNet model with Kaiming He initialized weights.
    """
    if monai_resnet is None:
        raise ImportError("MONAI is required for 3D ResNet. Install with 'pip install monai'.")

    resnet_map = {
        'resnet18': monai_resnet.resnet18,
        'resnet34': monai_resnet.resnet34,
        'resnet50': monai_resnet.resnet50,
        'resnet101': monai_resnet.resnet101,
        'resnet152': monai_resnet.resnet152,
    }
    if resnet_type not in resnet_map:
        raise ValueError(f"Unsupported resnet_type: {resnet_type}")

    model = resnet_map[resnet_type](spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)

    def kaiming_init(m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(kaiming_init)
    return model

def print_resnet3d_param_counts():
    """
    Prints the number of parameters for each 3D ResNet configuration using a random input of shape (1, 1, 256, 256, 64).
    """
    configs = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    in_channels = 1
    num_classes = 1
    input_shape = (1, in_channels, 256, 256, 64)
    print('ResNet3D Configurations:')
    for cfg in configs:
        model = build_network(cfg, in_channels, num_classes)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'{cfg}: {n_params} parameters')

if __name__ == '__main__':
    print_resnet3d_param_counts()
    # ResNet3D Configurations:
    # resnet18: 33161409 parameters
    # resnet34: 63471041 parameters (191% of resnet18)
    # resnet50: 46160961 parameters (139% of resnet18)
    # resnet101: 85207105 parameters (256% of resnet18)
    # resnet152: 117365825 parameters (354% of resnet18)

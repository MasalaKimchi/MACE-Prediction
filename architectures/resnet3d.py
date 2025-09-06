import torch
import torch.nn as nn
from typing import Literal

try:
    from monai.networks.nets import resnet as monai_resnet
except ImportError:
    monai_resnet = None


class ResNet3DEncoder(nn.Module):
    """
    ResNet3D-based encoder for 3D volumes.
    
    This class provides a complete encoder for feature extraction using various
    ResNet architectures. It removes the final classification layer and uses
    the original feature dimensions directly without projection.
    
    Parameters
    ----------
    resnet_type : str
        Type of ResNet backbone ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').
    in_channels : int
        Number of input channels for the initial volume.
        
    Note
    ----
    Output feature dimensions:
    - ResNet18/34: 512 features
    - ResNet50/101/152: 2048 features
    """
    
    def __init__(
        self,
        resnet_type: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18',
        in_channels: int = 1
    ) -> None:
        super().__init__()
        if monai_resnet is None:
            raise ImportError("MONAI is required for ResNet3DEncoder. Install with 'pip install monai'.")

        resnet_map = {
            'resnet18': monai_resnet.resnet18,
            'resnet34': monai_resnet.resnet34,
            'resnet50': monai_resnet.resnet50,
            'resnet101': monai_resnet.resnet101,
            'resnet152': monai_resnet.resnet152,
        }
        if resnet_type not in resnet_map:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")

        # Build the ResNet backbone
        self.backbone = resnet_map[resnet_type](
            spatial_dims=3, 
            n_input_channels=in_channels, 
            num_classes=1  # We'll remove this layer
        )
        
        # Get the feature dimension from the last layer before classification
        if hasattr(self.backbone, 'fc'):
            self.feature_dim = self.backbone.fc.in_features
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Could not find final classification layer in ResNet backbone")
        
        # No projection layer - retain original feature dimensions
        self.embed_dim = self.feature_dim
        
        # Apply Kaiming initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming He initialization."""
        def kaiming_init(m):
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.apply(kaiming_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input volume into a global embedding vector.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, D, H, W)``.
            
        Returns
        -------
        torch.Tensor
            A global embedding vector of shape ``(B, feature_dim)``.
        """
        # Extract features using ResNet backbone (without final classification layer)
        features = self.backbone(x)  # Shape: (B, feature_dim)
        
        # Return original feature dimensions (no projection)
        return features


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
    # Test ResNet3DEncoder with original feature dimensions
    print("Testing ResNet3DEncoder with original feature dimensions:")
    print("=" * 70)
    
    configs = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    input_volume = torch.randn(1, 1, 256, 256, 64)
    print(f"Input volume shape: {input_volume.shape}")
    print()
    
    # Expected feature dimensions for each ResNet variant
    expected_dims = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
    }
    
    for cfg in configs:
        try:
            print(f"Testing {cfg}:")
            model = ResNet3DEncoder(resnet_type=cfg, in_channels=1)
            model.eval()
            
            with torch.no_grad():
                output = model(input_volume)
            
            n_params = sum(p.numel() for p in model.parameters())
            expected_dim = expected_dims[cfg]
            print(f"  Parameters: {n_params:,}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected features: {expected_dim}")
            
            # Verify output shape
            assert output.shape == (1, expected_dim), f"Expected (1, {expected_dim}), got {output.shape}"
            print(f"  ✓ Success: {cfg} produced correct output shape")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            print()
    
    # Test with different input configurations
    print("Testing with different input configurations:")
    print("-" * 50)
    
    # Test multi-channel input
    print("Multi-channel input (4 channels):")
    input_4ch = torch.randn(1, 4, 128, 128, 64)
    model_4ch = ResNet3DEncoder(resnet_type='resnet18', in_channels=4)
    with torch.no_grad():
        output_4ch = model_4ch(input_4ch)
    print(f"  Input: {input_4ch.shape} → Output: {output_4ch.shape}")
    assert output_4ch.shape == (1, 512), "Multi-channel test failed"
    print("  ✓ Multi-channel test passed")
    
    # Test ResNet50 with different input size
    print("\nResNet50 with different input size:")
    model_50 = ResNet3DEncoder(resnet_type='resnet50', in_channels=1)
    with torch.no_grad():
        output_50 = model_50(input_volume)
    print(f"  Input: {input_volume.shape} → Output: {output_50.shape}")
    assert output_50.shape == (1, 2048), "ResNet50 test failed"
    print("  ✓ ResNet50 test passed")
    
    print("\nAll ResNet3DEncoder tests completed successfully!")
    print("\nSummary of original ResNet feature dimensions:")
    print("- ResNet18/34: 512 features")
    print("- ResNet50/101/152: 2048 features")

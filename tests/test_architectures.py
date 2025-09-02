import sys
import torch
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from architectures import build_network


def test_resnet_architectures():
    """Test all ResNet architectures."""
    resnet_types = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    in_channels = 1
    num_classes = 4
    input_shape = (1, in_channels, 32, 32, 16)
    x = torch.randn(input_shape)
    
    print("Testing ResNet architectures:")
    for resnet in resnet_types:
        model = build_network(resnet, in_channels, num_classes)
        out = model(x)
        print(f"  {resnet}: input={x.shape} -> output={out.shape}")
        
        # Verify output shape
        expected_shape = (input_shape[0], num_classes)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"


def test_different_input_channels():
    """Test ResNet with different input channel configurations."""
    resnet_type = 'resnet18'
    num_classes = 1
    
    # Test different input channels
    for in_channels in [1, 3]:
        input_shape = (1, in_channels, 32, 32, 16)
        x = torch.randn(input_shape)
        model = build_network(resnet_type, in_channels, num_classes)
        out = model(x)
        print(f"  {resnet_type} with {in_channels} channels: input={x.shape} -> output={out.shape}")
        
        expected_shape = (input_shape[0], num_classes)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"


def test_different_output_classes():
    """Test ResNet with different output class configurations."""
    resnet_type = 'resnet18'
    in_channels = 1
    
    # Test different output classes
    for num_classes in [1, 4, 10]:
        input_shape = (1, in_channels, 32, 32, 16)
        x = torch.randn(input_shape)
        model = build_network(resnet_type, in_channels, num_classes)
        out = model(x)
        print(f"  {resnet_type} with {num_classes} classes: input={x.shape} -> output={out.shape}")
        
        expected_shape = (input_shape[0], num_classes)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"


def test_model_parameters():
    """Test model parameter counts and initialization."""
    resnet_type = 'resnet18'
    in_channels = 1
    num_classes = 1
    
    model = build_network(resnet_type, in_channels, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  {resnet_type} parameter count:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Verify all parameters are trainable
    assert total_params == trainable_params, "All parameters should be trainable"


def test_model_device_compatibility():
    """Test model works on CPU (basic device compatibility)."""
    resnet_type = 'resnet18'
    in_channels = 1
    num_classes = 1
    input_shape = (1, in_channels, 32, 32, 16)
    
    model = build_network(resnet_type, in_channels, num_classes)
    x = torch.randn(input_shape)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    print(f"  {resnet_type} device compatibility test passed")
    assert out.shape == (input_shape[0], num_classes)


def main():
    """Run all architecture tests."""
    print("Testing architectures...")
    test_resnet_architectures()
    test_different_input_channels()
    test_different_output_classes()
    test_model_parameters()
    test_model_device_compatibility()
    print("All architecture tests completed successfully!")


if __name__ == "__main__":
    main()

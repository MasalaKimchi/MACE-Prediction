# Architectures

This folder contains neural network architectures for survival analysis on 3D medical imaging data.

## Overview

The architectures module provides 3D ResNet implementations optimized for medical imaging survival analysis. These models are designed to extract meaningful features from 3D medical volumes for downstream survival prediction tasks.

## Available Models

### 3D ResNet Architectures

- **ResNet18**: Lightweight model with 18 layers, suitable for smaller datasets
- **ResNet34**: Medium complexity with 34 layers, good balance of performance and speed
- **ResNet50**: Standard model with 50 layers, widely used in medical imaging
- **ResNet101**: Deeper model with 101 layers, for complex feature extraction
- **ResNet152**: Deepest model with 152 layers, maximum representational capacity

## Key Features

- **3D Convolutions**: All models use 3D convolutions for volumetric data processing
- **Batch Normalization**: Integrated batch normalization for stable training
- **Residual Connections**: Skip connections to enable training of very deep networks
- **Global Average Pooling**: Final pooling layer for feature vector extraction
- **Configurable Input Size**: Flexible input dimensions for different medical imaging modalities

## Usage

```python
from architectures import ResNet3D

# Create a ResNet18 model
model = ResNet3D(
    model_name='resnet18',
    input_channels=1,  # Number of input channels (e.g., 1 for CT, 4 for multi-modal MRI)
    num_classes=1,     # Output dimension for survival prediction
    pretrained=False   # Whether to use pretrained weights
)

# Forward pass
features = model(volume)  # volume: (batch_size, channels, depth, height, width)
```

## Model Specifications

| Model | Parameters | Input Size | Output Size | Memory Usage |
|-------|------------|------------|-------------|--------------|
| ResNet18 | ~11M | (B, C, D, H, W) | (B, 1) | Low |
| ResNet34 | ~21M | (B, C, D, H, W) | (B, 1) | Medium |
| ResNet50 | ~23M | (B, C, D, H, W) | (B, 1) | Medium |
| ResNet101 | ~42M | (B, C, D, H, W) | (B, 1) | High |
| ResNet152 | ~58M | (B, C, D, H, W) | (B, 1) | Very High |

## Implementation Details

### Architecture Components

1. **Initial Convolution**: 7x7x7 convolution with batch normalization and ReLU
2. **Max Pooling**: 3x3x3 max pooling with stride 2
3. **Residual Blocks**: Multiple residual blocks with 3D convolutions
4. **Global Average Pooling**: Reduces spatial dimensions to 1x1x1
5. **Final Linear Layer**: Maps features to survival prediction output

### Key Design Decisions

- **3D Convolutions**: Essential for capturing spatial relationships in volumetric data
- **Residual Connections**: Enable training of deep networks without vanishing gradients
- **Global Average Pooling**: Reduces overfitting and provides translation invariance
- **Batch Normalization**: Stabilizes training and improves convergence

## Integration with Survival Analysis

The architectures are designed to work seamlessly with:

- **Cox Proportional Hazards**: Output features can be used directly with Cox regression
- **Deep Survival Models**: Features can be fed into survival-specific layers
- **Multi-task Learning**: Can be extended for joint segmentation and survival prediction

## Performance Considerations

### Memory Usage
- ResNet18/34: Suitable for GPUs with 8GB+ memory
- ResNet50: Requires 12GB+ GPU memory
- ResNet101/152: Requires 16GB+ GPU memory

### Training Tips
- Use gradient accumulation for large models with limited memory
- Consider mixed precision training (FP16) to reduce memory usage
- Use data parallelism for multi-GPU training

## Extending the Architectures

To add new architectures:

1. Create a new class inheriting from `torch.nn.Module`
2. Implement the `forward` method
3. Add the model to the factory function
4. Update the `__init__.py` to export the new model
5. Add corresponding tests in `tests/test_architectures.py`

## Dependencies

- **PyTorch**: Core deep learning framework
- **torchvision**: ResNet implementations and utilities
- **NumPy**: Numerical operations

## Testing

Run architecture tests:
```bash
python -m pytest tests/test_architectures.py -v
```

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [3D Deep Learning for Medical Image Analysis](https://www.nature.com/articles/s41591-019-0447-9)
- [SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D): Architectural inspiration

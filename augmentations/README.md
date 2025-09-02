# Augmentations

This folder contains data augmentation transforms for 3D medical imaging survival analysis.

## Overview

The augmentations module provides comprehensive data augmentation pipelines for 3D medical imaging data, built on top of [MONAI](https://monai.io/)'s medical imaging augmentation framework. It includes both geometric and intensity-based augmentations specifically designed for medical imaging tasks.

## Available Components

### transforms.py

Main augmentation module containing transform pipelines and utilities for medical image augmentation.

#### Key Features

- **MONAI Integration**: Built on MONAI's medical imaging augmentation framework
- **3D Support**: Full support for 3D volumetric data augmentation
- **Medical Imaging Specific**: Augmentations designed for medical imaging characteristics
- **Configurable Pipelines**: Easy configuration of augmentation pipelines
- **Performance Optimized**: Efficient augmentation for large 3D volumes

## Augmentation Categories

### Geometric Augmentations

#### Spatial Transforms

```python
from augmentations import SpatialTransforms

# Create spatial augmentation pipeline
spatial_transforms = SpatialTransforms(
    rotation_range=15,        # Rotation in degrees
    translation_range=0.1,    # Translation as fraction of image size
    scale_range=0.1,          # Scale variation
    shear_range=5,            # Shear in degrees
    flip_probability=0.5      # Probability of flipping
)
```

#### Affine Transforms

```python
from augmentations import AffineTransforms

# Create affine augmentation pipeline
affine_transforms = AffineTransforms(
    rotation_prob=0.5,
    translation_prob=0.5,
    scale_prob=0.5,
    shear_prob=0.3,
    flip_prob=0.5
)
```

### Intensity Augmentations

#### Noise Augmentation

```python
from augmentations import NoiseTransforms

# Create noise augmentation pipeline
noise_transforms = NoiseTransforms(
    gaussian_noise_prob=0.3,
    gaussian_noise_std=0.1,
    poisson_noise_prob=0.3,
    salt_pepper_prob=0.2
)
```

#### Intensity Scaling

```python
from augmentations import IntensityTransforms

# Create intensity augmentation pipeline
intensity_transforms = IntensityTransforms(
    brightness_prob=0.5,
    brightness_range=0.2,
    contrast_prob=0.5,
    contrast_range=0.2,
    gamma_prob=0.5,
    gamma_range=(0.8, 1.2)
)
```

## Predefined Augmentation Pipelines

### Light Augmentation

Minimal augmentation for sensitive medical data:

```python
from augmentations import LightAugmentation

# Create light augmentation pipeline
light_aug = LightAugmentation(
    rotation_range=5,
    translation_range=0.05,
    noise_prob=0.2,
    noise_std=0.05
)
```

### Medium Augmentation

Balanced augmentation for most medical imaging tasks:

```python
from augmentations import MediumAugmentation

# Create medium augmentation pipeline
medium_aug = MediumAugmentation(
    rotation_range=15,
    translation_range=0.1,
    scale_range=0.1,
    noise_prob=0.3,
    intensity_prob=0.3
)
```

### Heavy Augmentation

Strong augmentation for data augmentation studies:

```python
from augmentations import HeavyAugmentation

# Create heavy augmentation pipeline
heavy_aug = HeavyAugmentation(
    rotation_range=30,
    translation_range=0.2,
    scale_range=0.2,
    noise_prob=0.5,
    intensity_prob=0.5,
    elastic_prob=0.3
)
```

## Modality-Specific Augmentations

### CT Augmentation

Optimized for CT imaging characteristics:

```python
from augmentations import CTAugmentation

# Create CT-specific augmentation
ct_aug = CTAugmentation(
    window_range=(-1000, 1000),  # HU window
    noise_prob=0.3,
    noise_std=0.1,
    rotation_range=10,
    translation_range=0.05
)
```

### MRI Augmentation

Optimized for MRI imaging characteristics:

```python
from augmentations import MRIAugmentation

# Create MRI-specific augmentation
mri_aug = MRIAugmentation(
    bias_field_prob=0.3,        # Bias field augmentation
    noise_prob=0.3,
    noise_std=0.05,
    rotation_range=15,
    translation_range=0.1
)
```

### Multi-modal Augmentation

For multi-modal imaging data:

```python
from augmentations import MultiModalAugmentation

# Create multi-modal augmentation
multimodal_aug = MultiModalAugmentation(
    modalities=['T1', 'T2', 'FLAIR', 'T1CE'],
    shared_transforms=['rotation', 'translation'],
    modality_specific_transforms={
        'T1': ['bias_field'],
        'T2': ['noise'],
        'FLAIR': ['intensity'],
        'T1CE': ['contrast']
    }
)
```

## Advanced Augmentations

### Elastic Deformation

```python
from augmentations import ElasticDeformation

# Create elastic deformation
elastic_deform = ElasticDeformation(
    prob=0.3,
    sigma_range=(10, 50),
    magnitude_range=(0, 0.2),
    spatial_size=(128, 128, 128)
)
```

### Random Crop and Pad

```python
from augmentations import RandomCropPad

# Create random crop and pad
crop_pad = RandomCropPad(
    prob=0.5,
    crop_size=(96, 96, 96),
    pad_size=(128, 128, 128),
    pad_mode='constant'
)
```

### Mixup Augmentation

```python
from augmentations import MixupAugmentation

# Create mixup augmentation
mixup = MixupAugmentation(
    prob=0.3,
    alpha=0.2,
    beta=0.2
)
```

## Configuration System

### YAML Configuration

```yaml
# config.yaml
augmentation:
  type: 'medium'
  spatial:
    rotation_range: 15
    translation_range: 0.1
    scale_range: 0.1
    flip_prob: 0.5
  intensity:
    noise_prob: 0.3
    noise_std: 0.1
    brightness_prob: 0.3
    contrast_prob: 0.3
  elastic:
    prob: 0.2
    sigma_range: [10, 50]
    magnitude_range: [0, 0.2]
```

### Python Configuration

```python
# Python configuration
aug_config = {
    'type': 'medium',
    'spatial': {
        'rotation_range': 15,
        'translation_range': 0.1,
        'scale_range': 0.1,
        'flip_prob': 0.5
    },
    'intensity': {
        'noise_prob': 0.3,
        'noise_std': 0.1,
        'brightness_prob': 0.3,
        'contrast_prob': 0.3
    }
}
```

## Augmentation Factory

### Basic Usage

```python
from augmentations import AugmentationFactory

# Create factory
factory = AugmentationFactory()

# Create augmentation pipeline
augmentation = factory.create_augmentation(
    config=aug_config
)
```

### Advanced Usage

```python
# Create custom augmentation pipeline
custom_aug = factory.create_custom_augmentation(
    transforms=[
        'rotation',
        'translation',
        'noise',
        'intensity'
    ],
    probabilities=[0.5, 0.5, 0.3, 0.3],
    parameters={
        'rotation': {'range': 15},
        'translation': {'range': 0.1},
        'noise': {'std': 0.1},
        'intensity': {'range': 0.2}
    }
)
```

## Performance Optimization

### Efficient Augmentation

```python
from augmentations import EfficientAugmentation

# Create efficient augmentation
efficient_aug = EfficientAugmentation(
    pipeline=augmentation,
    cache_size=100,        # Cache augmented samples
    num_workers=4,         # Parallel augmentation
    pin_memory=True        # Pin memory for GPU
)
```

### GPU Acceleration

```python
from augmentations import GPUAugmentation

# Create GPU-accelerated augmentation
gpu_aug = GPUAugmentation(
    pipeline=augmentation,
    device='cuda:0',
    batch_size=8
)
```

## Integration with Training

### Training Loop Integration

```python
from augmentations import AugmentationFactory

# Create augmentation
factory = AugmentationFactory()
augmentation = factory.create_augmentation(config)

# Apply to dataset
from dataloaders import SurvivalDataset

dataset = SurvivalDataset(
    csv_path='data/train.csv',
    transform=augmentation
)
```

### Experiment Configuration

```python
# Load augmentation from config
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create augmentation from config
augmentation = factory.create_augmentation(
    config=config['augmentation']
)
```

## Quality Control

### Augmentation Validation

```python
from augmentations import AugmentationValidator

# Create validator
validator = AugmentationValidator()

# Validate augmentation
is_valid, issues = validator.validate_augmentation(
    augmentation=augmentation,
    sample_data=sample_image
)

if not is_valid:
    print(f"Augmentation issues: {issues}")
```

### Augmentation Visualization

```python
from augmentations import AugmentationVisualizer

# Create visualizer
visualizer = AugmentationVisualizer()

# Visualize augmentation effects
visualizer.visualize_augmentation(
    original_image=sample_image,
    augmentation=augmentation,
    num_samples=5,
    save_path='augmentation_samples.png'
)
```

## Best Practices

### Augmentation Selection

1. **Light Augmentation**: For sensitive medical data
2. **Medium Augmentation**: For most medical imaging tasks
3. **Heavy Augmentation**: For data augmentation studies
4. **Modality-Specific**: Use modality-specific augmentations

### Performance Tips

1. **Cache Augmented Data**: Cache frequently used augmentations
2. **Parallel Processing**: Use multiple workers for augmentation
3. **GPU Acceleration**: Use GPU for intensity augmentations
4. **Memory Management**: Monitor memory usage during augmentation

### Medical Imaging Considerations

1. **Preserve Anatomy**: Avoid augmentations that change anatomical structure
2. **Maintain Intensity Relationships**: Preserve intensity relationships between tissues
3. **Consider Modality**: Use modality-specific augmentation strategies
4. **Validate Results**: Always validate augmentation results

## Testing

### Unit Tests

```python
# Test augmentation creation
def test_augmentation_creation():
    from augmentations import AugmentationFactory
    
    factory = AugmentationFactory()
    augmentation = factory.create_augmentation(
        config={'type': 'light'}
    )
    
    assert augmentation is not None
```

### Integration Tests

```python
# Test augmentation with data
def test_augmentation_integration():
    from augmentations import AugmentationFactory
    from dataloaders import SurvivalDataset
    
    factory = AugmentationFactory()
    augmentation = factory.create_augmentation(
        config={'type': 'medium'}
    )
    
    dataset = SurvivalDataset(
        csv_path='test_data.csv',
        transform=augmentation
    )
    
    # Test augmentation
    sample = dataset[0]
    assert sample['image'].shape[0] > 0
```

## Dependencies

- **[MONAI](https://monai.io/)**: Medical imaging AI toolkit
- **PyTorch**: Core deep learning framework
- **NumPy**: Numerical operations
- **Pillow**: Image processing utilities

## References

- [MONAI Transforms](https://docs.monai.io/en/stable/transforms.html)
- [Data Augmentation in Medical Imaging](https://www.nature.com/articles/s41591-019-0447-9)
- [3D Medical Image Augmentation](https://arxiv.org/abs/2003.01200)

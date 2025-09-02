# Dataloaders

This folder contains data loading utilities and dataset classes for survival analysis on 3D medical imaging data.

## Overview

The dataloaders module provides PyTorch dataset classes and data loading utilities specifically designed for survival analysis tasks. It integrates with [MONAI](https://monai.io/) for medical image processing and supports various medical imaging formats.

## Available Classes

### SurvivalDataset

The main dataset class for survival analysis with 3D medical imaging data.

#### Key Features

- **MONAI Integration**: Leverages MONAI's medical image processing capabilities
- **Multiple Formats**: Supports NIfTI, DICOM, and other medical imaging formats
- **Survival Data**: Handles time-to-event data with censoring information
- **Flexible Preprocessing**: Configurable data preprocessing pipelines
- **Memory Efficient**: Optimized for large 3D medical volumes

#### Usage

```python
from dataloaders import SurvivalDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = SurvivalDataset(
    csv_path='data/train.csv',
    image_col='image_path',
    time_col='survival_time',
    event_col='event',
    transform=transforms,
    cache_rate=0.1  # Cache 10% of data in memory
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through data
for batch in dataloader:
    images = batch['image']      # (B, C, D, H, W)
    times = batch['time']        # (B,)
    events = batch['event']      # (B,)
    patient_ids = batch['id']    # (B,)
```

## Data Format Requirements

### CSV File Structure

The dataset expects a CSV file with the following columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `patient_id` | Unique patient identifier | string | "P001" |
| `image_path` | Path to 3D medical image | string | "/data/images/P001.nii.gz" |
| `survival_time` | Time to event or censoring | float | 365.5 |
| `event` | Event indicator (1=event, 0=censored) | int | 1 |

### Image Format Support

- **NIfTI (.nii, .nii.gz)**: Standard neuroimaging format
- **DICOM**: Medical imaging standard
- **Numpy arrays**: Preprocessed volumes
- **Other formats**: Via MONAI's image readers

## Data Loading Pipeline

### 1. Image Loading
- Load 3D medical volumes using MONAI readers
- Handle different medical imaging formats
- Apply format-specific preprocessing

### 2. Preprocessing
- Resampling to consistent voxel spacing
- Normalization (z-score, min-max, etc.)
- Cropping/padding to standard size
- Data augmentation (if specified)

### 3. Survival Data Processing
- Parse survival times and event indicators
- Handle missing or invalid data
- Apply time transformations if needed

### 4. Caching
- Optional in-memory caching for frequently accessed data
- Configurable cache rate to balance memory usage and speed

## Configuration Options

### Dataset Parameters

```python
dataset = SurvivalDataset(
    csv_path='data/train.csv',           # Path to CSV file
    image_col='image_path',              # Column name for image paths
    time_col='survival_time',            # Column name for survival times
    event_col='event',                   # Column name for event indicators
    transform=transforms,                # MONAI transforms pipeline
    cache_rate=0.1,                      # Fraction of data to cache
    num_workers=4,                       # Number of worker processes
    pin_memory=True,                     # Pin memory for GPU transfer
    drop_last=False,                     # Drop last incomplete batch
    shuffle=True                         # Shuffle data
)
```

### DataLoader Parameters

```python
dataloader = DataLoader(
    dataset,
    batch_size=4,                        # Batch size
    shuffle=True,                        # Shuffle batches
    num_workers=4,                       # Number of worker processes
    pin_memory=True,                     # Pin memory for GPU
    persistent_workers=True,             # Keep workers alive
    prefetch_factor=2                    # Prefetch batches
)
```

## Memory Optimization

### Caching Strategies

1. **No Caching**: Load data on-demand (lowest memory usage)
2. **Partial Caching**: Cache frequently accessed data
3. **Full Caching**: Cache all data in memory (fastest, highest memory usage)

### Memory Management

- **Lazy Loading**: Load images only when needed
- **Memory Mapping**: Use memory-mapped files for large datasets
- **Gradient Accumulation**: Process smaller batches to reduce memory usage

## Data Augmentation

Integration with MONAI's augmentation pipeline:

```python
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd,
    ScaleIntensityRanged, RandRotated, RandFlipd
)

transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0)),
    ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0, b_max=1),
    RandRotated(keys=['image'], prob=0.5, range_x=0.1),
    RandFlipd(keys=['image'], prob=0.5, spatial_axis=0)
])
```

## Error Handling

### Common Issues and Solutions

1. **Missing Files**: Skip missing images with warning
2. **Corrupted Data**: Validate data integrity during loading
3. **Memory Errors**: Implement progressive loading strategies
4. **Format Errors**: Handle various medical imaging formats gracefully

## Performance Tips

### Optimization Strategies

1. **Use Multiple Workers**: Parallelize data loading
2. **Pin Memory**: Faster GPU transfer
3. **Prefetching**: Load next batch while processing current
4. **Caching**: Cache frequently accessed data
5. **Efficient Transforms**: Use vectorized operations

## Testing

Run dataloader tests:
```bash
python -m pytest tests/test_dataloaders.py -v
```

## Integration Examples

### With Training Scripts

```python
# In training script
from dataloaders import SurvivalDataset
from torch.utils.data import DataLoader

# Create datasets
train_dataset = SurvivalDataset('data/train.csv', transform=train_transforms)
val_dataset = SurvivalDataset('data/val.csv', transform=val_transforms)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training code here
        pass
```

### With Experiment Configuration

```yaml
# config.yaml
dataset:
  train_csv: "data/train.csv"
  val_csv: "data/val.csv"
  image_col: "image_path"
  time_col: "survival_time"
  event_col: "event"
  cache_rate: 0.1
  batch_size: 8
  num_workers: 4
```

## Dependencies

- **PyTorch**: Core deep learning framework
- **[MONAI](https://monai.io/)**: Medical imaging AI toolkit
- **Pandas**: CSV file handling
- **NumPy**: Numerical operations
- **Pillow**: Image processing utilities

## References

- [MONAI Documentation](https://docs.monai.io/)
- [PyTorch Data Loading](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Medical Image Processing with MONAI](https://monai.io/tutorials.html)

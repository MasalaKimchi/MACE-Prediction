# Pretraining Scripts

This directory contains scripts for pretraining 3D ResNet models on feature prediction tasks with multi-GPU support.

## Structure

- `pretrain_features.py` - Main pretraining script for feature prediction (single-GPU)
- `pretrain_features_distributed.py` - Distributed pretraining script with DDP support
- `launch_distributed_pretraining.py` - Launch script for distributed pretraining
- `quick_start_pretraining.py` - Super simple pretraining script (auto-detects everything)

## Usage

### 1. Super Simple (Recommended)
```bash
# Just specify your data and features - everything else is auto-detected!
python quick_start_pretraining.py \
    --csv_path data/dataset.csv \
    --feature_cols Age AgatstonScore2D MassScore
```

### 2. Basic Pretraining (Single-GPU)
```bash
# Use all available features (recommended)
python pretrain_features.py \
    --csv_path data/dataset.csv \
    --resnet resnet18 \
    --epochs 1000 \
    --batch_size 8

# Or specify specific features
python pretrain_features.py \
    --csv_path data/dataset.csv \
    --feature_cols Age AgatstonScore2D MassScore \
    --resnet resnet18 \
    --epochs 1000 \
    --batch_size 8
```

### 3. Multi-GPU Distributed Pretraining
```bash
# Launch distributed pretraining (auto-detects all GPUs, uses all features)
python launch_distributed_pretraining.py \
    --csv_path data/dataset.csv \
    --resnet resnet50 \
    --batch_size 32 \
    --epochs 1000 \
    --amp \
    --use_smart_cache
```

### 4. Advanced Pretraining with Optimizations
```bash
# Use all features with advanced optimizations
python pretrain_features.py \
    --csv_path data/dataset.csv \
    --resnet resnet50 \
    --epochs 1000 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --optimizer adamw \
    --scheduler cosine \
    --eta_min 1e-7 \
    --max_grad_norm 1.0 \
    --amp \
    --compile \
    --output pretrain_logs/pretrained_resnet50.pth
```

## Key Features

- **Auto-Feature Detection**: By default, uses ALL available features from your CSV (no need to specify long feature lists!)
- **Feature Prediction**: Learns to predict clinical/radiomic features from 3D medical images
- **MSE Loss**: Uses mean squared error for regression task
- **Scalable**: Designed for large datasets (100,000+ images)
- **Memory Efficient**: Configurable caching and batch processing
- **Multi-GPU Support**: Distributed Data Parallel (DDP) for multiple GPUs
- **Fold-based Splitting**: Uses single CSV with Fold_1 column for train/val split
- **Feature Scaling**: Proper Z-score normalization with scaler persistence
- **Complete Checkpoints**: Saves model weights, scaler, and metadata together
- **Advanced Optimization**: Mixed precision training, AdamW optimizer, cosine annealing
- **Performance Optimizations**: Gradient clipping, torch.compile, efficient data loading
- **Auto-Detection**: Automatically detects available GPUs and optimizes settings
- **MONAI Integration**: SmartCacheDataset for efficient memory management

## Output

The pretrained model can be used as initialization for fine-tuning:

```python
# In finetune_survival.py
--init pretrained \
--pretrained_path pretrain_logs/pretrained_resnet.pth
```

## Checkpoint Format

The pretraining script saves complete checkpoints that include:

```python
{
    'model_state_dict': {...},      # Model weights
    'feature_scaler': StandardScaler(),  # Fitted scaler for Z-score normalization
    'feature_columns': ['feat1', 'feat2', ...],  # Feature names
    'resnet_type': 'resnet18',      # Architecture type
    'epoch': 100,                   # Training epoch
    'loss': 0.0234                  # Final loss
}
```

## Feature Scaling

### Z-Score Normalization Process
1. **Training**: Features are Z-score normalized: `(x - mean) / std`
2. **Loss computation**: MSE loss is computed in normalized space (correct approach)
3. **Model learning**: Network learns to predict normalized features
4. **Inference**: Predictions are in normalized space
5. **Conversion**: Use `inverse_transform_features()` to get original scale

### Using Pretrained Models for Inference

```python
from data import load_pretrained_checkpoint, inverse_transform_features
from architectures import build_network

# Load checkpoint
checkpoint = load_pretrained_checkpoint('pretrained_model.pth')
model = build_network(checkpoint['resnet_type'], 1, len(checkpoint['feature_columns']))
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
predictions_normalized = model(image_tensor)

# Convert to original scale
predictions_original = inverse_transform_features(predictions_normalized, checkpoint['feature_scaler'])
```

### Validation

The implementation includes validation functions to ensure proper scaling:

```python
from data import validate_feature_scaling

# Validate that features are properly Z-score normalized
is_valid = validate_feature_scaling(normalized_features)
# Output: Mean ≈ 0, Std ≈ 1
```

## Data Format

The CSV file should contain:
- **"NIFTI path"** column with image file paths
- **"Fold_1"** column with values "train" and "val" for splitting
- **Feature columns** to predict (e.g., Age, AgatstonScore2D, MassScore)
- Any other metadata columns

Example:
```csv
NIFTI path,Fold_1,Age,AgatstonScore2D,MassScore,other_metadata
/path/to/image1.nii.gz,train,65,150.5,25.3,metadata1
/path/to/image2.nii.gz,val,72,89.2,18.7,metadata2
/path/to/image3.nii.gz,train,58,203.1,31.2,metadata3
```

## Multi-GPU Training Benefits

- **Automatic GPU Detection**: Uses all available GPUs by default
- **Distributed Data Parallel**: Efficient gradient synchronization across GPUs
- **Smart Caching**: MONAI SmartCacheDataset for optimal memory usage
- **Scalable**: Handles large datasets with multiple GPUs
- **Easy Launch**: Simple scripts with auto-detection and optimization

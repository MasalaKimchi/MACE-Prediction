# Pretraining Scripts

This directory contains scripts for pretraining 3D ResNet models on feature prediction tasks.

## Structure

- `pretrain_features.py` - Main pretraining script for feature prediction

## Usage

### Basic Pretraining
```bash
python pretrain_features.py \
    --csv_path path/to/data.csv \
    --feature_cols feature1 feature2 feature3 \
    --resnet resnet18 \
    --epochs 1000 \
    --batch_size 8
```

### Advanced Pretraining with Optimizations
```bash
python pretrain_features.py \
    --csv_path path/to/data.csv \
    --feature_cols feature1 feature2 feature3 \
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

- **Feature Prediction**: Learns to predict clinical/radiomic features from 3D medical images
- **MSE Loss**: Uses mean squared error for regression task
- **Scalable**: Designed for large datasets (100,000+ images)
- **Memory Efficient**: Configurable caching and batch processing
- **Multi-GPU Support**: Automatic DataParallel for multiple GPUs
- **Feature Scaling**: Proper Z-score normalization with scaler persistence
- **Complete Checkpoints**: Saves model weights, scaler, and metadata together
- **Advanced Optimization**: Mixed precision training, AdamW optimizer, cosine annealing
- **Performance Optimizations**: Gradient clipping, torch.compile, efficient data loading

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
- DICOM path column
- Feature columns to predict
- Any other metadata columns

Example:
```csv
dicom_path,feature1,feature2,feature3,other_metadata
/path/to/image1.dcm,0.5,1.2,0.8,metadata1
/path/to/image2.dcm,0.3,0.9,1.1,metadata2
```

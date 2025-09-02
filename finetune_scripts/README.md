# Fine-tuning Scripts

This directory contains scripts and utilities for fine-tuning pretrained 3D ResNet models for survival analysis.

## Structure

- `finetune_survival.py` - Main fine-tuning script (refactored for better organization)
- `survival_utils.py` - Core survival analysis functions (Cox loss, baseline estimation, etc.)
- `metrics_utils.py` - Evaluation metrics (C-index, Brier score, time-dependent AUC)
- `training_utils.py` - Training utilities (model loading, epoch execution, checkpointing)

## Usage

### Basic Fine-tuning
```bash
python finetune_survival.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --resnet resnet18 \
    --epochs 50 \
    --batch_size 6
```

### Fine-tuning with Pretrained Weights
```bash
python finetune_survival.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --resnet resnet18 \
    --init pretrained \
    --pretrained_path pretrain_logs/pretrained_resnet.pth \
    --epochs 50 \
    --batch_size 6
```

### Advanced Options with Optimizations
```bash
python finetune_survival.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --resnet resnet50 \
    --init pretrained \
    --pretrained_path pretrain_logs/pretrained_resnet.pth \
    --freeze_epochs 5 \
    --eval_years 1 2 3 4 5 \
    --optimizer adamw \
    --scheduler cosine \
    --eta_min 1e-7 \
    --max_grad_norm 1.0 \
    --amp \
    --compile \
    --log_dir custom_logs
```

## Key Features

- **Modular Design**: Utility functions are separated into focused modules
- **Comprehensive Metrics**: C-index, Integrated Brier Score, Time-dependent AUC
- **Flexible Training**: Support for encoder freezing, mixed precision, model compilation
- **Advanced Optimization**: AdamW optimizer, cosine annealing, gradient clipping
- **Robust Evaluation**: IPCW-based metrics for handling censored data
- **Multi-GPU Support**: Automatic DataParallel for multiple GPUs
- **Pretrained Model Support**: Seamless loading of pretrained checkpoints with feature scalers
- **Backward Compatibility**: Works with both old and new checkpoint formats
- **Performance Optimizations**: Mixed precision training, torch.compile, efficient data loading

## Pretrained Model Loading

The fine-tuning scripts support loading pretrained models with enhanced checkpoint handling:

### Loading Pretrained Checkpoints

```python
from finetune_scripts.training_utils import load_model

# Load model with pretrained weights
model = load_model(
    resnet_type='resnet18',
    init_mode='pretrained',
    pretrained_path='pretrain_logs/pretrained_resnet.pth',
    device=device
)
```

### Checkpoint Format Support

The training utilities handle both old and new checkpoint formats:

#### New Format (with feature scaler)
```python
{
    'model_state_dict': {...},      # Model weights
    'feature_scaler': StandardScaler(),  # Fitted scaler
    'feature_columns': ['feat1', 'feat2', ...],  # Feature names
    'resnet_type': 'resnet18',      # Architecture
    'epoch': 100,                   # Training epoch
    'loss': 0.0234                  # Final loss
}
```

#### Backward Compatibility
- **Old checkpoints**: Still work without feature scaler
- **New checkpoints**: Include complete metadata and scaler
- **Error handling**: Clear messages for missing components

### Enhanced Loading Features

- **Automatic detection**: Detects checkpoint format automatically
- **Validation**: Validates checkpoint integrity
- **Error handling**: Graceful handling of missing components
- **Metadata extraction**: Extracts architecture and training information

## Code Organization Benefits

The refactored structure provides:

1. **Better Readability**: Main script focuses on high-level training logic
2. **Reusability**: Utility functions can be imported and used elsewhere
3. **Maintainability**: Each module has a single responsibility
4. **Testability**: Individual functions can be unit tested
5. **Documentation**: Each module is well-documented with clear interfaces

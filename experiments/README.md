# Experiments

This directory contains structured experiment management for pretraining and fine-tuning with organized output, YAML configurations, and proper checkpoint storage.

## 📁 Directory Structure

```
experiments/
├── experiment_utils.py              # Experiment management utilities
├── manage_experiments.py            # CLI for managing experiments
├── run_pretraining_experiment.py    # Run pretraining experiments
├── run_finetuning_experiment.py     # Run fine-tuning experiments
├── pretraining_experiment/
│   └── config.yaml                  # Pretraining experiment template
├── finetuning_experiment/
│   └── config.yaml                  # Fine-tuning experiment template
└── [experiment_name]/               # Individual experiment results
    ├── configs/
    │   └── config.yaml              # Experiment configuration
    ├── checkpoints/
    │   ├── checkpoint_best.pth      # Best model checkpoint
    │   ├── checkpoint_epoch_*.pth   # Epoch checkpoints
    │   └── final_model.pth          # Final trained model
    ├── logs/
    │   └── tensorboard/             # TensorBoard logs
    ├── results/
    │   └── results.json             # Final results and metrics
    └── artifacts/                   # Additional artifacts (plots, etc.)
```

## 🚀 Quick Start

### 1. Run Pretraining Experiment

```bash
# Using template config
python experiments/run_pretraining_experiment.py \
    --config experiments/pretraining_experiment/config.yaml \
    --auto_name

# With custom experiment name
python experiments/run_pretraining_experiment.py \
    --config experiments/pretraining_experiment/config.yaml \
    --experiment_name "my_pretrain_experiment"
```

### 2. Run Fine-tuning Experiment

```bash
# Using template config
python experiments/run_finetuning_experiment.py \
    --config experiments/finetuning_experiment/config.yaml \
    --auto_name

# With custom experiment name
python experiments/run_finetuning_experiment.py \
    --config experiments/finetuning_experiment/config.yaml \
    --experiment_name "my_finetune_experiment"
```

### 3. Manage Experiments

```bash
# List all experiments
python experiments/manage_experiments.py list

# Show experiment details
python experiments/manage_experiments.py show pretrain_resnet18_features_20240101_120000

# Compare two experiments
python experiments/manage_experiments.py compare exp1 exp2

# Clean up experiments (keep checkpoints)
python experiments/manage_experiments.py cleanup exp1 exp2

# Clean up experiments (remove checkpoints too)
python experiments/manage_experiments.py cleanup exp1 exp2 --remove-checkpoints
```

## 📝 Configuration

### Pretraining Configuration

Edit `experiments/pretraining_experiment/config.yaml`:

```yaml
# Dataset configuration
dataset:
  csv_path: "data/dataset.csv"
  fold_column: "Fold_1"
  train_fold: "train"
  val_fold: "val"
  feature_cols: ["Age", "AgatstonScore2D", "MassScore"]
  image_size: [256, 256, 64]

# Model configuration
model:
  architecture: "resnet18"  # resnet18, resnet34, resnet50, resnet101, resnet152
  num_classes: 3  # Number of features to predict

# Training configuration
training:
  batch_size: 24
  epochs: 1000
  learning_rate: 1e-4
  amp: true  # Mixed precision
```

### Fine-tuning Configuration

Edit `experiments/finetuning_experiment/config.yaml`:

```yaml
# Dataset configuration
dataset:
  csv_path: "data/dataset.csv"
  fold_column: "Fold_1"
  train_fold: "train"
  val_fold: "val"
  time_col: "time"
  event_col: "event"

# Model configuration
model:
  architecture: "resnet18"
  init_mode: "pretrained"
  pretrained_path: "experiments/pretrain_experiment/checkpoints/final_model.pth"

# Training configuration
training:
  batch_size: 24
  epochs: 100
  learning_rate: 1e-4
  amp: true
```

## 🔧 Experiment Management

### ExperimentManager Class

The `ExperimentManager` class provides structured experiment organization:

```python
from experiments.experiment_utils import ExperimentManager

# Create experiment
exp_manager = ExperimentManager("my_experiment")

# Save configuration
config_path = exp_manager.save_config(config_dict)

# Save checkpoint
checkpoint_path = exp_manager.save_checkpoint(
    model, optimizer, epoch, metrics, is_best=True
)

# Save final model
final_model_path = exp_manager.save_final_model(model, final_metrics)

# Save results
results_path = exp_manager.save_results(results_dict)
```

### Automatic Experiment Naming

```python
from experiments.experiment_utils import create_experiment_name

# Generate automatic names
name1 = create_experiment_name('pretrain', 'resnet50', 'dataset1')
# Result: "pretrain_resnet50_dataset1_20240101_120000"

name2 = create_experiment_name('finetune', 'resnet18', timestamp=False)
# Result: "finetune_resnet18"
```

## 📊 Output Structure

### Checkpoint Format

Checkpoints include comprehensive metadata:

```python
{
    'epoch': 100,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'metrics': {
        'train_loss': 0.0234,
        'val_loss': 0.0289,
        'c_index': 0.756
    },
    'experiment_metadata': {
        'name': 'my_experiment',
        'created_at': '2024-01-01T12:00:00',
        'status': 'running'
    },
    'is_best': True,
    'timestamp': '2024-01-01T12:30:00'
}
```

### Results Format

Results are saved as JSON with metadata:

```python
{
    'experiment_metadata': {...},
    'timestamp': '2024-01-01T12:30:00',
    'results': {
        'final_metrics': {
            'c_index': 0.756,
            'ibs': 0.123,
            'td_auc': [0.712, 0.734, 0.756, 0.768, 0.771]
        },
        'best_epoch': 87,
        'training_time': '2h 15m 30s'
    }
}
```

## 🔄 Workflow Integration

### Complete Pretraining → Fine-tuning Workflow

```bash
# 1. Pretrain model
python experiments/run_pretraining_experiment.py \
    --config experiments/pretraining_experiment/config.yaml \
    --auto_name

# 2. Update fine-tuning config with pretrained model path
# Edit experiments/finetuning_experiment/config.yaml:
# model.pretrained_path: "experiments/pretrain_resnet18_dataset_20240101_120000/checkpoints/final_model.pth"

# 3. Fine-tune model
python experiments/run_finetuning_experiment.py \
    --config experiments/finetuning_experiment/config.yaml \
    --auto_name
```

### Using Pretrained Models

```python
# Load pretrained model for fine-tuning
import torch
from experiments.experiment_utils import load_experiment_config

# Load experiment config
config = load_experiment_config("experiments/pretrain_experiment")

# Load pretrained checkpoint
checkpoint_path = "experiments/pretrain_experiment/checkpoints/final_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract model weights
model_state_dict = checkpoint['model_state_dict']
```

## 🧹 Cleanup and Maintenance

### List Experiments

```bash
python experiments/manage_experiments.py list
```

### Clean Up Old Experiments

```bash
# Keep checkpoints, remove logs
python experiments/manage_experiments.py cleanup old_exp1 old_exp2

# Remove everything including checkpoints
python experiments/manage_experiments.py cleanup old_exp1 old_exp2 --remove-checkpoints
```

### Compare Experiments

```bash
python experiments/manage_experiments.py compare exp1 exp2
```

## 📈 Monitoring and Logging

- **TensorBoard**: Logs are saved in `experiments/[name]/logs/`
- **Checkpoints**: Saved in `experiments/[name]/checkpoints/`
- **Results**: Final metrics saved in `experiments/[name]/results/`
- **Configs**: All configurations saved in `experiments/[name]/configs/`

## 🎯 Best Practices

1. **Use descriptive experiment names** or enable `--auto_name`
2. **Save configurations** for reproducibility
3. **Monitor experiments** with TensorBoard
4. **Clean up regularly** to save disk space
5. **Compare experiments** to understand performance differences
6. **Use structured output** for easy result analysis

## 🔗 Integration with Existing Scripts

The experiment system integrates seamlessly with existing pretraining and fine-tuning scripts:

- **Pretraining**: `pretrain_scripts/pretrain_features_distributed.py`
- **Fine-tuning**: `finetune_scripts/finetune_survival_distributed.py`
- **Multi-GPU**: All distributed training features are preserved
- **Fold-based splitting**: Compatible with single CSV approach
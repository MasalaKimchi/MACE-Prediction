# Experiments

This folder contains experiment configurations and templates for survival analysis experiments.

## Overview

The experiments module provides a comprehensive framework for configuring and running survival analysis experiments. It includes template experiments, configuration management, and experiment tracking capabilities.

## Available Components

### Template Experiments

#### survival_analysis/template_experiment/

A complete template experiment setup with:

- **config.yaml**: YAML-based experiment configuration
- **run_experiment.py**: Config-driven training script
- **README.md**: Experiment-specific documentation

## Experiment Structure

```
experiments/
├── survival_analysis/
│   ├── template_experiment/
│   │   ├── config.yaml          # Experiment configuration
│   │   ├── run_experiment.py    # Training script
│   │   └── README.md           # Experiment documentation
│   └── my_experiment/          # Your custom experiment
│       ├── config.yaml
│       ├── run_experiment.py
│       └── README.md
```

## Configuration System

### YAML Configuration

The `config.yaml` file provides comprehensive configuration for all aspects of the experiment:

```yaml
# Experiment metadata
experiment:
  name: "survival_analysis_experiment"
  description: "3D ResNet survival analysis on medical imaging data"
  tags: ["survival", "medical_imaging", "resnet3d"]

# Dataset configuration
dataset:
  train_csv: "data/train.csv"
  val_csv: "data/val.csv"
  test_csv: "data/test.csv"
  image_col: "image_path"
  time_col: "survival_time"
  event_col: "event"
  cache_rate: 0.1
  batch_size: 8
  num_workers: 4

# Model configuration
model:
  architecture: "resnet3d"
  model_name: "resnet18"
  input_channels: 1
  num_classes: 1
  pretrained: false
  dropout_rate: 0.1

# Training configuration
training:
  epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-5
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clipping:
    enabled: true
    max_norm: 1.0

# Loss configuration
loss:
  type: "cox"
  reduction: "mean"
  eps: 1e-8

# Metrics configuration
metrics:
  - "c_index"
  - "brier_score"
  - "td_auc"
  time_points: [30, 90, 180, 365]

# Augmentation configuration
augmentation:
  type: "medium"
  spatial:
    rotation_range: 15
    translation_range: 0.1
    scale_range: 0.1
    flip_prob: 0.5
  intensity:
    noise_prob: 0.3
    noise_std: 0.1

# Logging configuration
logging:
  log_dir: "logs"
  tensorboard: true
  wandb:
    enabled: false
    project: "survival_analysis"
    entity: "your_entity"

# Output configuration
output:
  model_dir: "models"
  checkpoint_dir: "checkpoints"
  results_dir: "results"
  save_best: true
  save_last: true
```

## Running Experiments

### Basic Usage

1. **Copy Template Experiment**:
```bash
cp -r experiments/survival_analysis/template_experiment experiments/survival_analysis/my_experiment
```

2. **Edit Configuration**:
```bash
nano experiments/survival_analysis/my_experiment/config.yaml
```

3. **Run Experiment**:
```bash
cd experiments/survival_analysis/my_experiment
python run_experiment.py --config config.yaml
```

### Advanced Usage

#### Command Line Arguments

```bash
python run_experiment.py \
    --config config.yaml \
    --gpu 0 \
    --resume checkpoint.pth \
    --eval-only \
    --seed 42
```

#### Programmatic Usage

```python
from experiments.survival_analysis.template_experiment.run_experiment import run_experiment

# Run experiment programmatically
results = run_experiment(
    config_path='config.yaml',
    gpu_id=0,
    resume_path=None,
    eval_only=False,
    seed=42
)
```

## Experiment Tracking

### TensorBoard Integration

```yaml
# config.yaml
logging:
  tensorboard: true
  log_dir: "logs"
  log_interval: 100
```

View results:
```bash
tensorboard --logdir logs
```

### Weights & Biases Integration

```yaml
# config.yaml
logging:
  wandb:
    enabled: true
    project: "survival_analysis"
    entity: "your_entity"
    tags: ["survival", "medical_imaging"]
```

### Custom Logging

```python
# Custom logging in run_experiment.py
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
```

## Experiment Management

### Experiment Registry

```python
from experiments import ExperimentRegistry

# Register experiment
registry = ExperimentRegistry()
registry.register_experiment(
    name="my_experiment",
    config_path="config.yaml",
    status="running"
)

# List experiments
experiments = registry.list_experiments()
```

### Experiment Comparison

```python
from experiments import ExperimentComparator

# Compare experiments
comparator = ExperimentComparator()
results = comparator.compare_experiments(
    experiment_ids=["exp1", "exp2", "exp3"],
    metrics=["c_index", "brier_score"]
)
```

## Configuration Validation

### Schema Validation

```python
from experiments import ConfigValidator

# Validate configuration
validator = ConfigValidator()
is_valid, errors = validator.validate_config("config.yaml")

if not is_valid:
    print(f"Configuration errors: {errors}")
```

### Default Configuration

```python
from experiments import DefaultConfig

# Get default configuration
default_config = DefaultConfig.get_default_config()
print(default_config)
```

## Experiment Templates

### Survival Analysis Template

The template experiment includes:

- **Complete Configuration**: All necessary parameters
- **Training Script**: Ready-to-run training code
- **Documentation**: Detailed README
- **Best Practices**: Optimized settings

### Custom Templates

Create custom templates for specific use cases:

```bash
# Create custom template
mkdir -p experiments/survival_analysis/custom_template
cp -r experiments/survival_analysis/template_experiment/* experiments/survival_analysis/custom_template/
```

## Best Practices

### Configuration Management

1. **Version Control**: Keep config files in version control
2. **Documentation**: Document all configuration parameters
3. **Validation**: Validate configurations before running
4. **Backup**: Backup successful configurations

### Experiment Organization

1. **Naming Convention**: Use descriptive experiment names
2. **Folder Structure**: Organize experiments by project/study
3. **Documentation**: Document experiment purpose and results
4. **Reproducibility**: Ensure experiments are reproducible

### Performance Optimization

1. **Resource Management**: Monitor GPU/CPU usage
2. **Checkpointing**: Save checkpoints regularly
3. **Early Stopping**: Use early stopping to prevent overfitting
4. **Hyperparameter Tuning**: Use systematic hyperparameter search

## Integration Examples

### With Training Scripts

```python
# In run_experiment.py
from dataloaders import SurvivalDataset
from architectures import ResNet3D
from losses import CoxLoss
from metrics import SurvivalEvaluator

# Load configuration
config = load_config('config.yaml')

# Create components
dataset = SurvivalDataset(config['dataset'])
model = ResNet3D(config['model'])
loss_fn = CoxLoss(config['loss'])
evaluator = SurvivalEvaluator(config['metrics'])

# Training loop
for epoch in range(config['training']['epochs']):
    # Training code here
    pass
```

### With External Tools

```python
# Integration with external tools
from experiments import ExternalToolIntegration

# Weights & Biases
wandb_integration = ExternalToolIntegration.wandb()
wandb_integration.init(config)

# MLflow
mlflow_integration = ExternalToolIntegration.mlflow()
mlflow_integration.log_experiment(config, results)
```

## Testing

### Configuration Tests

```python
# Test configuration loading
def test_config_loading():
    from experiments import load_config
    
    config = load_config('config.yaml')
    assert 'experiment' in config
    assert 'dataset' in config
    assert 'model' in config
```

### Experiment Tests

```python
# Test experiment execution
def test_experiment_execution():
    from experiments.survival_analysis.template_experiment.run_experiment import run_experiment
    
    # Run with test configuration
    results = run_experiment('test_config.yaml')
    assert results is not None
```

## Dependencies

- **PyYAML**: YAML configuration parsing
- **PyTorch**: Deep learning framework
- **TensorBoard**: Experiment logging
- **Weights & Biases**: Experiment tracking (optional)
- **MLflow**: Experiment management (optional)

## References

- [YAML Configuration](https://yaml.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Weights & Biases](https://wandb.ai/)
- [MLflow](https://mlflow.org/)

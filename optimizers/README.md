# Optimizers

This folder contains optimizer configurations and factory for training survival analysis models.

## Overview

The optimizers module provides a comprehensive optimizer factory and configuration system for training deep survival models. It supports various optimization algorithms with medical imaging-specific configurations and learning rate scheduling strategies.

## Available Components

### optimizer_factory.py

Main optimizer factory providing easy configuration and instantiation of optimizers.

#### Key Features

- **Multiple Optimizers**: Support for Adam, AdamW, SGD, RMSprop, and more
- **Learning Rate Scheduling**: Built-in support for various LR schedulers
- **Medical Imaging Optimized**: Configurations optimized for medical imaging tasks
- **Easy Configuration**: YAML-based configuration system
- **Gradient Clipping**: Built-in gradient clipping for stable training

## Supported Optimizers

### Adam Optimizer

Default choice for most deep learning tasks, including survival analysis.

```python
from optimizers import create_optimizer

# Create Adam optimizer
optimizer = create_optimizer(
    model=model,
    optimizer_type='adam',
    learning_rate=1e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### AdamW Optimizer

Improved version of Adam with decoupled weight decay.

```python
# Create AdamW optimizer
optimizer = create_optimizer(
    model=model,
    optimizer_type='adamw',
    learning_rate=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### SGD Optimizer

Stochastic Gradient Descent with momentum, often used for fine-tuning.

```python
# Create SGD optimizer
optimizer = create_optimizer(
    model=model,
    optimizer_type='sgd',
    learning_rate=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

### RMSprop Optimizer

Root Mean Square Propagation, good for recurrent networks.

```python
# Create RMSprop optimizer
optimizer = create_optimizer(
    model=model,
    optimizer_type='rmsprop',
    learning_rate=1e-3,
    alpha=0.99,
    eps=1e-8,
    weight_decay=1e-4
)
```

## Learning Rate Schedulers

### StepLR Scheduler

Reduces learning rate by a factor at specified epochs.

```python
from optimizers import create_scheduler

# Create StepLR scheduler
scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type='steplr',
    step_size=30,
    gamma=0.1
)
```

### CosineAnnealingLR Scheduler

Cosine annealing learning rate schedule.

```python
# Create CosineAnnealingLR scheduler
scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type='cosine',
    T_max=100,
    eta_min=1e-6
)
```

### ReduceLROnPlateau Scheduler

Reduces learning rate when a metric has stopped improving.

```python
# Create ReduceLROnPlateau scheduler
scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type='plateau',
    mode='max',
    factor=0.5,
    patience=10,
    threshold=1e-4
)
```

### Warmup Scheduler

Combines warmup with other schedulers for stable training.

```python
# Create warmup scheduler
scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type='warmup_cosine',
    warmup_epochs=10,
    max_epochs=100,
    eta_min=1e-6
)
```

## Optimizer Factory

### Basic Usage

```python
from optimizers import OptimizerFactory

# Create factory
factory = OptimizerFactory()

# Create optimizer
optimizer = factory.create_optimizer(
    model=model,
    config={
        'type': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    }
)
```

### Advanced Configuration

```python
# Advanced optimizer configuration
config = {
    'optimizer': {
        'type': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.999],
        'eps': 1e-8
    },
    'scheduler': {
        'type': 'cosine',
        'T_max': 100,
        'eta_min': 1e-6
    },
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 1.0
    }
}

optimizer, scheduler = factory.create_optimizer_and_scheduler(
    model=model,
    config=config
)
```

## Medical Imaging Optimizations

### Pre-trained Model Fine-tuning

```python
# Fine-tuning configuration for pre-trained models
finetune_config = {
    'optimizer': {
        'type': 'adamw',
        'learning_rate': 1e-5,  # Lower LR for fine-tuning
        'weight_decay': 1e-4
    },
    'scheduler': {
        'type': 'warmup_cosine',
        'warmup_epochs': 5,
        'max_epochs': 50
    },
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 0.5
    }
}
```

### Large Model Training

```python
# Configuration for large models
large_model_config = {
    'optimizer': {
        'type': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.95]  # Adjusted betas for large models
    },
    'scheduler': {
        'type': 'cosine',
        'T_max': 200,
        'eta_min': 1e-6
    },
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 1.0
    }
}
```

## Configuration System

### YAML Configuration

```yaml
# config.yaml
optimizer:
  type: 'adamw'
  learning_rate: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: 'cosine'
  T_max: 100
  eta_min: 1e-6

gradient_clipping:
  enabled: true
  max_norm: 1.0
```

### Python Configuration

```python
# Python configuration
config = {
    'optimizer': {
        'type': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    },
    'scheduler': {
        'type': 'cosine',
        'T_max': 100
    }
}
```

## Gradient Clipping

### Automatic Gradient Clipping

```python
from optimizers import GradientClipper

# Create gradient clipper
clipper = GradientClipper(
    max_norm=1.0,
    norm_type=2
)

# Use in training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    # Clip gradients
    clipper.clip_gradients(model.parameters())
    
    optimizer.step()
```

### Adaptive Gradient Clipping

```python
# Adaptive gradient clipping
adaptive_clipper = GradientClipper(
    max_norm=1.0,
    adaptive=True,
    clip_factor=0.1
)
```

## Optimizer Monitoring

### Learning Rate Monitoring

```python
from optimizers import OptimizerMonitor

# Create monitor
monitor = OptimizerMonitor(
    optimizer=optimizer,
    log_interval=100
)

# Use in training
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Training code here
        monitor.log_lr(batch_idx)
```

### Gradient Monitoring

```python
# Monitor gradient norms
gradient_monitor = GradientMonitor(
    model=model,
    log_interval=100
)

# Use in training
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    # Log gradient norms
    gradient_monitor.log_gradients()
    
    optimizer.step()
```

## Performance Optimization

### Mixed Precision Training

```python
from optimizers import MixedPrecisionOptimizer

# Create mixed precision optimizer
mp_optimizer = MixedPrecisionOptimizer(
    model=model,
    optimizer_type='adamw',
    learning_rate=1e-4,
    use_amp=True  # Automatic Mixed Precision
)
```

### Distributed Training

```python
from optimizers import DistributedOptimizer

# Create distributed optimizer
dist_optimizer = DistributedOptimizer(
    model=model,
    optimizer_type='adamw',
    learning_rate=1e-4,
    world_size=4
)
```

## Integration with Training

### Training Loop Integration

```python
from optimizers import OptimizerFactory

# Create optimizer and scheduler
factory = OptimizerFactory()
optimizer, scheduler = factory.create_optimizer_and_scheduler(
    model=model,
    config=config
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()
```

### Experiment Configuration

```python
# Load configuration from YAML
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create optimizer from config
optimizer, scheduler = factory.create_optimizer_and_scheduler(
    model=model,
    config=config
)
```

## Best Practices

### Learning Rate Selection

1. **Pre-training**: Start with 1e-3 to 1e-4
2. **Fine-tuning**: Use 1e-5 to 1e-6
3. **Large models**: Use lower learning rates
4. **Small datasets**: Use higher learning rates

### Weight Decay

1. **AdamW**: Use 1e-4 to 1e-3
2. **Adam**: Use 1e-5 to 1e-4
3. **SGD**: Use 1e-4 to 1e-3

### Scheduler Selection

1. **Cosine**: Good for most tasks
2. **StepLR**: Good for fine-tuning
3. **Plateau**: Good for validation-based scheduling
4. **Warmup**: Good for large models

## Testing

### Unit Tests

```python
# Test optimizer creation
def test_optimizer_creation():
    from optimizers import OptimizerFactory
    
    factory = OptimizerFactory()
    optimizer = factory.create_optimizer(
        model=model,
        config={'type': 'adam', 'learning_rate': 1e-4}
    )
    
    assert optimizer is not None
    assert optimizer.param_groups[0]['lr'] == 1e-4
```

### Integration Tests

```python
# Test optimizer with training
def test_optimizer_training():
    from optimizers import OptimizerFactory
    
    factory = OptimizerFactory()
    optimizer, scheduler = factory.create_optimizer_and_scheduler(
        model=model,
        config=config
    )
    
    # Test training step
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    assert not torch.isnan(loss)
```

## Dependencies

- **PyTorch**: Core deep learning framework
- **NumPy**: Numerical operations
- **PyYAML**: YAML configuration parsing

## References

- [Adam Optimizer](https://arxiv.org/abs/1412.6980)
- [AdamW Optimizer](https://arxiv.org/abs/1711.05101)
- [Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Gradient Clipping](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_)

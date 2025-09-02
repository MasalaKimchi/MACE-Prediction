# Loss Functions

This folder contains loss functions specifically designed for survival analysis tasks in medical imaging.

## Overview

The losses module provides comprehensive loss functions for survival analysis, with a focus on Cox proportional hazards modeling and deep survival learning. It integrates with [torchsurv](https://github.com/autonlab/torchsurv) for robust survival analysis implementations.

## Available Loss Functions

### Cox Proportional Hazards Loss

The primary loss function for survival analysis, implementing the Cox proportional hazards model.

#### Key Features

- **Cox Regression**: Implements the partial likelihood for Cox proportional hazards
- **Censoring Handling**: Properly handles right-censored survival data
- **Risk Set Management**: Efficient computation of risk sets for each time point
- **Gradient Optimization**: Optimized for deep learning with automatic differentiation

#### Usage

```python
from losses import CoxLoss

# Create Cox loss function
cox_loss = CoxLoss()

# Compute loss
loss = cox_loss(
    risk_scores=model_output,    # (batch_size,) - risk scores from model
    times=survival_times,        # (batch_size,) - survival times
    events=event_indicators      # (batch_size,) - event indicators (1=event, 0=censored)
)

# Backward pass
loss.backward()
```

### Deep Survival Loss

Extended loss function for deep survival models with additional regularization terms.

```python
from losses import DeepSurvivalLoss

# Create deep survival loss with regularization
deep_loss = DeepSurvivalLoss(
    cox_weight=1.0,              # Weight for Cox loss
    l2_weight=0.01,              # L2 regularization weight
    l1_weight=0.001              # L1 regularization weight
)

# Compute loss
loss = deep_loss(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators,
    features=extracted_features  # Optional: for regularization
)
```

## Loss Function Details

### Cox Proportional Hazards Loss

The Cox loss implements the negative log partial likelihood:

```
L(β) = -∑[i: δᵢ=1] [βᵀxᵢ - log(∑[j: tⱼ≥tᵢ] exp(βᵀxⱼ))]
```

Where:
- `β` are the model parameters
- `xᵢ` are the feature vectors
- `tᵢ` are the survival times
- `δᵢ` are the event indicators

#### Implementation Features

- **Numerical Stability**: Uses log-sum-exp trick for numerical stability
- **Efficient Computation**: Vectorized operations for batch processing
- **Memory Optimization**: Efficient risk set computation
- **Gradient Computation**: Automatic differentiation support

### Loss Function Variants

#### 1. Standard Cox Loss
```python
from losses import CoxLoss

loss_fn = CoxLoss(
    reduction='mean',        # 'mean', 'sum', or 'none'
    eps=1e-8                # Numerical stability epsilon
)
```

#### 2. Weighted Cox Loss
```python
from losses import WeightedCoxLoss

loss_fn = WeightedCoxLoss(
    sample_weights=None,     # Optional sample weights
    class_weights=None,      # Optional class weights
    reduction='mean'
)
```

#### 3. Regularized Cox Loss
```python
from losses import RegularizedCoxLoss

loss_fn = RegularizedCoxLoss(
    l2_weight=0.01,         # L2 regularization
    l1_weight=0.001,        # L1 regularization
    dropout_weight=0.1      # Dropout regularization
)
```

## Advanced Loss Functions

### Multi-task Loss

For joint learning of survival prediction and other tasks:

```python
from losses import MultiTaskLoss

loss_fn = MultiTaskLoss(
    survival_weight=1.0,     # Weight for survival loss
    auxiliary_weight=0.5,    # Weight for auxiliary task
    tasks=['survival', 'classification']
)
```

### Time-dependent Loss

For time-dependent survival analysis:

```python
from losses import TimeDependentLoss

loss_fn = TimeDependentLoss(
    time_points=[30, 90, 180, 365],  # Time points for evaluation
    weights=[1.0, 1.0, 1.0, 1.0]     # Weights for each time point
)
```

## Loss Function Configuration

### Standard Configuration

```python
loss_config = {
    'type': 'cox',                    # Loss function type
    'reduction': 'mean',              # Reduction method
    'eps': 1e-8,                     # Numerical stability
    'weight': 1.0,                   # Loss weight
    'label_smoothing': 0.0           # Label smoothing
}
```

### Advanced Configuration

```python
advanced_config = {
    'type': 'deep_survival',
    'cox_weight': 1.0,
    'l2_weight': 0.01,
    'l1_weight': 0.001,
    'dropout_weight': 0.1,
    'auxiliary_weight': 0.5,
    'time_points': [30, 90, 180, 365],
    'reduction': 'mean'
}
```

## Integration with Training

### Training Loop Integration

```python
from losses import CoxLoss
import torch

# Initialize loss function
criterion = CoxLoss()

# Training loop
for batch in dataloader:
    images = batch['image']
    times = batch['time']
    events = batch['event']
    
    # Forward pass
    risk_scores = model(images)
    
    # Compute loss
    loss = criterion(risk_scores, times, events)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Loss Monitoring

```python
from losses import LossMonitor

# Create loss monitor
monitor = LossMonitor(
    loss_fn=criterion,
    log_interval=100,
    log_file='loss_log.txt'
)

# Use in training
for batch_idx, batch in enumerate(dataloader):
    loss = criterion(model_output, times, events)
    monitor.log_loss(loss, batch_idx)
```

## Performance Optimization

### Efficient Computation

1. **Vectorized Operations**: Use batch processing for efficiency
2. **Memory Management**: Efficient risk set computation
3. **Numerical Stability**: Log-sum-exp trick for stability
4. **GPU Acceleration**: CUDA-optimized implementations

## Error Handling

### Robust Loss Computation

```python
from losses import RobustCoxLoss

loss_fn = RobustCoxLoss(
    handle_errors=True,      # Handle numerical errors
    fallback_value=1e6,      # Fallback value for errors
    log_errors=True          # Log errors
)
```

### Validation

```python
from losses import validate_loss_inputs

# Validate inputs before loss computation
is_valid, error_msg = validate_loss_inputs(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators
)

if is_valid:
    loss = criterion(model_output, survival_times, event_indicators)
else:
    print(f"Invalid inputs: {error_msg}")
```

## Testing and Validation

### Unit Tests

```python
# Test loss function
def test_cox_loss():
    from losses import CoxLoss
    
    loss_fn = CoxLoss()
    
    # Test with known values
    risk_scores = torch.tensor([1.0, 2.0, 0.5])
    times = torch.tensor([100, 200, 150])
    events = torch.tensor([1, 1, 0])
    
    loss = loss_fn(risk_scores, times, events)
    assert loss.item() > 0, "Loss should be positive"
```

### Integration Tests

```python
# Test with real data
def test_loss_integration():
    from dataloaders import SurvivalDataset
    from losses import CoxLoss
    
    dataset = SurvivalDataset('test_data.csv')
    loss_fn = CoxLoss()
    
    for batch in dataset:
        loss = loss_fn(batch['risk_scores'], batch['time'], batch['event'])
        assert not torch.isnan(loss), "Loss should not be NaN"
```

## Dependencies

- **PyTorch**: Core deep learning framework
- **[torchsurv](https://github.com/autonlab/torchsurv)**: Survival analysis library
- **NumPy**: Numerical operations
- **SciPy**: Statistical functions

## References

- [Cox Proportional Hazards Model](https://en.wikipedia.org/wiki/Proportional_hazards_model)
- [Deep Survival Analysis](https://arxiv.org/abs/1608.02158)
- [torchsurv Documentation](https://torchsurv.readthedocs.io/)
- [Survival Analysis in Medical Imaging](https://www.nature.com/articles/s41591-019-0447-9)

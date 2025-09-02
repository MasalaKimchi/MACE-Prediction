# Metrics

This folder contains evaluation metrics for survival analysis tasks in medical imaging.

## Overview

The metrics module provides comprehensive evaluation metrics for survival analysis, including concordance index (C-index), Brier score, time-dependent AUC, and integrated Brier score. These metrics are essential for evaluating the performance of survival prediction models.

## Available Metrics

### Concordance Index (C-index)

The most widely used metric for survival analysis, measuring the proportion of concordant pairs.

#### Key Features

- **Concordance Measurement**: Measures how well the model ranks survival times
- **Censoring Handling**: Properly handles right-censored data
- **Time-dependent**: Can be computed at specific time points
- **Robust Implementation**: Handles edge cases and missing data

#### Usage

```python
from metrics import ConcordanceIndex

# Create C-index metric
c_index = ConcordanceIndex()

# Compute C-index
score = c_index(
    risk_scores=model_output,    # (batch_size,) - risk scores from model
    times=survival_times,        # (batch_size,) - survival times
    events=event_indicators      # (batch_size,) - event indicators
)

print(f"C-index: {score:.4f}")
```

### Brier Score

Measures the accuracy of probabilistic predictions for survival analysis.

```python
from metrics import BrierScore

# Create Brier score metric
brier_score = BrierScore(time_points=[30, 90, 180, 365])

# Compute Brier score
score = brier_score(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators
)

print(f"Brier Score: {score:.4f}")
```

### Time-dependent AUC

Area under the ROC curve at specific time points.

```python
from metrics import TimeDependentAUC

# Create time-dependent AUC metric
td_auc = TimeDependentAUC(time_points=[30, 90, 180, 365])

# Compute time-dependent AUC
auc_scores = td_auc(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators
)

print(f"Time-dependent AUC: {auc_scores}")
```

## Metric Details

### Concordance Index (C-index)

The C-index measures the proportion of concordant pairs in the dataset:

```
C-index = (Number of concordant pairs) / (Total number of comparable pairs)
```

A pair (i, j) is concordant if:
- Patient i has shorter survival time than patient j AND
- Patient i has higher risk score than patient j

#### Implementation Features

- **Efficient Computation**: O(n²) algorithm with optimizations
- **Censoring Handling**: Proper handling of right-censored data
- **Memory Efficient**: Optimized for large datasets
- **Numerical Stability**: Robust against numerical issues

### Brier Score

The Brier score measures the mean squared difference between predicted probabilities and actual outcomes:

```
BS(t) = (1/n) * Σ[I(Ti ≤ t, δi = 1) - S(t|xi)]²
```

Where:
- `Ti` is the survival time
- `δi` is the event indicator
- `S(t|xi)` is the predicted survival probability

#### Time Points

The Brier score can be computed at multiple time points:
- **30 days**: Short-term survival
- **90 days**: Medium-term survival
- **180 days**: Long-term survival
- **365 days**: One-year survival

### Time-dependent AUC

Measures the area under the ROC curve at specific time points:

```
AUC(t) = P(risk_score_i > risk_score_j | T_i ≤ t < T_j)
```

## Advanced Metrics

### Integrated Brier Score

Combines Brier scores across multiple time points:

```python
from metrics import IntegratedBrierScore

# Create integrated Brier score
ibs = IntegratedBrierScore(
    time_points=[30, 90, 180, 365],
    weights=[1.0, 1.0, 1.0, 1.0]  # Equal weights
)

# Compute integrated Brier score
score = ibs(risk_scores, times, events)
```

### Harrell's C-index

Extended C-index with additional features:

```python
from metrics import HarrellsCIndex

# Create Harrell's C-index
harrell_c = HarrellsCIndex(
    tied_times='omit',      # How to handle tied times
    weight_function=None     # Optional weight function
)

# Compute Harrell's C-index
score = harrell_c(risk_scores, times, events)
```

### Uno's C-index

Time-dependent C-index for specific time points:

```python
from metrics import UnosCIndex

# Create Uno's C-index
uno_c = UnosCIndex(time_points=[30, 90, 180, 365])

# Compute Uno's C-index
scores = uno_c(risk_scores, times, events)
```

## Metric Configuration

### Standard Configuration

```python
metric_config = {
    'c_index': {
        'tied_times': 'omit',        # Handle tied times
        'weight_function': None      # Optional weight function
    },
    'brier_score': {
        'time_points': [30, 90, 180, 365],
        'weights': [1.0, 1.0, 1.0, 1.0]
    },
    'td_auc': {
        'time_points': [30, 90, 180, 365],
        'method': 'nearest'          # Interpolation method
    }
}
```

### Advanced Configuration

```python
advanced_config = {
    'metrics': ['c_index', 'brier_score', 'td_auc'],
    'time_points': [30, 90, 180, 365],
    'weights': [1.0, 1.0, 1.0, 1.0],
    'bootstrap_samples': 1000,       # Bootstrap for confidence intervals
    'confidence_level': 0.95
}
```

## Evaluation Pipeline

### Comprehensive Evaluation

```python
from metrics import SurvivalEvaluator

# Create evaluator
evaluator = SurvivalEvaluator(
    metrics=['c_index', 'brier_score', 'td_auc'],
    time_points=[30, 90, 180, 365]
)

# Evaluate model
results = evaluator.evaluate(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators
)

# Print results
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
```

### Batch Evaluation

```python
from metrics import BatchEvaluator

# Create batch evaluator
batch_evaluator = BatchEvaluator(
    metrics=['c_index', 'brier_score'],
    batch_size=1000
)

# Evaluate in batches
results = batch_evaluator.evaluate_batch(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators
)
```

## Statistical Analysis

### Confidence Intervals

```python
from metrics import ConfidenceInterval

# Create confidence interval calculator
ci_calculator = ConfidenceInterval(
    metric='c_index',
    confidence_level=0.95,
    bootstrap_samples=1000
)

# Compute confidence interval
ci = ci_calculator.compute_ci(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators
)

print(f"C-index: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
```

### Statistical Tests

```python
from metrics import StatisticalTests

# Create statistical test calculator
stats_tests = StatisticalTests()

# Compare two models
p_value = stats_tests.compare_models(
    model1_scores=model1_output,
    model2_scores=model2_output,
    times=survival_times,
    events=event_indicators,
    metric='c_index'
)

print(f"P-value: {p_value:.4f}")
```

## Visualization

### Metric Plots

```python
from metrics import MetricVisualizer

# Create visualizer
visualizer = MetricVisualizer()

# Plot time-dependent metrics
visualizer.plot_time_dependent_metrics(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators,
    time_points=[30, 90, 180, 365],
    save_path='metrics_plot.png'
)
```

### ROC Curves

```python
# Plot ROC curves at different time points
visualizer.plot_roc_curves(
    risk_scores=model_output,
    times=survival_times,
    events=event_indicators,
    time_points=[30, 90, 180, 365],
    save_path='roc_curves.png'
)
```

## Performance Optimization

### Efficient Computation

1. **Vectorized Operations**: Use NumPy/PyTorch for efficient computation
2. **Memory Management**: Optimize memory usage for large datasets
3. **Parallel Processing**: Use multiprocessing for bootstrap sampling
4. **Caching**: Cache intermediate results for repeated computations

### Benchmarking

Typical performance on different hardware:

| Hardware | Dataset Size | C-index (ms) | Brier Score (ms) |
|----------|--------------|--------------|------------------|
| CPU | 1,000 | ~50 | ~100 |
| CPU | 10,000 | ~500 | ~1,000 |
| GPU | 1,000 | ~10 | ~20 |
| GPU | 10,000 | ~100 | ~200 |

## Integration with Training

### Training Loop Integration

```python
from metrics import SurvivalEvaluator

# Create evaluator
evaluator = SurvivalEvaluator(['c_index', 'brier_score'])

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training code here
        pass
    
    # Validation
    val_results = evaluator.evaluate(
        risk_scores=val_output,
        times=val_times,
        events=val_events
    )
    
    print(f"Epoch {epoch}: C-index = {val_results['c_index']:.4f}")
```

### Experiment Tracking

```python
from metrics import ExperimentTracker

# Create tracker
tracker = ExperimentTracker(
    experiment_name='survival_experiment',
    metrics=['c_index', 'brier_score', 'td_auc']
)

# Track metrics
tracker.log_metrics(
    epoch=epoch,
    metrics=val_results,
    split='validation'
)
```

## Testing and Validation

### Unit Tests

```python
# Test metric computation
def test_c_index():
    from metrics import ConcordanceIndex
    
    c_index = ConcordanceIndex()
    
    # Test with known values
    risk_scores = torch.tensor([1.0, 2.0, 0.5])
    times = torch.tensor([100, 200, 150])
    events = torch.tensor([1, 1, 0])
    
    score = c_index(risk_scores, times, events)
    assert 0 <= score <= 1, "C-index should be between 0 and 1"
```

### Integration Tests

```python
# Test with real data
def test_metrics_integration():
    from dataloaders import SurvivalDataset
    from metrics import SurvivalEvaluator
    
    dataset = SurvivalDataset('test_data.csv')
    evaluator = SurvivalEvaluator(['c_index', 'brier_score'])
    
    for batch in dataset:
        results = evaluator.evaluate(
            batch['risk_scores'], 
            batch['time'], 
            batch['event']
        )
        assert all(0 <= v <= 1 for v in results.values())
```

## Dependencies

- **NumPy**: Numerical operations
- **SciPy**: Statistical functions
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization

## References

- [Concordance Index](https://en.wikipedia.org/wiki/Concordance_index)
- [Brier Score](https://en.wikipedia.org/wiki/Brier_score)
- [Time-dependent ROC Curves](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3079915/)
- [Survival Analysis Metrics](https://www.nature.com/articles/s41591-019-0447-9)

# Tests

This folder contains comprehensive test suites for all components of the survival analysis framework.

## Overview

The tests module provides unit tests, integration tests, and end-to-end tests for all components of the survival analysis framework. It ensures code quality, reliability, and correctness of the implementation.

## Available Test Files

### Core Component Tests

#### test_architectures.py
Tests for neural network architectures and model implementations.

```python
# Example test
def test_resnet3d_creation():
    from architectures import ResNet3D
    
    model = ResNet3D(model_name='resnet18', input_channels=1, num_classes=1)
    assert model is not None
    assert model.__class__.__name__ == 'ResNet3D'
```

#### test_data_preprocessing.py
Tests for data preprocessing utilities and pipelines.

```python
# Example test
def test_preprocessing_pipeline():
    from data.preprocessing import BasicPreprocessingPipeline
    
    pipeline = BasicPreprocessingPipeline()
    assert pipeline is not None
```

#### test_dataloaders.py
Tests for data loading and dataset classes.

```python
# Example test
def test_survival_dataset():
    from dataloaders import SurvivalDataset
    
    dataset = SurvivalDataset('test_data.csv')
    assert len(dataset) > 0
```

### Additional Test Files

#### test_losses.py
Tests for loss functions and survival analysis losses.

#### test_metrics.py
Tests for evaluation metrics and scoring functions.

#### test_optimizers.py
Tests for optimizer configurations and factory.

#### test_augmentations.py
Tests for data augmentation transforms.

#### test_experiments.py
Tests for experiment configurations and execution.

## Test Structure

```
tests/
├── __init__.py
├── README.md
├── run_all_tests.py          # Test runner
├── test_architectures.py     # Architecture tests
├── test_data_preprocessing.py # Preprocessing tests
├── test_dataloaders.py       # Data loading tests
├── test_losses.py           # Loss function tests
├── test_metrics.py          # Metrics tests
├── test_optimizers.py       # Optimizer tests
├── test_augmentations.py    # Augmentation tests
├── test_experiments.py      # Experiment tests
├── fixtures/                # Test fixtures and data
│   ├── sample_data.csv
│   ├── sample_image.nii.gz
│   └── test_config.yaml
└── utils/                   # Test utilities
    ├── test_helpers.py
    └── mock_data.py
```

## Running Tests

### Run All Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_architectures.py -v
```

### Using the Test Runner

```bash
# Use the provided test runner
python tests/run_all_tests.py

# Run with specific options
python tests/run_all_tests.py --verbose --coverage
```

### Individual Test Execution

```bash
# Run specific test
python -m pytest tests/test_architectures.py::test_resnet3d_creation -v

# Run tests matching pattern
python -m pytest tests/ -k "test_resnet" -v

# Run tests with specific markers
python -m pytest tests/ -m "slow" -v
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
def test_cox_loss_computation():
    from losses import CoxLoss
    import torch
    
    loss_fn = CoxLoss()
    risk_scores = torch.tensor([1.0, 2.0, 0.5])
    times = torch.tensor([100, 200, 150])
    events = torch.tensor([1, 1, 0])
    
    loss = loss_fn(risk_scores, times, events)
    assert loss.item() > 0
    assert not torch.isnan(loss)
```

### Integration Tests

Test component interactions:

```python
def test_training_pipeline():
    from dataloaders import SurvivalDataset
    from architectures import ResNet3D
    from losses import CoxLoss
    
    # Create components
    dataset = SurvivalDataset('test_data.csv')
    model = ResNet3D(model_name='resnet18')
    loss_fn = CoxLoss()
    
    # Test integration
    sample = dataset[0]
    output = model(sample['image'].unsqueeze(0))
    loss = loss_fn(output, sample['time'], sample['event'])
    
    assert not torch.isnan(loss)
```

### End-to-End Tests

Test complete workflows:

```python
def test_experiment_execution():
    from experiments.survival_analysis.template_experiment.run_experiment import run_experiment
    
    # Run experiment with test configuration
    results = run_experiment('test_config.yaml')
    
    assert results is not None
    assert 'c_index' in results
    assert results['c_index'] > 0
```

## Test Fixtures

### Sample Data

```python
# fixtures/sample_data.csv
patient_id,image_path,survival_time,event
P001,fixtures/sample_image.nii.gz,365.5,1
P002,fixtures/sample_image.nii.gz,200.0,0
P003,fixtures/sample_image.nii.gz,500.0,1
```

### Test Configuration

```yaml
# fixtures/test_config.yaml
experiment:
  name: "test_experiment"
  description: "Test experiment configuration"

dataset:
  train_csv: "fixtures/sample_data.csv"
  batch_size: 2
  num_workers: 0

model:
  architecture: "resnet3d"
  model_name: "resnet18"
  input_channels: 1
  num_classes: 1

training:
  epochs: 1
  learning_rate: 1e-4
```

## Test Utilities

### Test Helpers

```python
# utils/test_helpers.py
import torch
import numpy as np

def create_dummy_volume(shape=(64, 64, 64)):
    """Create dummy 3D volume for testing."""
    return torch.randn(shape)

def create_dummy_survival_data(num_samples=10):
    """Create dummy survival data for testing."""
    return {
        'times': torch.rand(num_samples) * 365,
        'events': torch.randint(0, 2, (num_samples,)),
        'risk_scores': torch.randn(num_samples)
    }
```

### Mock Data

```python
# utils/mock_data.py
class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'image': torch.randn(1, 64, 64, 64),
            'time': torch.rand(1) * 365,
            'event': torch.randint(0, 2, (1,))
        }
```

## Test Markers

### Performance Markers

```python
import pytest

@pytest.mark.slow
def test_large_dataset_processing():
    """Test that takes a long time to run."""
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test that requires GPU."""
    pass
```

### Category Markers

```python
@pytest.mark.unit
def test_unit_function():
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test."""
    pass

@pytest.mark.e2e
def test_end_to_end():
    """End-to-end test."""
    pass
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        python -m pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### Local CI

```bash
# Run CI locally
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term
```

## Test Coverage

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Components**: 90% coverage for core modules
- **New Code**: 100% coverage for new features

### Coverage Reports

```bash
# Generate coverage report
python -m pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Performance Testing

### Benchmark Tests

```python
import time
import pytest

def test_model_inference_speed():
    """Test model inference speed."""
    from architectures import ResNet3D
    
    model = ResNet3D(model_name='resnet18')
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    
    inference_time = end_time - start_time
    assert inference_time < 1.0  # Should be faster than 1 second
```

### Memory Tests

```python
def test_memory_usage():
    """Test memory usage during training."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run memory-intensive operation
    # ... training code ...
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB increase
```

## Best Practices

### Test Organization

1. **One Test File Per Module**: Keep tests organized by module
2. **Descriptive Test Names**: Use clear, descriptive test names
3. **Test Fixtures**: Use fixtures for common test data
4. **Test Isolation**: Ensure tests don't depend on each other

### Test Quality

1. **Comprehensive Coverage**: Test all code paths
2. **Edge Cases**: Test boundary conditions and edge cases
3. **Error Handling**: Test error conditions and exceptions
4. **Performance**: Include performance and memory tests

### Maintenance

1. **Regular Updates**: Keep tests updated with code changes
2. **Documentation**: Document test purpose and expected behavior
3. **CI Integration**: Integrate tests with continuous integration
4. **Coverage Monitoring**: Monitor test coverage regularly

## Dependencies

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Parallel test execution
- **torch**: PyTorch for model testing
- **numpy**: Numerical testing utilities

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python.org/3/library/unittest.html)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)
# TabM Encoder for Clinical Data

This directory contains a TabM-based encoder specifically designed for clinical data, including Agatston scores and calcium-omics data. The implementation extends the original [TabM model](https://github.com/yandex-research/tabm) to provide powerful encoding capabilities for tabular medical data.

## Overview

The TabM Encoder leverages the parameter-efficient ensembling approach of TabM to encode rich, high-dimensional features from tabular clinical data. It's particularly well-suited for:

- **Clinical variables**: Age, BMI, blood pressure, lab values, etc.
- **Agatston scores**: Calcium scoring measurements from CT scans
- **Calcium-omics data**: Multi-omics data related to calcium metabolism
- **Categorical variables**: Gender, race, medical history, etc.

## Key Features

- **Ensemble-based encoding**: Provides k different encoded representations for uncertainty quantification
- **Multi-modal data support**: Handles both numerical and categorical features seamlessly
- **Optional numerical embeddings**: Uses piecewise linear embeddings for better representation learning
- **Flexible architecture**: Supports different TabM architectures (tabm, tabm-mini, tabm-packed)
- **Clinical data optimized**: Specifically designed for medical tabular data
- **Survival analysis ready**: Can output single vectors directly for survival modeling

## Files

- `tabm.py`: Original TabM implementation (from yandex-research/tabm)
- `tabm_encoder.py`: TabM encoder implementation

## Installation

First, install the required dependencies:

```bash
pip install torch
pip install rtdl-num-embeddings
```

## Quick Start

### Basic Usage

```python
import torch
from tabm_encoder import TabMEncoder

# Initialize the encoder
model = TabMEncoder(
    n_num_features=20,  # Number of numerical features
    cat_cardinalities=[2, 3, 4],  # Categorical feature cardinalities
    d_block=512,  # Feature dimension
    k=32,  # Ensemble size
    n_blocks=3,  # Number of MLP blocks
    dropout=0.1
)

# Create sample data
batch_size = 64
x_num = torch.randn(batch_size, 20)  # Numerical features
x_cat = torch.randint(0, 3, (batch_size, 3))  # Categorical features

# Encode data
with torch.no_grad():
    # Get ensemble encoding (batch_size, k, d_block) - core TabM output
    ensemble_encoded = model.encode_ensemble(x_num, x_cat)
    
    # Get flattened encoding (batch_size, k * d_block) - for downstream ML
    flattened_encoded = model.encode_flattened(x_num, x_cat)
    
    # For survival analysis, use mean across ensemble
    survival_features = ensemble_encoded.mean(dim=1)  # (batch_size, d_block)
```

### Clinical Data Example

```python
import rtdl_num_embeddings

# Initialize for clinical data with embeddings
# First, compute bins from training data
x_train = torch.randn(1000, 40)  # Your training data
bins = rtdl_num_embeddings.compute_bins(x_train, n_bins=32)
num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    bins=bins,
    d_embedding=16,
    activation=True,
    version="B"
)

model = TabMEncoder(
    n_num_features=40,  # Clinical + Agatston + calcium-omics
    cat_cardinalities=[2, 3, 4],  # Gender, race, etc.
    num_embeddings=num_embeddings,
    d_block=512,
    k=32,
    n_blocks=3,
    dropout=0.1
)

# Prepare your clinical data
clinical_data = torch.randn(batch_size, 15)
agatston_data = torch.abs(torch.randn(batch_size, 5))  # Non-negative
calcium_omics_data = torch.randn(batch_size, 20)
categorical_data = torch.randint(0, 3, (batch_size, 3))

# Concatenate numerical features
x_num = torch.cat([clinical_data, agatston_data, calcium_omics_data], dim=1)

# Encode data
with torch.no_grad():
    # Get ensemble encoding (core TabM output)
    ensemble_features = model.encode_ensemble(x_num, categorical_data)
    
    # For survival analysis - use mean across ensemble
    survival_features = ensemble_features.mean(dim=1)  # (batch_size, d_block)
    
    # For uncertainty quantification
    mean_features = ensemble_features.mean(dim=1)  # Mean across ensemble
    std_features = ensemble_features.std(dim=1)    # Uncertainty measure
```

## Integration with MACE-Prediction

The TabM encoder can be easily integrated into your MACE-Prediction pipeline:

### 1. Direct Survival Analysis Integration

```python
# Get ensemble encoding (core TabM output)
ensemble_features = model.encode_ensemble(x_num, x_cat)

# For survival analysis - use mean across ensemble
survival_features = ensemble_features.mean(dim=1)  # (batch_size, d_block)

# Use directly with Cox proportional hazards model
cox_model.fit(survival_features, survival_times, events)

# Or with any other survival model
from sksurv.linear_model import CoxPHSurvivalAnalysis
survival_model = CoxPHSurvivalAnalysis()
survival_model.fit(survival_features, survival_times, events)
```

### 2. Ensemble-based Uncertainty Quantification

```python
# Get ensemble encoding for uncertainty quantification
ensemble_features = model.encode_ensemble(x_num, x_cat)

# Use mean features for predictions
mean_features = ensemble_features.mean(dim=1)
predictions = downstream_model(mean_features)

# Use standard deviation for uncertainty quantification
uncertainty = ensemble_features.std(dim=1)
```

### 3. Multi-Modal Integration

```python
# Encode different modalities separately
clinical_ensemble = model.encode_ensemble(clinical_data, categorical_data)
imaging_ensemble = model.encode_ensemble(agatston_data, None)
omics_ensemble = model.encode_ensemble(calcium_omics_data, None)

# Use mean across ensemble for each modality
clinical_encoded = clinical_ensemble.mean(dim=1)
imaging_encoded = imaging_ensemble.mean(dim=1)
omics_encoded = omics_ensemble.mean(dim=1)

# Combine modalities
combined_features = torch.cat([
    clinical_encoded, imaging_encoded, omics_encoded
], dim=1)
```

## Advanced Usage

### Custom Numerical Embeddings

```python
import rtdl_num_embeddings

# Create custom embeddings for numerical features
# First, compute bins from your training data
x_train = torch.randn(1000, 20)  # Your training data
bins = rtdl_num_embeddings.compute_bins(x_train, n_bins=32)
num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    bins=bins,
    d_embedding=16,
    activation=True,
    version="B"
)

model = TabMEncoder(
    n_num_features=20,
    num_embeddings=num_embeddings,
    d_block=512,
    k=32
)
```

### Different TabM Architectures

```python
# Use TabM-mini for faster training/inference
model_mini = TabMEncoder(
    n_num_features=20,
    d_block=512,
    k=32,
    arch_type='tabm-mini'
)

# Use TabM-packed for maximum performance
model_packed = TabMEncoder(
    n_num_features=20,
    d_block=512,
    k=32,
    arch_type='tabm-packed'
)
```

## Performance Considerations

- **Memory usage**: Ensemble size k affects memory usage. Use smaller k for memory-constrained environments
- **Training time**: TabM-mini trains faster but may have slightly lower performance
- **Feature dimension**: Larger d_block provides richer features but increases computational cost
- **Batch size**: Larger batches improve training efficiency

## Testing

The encoder can be tested by importing and using it directly:

```python
from tabm_encoder import TabMEncoder
import torch

# Test basic functionality
model = TabMEncoder(n_num_features=10, d_block=256, k=8)
x_num = torch.randn(32, 10)
encoded = model.encode_ensemble(x_num, None)
print(f"Encoded shape: {encoded.shape}")  # Should be (32, 8, 256)
print(f"Mean shape: {encoded.mean(dim=1).shape}")  # Should be (32, 256)
```

## References

- [TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)
- [Original TabM Repository](https://github.com/yandex-research/tabm)

## License

This implementation follows the same Apache 2.0 license as the original TabM repository.

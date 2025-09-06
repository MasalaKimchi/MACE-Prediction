# TabM Encoder

The TabM encoder provides an ensemble-based representation of tabular clinical data. It extends the original [TabM](https://github.com/yandex-research/tabm) implementation for use within MACE-Prediction.

## Installation

```bash
pip install torch rtdl-num-embeddings
```

## Basic usage

```python
from architectures.tabm_encoder import TabMEncoder
import torch

encoder = TabMEncoder(
    n_num_features=20,
    cat_cardinalities=[2, 5],
    d_block=512,
    k=16,
)

x_num = torch.randn(8, 20)
x_cat = torch.randint(0, 5, (8, 2))

# Ensemble features: (batch, k, d_block)
features = encoder.encode_ensemble(x_num, x_cat)

# Mean across ensemble for survival models
survival_feat = features.mean(dim=1)
```

## Notes

- Handles numerical and categorical inputs; optional numerical embeddings are supported via `rtdl-num-embeddings`.
- Reduce `k` or `d_block` for lower memory usage.

## Reference

- [TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)


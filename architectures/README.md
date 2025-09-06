# Architectures

This directory contains the neural network modules used in MACE-Prediction. It includes image encoders for 3D medical volumes, encoders for tabular data and a multimodal network that fuses both modalities for survival prediction.

## Components

### Image encoders
- **ResNet3D** – 3D ResNet variants (18/34/50/101/152)
- **Swin3D** – 3D Swin Transformer
- **SegFormer3D** – transformer-based encoder

All image encoders output a feature vector without a classification head.

### Tabular encoders
- **MLP** – simple feed-forward network
- **FT-Transformer** – transformer for tabular data
- **TabM** – ensemble tabular encoder (see `README_TabM.md`)

### Multimodal survival network
`MultimodalSurvivalNet` combines an image encoder, a tabular encoder, a fusion module and a linear risk head that produces `log_risk` scores.

## Quick start

```python
from architectures import build_multimodal_network

model = build_multimodal_network(
    img_encoder="resnet3d",
    tab_encoder="tabm",
    resnet_type="resnet50",
    tabular_dim=20,
    tab_kwargs={"n_num_features": 20, "k": 16},
)

log_risk = model(image_volume, tabular_data, tab_mask)
```

Setting `embed_dim=None` (default) preserves the native output dimension of each encoder.

## Testing

Run architecture tests:

```bash
python -m pytest tests/test_architectures.py -v
```

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)


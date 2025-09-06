# Multimodal Survival Network Architecture

## Overview

This document provides a comprehensive explanation of the multimodal survival network architecture, including how `log_risk` is produced and how different image encoders preserve their unique feature dimensions.

## 1. Log Risk Production Process

The `log_risk` is produced through a **4-step pipeline**:

```
Input Data → Encoder → Fusion → Risk Head → log_risk
```

### Step-by-Step Process:

1. **Image Encoding**: `img_feat = image_encoder(image)`
   - Input: `(B, C, D, H, W)` - 3D medical image
   - Output: `(B, img_dim)` - Image features

2. **Tabular Encoding**: `tab_feat = tabular_encoder(tabular)`
   - Input: `(B, tabular_dim)` - Clinical/tabular data
   - Output: `(B, tab_dim)` - Tabular features

3. **Fusion**: `fused = fusion(img_feat, tab_feat, tab_mask)`
   - Cross-Attention + FiLM fusion
   - Output: `(B, embed_dim)` - Fused multimodal features

4. **Risk Prediction**: `log_risk = risk_head(fused).squeeze(-1)`
   - Simple linear layer: `nn.Linear(embed_dim, 1)`
   - Output: `(B,)` - Raw logit risk scores

### Risk Head Details:
```python
self.risk_head = nn.Linear(embed_dim, 1)  # embed_dim → 1
log_risk = self.risk_head(fused).squeeze(-1)  # Remove last dimension
```

**Important**: `log_risk` is the **raw logit output** - not a probability. For survival analysis, you typically apply:
- **Cox model**: Use `log_risk` directly as hazard ratio
- **DeepSurv**: Use `log_risk` as risk score
- **Probability**: Apply `sigmoid(log_risk)` for binary classification

## 2. Image Encoder Dimension Preservation

### The Problem (Fixed)
Previously, all encoders were forced to output the same `embed_dim` (default 128), which:
- **Wasted ResNet3D's rich 512/2048 features** by projecting down to 128
- **Lost Swin3D's 768 features** by projecting down to 128
- **Reduced SegFormer3D's 256 features** to 128

### The Solution (Current)
Each image encoder now **preserves its unique feature dimensions**:

| Encoder | Original Dimensions | Preserved Output |
|---------|-------------------|------------------|
| **ResNet3D-18/34** | 512 features | `(B, 512)` |
| **ResNet3D-50/101/152** | 2048 features | `(B, 2048)` |
| **Swin3D** | 768 features (48×2⁴) | `(B, 768)` |
| **SegFormer3D** | 256 features | `(B, 256)` |

### Implementation:
```python
# When embed_dim=None, preserve encoder-specific dimensions
if embed_dim is None:
    embed_dim = img_dim  # Use encoder's natural dimension
```

## 3. Complete Architecture Diagram

```mermaid
graph TB
    subgraph "Input Data"
        IMG[3D Medical Image<br/>B×C×D×H×W]
        TAB[Tabular Data<br/>B×tabular_dim]
        MASK[Tabular Mask<br/>B×1]
    end
    
    subgraph "Image Encoders (Preserved Dimensions)"
        RESNET[ResNet3D<br/>512/2048 features]
        SWIN[Swin3D<br/>768 features]
        SEG[SegFormer3D<br/>256 features]
    end
    
    subgraph "Tabular Encoders"
        MLP[MLP<br/>embed_dim features]
        TABM[TabM Ensemble<br/>B×k×embed_dim]
    end
    
    subgraph "Fusion Module"
        CA[Cross-Attention<br/>q=img, k=v=tab]
        FILM[FiLM Modulation<br/>γ×x + β + residual]
        POOL[Pooling<br/>mean/attention]
    end
    
    subgraph "Risk Prediction"
        RISK[Risk Head<br/>Linear(embed_dim, 1)]
        LOG[log_risk<br/>B×1]
    end
    
    IMG --> RESNET
    IMG --> SWIN
    IMG --> SEG
    
    TAB --> MLP
    TAB --> TABM
    
    RESNET --> CA
    SWIN --> CA
    SEG --> CA
    MLP --> CA
    TABM --> CA
    
    CA --> FILM
    FILM --> POOL
    POOL --> RISK
    RISK --> LOG
    
    MASK -.-> CA
    MASK -.-> FILM
```

## 4. Detailed Information Flow

### 4.1 Image Encoder Processing

#### ResNet3D:
```
Input: (B, 1, D, H, W)
↓
3D Convolutional Layers
↓
Global Average Pooling
↓
Output: (B, 512) or (B, 2048)
```

#### Swin3D:
```
Input: (B, 1, D, H, W)
↓
Patch Embedding + Swin Transformer
↓
Multi-scale Features
↓
Global Average Pooling
↓
Output: (B, 768)
```

#### SegFormer3D:
```
Input: (B, 1, D, H, W)
↓
MixVision Transformer
↓
Multi-scale Features
↓
Global Average Pooling
↓
Linear Projection
↓
Output: (B, 256)
```

### 4.2 Tabular Encoder Processing

#### MLP:
```
Input: (B, tabular_dim)
↓
Linear(tabular_dim, embed_dim)
↓
ReLU
↓
Linear(embed_dim, embed_dim)
↓
ReLU
↓
Output: (B, embed_dim)
```

#### TabM:
```
Input: (B, tabular_dim)
↓
TabM Ensemble (k=16)
↓
Output: (B, k, embed_dim)
↓
Mean Pooling (optional)
↓
Output: (B, embed_dim)
```

### 4.3 Fusion Process

#### Cross-Attention:
```
Image Features: (B, img_dim)
Tabular Features: (B, tab_dim)
↓
Project tabular to img_dim
↓
Q = Image Features
K, V = Tabular Features
↓
Attention(Q, K, V)
↓
Residual Connection
↓
Output: (B, img_dim)
```

#### FiLM Modulation:
```
Image Features: (B, img_dim)
Tabular Features: (B, tab_dim)
↓
LayerNorm(Image Features)
↓
MLP(Tabular Features) → γ, β
↓
γ × Image + β
↓
Residual Connection
↓
Output: (B, img_dim)
```

## 5. Usage Examples

### 5.1 Preserve Encoder Dimensions (Recommended)
```python
# Let each encoder use its natural dimensions
model = build_multimodal_network(
    img_encoder="resnet3d",
    tab_encoder="tabm",
    embed_dim=None,  # Preserve encoder-specific dimensions
    resnet_type="resnet50",  # Will use 2048 features
    tab_kwargs={"n_num_features": 20, "k": 16}
)
# Result: ResNet3D(2048) + TabM(2048) + Fusion(2048) + Risk(1)
```

### 5.2 Force Specific Dimension
```python
# Force all encoders to use 512 dimensions
model = build_multimodal_network(
    img_encoder="swin3d",
    tab_encoder="tabm", 
    embed_dim=512,  # Force 512 dimensions
    tab_kwargs={"n_num_features": 20, "k": 16}
)
# Result: Swin3D(768→512) + TabM(512) + Fusion(512) + Risk(1)
```

### 5.3 Different Encoder Combinations
```python
# ResNet3D + TabM (preserves 512 features)
model1 = build_multimodal_network(
    img_encoder="resnet3d", tab_encoder="tabm", 
    resnet_type="resnet18", embed_dim=None
)

# Swin3D + MLP (preserves 768 features)  
model2 = build_multimodal_network(
    img_encoder="swin3d", tab_encoder="mlp",
    embed_dim=None
)

# SegFormer3D + TabM (preserves 256 features)
model3 = build_multimodal_network(
    img_encoder="segformer3d", tab_encoder="tabm",
    embed_dim=None
)
```

## 6. Modality Dropout Handling

The architecture is **robust to missing tabular data**:

```python
# Some samples missing tabular data
tab_mask = torch.tensor([True, True, False, False])

# Forward pass handles missing data gracefully
result = model(image, tabular, tab_mask)
# Samples 2,3 will use only image features
# Samples 0,1 will use both image + tabular features
```

**Fast Path**: When `tab_mask.sum() == 0`, the fusion module returns pooled image features directly, avoiding unnecessary computation.

## 7. Performance Considerations

### 7.1 Memory Usage
- **ResNet3D-50**: ~2048 features (higher memory)
- **Swin3D**: ~768 features (balanced)
- **SegFormer3D**: ~256 features (lower memory)

### 7.2 Computational Cost
- **ResNet3D**: Fastest (convolutional)
- **Swin3D**: Moderate (attention-based)
- **SegFormer3D**: Slower (complex transformer)

### 7.3 Feature Richness
- **ResNet3D**: Rich spatial features, good for anatomical structures
- **Swin3D**: Hierarchical features, good for multi-scale patterns
- **SegFormer3D**: Semantic features, good for complex textures

## 8. Recommendations

1. **Use `embed_dim=None`** to preserve encoder-specific dimensions
2. **Choose encoder based on data**:
   - ResNet3D: For anatomical/structural analysis
   - Swin3D: For multi-scale pattern recognition
   - SegFormer3D: For texture/semantic analysis
3. **Use TabM** for uncertainty quantification in tabular data
4. **Monitor attention weights** for interpretability
5. **Test modality dropout** scenarios for robustness

This architecture provides maximum flexibility while preserving the unique strengths of each encoder type!

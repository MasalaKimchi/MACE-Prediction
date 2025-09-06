"""Multimodal survival network with cross-attention + FiLM fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .fusion import CrossAttentionFiLMFusion
from .resnet3d import build_network


class MultimodalSurvivalNet(nn.Module):
    """Single-stage multimodal survival model with pluggable encoders."""

    def __init__(
        self,
        image_encoder: nn.Module,
        tabular_encoder: Optional[nn.Module],
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.fusion = CrossAttentionFiLMFusion(
            img_dim=embed_dim, tab_dim=embed_dim, hidden_dim=embed_dim
        )
        self.risk_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        image: torch.Tensor,
        tabular: Optional[torch.Tensor],
        tab_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        img_feat = self.image_encoder(image)
        tab_feat = None
        if tabular is not None and self.tabular_encoder is not None:
            try:
                tab_feat = self.tabular_encoder(tabular)
            except TypeError:  # e.g. FTTransformer expects x_num, x_cat
                tab_feat = self.tabular_encoder(tabular, None)
        fused = self.fusion(img_feat, tab_feat, tab_mask)
        log_risk = self.risk_head(fused).squeeze(-1)
        return {
            "log_risk": log_risk,
            "image_embed": img_feat,
            "tab_embed": tab_feat,
            "fused": fused,
        }


def build_multimodal_network(
    img_encoder: str = "resnet3d",
    tab_encoder: str = "mlp",
    resnet_type: str = "resnet18",
    in_channels: int = 1,
    tabular_dim: int = 16,
    embed_dim: int = 128,
    img_kwargs: dict | None = None,
    tab_kwargs: dict | None = None,
) -> MultimodalSurvivalNet:
    """Factory for :class:`MultimodalSurvivalNet` with configurable encoders."""

    img_kwargs = img_kwargs or {}
    tab_kwargs = tab_kwargs or {}

    if img_encoder == "resnet3d":
        image_model = build_network(
            resnet_type=resnet_type, in_channels=in_channels, num_classes=embed_dim
        )
    elif img_encoder == "swin3d":
        from .swin3d import Swin3DEncoder

        image_model = Swin3DEncoder(
            in_ch=in_channels, embed_dim=embed_dim, **img_kwargs
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported image encoder: {img_encoder}")

    if tab_encoder == "mlp":
        tab_model = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
    elif tab_encoder == "ft_transformer":
        from .ft_transformer import FTTransformer

        num_numeric = tab_kwargs.get("num_numeric", tabular_dim)
        cat_card = tab_kwargs.get("cat_cardinalities")
        tab_model = FTTransformer(
            num_numeric=num_numeric,
            cat_cardinalities=cat_card,
            embed_dim=embed_dim,
            n_heads=tab_kwargs.get("n_heads", 4),
            depth=tab_kwargs.get("depth", 2),
            dropout=tab_kwargs.get("dropout", 0.1),
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported tabular encoder: {tab_encoder}")

    return MultimodalSurvivalNet(image_model, tab_model, embed_dim)

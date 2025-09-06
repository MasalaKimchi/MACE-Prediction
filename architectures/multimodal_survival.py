"""Multimodal survival network with cross-attention + FiLM fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .resnet3d import build_network
from .fusion import CrossAttentionFiLMFusion


class MultimodalSurvivalNet(nn.Module):
    """Single-stage multimodal survival model.

    The model encodes imaging data with a 3D ResNet encoder and tabular data
    with a small MLP. Features are fused using cross-attention followed by
    FiLM modulation. The fused representation is used to predict the log-risk
    for Cox-based objectives and can also serve as embedding for contrastive
    losses.
    """

    def __init__(
        self,
        image_arch: str,
        in_channels: int,
        tabular_dim: int,
        fusion_dim: int = 128,
    ) -> None:
        super().__init__()
        # Image encoder produces feature vector of size fusion_dim
        self.image_encoder = build_network(
            resnet_type=image_arch,
            in_channels=in_channels,
            num_classes=fusion_dim,
        )
        # Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = CrossAttentionFiLMFusion(
            img_dim=fusion_dim, tab_dim=fusion_dim, hidden_dim=fusion_dim
        )
        self.risk_head = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        image: torch.Tensor,
        tabular: Optional[torch.Tensor],
        tab_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Parameters
        ----------
        image : torch.Tensor
            Image tensor ``(B, C, H, W, D)``.
        tabular : torch.Tensor, optional
            Tabular data ``(B, F)``.
        tab_mask : torch.Tensor, optional
            Boolean mask indicating which samples contain tabular data.

        Returns
        -------
        dict
            Dictionary with keys ``log_risk`` and modality embeddings.
        """
        img_feat = self.image_encoder(image)
        img_feat = img_feat.view(img_feat.size(0), -1)
        tab_feat = self.tabular_encoder(tabular) if tabular is not None else None
        fused = self.fusion(img_feat, tab_feat, tab_mask)
        log_risk = self.risk_head(fused).squeeze(-1)
        return {
            "log_risk": log_risk,
            "image_embed": img_feat,
            "tab_embed": tab_feat,
            "fused": fused,
        }


def build_multimodal_network(
    image_arch: str = "resnet18",
    in_channels: int = 1,
    tabular_dim: int = 16,
    fusion_dim: int = 128,
) -> MultimodalSurvivalNet:
    """Factory for :class:`MultimodalSurvivalNet`.

    Parameters
    ----------
    image_arch : str, optional
        Backbone architecture for images.
    in_channels : int, optional
        Number of image channels.
    tabular_dim : int, optional
        Dimension of tabular input features.
    fusion_dim : int, optional
        Dimension of intermediate fusion representation.
    """
    return MultimodalSurvivalNet(image_arch, in_channels, tabular_dim, fusion_dim)

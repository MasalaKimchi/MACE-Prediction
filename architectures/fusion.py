import torch
import torch.nn as nn
from typing import Optional

class CrossAttentionFiLMFusion(nn.Module):
    """Cross-attention followed by FiLM fusion for two modalities.

    Parameters
    ----------
    img_dim : int
        Dimension of image features.
    tab_dim : int
        Dimension of tabular features.
    hidden_dim : int
        Dimension used inside the attention module.
    dropout : float, optional
        Dropout applied after attention, by default 0.1.

    Notes
    -----
    This module is modality-dropout aware. If the tabular modality is missing
    (``tab_feats`` is ``None``), the image features are passed through
    unchanged. ``tab_mask`` can be provided to indicate which samples contain
    tabular information.
    """

    def __init__(self, img_dim: int, tab_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=img_dim, num_heads=1, batch_first=True)
        self.tab_proj = nn.Linear(tab_dim, img_dim)
        self.gamma = nn.Linear(tab_dim, img_dim)
        self.beta = nn.Linear(tab_dim, img_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        img_feats: torch.Tensor,
        tab_feats: Optional[torch.Tensor],
        tab_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse modalities.

        Parameters
        ----------
        img_feats : torch.Tensor
            Image feature tensor of shape ``(B, C)``.
        tab_feats : torch.Tensor, optional
            Tabular feature tensor of shape ``(B, D)``.
        tab_mask : torch.Tensor, optional
            Boolean mask of shape ``(B,)`` indicating presence of tabular
            features.

        Returns
        -------
        torch.Tensor
            Fused feature representation ``(B, C)``.
        """
        if tab_feats is None:
            return img_feats

        if tab_mask is not None:
            tab_feats = tab_feats * tab_mask.unsqueeze(1)

        # Cross attention: tabular query attends to image features
        q = self.tab_proj(tab_feats).unsqueeze(1)  # (B,1,C)
        k = v = img_feats.unsqueeze(1)  # (B,1,C)
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.dropout(attn_out.squeeze(1))  # (B,C)

        # FiLM modulation
        gamma = self.gamma(tab_feats)
        beta = self.beta(tab_feats)
        fused = (1 + gamma) * attn_out + beta + img_feats
        return fused

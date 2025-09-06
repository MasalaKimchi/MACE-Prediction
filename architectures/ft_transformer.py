from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


class FTTransformer(nn.Module):
    """Implementation of the FT-Transformer for tabular data.

    References
    ----------
    Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*,
    NeurIPS 2021.
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: Optional[List[int]] | None,
        embed_dim: int = 256,
        n_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities or []

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_linear = nn.Linear(1, embed_dim) if num_numeric > 0 else None
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, embed_dim) for card in self.cat_cardinalities]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning the CLS token embedding."""
        bsz = x_num.size(0) if x_num is not None else x_cat.size(0)  # type: ignore[arg-type]
        tokens: list[torch.Tensor] = []

        if self.num_linear is not None and x_num is not None:
            num_tok = self.num_linear(x_num.unsqueeze(-1))  # (B, N_num, E)
            tokens.append(num_tok)

        if self.cat_embeddings and x_cat is not None:
            cat_tok = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            tokens.append(torch.stack(cat_tok, dim=1))

        if tokens:
            x = torch.cat(tokens, dim=1)
        else:
            x = torch.zeros(bsz, 0, self.embed_dim, device=self.cls_token.device, dtype=self.cls_token.dtype)

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.norm(x[:, 0])

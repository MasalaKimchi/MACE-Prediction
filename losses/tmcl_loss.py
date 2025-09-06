"""Time-aware multimodal contrastive losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def time_aware_triplet_loss(
    img_embed: torch.Tensor,
    tab_embed: torch.Tensor,
    times: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet contrastive loss with time-scaled margin.

    Uses a simple negative sampling by rolling the batch.
    """
    neg_tab = tab_embed.roll(shifts=1, dims=0)
    neg_time = times.roll(shifts=1, dims=0)
    d_pos = F.pairwise_distance(img_embed, tab_embed)
    d_neg = F.pairwise_distance(img_embed, neg_tab)
    time_scale = 1.0 + torch.abs(times - neg_time)
    loss = F.relu(d_pos - d_neg + margin * time_scale)
    return loss.mean()


def time_aware_pairwise_contrastive(
    img_embed: torch.Tensor,
    tab_embed: torch.Tensor,
    times: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pairwise InfoNCE-style contrastive loss with time-aware temperature."""
    logits = img_embed @ tab_embed.t()
    dt = torch.abs(times.unsqueeze(1) - times.unsqueeze(0))
    temp = temperature / (1.0 + dt)
    logits = logits / temp
    labels = torch.arange(img_embed.size(0), device=img_embed.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_j)

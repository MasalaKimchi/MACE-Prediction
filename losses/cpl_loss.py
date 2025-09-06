"""Concordance-pairwise logistic loss with IPCW and time-aware temperature."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from .ddp_utils import gather_tensor


def _ipcw(times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Estimate inverse probability of censoring weights via KM."""
    order = torch.argsort(times)
    t = times[order]
    e = events[order]
    censor = 1.0 - e
    n = t.numel()
    at_risk = n - torch.arange(n, device=t.device)
    hazard = censor / torch.clamp(at_risk, min=1.0)
    surv = torch.cumprod(1.0 - hazard, dim=0)
    ipcw = 1.0 / torch.clamp(surv, min=1e-12)
    return ipcw[torch.argsort(order)]


def concordance_pairwise_loss(
    log_risks: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    temperature: float = 1.0,
    gather: bool = False,
) -> torch.Tensor:
    """Concordance pairwise logistic loss.

    Parameters
    ----------
    log_risks : torch.Tensor
        Predicted log-risk scores ``(N,)``.
    times : torch.Tensor
        Event or censoring times ``(N,)``.
    events : torch.Tensor
        Event indicators ``(N,)``.
    temperature : float, optional
        Base temperature for logistic function.
    gather : bool, optional
        If ``True``, gather tensors across distributed processes.
    """
    if gather:
        log_risks = gather_tensor(log_risks)
        times = gather_tensor(times)
        events = gather_tensor(events)

    ipcw = _ipcw(times, events)
    # Construct pairwise comparisons: i event and j with time > t_i
    t_i = times.unsqueeze(1)
    t_j = times.unsqueeze(0)
    mask = (t_i < t_j) & (events.unsqueeze(1) > 0.5)
    if not mask.any():
        return torch.tensor(0.0, device=times.device, dtype=times.dtype)

    risk_diff = log_risks.unsqueeze(1) - log_risks.unsqueeze(0)
    dt = torch.abs(t_i - t_j)
    temp = temperature / (dt + 1.0)
    logits = risk_diff / temp
    weights = ipcw.unsqueeze(1)
    loss = F.softplus(-logits) * weights
    return loss[mask].mean()

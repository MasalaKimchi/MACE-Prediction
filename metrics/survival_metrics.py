from __future__ import annotations

import torch
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore


@torch.no_grad()
def cindex_torchsurv(log_risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> float:
    """Harrell-style C-index computed by TorchSurv on concatenated tensors."""
    c = ConcordanceIndex()
    return float(c(log_risk.view(-1, 1), event.bool(), time).item())


@torch.no_grad()
def td_auc_torchsurv(
    log_risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor, horizons: list[float]
) -> dict[float, float]:
    """Time-dependent AUC at specified horizons; returns {h: auc}."""
    auc = Auc()
    out: dict[float, float] = {}
    for h in horizons:
        out[h] = float(
            auc(
                log_risk.view(-1, 1),
                event.bool(),
                time,
                new_time=torch.tensor(h, device=log_risk.device, dtype=time.dtype),
            ).item()
        )
    return out


@torch.no_grad()
def brier_ibs_torchsurv(
    surv_matrix: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    time_grid: torch.Tensor,
) -> tuple[dict[float, float], float]:
    """
    BrierScore expects S(t|x) over a grid T (shape [N, |T|]).
    Returns BS at each grid point and the integrated Brier score over the grid.
    """
    bs = BrierScore()
    vals = bs(surv_matrix, event.bool(), time, new_time=time_grid)  # shape [|T|]
    ibs = float(bs.integral().item())
    out = {float(t.cpu().item()): float(v.cpu().item()) for t, v in zip(time_grid, vals)}
    return out, ibs


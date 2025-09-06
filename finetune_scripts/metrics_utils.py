"""
Evaluation metrics for survival analysis.
Contains functions for computing concordance index, Brier score, and time-dependent AUC.
"""

import torch
from typing import Callable

# Optional: use torchsurv package metrics if available (class-based API)
try:
    from torchsurv.metrics import ConcordanceIndex as TSConcordanceIndex
except Exception:
    TSConcordanceIndex = None

try:
    from torchsurv.metrics import BrierScore as TSBrierScore
except Exception:
    TSBrierScore = None

try:
    from torchsurv.metrics import Auc as TSAuc
except Exception:
    TSAuc = None

from .survival_utils import estimate_breslow_baseline, baseline_cumhaz_at


@torch.no_grad()
def concordance_index(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> float:
    """
    Compute C-index. Higher risk -> earlier event expected.
    Ignores non-comparable pairs (e.g., both censored or no ordering).
    
    Args:
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
        risks: (N,) predicted risk scores
    
    Returns:
        Concordance index (float between 0 and 1)
    """
    # Prefer torchsurv implementation if available
    if TSConcordanceIndex is not None:
        try:
            ci = TSConcordanceIndex()
            # torchsurv typically expects (predictions, event, time)
            return float(ci(risks, events, times))
        except Exception:
            pass
    
    t = times.cpu().numpy()
    e = events.cpu().numpy().astype(bool)
    r = risks.cpu().numpy()

    num, den = 0, 0
    n = len(t)
    for i in range(n):
        for j in range(n):
            if t[i] == t[j]:
                continue
            # Comparable if the earlier time had an event
            if t[i] < t[j] and e[i]:
                den += 1
                if r[i] > r[j]:
                    num += 1
                elif r[i] == r[j]:
                    num += 0.5
            elif t[j] < t[i] and e[j]:
                den += 1
                if r[j] > r[i]:
                    num += 1
                elif r[i] == r[j]:
                    num += 0.5
    return float(num / den) if den > 0 else 0.0


def brier_score_ipcw(times: torch.Tensor, events: torch.Tensor, S_probs: torch.Tensor, 
                     t_eval: torch.Tensor, G_of_t: Callable) -> torch.Tensor:
    """
    IPCW Brier score at times t_eval. Returns vector (T,).
    
    Args:
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
        S_probs: (N, T) survival probabilities at each eval time
        t_eval: (T,) evaluation times
        G_of_t: Function to evaluate censoring survival
    
    Returns:
        (T,) Brier scores at each evaluation time
    """
    N, T = S_probs.shape
    device = times.device
    bs = torch.zeros(T, device=device, dtype=times.dtype)
    for k in range(T):
        t = t_eval[k]
        S_t = S_probs[:, k]
        # Indicators
        event_obs = (times <= t) & (events > 0.5)
        cens_after_t = times > t
        # Weights
        w1 = torch.tensor(0.0, device=device, dtype=times.dtype)
        if event_obs.any():
            G_Tminus = G_of_t(times[event_obs], left_limit=True)
            comp1 = ((0.0 - S_t[event_obs]) ** 2) / torch.clamp(G_Tminus, min=1e-12)
            w1 = comp1.mean() * (event_obs.float().mean() / max(event_obs.float().mean(), torch.tensor(1e-12, device=device)))  # stabilize
        w2 = torch.tensor(0.0, device=device, dtype=times.dtype)
        if cens_after_t.any():
            G_t = torch.clamp(G_of_t(t.repeat(int(cens_after_t.sum().item()))), min=1e-12)
            comp2 = ((1.0 - S_t[cens_after_t]) ** 2) / G_t
            w2 = comp2.mean() * (cens_after_t.float().mean() / max(cens_after_t.float().mean(), torch.tensor(1e-12, device=device)))
        # Average of the two components weighted by their prevalence
        p1 = event_obs.float().mean()
        p2 = cens_after_t.float().mean()
        denom = torch.clamp(p1 + p2, min=1e-12)
        bs[k] = (p1 * w1 + p2 * w2) / denom
    return bs


def integrated_brier_score(times: torch.Tensor, events: torch.Tensor, S_probs: torch.Tensor, 
                          t_eval: torch.Tensor, G_of_t: Callable) -> float:
    """
    Compute integrated Brier score over time.
    
    Args:
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
        S_probs: (N, T) survival probabilities at each eval time
        t_eval: (T,) evaluation times
        G_of_t: Function to evaluate censoring survival
    
    Returns:
        Integrated Brier score (float)
    """
    # Prefer torchsurv implementation if available (BrierScore class with integral())
    if TSBrierScore is not None:
        try:
            bs = TSBrierScore()
            _ = bs(S_probs, events, times, new_time=t_eval)
            return float(bs.integral())
        except Exception:
            pass
    
    bs = brier_score_ipcw(times, events, S_probs, t_eval, G_of_t)
    # Trapezoidal rule over t_eval
    if len(t_eval) < 2:
        return float(bs.mean().item())
    # Ensure ascending
    order = torch.argsort(t_eval)
    te = t_eval[order]
    bs_sorted = bs[order]
    area = torch.trapz(bs_sorted, te)
    total_time = te[-1] - te[0]
    return float((area / torch.clamp(total_time, min=1e-12)).item())


def time_dependent_auc(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor,
                      t_eval: torch.Tensor, G_of_t: Callable) -> torch.Tensor:
    """
    Dynamic AUC at times t_eval using IPCW.
    
    Args:
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
        risks: (N,) predicted risk scores
        t_eval: (T,) evaluation times
        G_of_t: Function to evaluate censoring survival
    
    Returns:
        (T,) time-dependent AUC values at each evaluation time
    """
    # Prefer torchsurv implementation if available (Auc class, vectorized new_time)
    if TSAuc is not None:
        try:
            auc = TSAuc()
            vals = auc(risks, events, times, new_time=t_eval)
            return vals if isinstance(vals, torch.Tensor) else torch.tensor(vals, device=times.device, dtype=times.dtype)
        except Exception:
            pass
    
    device = times.device
    aucs = torch.zeros_like(t_eval, dtype=times.dtype, device=device)
    for k in range(t_eval.numel()):
        t = t_eval[k]
        cases = (times <= t) & (events > 0.5)
        controls = times > t
        if not cases.any() or not controls.any():
            aucs[k] = torch.tensor(0.0, device=device, dtype=times.dtype)
            continue
        r_cases = risks[cases]
        r_ctrls = risks[controls]
        # IPCW weights
        w_i = 1.0 / torch.clamp(G_of_t(times[cases], left_limit=True), min=1e-12)
        v_j = 1.0 / torch.clamp(G_of_t(t.repeat(int(controls.sum().item()))), min=1e-12)
        # Compute weighted concordance for all pairs
        # Expand to pairwise matrix
        rc = r_cases.unsqueeze(1)
        rj = r_ctrls.unsqueeze(0)
        Wij = w_i.unsqueeze(1) * v_j.unsqueeze(0)
        concordant = (rc > rj).to(times.dtype)
        ties = (rc == rj).to(times.dtype) * 0.5
        num = (Wij * (concordant + ties)).sum()
        den = Wij.sum()
        aucs[k] = (num / torch.clamp(den, min=1e-12)).to(times.dtype)
    return aucs


@torch.no_grad()
def cox_survival_matrix(
    log_risks: torch.Tensor,
    events: torch.Tensor,
    times: torch.Tensor,
    time_grid: torch.Tensor,
) -> torch.Tensor:
    """Compute S(t|x) for a Cox model using Breslow baseline on a grid.

    Args:
        log_risks: (N,) log-risk scores.
        events: (N,) event indicators.
        times: (N,) observed times.
        time_grid: (T,) grid of times at which to evaluate survival.

    Returns:
        (N, T) survival probability matrix.
    """
    event_times, H0 = estimate_breslow_baseline(times, events, log_risks)
    H_t = baseline_cumhaz_at(event_times, H0, time_grid)
    S0 = torch.exp(-H_t)  # (T,)
    # S(t|x) = S0(t) ** exp(r)
    return torch.pow(S0.unsqueeze(0), torch.exp(log_risks).unsqueeze(1))

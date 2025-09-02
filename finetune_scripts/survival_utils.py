"""
Survival analysis utility functions for Cox proportional hazards model.
Contains core survival analysis functions like Cox loss, baseline estimation, and survival probability prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple


def cox_ph_loss(log_risks: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Breslow approximation for Cox partial likelihood.
    C-index explanation: compares all comparable patient pairs; a pair is concordant
    if the patient who experiences the event earlier has a higher predicted risk.
    
    Args:
        log_risks: (N,) predicted log-risk scores
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
    
    Returns:
        Negative partial log-likelihood (scalar)
    """
    # Sort by descending time so that risk set for i is [0..i]
    order = torch.argsort(times, descending=True)
    log_risks = log_risks[order]
    events = events[order]
    times = times[order]

    # Group ties by time
    unique_times, inverse_idx, counts = torch.unique_consecutive(times, return_inverse=True, return_counts=True)

    # CumSum of exp(log_risk) over sorted order provides denominator for risk sets
    exp_log_risks = torch.exp(log_risks)
    cum_sum = torch.cumsum(exp_log_risks, dim=0)

    # For each tied group, denominator is sum of risk for all samples with time >= that group's time
    # Build mapping from sample index -> denom using cum_sum at the last index of each tie group
    group_last_index = torch.cumsum(counts, dim=0) - 1  # last index per group
    denom_per_group = cum_sum[group_last_index]  # shape (G,)
    denom = denom_per_group[inverse_idx]

    # Numerator: sum of log_risk over events, grouped by tied time
    # For Breslow, events within a tied group share the same denominator raised to num_events_in_group
    # Implement by summing over events and subtracting event_count * log(denom_group)
    event_mask = events > 0.5
    if not torch.any(event_mask):
        return torch.tensor(0.0, device=log_risks.device, dtype=log_risks.dtype)

    # Sum log_risk over events
    log_num = log_risks[event_mask].sum()

    # Count events per group, then compute sum over groups: e_g * log(denom_g)
    event_groups = inverse_idx[event_mask]
    # Compute counts per group efficiently
    max_group = int(unique_times.shape[0])
    event_counts_per_group = torch.zeros(max_group, device=log_risks.device, dtype=log_risks.dtype)
    event_counts_per_group.scatter_add_(0, event_groups, torch.ones_like(event_groups, dtype=log_risks.dtype))
    log_denom_sum = (event_counts_per_group * torch.log(denom_per_group)).sum()

    neg_log_likelihood = -(log_num - log_denom_sum)
    return neg_log_likelihood / events.sum()


def estimate_breslow_baseline(times: torch.Tensor, events: torch.Tensor, log_risks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate baseline cumulative hazard H0(t) via Breslow.
    
    Args:
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
        log_risks: (N,) predicted log-risk scores
    
    Returns:
        event_times (ascending) and H0 values at those times (same length)
    """
    device = times.device
    # Sort by time ascending
    order = torch.argsort(times, descending=False)
    t_sorted = times[order]
    e_sorted = events[order]
    lr_sorted = log_risks[order]

    # Unique event times
    is_event = e_sorted > 0.5
    event_times = torch.unique(t_sorted[is_event])
    if event_times.numel() == 0:
        return event_times, torch.zeros(0, device=device, dtype=times.dtype)

    # For each event time, compute number of events and risk set sum
    H0 = []
    cumH = 0.0
    exp_lr = torch.exp(lr_sorted)
    n = len(t_sorted)
    for t in event_times:
        # d_k: number of events at time t
        mask_t_event = (t_sorted == t) & is_event
        d_k = mask_t_event.sum().item()
        # risk set: subjects with time >= t
        risk_mask = t_sorted >= t
        denom = exp_lr[risk_mask].sum().item()
        if denom <= 0:
            continue
        cumH += d_k / denom
        H0.append(cumH)
    return event_times, torch.tensor(H0, device=device, dtype=times.dtype)


def baseline_cumhaz_at(event_times: torch.Tensor, H0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
    """
    Piecewise-constant cumulative hazard evaluated at t_eval using step function defined at event_times.
    
    Args:
        event_times: (G,) unique event times in ascending order
        H0: (G,) cumulative hazard values at event_times
        t_eval: (T,) evaluation times
    
    Returns:
        (T,) cumulative hazard values at t_eval
    """
    if event_times.numel() == 0:
        return torch.zeros_like(t_eval)
    # For each t_eval, find rightmost event_time <= t
    idx = torch.searchsorted(event_times, t_eval, right=True) - 1
    idx = torch.clamp(idx, min=0)
    return H0[idx]


def predict_survival_probs(log_risks: torch.Tensor, event_times: torch.Tensor, H0: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
    """
    Compute survival probabilities S_i(t) = exp(-H0(t) * exp(lp_i)).
    
    Args:
        log_risks: (N,) predicted log-risk scores
        event_times: (G,) unique event times in ascending order
        H0: (G,) cumulative hazard values at event_times
        t_eval: (T,) evaluation times
    
    Returns:
        (N, T) survival probabilities for each patient at each evaluation time
    """
    H_t = baseline_cumhaz_at(event_times, H0, t_eval)  # (T,)
    exp_lr = torch.exp(log_risks).unsqueeze(1)  # (N,1)
    S = torch.exp(-exp_lr * H_t.unsqueeze(0))
    return S


def km_censoring(times: torch.Tensor, events: torch.Tensor):
    """
    Kaplan-Meier estimator for censoring survival G(t) = P(C >= t).
    
    Args:
        times: (N,) survival/censoring times
        events: (N,) 1 if event occurred, 0 if censored
    
    Returns:
        Function G_of_t(query_t, left_limit=False) that evaluates censoring survival
    """
    device = times.device
    # Censoring indicator: 1 if censored, 0 if event
    cens = (events < 0.5).to(torch.float32)
    order = torch.argsort(times)
    t = times[order]
    c = cens[order]
    # Unique times
    uniq, counts = torch.unique_consecutive(t, return_counts=True)
    # At each unique time, number censored and at risk
    idx_start = 0
    at_risk = t.numel()
    G_vals = []
    G = 1.0
    for i, tt in enumerate(uniq):
        count_t = int(counts[i].item())
        slice_mask = slice(idx_start, idx_start + count_t)
        num_censored = c[slice_mask].sum().item()
        # KM step: multiply by (1 - d_c / R), where d_c censored at t, R at risk just before t
        if at_risk > 0:
            G *= max(1.0 - (num_censored / at_risk), 1e-12)
        G_vals.append(G)
        at_risk -= count_t
        idx_start += count_t
    uniq = uniq.to(device)
    G_tensor = torch.tensor(G_vals, device=device, dtype=times.dtype)

    def G_of_t(query_t: torch.Tensor, left_limit: bool = False) -> torch.Tensor:
        if uniq.numel() == 0:
            return torch.ones_like(query_t)
        if left_limit:
            idx = torch.searchsorted(uniq, query_t, right=False) - 1
        else:
            idx = torch.searchsorted(uniq, query_t, right=True) - 1
        idx = torch.clamp(idx, min=0)
        return G_tensor[idx]

    return G_of_t

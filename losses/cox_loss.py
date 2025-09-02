import torch


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

import torch
import numpy as np

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


@torch.no_grad()
def concordance_index(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> float:
    """
    Compute C-index. Higher risk -> earlier event expected.
    Ignores non-comparable pairs (e.g., both censored or no ordering).
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


def estimate_breslow_baseline(times: torch.Tensor, events: torch.Tensor, log_risks: torch.Tensor):
    """
    Estimate baseline cumulative hazard H0(t) via Breslow.
    Returns event_times (ascending) and H0 values at those times (same length).
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
    Returns tensor of shape (N, T).
    """
    H_t = baseline_cumhaz_at(event_times, H0, t_eval)  # (T,)
    exp_lr = torch.exp(log_risks).unsqueeze(1)  # (N,1)
    S = torch.exp(-exp_lr * H_t.unsqueeze(0))
    return S


def km_censoring(times: torch.Tensor, events: torch.Tensor):
    """
    Kaplan-Meier estimator for censoring survival G(t) = P(C >= t).
    Returns step function evaluator G(t) and G(t-).
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


def brier_score_ipcw(times: torch.Tensor, events: torch.Tensor, S_probs: torch.Tensor, t_eval: torch.Tensor, G_of_t) -> torch.Tensor:
    """
    IPCW Brier score at times t_eval. Returns vector (T,).
    S_probs: (N, T) survival probabilities at each eval time.
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


def integrated_brier_score(times: torch.Tensor, events: torch.Tensor, S_probs: torch.Tensor, t_eval: torch.Tensor, G_of_t) -> float:
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


def time_dependent_auc(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor, t_eval: torch.Tensor, G_of_t) -> torch.Tensor:
    """
    Dynamic AUC at times t_eval using IPCW.
    Returns vector (T,).
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

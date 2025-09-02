import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from dataloaders import MonaiSurvivalDataset
from architectures import build_network

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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune 3D ResNet for survival prediction (CoxPH).")
    parser.add_argument('--train_csv', type=str, required=True, help='Training CSV path')
    parser.add_argument('--val_csv', type=str, required=True, help='Validation CSV path')
    parser.add_argument('--resnet', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='ResNet architecture')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64], help='Image size (D, H, W)')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--init', type=str, default='random', choices=['random', 'pretrained'], help='Initialization mode')
    parser.add_argument('--pretrained_path', type=str, default='', help='Path to pretrained checkpoint (for init=pretrained)')
    parser.add_argument('--freeze_epochs', type=int, default=0, help='Freeze encoder for N warmup epochs')
    parser.add_argument('--eval_years', type=float, nargs='+', default=[1, 2, 3, 4, 5], help='Years at which to compute time-dependent AUC and Brier score')
    parser.add_argument('--log_dir', type=str, default='finetune_logs', help='Directory for logs/checkpoints')
    parser.add_argument('--output', type=str, default='finetune_logs/finetuned_model.pth', help='Path to save final model')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for model')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor')
    parser.add_argument('--persistent_workers', action='store_true', help='Use persistent workers in DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory in DataLoader')
    return parser.parse_args()


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


def load_model(resnet_type: str, init_mode: str, pretrained_path: str, device: torch.device) -> nn.Module:
    # Output 1 log-risk value per sample
    model = build_network(resnet_type=resnet_type, in_channels=1, num_classes=1)

    if init_mode == 'pretrained':
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained encoder weights from {pretrained_path}")

    return model


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None, return_collections: bool = False, amp: bool = False):
    is_train = optimizer is not None
    total_loss = 0.0
    total_events = 0.0

    if is_train:
        model.train()
    else:
        model.eval()

    all_times = []
    all_events = []
    all_risks = []

    for batch in loader:
        # Dataset returns: features, time, event, image, dicom_path
        _, times, events, images, _ = batch
        images = images.to(device, non_blocking=True)
        times = times.to(device, non_blocking=True)
        events = events.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            log_risks = model(images).squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
            loss = cox_ph_loss(log_risks, times, events)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_events = events.sum().item()
        total_loss += loss.item() * max(batch_events, 1.0)
        total_events += batch_events

        all_times.append(times)
        all_events.append(events)
        all_risks.append(log_risks)

    times_cat = torch.cat(all_times)
    events_cat = torch.cat(all_events)
    risks_cat = torch.cat(all_risks)
    c_idx = concordance_index(times_cat, events_cat, risks_cat)
    avg_loss = total_loss / max(total_events, 1.0)
    if return_collections:
        return avg_loss, c_idx, times_cat, events_cat, risks_cat
    return avg_loss, c_idx


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


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_best = os.path.join(args.log_dir, 'checkpoint_best.pth')

    # Datasets and loaders
    common_ds_kwargs = dict(image_size=tuple(args.image_size), use_cache=False, augment=True)
    train_ds = MonaiSurvivalDataset(csv_path=args.train_csv, **common_ds_kwargs)
    val_ds = MonaiSurvivalDataset(csv_path=args.val_csv, **{**common_ds_kwargs, 'augment': False})

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = load_model(args.resnet, args.init, args.pretrained_path, device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = DataParallel(model)
    model = model.to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode='max-autotune')
            print('Enabled torch.compile')
        except Exception as e:
            print(f'torch.compile not available: {e}')

    # Optionally freeze encoder for a few warmup epochs
    def set_encoder_trainable(trainable: bool):
        for p in model.parameters():
            p.requires_grad = trainable

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Optional tensorboard logging for finetuning
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_c = -1.0

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_epochs > 0:
            set_encoder_trainable(False)
        if epoch == args.freeze_epochs + 1:
            set_encoder_trainable(True)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = None

        train_loss, train_c = run_epoch(model, train_loader, device, optimizer, amp=args.amp)
        val_loss, val_c, v_times, v_events, v_risks = run_epoch(model, val_loader, device, optimizer=None, return_collections=True, amp=args.amp)

        # Build baseline using validation set predictions
        v_event_times, v_H0 = estimate_breslow_baseline(v_times, v_events, v_risks)
        # Evaluation grid: years -> same units as input times; assume times are in years
        t_eval = torch.tensor(args.eval_years, device=v_times.device, dtype=v_times.dtype)
        # Survival probabilities using Cox model
        S_probs = predict_survival_probs(v_risks, v_event_times, v_H0, t_eval)
        # IPCW using KM for censoring
        G_of_t = km_censoring(v_times, v_events)
        ibs = integrated_brier_score(v_times, v_events, S_probs, t_eval, G_of_t)
        td_auc = time_dependent_auc(v_times, v_events, v_risks, t_eval, G_of_t)
        td_auc_str = ", ".join([f"{float(a):.3f}" for a in td_auc.cpu()])

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} C: {train_c:.4f} | "
            f"Val Loss: {val_loss:.4f} C: {val_c:.4f} | IBS: {ibs:.4f} | tAUC@{args.eval_years}: [{td_auc_str}]"
        )
        # Log metrics
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Cindex', train_c, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Cindex', val_c, epoch)
        writer.add_scalar('Val/IBS', ibs, epoch)
        for i, t in enumerate(args.eval_years):
            writer.add_scalar(f'Val/tAUC@{t}', float(td_auc[i]), epoch)

        if val_c > best_val_c:
            best_val_c = val_c
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), ckpt_best)
            else:
                torch.save(model.state_dict(), ckpt_best)
            print(f"Saved best model to {ckpt_best} (C-index={best_val_c:.4f})")

    # Save final
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), args.output)
    else:
        torch.save(model.state_dict(), args.output)
    print(f"Saved final model to {args.output}")
    writer.close()


if __name__ == '__main__':
    main()



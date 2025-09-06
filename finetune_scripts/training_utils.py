"""
Training utilities for survival analysis fine-tuning.
Contains functions for model loading, epoch execution, and training loop management.
"""

import os
import math
from typing import Iterator, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from architectures import build_multimodal_network
from losses import cox_ph_loss, concordance_pairwise_loss, time_aware_triplet_loss
from metrics.survival_metrics import cindex_torchsurv
from optimizers import create_optimizer_and_scheduler
from .distributed_utils import (
    setup_ddp_model, reduce_tensor, synchronize_distributed,
    get_optimal_batch_size_per_gpu, get_optimal_num_workers
)


class EventAwareBatchSampler(Sampler[List[int]]):
    """Batch sampler ensuring a minimum number of event samples per batch.

    Parameters
    ----------
    events : Sequence[int]
        Binary event indicators for the dataset.
    batch_size : int
        Number of samples per batch.
    m_events_per_batch : int, optional
        Minimum number of event samples per batch. Defaults to 1.
    num_batches : int, optional
        Number of batches per epoch. If ``None`` it is computed from the
        dataset length.
    agatston_bins : Sequence[int], optional
        Optional Agatston bin label for each sample enabling stratified
        sampling across bins.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        events: Sequence[int],
        batch_size: int,
        m_events_per_batch: int = 1,
        num_batches: int | None = None,
        agatston_bins: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self.events = np.asarray(events).astype(bool)
        self.batch_size = int(batch_size)
        self.m_events = int(m_events_per_batch)
        n = len(self.events)
        self.num_batches = num_batches or math.ceil(n / batch_size)
        self.event_idx = np.where(self.events)[0]
        self.censor_idx = np.where(~self.events)[0]
        self.seed = seed

        if agatston_bins is not None:
            bins = np.asarray(agatston_bins)
            self.unique_bins = np.unique(bins)
            self.event_bins = {
                b: self.event_idx[bins[self.event_idx] == b] for b in self.unique_bins
            }
            self.censor_bins = {
                b: self.censor_idx[bins[self.censor_idx] == b] for b in self.unique_bins
            }
            bin_counts = np.array([np.sum(bins == b) for b in self.unique_bins], dtype=float)
            self.bin_probs = bin_counts / bin_counts.sum()
        else:
            self.unique_bins = None
            self.event_bins = None
            self.censor_bins = None

    def _sample_from_pool(
        self, pool: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        if n <= 0 or len(pool) == 0:
            return np.array([], dtype=int)
        replace = len(pool) < n
        return rng.choice(pool, size=n, replace=replace)

    def _sample_stratified(
        self, pools: dict[int, np.ndarray], n: int, rng: np.random.Generator
    ) -> np.ndarray:
        if n <= 0 or pools is None:
            return np.array([], dtype=int)
        counts = rng.multinomial(n, self.bin_probs)
        samples: list[np.ndarray] = []
        for b, c in zip(self.unique_bins, counts):
            if c == 0:
                continue
            pool = pools.get(b, np.array([], dtype=int))
            if len(pool) == 0:
                continue
            replace = len(pool) < c
            samples.append(rng.choice(pool, size=c, replace=replace))
        if samples:
            return np.concatenate(samples)
        return np.array([], dtype=int)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_batches):
            n_events = self.m_events if len(self.event_idx) > 0 else 0
            if self.unique_bins is not None:
                ev = self._sample_stratified(self.event_bins, n_events, rng)
                n_non = self.batch_size - ev.shape[0]
                non = self._sample_stratified(self.censor_bins, n_non, rng)
            else:
                ev = self._sample_from_pool(self.event_idx, n_events, rng)
                n_non = self.batch_size - ev.shape[0]
                non = self._sample_from_pool(self.censor_idx, n_non, rng)
            batch = np.concatenate([ev, non])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


class EventAwareDistributedBatchSampler(Sampler[List[int]]):
    """Distributed version of :class:`EventAwareBatchSampler` for DDP."""

    def __init__(
        self,
        events: Sequence[int],
        batch_size: int,
        m_events_per_batch: int = 1,
        num_batches: int | None = None,
        agatston_bins: Sequence[int] | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.rank = rank
        self.world_size = world_size
        self.sampler = EventAwareBatchSampler(
            events,
            batch_size,
            m_events_per_batch,
            num_batches=num_batches,
            agatston_bins=agatston_bins,
            seed=seed + rank,
        )

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling across epochs."""
        self.sampler.seed = epoch + self.rank


def all_gather_no_grad(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """All-gather tensors across processes without tracking gradients."""

    if not (dist.is_available() and dist.is_initialized()):
        return list(tensors)

    world_size = dist.get_world_size()
    results: list[torch.Tensor] = []
    with torch.no_grad():
        for tensor in tensors:
            local_size = torch.tensor(tensor.shape[0], device=tensor.device, dtype=torch.long)
            size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            max_size = int(torch.stack(size_list).max().item())
            if tensor.shape[0] < max_size:
                pad_shape = (max_size - tensor.shape[0],) + tensor.shape[1:]
                padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
                tensor_padded = torch.cat([tensor, padding], dim=0)
            else:
                tensor_padded = tensor
            gather_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
            dist.all_gather(gather_list, tensor_padded)
            trimmed = [t[:s.item()] for t, s in zip(gather_list, size_list)]
            results.append(torch.cat(trimmed, dim=0))
    return results


def build_global_risk_tensors(
    log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather per-process risk tensors into global tensors."""

    if not (dist.is_available() and dist.is_initialized()):
        return log_risk, time, event
    gathered = all_gather_no_grad(log_risk, time, event)
    return tuple(gathered)  # type: ignore[return-value]


def load_model(
    img_encoder: str,
    tab_encoder: str,
    resnet_type: str,
    init_mode: str,
    pretrained_path: str,
    device: torch.device,
    tabular_dim: int,
    embed_dim: int,
) -> nn.Module:
    """Build multimodal model with optional pretrained image weights."""

    model = build_multimodal_network(
        img_encoder=img_encoder,
        tab_encoder=tab_encoder,
        resnet_type=resnet_type,
        in_channels=1,
        tabular_dim=tabular_dim,
        embed_dim=embed_dim,
    )

    if init_mode == "pretrained":
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained encoder weights from {pretrained_path}")

    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer=None,
    return_collections: bool = False,
    amp: bool = False,
    scaler=None,
    max_grad_norm: float | None = None,
    is_distributed: bool = False,
    loss_weights: dict | None = None,
    loss_params: dict | None = None,
):
    """
    Run one epoch of training or validation with distributed training support.
    
    Args:
        model: Model to train/evaluate
        loader: DataLoader for the epoch
        device: Device to run on
        optimizer: Optimizer (None for validation)
        return_collections: Whether to return all predictions for evaluation
        amp: Whether to use mixed precision
        scaler: GradScaler for mixed precision (required if amp=True)
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
        is_distributed: Whether running in distributed mode
    
    Returns:
        If return_collections=False: (avg_loss, c_index)
        If return_collections=True: (avg_loss, c_index, times, events, risks)
    """
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
        features, times, events, images, _ = batch

        # Move to device (non_blocking for better performance)
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        times = times.to(device, non_blocking=True)
        events = events.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images, features)
            log_risks = outputs["log_risk"]
            lw = loss_weights or {"cox": 1.0}
            lp = loss_params or {}
            loss = torch.tensor(0.0, device=device, dtype=log_risks.dtype)
            if lw.get("cox", 0.0) > 0:
                loss = loss + lw["cox"] * cox_ph_loss(log_risks, times, events, gather=is_distributed)
            if lw.get("cpl", 0.0) > 0:
                temp = lp.get("cpl", {}).get("temperature", 1.0)
                loss = loss + lw["cpl"] * concordance_pairwise_loss(
                    log_risks, times, events, temperature=temp, gather=is_distributed
                )
            if lw.get("tmcl", 0.0) > 0 and outputs.get("image_embed") is not None and outputs.get("tab_embed") is not None:
                margin = lp.get("tmcl", {}).get("margin", 1.0)
                loss = loss + lw["tmcl"] * time_aware_triplet_loss(
                    outputs["image_embed"], outputs["tab_embed"], times, margin=margin
                )

        if is_train:
            if scaler is not None:
                # Mixed precision training
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        batch_events = events.sum().item()
        total_loss += loss.item() * max(batch_events, 1.0)
        total_events += batch_events

        all_times.append(times)
        all_events.append(events)
        all_risks.append(log_risks)

    # Concatenate all predictions
    times_cat = torch.cat(all_times)
    events_cat = torch.cat(all_events)
    risks_cat = torch.cat(all_risks)
    
    # Compute metrics
    c_idx = cindex_torchsurv(risks_cat, events_cat, times_cat)
    avg_loss = total_loss / max(total_events, 1.0)
    
    # Reduce metrics across all processes in distributed training
    if is_distributed:
        avg_loss = reduce_tensor(torch.tensor(avg_loss, device=device), op='mean').item()
        c_idx = reduce_tensor(torch.tensor(c_idx, device=device), op='mean').item()
        
        # Synchronize all processes
        synchronize_distributed()
    
    if return_collections:
        return avg_loss, c_idx, times_cat, events_cat, risks_cat
    return avg_loss, c_idx


def setup_model_and_optimizer(
    args, device: torch.device, is_distributed: bool = False, tabular_dim: int | None = None
):
    """
    Setup model, optimizer, scheduler, and multi-GPU support.
    
    Args:
        args: Parsed command line arguments
        device: Device to run on
        is_distributed: Whether running in distributed mode
    
    Returns:
        Tuple of (model, optimizer, scheduler)
    """
    tab_dim = tabular_dim if tabular_dim is not None else getattr(args, "tab_dim", 0)
    model = load_model(
        args.img_encoder,
        args.tab_encoder,
        args.resnet,
        args.init,
        args.pretrained_path,
        device,
        tab_dim,
        args.embed_dim,
    )
    
    # Setup multi-GPU support
    if is_distributed:
        # Use DistributedDataParallel for distributed training
        model = setup_ddp_model(model, device)
        print(f"Using DistributedDataParallel on device {device}")
    elif torch.cuda.device_count() > 1:
        # Use DataParallel for single-node multi-GPU
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    # Enable torch.compile if requested
    if getattr(args, 'compile', False):
        try:
            model = torch.compile(model, mode='max-autotune')
            print('Enabled torch.compile')
        except Exception as e:
            print(f'torch.compile not available: {e}')

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model.parameters(),
        optimizer_name=getattr(args, 'optimizer', 'adamw'),
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_name=getattr(args, 'scheduler', 'cosine'),
        epochs=args.epochs,
        eta_min=getattr(args, 'eta_min', 1e-7)
    )
    
    return model, optimizer, scheduler


def create_encoder_freeze_function(model):
    """
    Create a function to freeze/unfreeze encoder parameters.
    
    Args:
        model: Model to control
    
    Returns:
        Function set_encoder_trainable(trainable: bool)
    """
    def set_encoder_trainable(trainable: bool):
        for p in model.parameters():
            p.requires_grad = trainable
    
    return set_encoder_trainable


def save_model_checkpoint(model, path: str, is_best: bool = False, metric_value: float = None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        metric_value: Metric value for logging
    """
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    
    if is_best and metric_value is not None:
        print(f"Saved best model to {path} (C-index={metric_value:.4f})")
    else:
        print(f"Saved model to {path}")

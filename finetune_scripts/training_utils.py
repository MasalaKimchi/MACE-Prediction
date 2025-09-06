"""
Training utilities for survival analysis fine-tuning.
Contains functions for model loading, epoch execution, and training loop management.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from architectures import build_network, build_multimodal_network
from losses import cox_ph_loss, concordance_pairwise_loss, time_aware_triplet_loss
from .metrics_utils import concordance_index
from optimizers import create_optimizer_and_scheduler
from .distributed_utils import (
    setup_ddp_model, reduce_tensor, synchronize_distributed,
    get_optimal_batch_size_per_gpu, get_optimal_num_workers
)


def load_model(resnet_type: str, init_mode: str, pretrained_path: str, device: torch.device) -> nn.Module:
    """
    Load model for fine-tuning with optional pretrained weights.
    
    Args:
        resnet_type: Type of ResNet architecture
        init_mode: 'random' or 'pretrained'
        pretrained_path: Path to pretrained checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model ready for fine-tuning
    """
    # Output 1 log-risk value per sample
    model = build_network(resnet_type=resnet_type, in_channels=1, num_classes=1)

    if init_mode == 'pretrained':
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Handle both old format (just state_dict) and new format (with scaler)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded pretrained encoder weights from {pretrained_path}")
            if 'feature_scaler' in checkpoint:
                print(f"Note: Feature scaler available in checkpoint (not used in fine-tuning)")
        else:
            # Old format - just state dict
            model.load_state_dict(checkpoint, strict=False)
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
    c_idx = concordance_index(times_cat, events_cat, risks_cat)
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


def setup_model_and_optimizer(args, device: torch.device, is_distributed: bool = False):
    """
    Setup model, optimizer, scheduler, and multi-GPU support.
    
    Args:
        args: Parsed command line arguments
        device: Device to run on
        is_distributed: Whether running in distributed mode
    
    Returns:
        Tuple of (model, optimizer, scheduler)
    """
    model = load_model(args.resnet, args.init, args.pretrained_path, device)
    
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

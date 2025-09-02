"""
Training utilities for survival analysis fine-tuning.
Contains functions for model loading, epoch execution, and training loop management.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from architectures import build_network
from survival_utils import cox_ph_loss
from metrics_utils import concordance_index
from optimizers import create_optimizer_and_scheduler


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


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, 
              optimizer=None, return_collections: bool = False, amp: bool = False,
              scaler=None, max_grad_norm: float = None):
    """
    Run one epoch of training or validation.
    
    Args:
        model: Model to train/evaluate
        loader: DataLoader for the epoch
        device: Device to run on
        optimizer: Optimizer (None for validation)
        return_collections: Whether to return all predictions for evaluation
        amp: Whether to use mixed precision
        scaler: GradScaler for mixed precision (required if amp=True)
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
    
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

    times_cat = torch.cat(all_times)
    events_cat = torch.cat(all_events)
    risks_cat = torch.cat(all_risks)
    c_idx = concordance_index(times_cat, events_cat, risks_cat)
    avg_loss = total_loss / max(total_events, 1.0)
    
    if return_collections:
        return avg_loss, c_idx, times_cat, events_cat, risks_cat
    return avg_loss, c_idx


def setup_model_and_optimizer(args, device: torch.device):
    """
    Setup model, optimizer, scheduler, and optional multi-GPU support.
    
    Args:
        args: Parsed command line arguments
        device: Device to run on
    
    Returns:
        Tuple of (model, optimizer, scheduler)
    """
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

#!/usr/bin/env python3
"""
Config-based survival analysis training script.
Based on SegFormer3D compartmentalized structure.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from architectures import build_network
from dataloaders import MonaiSurvivalDataset
from losses import cox_ph_loss
from metrics import (
    concordance_index,
    estimate_breslow_baseline,
    predict_survival_probs,
    km_censoring,
    integrated_brier_score,
    time_dependent_auc
)
from optimizers import create_optimizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict) -> tuple:
    """Create training and validation dataloaders."""
    dataset_config = config['dataset']
    
    # Common dataset kwargs
    common_kwargs = {
        'dicom_col': dataset_config['dicom_col'],
        'time_col': dataset_config['time_col'],
        'event_col': dataset_config['event_col'],
        'feature_cols': dataset_config['feature_cols'],
        'categorical_cols': dataset_config['categorical_cols'],
        'numerical_cols': dataset_config['numerical_cols'],
        'image_size': tuple(dataset_config['image_size']),
        'use_cache': dataset_config['use_cache'],
        'cache_rate': dataset_config['cache_rate'],
        'num_workers': dataset_config['num_workers']
    }
    
    # Training dataset with augmentation
    train_ds = MonaiSurvivalDataset(
        csv_path=dataset_config['train_csv'],
        augment=True,
        **common_kwargs
    )
    
    # Validation dataset without augmentation
    val_ds = MonaiSurvivalDataset(
        csv_path=dataset_config['val_csv'],
        augment=False,
        **common_kwargs
    )
    
    # DataLoader configuration
    dataloader_config = config['dataloader']
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        persistent_workers=dataloader_config['persistent_workers'] and dataloader_config['num_workers'] > 0,
        prefetch_factor=dataloader_config['prefetch_factor'] if dataloader_config['num_workers'] > 0 else None,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        persistent_workers=dataloader_config['persistent_workers'] and dataloader_config['num_workers'] > 0,
        prefetch_factor=dataloader_config['prefetch_factor'] if dataloader_config['num_workers'] > 0 else None,
    )
    
    return train_loader, val_loader


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create and configure the model."""
    model_config = config['model']
    
    model = build_network(
        resnet_type=model_config['architecture'],
        in_channels=model_config['in_channels'],
        num_classes=model_config['num_classes']
    )
    
    # Load pretrained weights if specified
    if model_config['init_mode'] == 'pretrained':
        pretrained_path = model_config['pretrained_path']
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained encoder weights from {pretrained_path}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = DataParallel(model)
    
    model = model.to(device)
    
    # Optional torch.compile
    if config['training']['compile']:
        try:
            model = torch.compile(model, mode='max-autotune')
            print('Enabled torch.compile')
        except Exception as e:
            print(f'torch.compile not available: {e}')
    
    return model


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, 
              optimizer=None, return_collections: bool = False, amp: bool = False):
    """Run one epoch of training or validation."""
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


def main():
    parser = argparse.ArgumentParser(description="Run survival analysis experiment")
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    ckpt_best = os.path.join(log_dir, 'checkpoint_best.pth')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer
    training_config = config['training']
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name=training_config['optimizer'],
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Optional tensorboard logging
    writer = SummaryWriter(log_dir=log_dir) if config['logging']['tensorboard'] else None
    best_val_c = -1.0
    
    # Training loop
    for epoch in range(1, training_config['epochs'] + 1):
        # Optional encoder freezing for warmup epochs
        if epoch == 1 and training_config['freeze_epochs'] > 0:
            for p in model.parameters():
                p.requires_grad = False
        if epoch == training_config['freeze_epochs'] + 1:
            for p in model.parameters():
                p.requires_grad = True
        
        # Clear gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = None
        
        # Run epochs
        train_loss, train_c = run_epoch(
            model, train_loader, device, optimizer, 
            amp=training_config['amp']
        )
        val_loss, val_c, v_times, v_events, v_risks = run_epoch(
            model, val_loader, device, optimizer=None, 
            return_collections=True, amp=training_config['amp']
        )
        
        # Compute additional metrics
        v_event_times, v_H0 = estimate_breslow_baseline(v_times, v_events, v_risks)
        t_eval = torch.tensor(config['evaluation']['eval_years'], device=v_times.device, dtype=v_times.dtype)
        S_probs = predict_survival_probs(v_risks, v_event_times, v_H0, t_eval)
        G_of_t = km_censoring(v_times, v_events)
        ibs = integrated_brier_score(v_times, v_events, S_probs, t_eval, G_of_t)
        td_auc = time_dependent_auc(v_times, v_events, v_risks, t_eval, G_of_t)
        td_auc_str = ", ".join([f"{float(a):.3f}" for a in td_auc.cpu()])
        
        print(
            f"Epoch {epoch}/{training_config['epochs']} | Train Loss: {train_loss:.4f} C: {train_c:.4f} | "
            f"Val Loss: {val_loss:.4f} C: {val_c:.4f} | IBS: {ibs:.4f} | tAUC@{config['evaluation']['eval_years']}: [{td_auc_str}]"
        )
        
        # Log metrics
        if writer:
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Cindex', train_c, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Cindex', val_c, epoch)
            writer.add_scalar('Val/IBS', ibs, epoch)
            for i, t in enumerate(config['evaluation']['eval_years']):
                writer.add_scalar(f'Val/tAUC@{t}', float(td_auc[i]), epoch)
        
        # Save best model
        if val_c > best_val_c:
            best_val_c = val_c
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), ckpt_best)
            else:
                torch.save(model.state_dict(), ckpt_best)
            print(f"Saved best model to {ckpt_best} (C-index={best_val_c:.4f})")
    
    # Save final model
    output_path = config['logging']['output_path']
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), output_path)
    else:
        torch.save(model.state_dict(), output_path)
    print(f"Saved final model to {output_path}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

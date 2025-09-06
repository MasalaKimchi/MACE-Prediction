"""
Fine-tune 3D ResNet for survival prediction using Cox proportional hazards model.
This script has been refactored to use modular utility functions for better code organization.
"""

import argparse
import os
import csv
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloaders import MonaiSurvivalDataset
from training_utils import (
    setup_model_and_optimizer,
    run_epoch,
    create_encoder_freeze_function,
    save_model_checkpoint
)
from metrics_utils import cox_survival_matrix
from metrics.survival_metrics import (
    cindex_torchsurv,
    td_auc_torchsurv,
    brier_ibs_torchsurv,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune 3D ResNet for survival prediction (CoxPH).")
    parser.add_argument('--csv_path', type=str, required=True, help='CSV path with Fold_1 column for train/val split')
    parser.add_argument('--fold_column', type=str, default='Fold_1', help='Column name for fold split (default: Fold_1)')
    parser.add_argument('--train_fold', type=str, default='train', help='Value in fold column for training data (default: train)')
    parser.add_argument('--val_fold', type=str, default='val', help='Value in fold column for validation data (default: val)')
    parser.add_argument('--resnet', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 
                       help='ResNet architecture')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64], 
                       help='Image size (D, H, W)')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--init', type=str, default='random', 
                       choices=['random', 'pretrained'], help='Initialization mode')
    parser.add_argument('--pretrained_path', type=str, default='', 
                       help='Path to pretrained checkpoint (for init=pretrained)')
    parser.add_argument('--freeze_epochs', type=int, default=0, 
                       help='Freeze encoder for N warmup epochs')
    parser.add_argument('--eval_years', type=float, nargs='+', default=[1, 2, 3, 4, 5], 
                       help='Years at which to compute time-dependent AUC and Brier score')
    parser.add_argument('--log_dir', type=str, default='finetune_logs', 
                       help='Directory for logs/checkpoints')
    parser.add_argument('--output', type=str, default='finetune_logs/finetuned_model.pth', 
                       help='Path to save final model')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for model')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor')
    parser.add_argument('--persistent_workers', action='store_true', 
                       help='Use persistent workers in DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adamw', 'adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'cosine_warm_restarts', 'onecycle', 'none'], 
                       help='Learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                       help='Maximum gradient norm for clipping (0 to disable)')
    parser.add_argument('--eta_min', type=float, default=1e-7, 
                       help='Minimum learning rate for cosine scheduler')
    return parser.parse_args()


def create_dataloaders(args):
    """Create training and validation dataloaders."""
    common_ds_kwargs = dict(
        nifti_col='NIFTI path',  # Default NIFTI column name
        image_size=tuple(args.image_size), 
        use_cache=False, 
        augment=True
    )
    train_ds = MonaiSurvivalDataset(
        csv_path=args.csv_path,
        fold_column=args.fold_column,
        fold_value=args.train_fold,
        **common_ds_kwargs
    )
    val_ds = MonaiSurvivalDataset(
        csv_path=args.csv_path,
        fold_column=args.fold_column,
        fold_value=args.val_fold,
        **{**common_ds_kwargs, 'augment': False}
    )

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
    
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, args):
    """Evaluate model and compute survival metrics."""
    val_loss, _, v_times, v_events, v_risks = run_epoch(
        model, val_loader, device, optimizer=None, return_collections=True, amp=args.amp
    )

    val_c = cindex_torchsurv(v_risks, v_events, v_times)

    horizons = [1, 2, 3, 4, 5]
    td_auc = td_auc_torchsurv(v_risks, v_events, v_times, horizons)

    time_grid = torch.linspace(0, 5, steps=51, device=v_times.device, dtype=v_times.dtype)
    surv_matrix = cox_survival_matrix(v_risks, v_events, v_times, time_grid)
    brier_dict, ibs = brier_ibs_torchsurv(surv_matrix, v_events, v_times, time_grid)

    return val_loss, val_c, td_auc, brier_dict, ibs


def log_metrics(
    writer,
    epoch,
    train_loss,
    train_c,
    val_loss,
    val_c,
    td_auc,
    brier_dict,
    ibs,
    log_dir,
):
    """Log metrics to TensorBoard and persist to CSV/JSON."""

    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Cindex', train_c, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Cindex', val_c, epoch)
    for h, v in td_auc.items():
        writer.add_scalar(f'Val/tdAUC@{h}', v, epoch)
    for h in [1, 3, 5]:
        if h in brier_dict:
            writer.add_scalar(f'Val/Brier@{h}', brier_dict[h], epoch)
    writer.add_scalar('Val/IBS_0_5y', ibs, epoch)

    metrics = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'train_cindex': float(train_c),
        'val_loss': float(val_loss),
        'val_cindex': float(val_c),
        'val_ibs_0_5y': float(ibs),
    }
    for h, v in td_auc.items():
        metrics[f'val_td_auc@{int(h)}'] = float(v)
    for h in [1, 3, 5]:
        metrics[f'val_brier@{h}'] = float(brier_dict.get(h, float('nan')))

    csv_path = os.path.join(log_dir, 'metrics.csv')
    json_path = os.path.join(log_dir, 'metrics.json')
    fieldnames = list(metrics.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer_csv.writeheader()
        writer_csv.writerow(metrics)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    """Main training loop."""
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_best = os.path.join(args.log_dir, 'checkpoint_best.pth')

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Setup model, optimizer, and scheduler
    model, optimizer, scheduler = setup_model_and_optimizer(args, device)
    
    # Create mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if args.amp else None
    
    # Create encoder freeze function
    set_encoder_trainable = create_encoder_freeze_function(model)

    # Setup logging
    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_c = -1.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Handle encoder freezing
        if epoch == 1 and args.freeze_epochs > 0:
            set_encoder_trainable(False)
        if epoch == args.freeze_epochs + 1:
            set_encoder_trainable(True)
        
        # Clear gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = None

        # Training epoch
        train_loss, train_c = run_epoch(
            model, train_loader, device, optimizer, 
            amp=args.amp, scaler=scaler, 
            max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None
        )
        
        # Validation epoch
        val_loss, val_c, td_auc, brier_dict, ibs = evaluate_model(model, val_loader, device, args)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Format time-dependent AUC for printing
        td_auc_str = ", ".join([f"{int(h)}:{auc:.3f}" for h, auc in td_auc.items()])
        brier_str = ", ".join([f"{h}:{brier_dict.get(h, float('nan')):.3f}" for h in [1, 3, 5]])

        # Print progress
        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} C: {train_c:.4f} | "
            f"Val Loss: {val_loss:.4f} C: {val_c:.4f} | IBS: {ibs:.4f} | "
            f"Brier@{{1,3,5}}: [{brier_str}] | tAUC: [{td_auc_str}]"
        )

        # Log metrics
        log_metrics(writer, epoch, train_loss, train_c, val_loss, val_c, td_auc, brier_dict, ibs, args.log_dir)

        # Save best model
        if val_c > best_val_c:
            best_val_c = val_c
            save_model_checkpoint(model, ckpt_best, is_best=True, metric_value=best_val_c)

    # Save final model
    save_model_checkpoint(model, args.output)
    writer.close()


if __name__ == '__main__':
    main()

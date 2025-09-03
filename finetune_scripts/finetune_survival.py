"""
Fine-tune 3D ResNet for survival prediction using Cox proportional hazards model.
This script has been refactored to use modular utility functions for better code organization.
"""

import argparse
import os
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
from survival_utils import (
    estimate_breslow_baseline,
    predict_survival_probs,
    km_censoring
)
from metrics_utils import (
    integrated_brier_score,
    time_dependent_auc
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune 3D ResNet for survival prediction (CoxPH).")
    parser.add_argument('--train_csv', type=str, required=True, help='Training CSV path')
    parser.add_argument('--val_csv', type=str, required=True, help='Validation CSV path')
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
    train_ds = MonaiSurvivalDataset(csv_path=args.train_csv, **common_ds_kwargs)
    val_ds = MonaiSurvivalDataset(
        csv_path=args.val_csv, 
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
    val_loss, val_c, v_times, v_events, v_risks = run_epoch(
        model, val_loader, device, optimizer=None, return_collections=True, amp=args.amp
    )

    # Build baseline using validation set predictions
    v_event_times, v_H0 = estimate_breslow_baseline(v_times, v_events, v_risks)
    
    # Evaluation grid: years -> same units as input times; assume times are in years
    t_eval = torch.tensor(args.eval_years, device=v_times.device, dtype=v_times.dtype)
    
    # Survival probabilities using Cox model
    S_probs = predict_survival_probs(v_risks, v_event_times, v_H0, t_eval)
    
    # IPCW using KM for censoring
    G_of_t = km_censoring(v_times, v_events)
    
    # Compute metrics
    ibs = integrated_brier_score(v_times, v_events, S_probs, t_eval, G_of_t)
    td_auc = time_dependent_auc(v_times, v_events, v_risks, t_eval, G_of_t)
    
    return val_loss, val_c, ibs, td_auc


def log_metrics(writer, epoch, train_loss, train_c, val_loss, val_c, ibs, td_auc, eval_years):
    """Log metrics to tensorboard."""
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Cindex', train_c, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Cindex', val_c, epoch)
    writer.add_scalar('Val/IBS', ibs, epoch)
    for i, t in enumerate(eval_years):
        writer.add_scalar(f'Val/tAUC@{t}', float(td_auc[i]), epoch)


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
        val_loss, val_c, ibs, td_auc = evaluate_model(model, val_loader, device, args)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Format time-dependent AUC for printing
        td_auc_str = ", ".join([f"{float(a):.3f}" for a in td_auc.cpu()])

        # Print progress
        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} C: {train_c:.4f} | "
            f"Val Loss: {val_loss:.4f} C: {val_c:.4f} | IBS: {ibs:.4f} | "
            f"tAUC@{args.eval_years}: [{td_auc_str}]"
        )
        
        # Log metrics
        log_metrics(writer, epoch, train_loss, train_c, val_loss, val_c, ibs, td_auc, args.eval_years)

        # Save best model
        if val_c > best_val_c:
            best_val_c = val_c
            save_model_checkpoint(model, ckpt_best, is_best=True, metric_value=best_val_c)

    # Save final model
    save_model_checkpoint(model, args.output)
    writer.close()


if __name__ == '__main__':
    main()

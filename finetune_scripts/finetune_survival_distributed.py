"""
Distributed fine-tuning script for 3D ResNet survival prediction using Cox proportional hazards model.
Supports multi-GPU training with MONAI optimizations and Distributed Data Parallel (DDP).
"""

import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloaders import MultiGPUSurvivalDataset
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
from distributed_utils import (
    setup_distributed, cleanup_distributed, get_distributed_info,
    create_distributed_dataloader, save_checkpoint_distributed,
    get_optimal_batch_size_per_gpu, get_optimal_num_workers
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed fine-tune 3D ResNet for survival prediction (CoxPH).")
    parser.add_argument('--csv_path', type=str, required=True, help='CSV path with Fold_1 column for train/val split')
    parser.add_argument('--fold_column', type=str, default='Fold_1', help='Column name for fold split (default: Fold_1)')
    parser.add_argument('--train_fold', type=str, default='train', help='Value in fold column for training data (default: train)')
    parser.add_argument('--val_fold', type=str, default='val', help='Value in fold column for validation data (default: val)')
    parser.add_argument('--resnet', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 
                       help='ResNet architecture')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64], 
                       help='Image size (D, H, W)')
    parser.add_argument('--batch_size', type=int, default=24, help='Total batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader workers per GPU (auto if None)')
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
    parser.add_argument('--log_dir', type=str, default='finetune_logs_distributed', 
                       help='Directory for logs/checkpoints')
    parser.add_argument('--output', type=str, default='finetune_logs_distributed/finetuned_model.pth', 
                       help='Path to save final model')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for model')
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
    
    # Multi-GPU specific arguments (auto-detected)
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs (auto-detected if None)')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], 
                       help='Distributed backend (auto-selected)')
    parser.add_argument('--use_smart_cache', action='store_true', help='Use MONAI SmartCacheDataset (recommended)')
    parser.add_argument('--cache_rate', type=float, default=0.5, help='Cache rate for SmartCacheDataset')
    parser.add_argument('--replace_rate', type=float, default=0.5, help='Replace rate for SmartCacheDataset')
    
    return parser.parse_args()


def create_dataloaders(args, device):
    """Create training and validation dataloaders with distributed support."""
    # Get distributed info
    dist_info = get_distributed_info()
    world_size = dist_info['world_size']
    
    # Calculate optimal batch size per GPU
    batch_size_per_gpu = get_optimal_batch_size_per_gpu(args.batch_size, world_size)
    
    # Calculate optimal number of workers
    if args.num_workers is None:
        num_workers = get_optimal_num_workers(world_size)
    else:
        num_workers = args.num_workers
    
    print(f"Using batch size {batch_size_per_gpu} per GPU (total effective batch size: {batch_size_per_gpu * world_size})")
    print(f"Using {num_workers} workers per GPU")
    
    # Common dataset arguments
    common_ds_kwargs = dict(
        nifti_col='NIFTI path',
        image_size=tuple(args.image_size), 
        use_smart_cache=args.use_smart_cache,
        cache_rate=args.cache_rate,
        replace_rate=args.replace_rate,
        num_init_workers=num_workers,
        num_replace_workers=num_workers,
        device=device,
        augment=True
    )
    
    # Create datasets with fold-based splitting
    train_ds = MultiGPUSurvivalDataset(
        csv_path=args.csv_path, 
        fold_column=args.fold_column,
        fold_value=args.train_fold,
        **common_ds_kwargs
    )
    val_ds = MultiGPUSurvivalDataset(
        csv_path=args.csv_path,
        fold_column=args.fold_column,
        fold_value=args.val_fold,
        **{**common_ds_kwargs, 'augment': False}
    )

    # Create distributed dataloaders
    train_loader = create_distributed_dataloader(
        train_ds,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
        drop_last=True
    )
    val_loader = create_distributed_dataloader(
        val_ds,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, args):
    """Evaluate model and compute survival metrics."""
    val_loss, val_c, v_times, v_events, v_risks = run_epoch(
        model, val_loader, device, optimizer=None, return_collections=True, 
        amp=args.amp, is_distributed=True
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


def log_metrics(writer, epoch, train_loss, train_c, val_loss, val_c, ibs, td_auc, eval_years, rank):
    """Log metrics to tensorboard (only on rank 0)."""
    if rank != 0:
        return
        
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Cindex', train_c, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Cindex', val_c, epoch)
    writer.add_scalar('Val/IBS', ibs, epoch)
    for i, t in enumerate(eval_years):
        writer.add_scalar(f'Val/tAUC@{t}', float(td_auc[i]), epoch)


def train_worker(rank, world_size, args):
    """Training worker function for distributed training."""
    # Setup distributed environment
    setup_distributed(rank, world_size, args.backend)
    
    # Get device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Get distributed info
    dist_info = get_distributed_info()
    
    # Create log directory (only on rank 0)
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        ckpt_best = os.path.join(args.log_dir, 'checkpoint_best.pth')
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
        ckpt_best = None

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args, device)

    # Setup model, optimizer, and scheduler
    model, optimizer, scheduler = setup_model_and_optimizer(args, device, is_distributed=True)
    
    # Create mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if args.amp else None
    
    # Create encoder freeze function
    set_encoder_trainable = create_encoder_freeze_function(model)

    best_val_c = -1.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
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
            max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
            is_distributed=True
        )
        
        # Validation epoch
        val_loss, val_c, ibs, td_auc = evaluate_model(model, val_loader, device, args)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Format time-dependent AUC for printing
        td_auc_str = ", ".join([f"{float(a):.3f}" for a in td_auc.cpu()])

        # Print progress (only on rank 0)
        if rank == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} C: {train_c:.4f} | "
                f"Val Loss: {val_loss:.4f} C: {val_c:.4f} | IBS: {ibs:.4f} | "
                f"tAUC@{args.eval_years}: [{td_auc_str}]"
            )
            
            # Log metrics
            log_metrics(writer, epoch, train_loss, train_c, val_loss, val_c, ibs, td_auc, args.eval_years, rank)

            # Save best model
            if val_c > best_val_c:
                best_val_c = val_c
                save_checkpoint_distributed(
                    model, optimizer, epoch, val_c, ckpt_best, is_best=True
                )

    # Save final model (only on rank 0)
    if rank == 0:
        save_checkpoint_distributed(model, optimizer, args.epochs, val_c, args.output)
        if writer:
            writer.close()

    # Cleanup distributed environment
    cleanup_distributed()


def main():
    """Main function to launch distributed training with automatic GPU detection."""
    args = parse_args()
    
    # Auto-detect GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("❌ No GPUs detected. Please ensure CUDA is properly installed.")
        sys.exit(1)
    
    # Use all available GPUs by default
    world_size = args.world_size if args.world_size is not None else available_gpus
    
    # Ensure we don't exceed available GPUs
    if world_size > available_gpus:
        print(f"⚠️  Requested {world_size} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        world_size = available_gpus
    
    print(f"🚀 Starting training with {world_size} GPU(s) (auto-detected)")
    print(f"   Backend: {args.backend}")
    print(f"   Total batch size: {args.batch_size}")
    print(f"   Batch size per GPU: {args.batch_size // world_size}")
    print(f"   Mixed precision: {'✅' if args.amp else '❌'}")
    print(f"   Smart cache: {'✅' if args.use_smart_cache else '❌'}")
    
    if world_size == 1:
        # Single GPU training - use regular training script
        print("📱 Single GPU detected, using optimized single-GPU training...")
        from finetune_survival import main as single_gpu_main
        single_gpu_main()
    else:
        # Multi-GPU distributed training
        print(f"🔥 Multi-GPU distributed training with {world_size} GPUs")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()

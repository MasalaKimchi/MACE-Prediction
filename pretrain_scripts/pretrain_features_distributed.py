"""
Distributed pretraining script for 3D ResNet feature prediction.
Supports multi-GPU training with MONAI optimizations and Distributed Data Parallel (DDP).
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders import MultiGPUSurvivalDataset
from architectures import build_network
from optimizers import create_optimizer_and_scheduler
from finetune_scripts.distributed_utils import (
    setup_distributed, cleanup_distributed, get_distributed_info,
    create_distributed_dataloader, save_checkpoint_distributed,
    get_optimal_batch_size_per_gpu, get_optimal_num_workers
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed pretrain 3D ResNet for feature prediction.")
    parser.add_argument('--csv_path', type=str, required=True, help='CSV path with Fold_1 column for train/val split')
    parser.add_argument('--fold_column', type=str, default='Fold_1', help='Column name for fold split (default: Fold_1)')
    parser.add_argument('--train_fold', type=str, default='train', help='Value in fold column for training data (default: train)')
    parser.add_argument('--val_fold', type=str, default='val', help='Value in fold column for validation data (default: val)')
    parser.add_argument('--resnet', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 
                       help='ResNet architecture')
    parser.add_argument('--feature_cols', type=str, nargs='+', required=True, help='List of feature columns to predict')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64], 
                       help='Image size (D, H, W)')
    parser.add_argument('--batch_size', type=int, default=24, help='Total batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader workers per GPU (auto if None)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--log_dir', type=str, default='pretrain_logs_distributed', 
                       help='Directory for logs/checkpoints')
    parser.add_argument('--output', type=str, default='pretrain_logs_distributed/pretrained_model.pth', 
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
        feature_cols=args.feature_cols,
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
    
    return train_loader, val_loader, train_ds.get_feature_scaler()


def train_epoch(model, loader, device, optimizer, criterion, scaler, max_grad_norm, amp):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        # Dataset returns: features, time, event, image, dicom_path
        features, _, _, images, _ = batch
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, features)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    
    return total_loss / total_samples


def validate_epoch(model, loader, device, criterion, amp):
    """Run one validation epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            features, _, _, images, _ = batch
            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, features)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    return total_loss / total_samples


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
    train_loader, val_loader, feature_scaler = create_dataloaders(args, device)

    # Build model
    model = build_network(resnet_type=args.resnet, in_channels=1, num_classes=len(args.feature_cols))
    model = model.to(device)
    
    # Wrap with DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[device], find_unused_parameters=False)
    
    # Enable torch.compile if requested
    if args.compile:
        try:
            model = torch.compile(model, mode='max-autotune')
            if rank == 0:
                print('Enabled torch.compile')
        except Exception as e:
            if rank == 0:
                print(f'torch.compile not available: {e}')

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        eta_min=args.eta_min
    )
    
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if args.amp else None

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Training epoch
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion, scaler, args.max_grad_norm, args.amp)
        
        # Validation epoch
        val_loss = validate_epoch(model, val_loader, device, criterion, args.amp)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print progress (only on rank 0)
        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Log metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint_distributed(
                    model, optimizer, epoch, val_loss, ckpt_best, is_best=True
                )

    # Save final model (only on rank 0)
    if rank == 0:
        save_checkpoint_distributed(model, optimizer, args.epochs, val_loss, args.output)
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
        return
    
    # Use all available GPUs by default
    world_size = args.world_size if args.world_size is not None else available_gpus
    
    # Ensure we don't exceed available GPUs
    if world_size > available_gpus:
        print(f"⚠️  Requested {world_size} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        world_size = available_gpus
    
    print(f"🚀 Starting pretraining with {world_size} GPU(s) (auto-detected)")
    print(f"   Backend: {args.backend}")
    print(f"   Total batch size: {args.batch_size}")
    print(f"   Batch size per GPU: {args.batch_size // world_size}")
    print(f"   Mixed precision: {'✅' if args.amp else '❌'}")
    print(f"   Smart cache: {'✅' if args.use_smart_cache else '❌'}")
    print(f"   Features to predict: {args.feature_cols}")
    
    if world_size == 1:
        # Single GPU training - use regular pretraining script
        print("📱 Single GPU detected, using optimized single-GPU pretraining...")
        from pretrain_features import main as single_gpu_main
        single_gpu_main()
    else:
        # Multi-GPU distributed training
        print(f"🔥 Multi-GPU distributed pretraining with {world_size} GPUs")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()

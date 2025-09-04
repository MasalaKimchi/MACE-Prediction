#!/usr/bin/env python3
"""
Launch script for distributed multi-GPU pretraining.
Provides easy configuration and launching of distributed pretraining jobs.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_gpu_count():
    """Get the number of available GPUs."""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def create_pretraining_command(args):
    """Create the pretraining command with proper arguments."""
    cmd = [
        sys.executable, 'pretrain_scripts/pretrain_features_distributed.py',
        '--csv_path', args.csv_path,
        '--fold_column', args.fold_column,
        '--train_fold', args.train_fold,
        '--val_fold', args.val_fold,
        '--resnet', args.resnet,
        '--feature_cols'] + args.feature_cols + [
        '--image_size'] + [str(x) for x in args.image_size] + [
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--log_dir', args.log_dir,
        '--output', args.output,
        '--backend', args.backend
    ]
    
    # Add optional arguments
    if args.num_workers:
        cmd.extend(['--num_workers', str(args.num_workers)])
    if args.amp:
        cmd.append('--amp')
    if args.compile:
        cmd.append('--compile')
    if args.pin_memory:
        cmd.append('--pin_memory')
    if args.use_smart_cache:
        cmd.append('--use_smart_cache')
    if args.cache_rate != 0.5:
        cmd.extend(['--cache_rate', str(args.cache_rate)])
    if args.replace_rate != 0.5:
        cmd.extend(['--replace_rate', str(args.replace_rate)])
    if args.max_grad_norm != 1.0:
        cmd.extend(['--max_grad_norm', str(args.max_grad_norm)])
    if args.eta_min != 1e-7:
        cmd.extend(['--eta_min', str(args.eta_min)])
    
    return cmd


def main():
    """Main function to launch distributed pretraining with automatic GPU detection."""
    parser = argparse.ArgumentParser(description="🚀 Launch multi-GPU pretraining (auto-detects GPUs)")
    
    # Required arguments
    parser.add_argument('--csv_path', type=str, required=True, help='CSV path with Fold_1 column for train/val split')
    parser.add_argument('--feature_cols', type=str, nargs='+', required=True, help='List of feature columns to predict')
    
    # Fold arguments
    parser.add_argument('--fold_column', type=str, default='Fold_1', help='Column name for fold split (default: Fold_1)')
    parser.add_argument('--train_fold', type=str, default='train', help='Value in fold column for training data (default: train)')
    parser.add_argument('--val_fold', type=str, default='val', help='Value in fold column for validation data (default: val)')
    
    # Model arguments
    parser.add_argument('--resnet', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet architecture (default: resnet18)')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64],
                       help='Image size D H W (default: 256 256 64)')
    parser.add_argument('--batch_size', type=int, default=24, 
                       help='Total batch size across all GPUs (auto-divided per GPU)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    # Output arguments
    parser.add_argument('--log_dir', type=str, default='pretrain_logs_distributed',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--output', type=str, default='pretrain_logs_distributed/pretrained_model.pth',
                       help='Path to save final model')
    
    # Multi-GPU arguments (auto-detected)
    parser.add_argument('--world_size', type=int, default=None, 
                       help='Number of GPUs (auto-detects all available if None)')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                       help='Distributed backend (auto-selected)')
    parser.add_argument('--num_workers', type=int, default=None, 
                       help='DataLoader workers per GPU (auto-optimized if None)')
    
    # Optimization arguments (recommended settings)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (recommended)')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile (PyTorch 2.0+)')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster transfers')
    parser.add_argument('--use_smart_cache', action='store_true', 
                       help='Use MONAI SmartCacheDataset (recommended for large datasets)')
    parser.add_argument('--cache_rate', type=float, default=0.5, 
                       help='Cache rate for SmartCacheDataset (0.3-0.8)')
    parser.add_argument('--replace_rate', type=float, default=0.5, 
                       help='Replace rate for SmartCacheDataset (0.2-0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                       help='Maximum gradient norm for clipping (0 to disable)')
    parser.add_argument('--eta_min', type=float, default=1e-7, 
                       help='Minimum learning rate for cosine scheduler')
    
    args = parser.parse_args()
    
    # Auto-detect GPUs
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        print("❌ No GPUs detected. Please ensure NVIDIA drivers and CUDA are properly installed.")
        sys.exit(1)
    
    # Use all available GPUs by default
    if args.world_size is None:
        args.world_size = gpu_count
    elif args.world_size > gpu_count:
        print(f"⚠️  Requested {args.world_size} GPUs but only {gpu_count} available. Using {gpu_count} GPUs.")
        args.world_size = gpu_count
    
    print(f"🚀 Multi-GPU Pretraining Setup")
    print(f"   📊 Detected {gpu_count} GPU(s), using {args.world_size} for pretraining")
    print(f"   🔧 Backend: {args.backend}")
    print(f"   📦 Total batch size: {args.batch_size}")
    print(f"   📦 Batch size per GPU: {args.batch_size // args.world_size}")
    print(f"   ⚡ Mixed precision: {'✅' if args.amp else '❌'}")
    print(f"   🧠 Smart cache: {'✅' if args.use_smart_cache else '❌'}")
    print(f"   🔄 torch.compile: {'✅' if args.compile else '❌'}")
    print(f"   🎯 Features to predict: {args.feature_cols}")
    
    # Create pretraining command
    cmd = create_pretraining_command(args)
    
    print(f"\n🚀 Launching pretraining...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Set environment variables for distributed training
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.world_size)))
    
    # Launch pretraining
    try:
        subprocess.run(cmd, env=env, check=True)
        print("🎉 Pretraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pretraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("⏹️  Pretraining interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()

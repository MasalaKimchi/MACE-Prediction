#!/usr/bin/env python3
"""
🚀 Quick Start Pretraining Script
The simplest way to start multi-GPU pretraining with automatic GPU detection.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Quick start pretraining with minimal configuration."""
    parser = argparse.ArgumentParser(
        description="🚀 Quick Start Multi-GPU Pretraining (auto-detects everything!)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pretraining with all available GPUs
  python quick_start_pretraining.py --csv_path data/dataset.csv --feature_cols Age AgatstonScore2D MassScore
  
  # With recommended optimizations
  python quick_start_pretraining.py --csv_path data/dataset.csv --feature_cols Age AgatstonScore2D MassScore --amp --use_smart_cache
  
  # Custom model and batch size
  python quick_start_pretraining.py --csv_path data/dataset.csv --feature_cols Age AgatstonScore2D MassScore --resnet resnet50 --batch_size 32 --amp
        """
    )
    
    # Only essential arguments
    parser.add_argument('--csv_path', type=str, required=True, help='CSV path with Fold_1 column for train/val split')
    parser.add_argument('--feature_cols', type=str, nargs='+', required=True, help='List of feature columns to predict')
    
    # Optional model settings
    parser.add_argument('--resnet', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet architecture (default: resnet18)')
    parser.add_argument('--batch_size', type=int, default=24, 
                       help='Total batch size (auto-divided across GPUs)')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Recommended optimizations (enabled by default)
    parser.add_argument('--amp', action='store_true', default=True, 
                       help='Enable mixed precision (recommended)')
    parser.add_argument('--use_smart_cache', action='store_true', default=True,
                       help='Use MONAI SmartCacheDataset (recommended)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for faster transfers (recommended)')
    
    # Advanced options (optional)
    parser.add_argument('--log_dir', type=str, default='quick_start_pretrain_logs',
                       help='Output directory for logs')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_path):
        print(f"❌ CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Build command with recommended settings
    cmd = [
        sys.executable, 'pretrain_scripts/launch_distributed_pretraining.py',
        '--csv_path', args.csv_path,
        '--feature_cols'] + args.feature_cols + [
        '--resnet', args.resnet,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--log_dir', args.log_dir,
        '--amp',  # Always enable mixed precision
        '--use_smart_cache',  # Always enable smart cache
        '--pin_memory',  # Always enable pin memory
    ]
    
    print("🚀 Quick Start Multi-GPU Pretraining")
    print("=" * 50)
    print(f"📊 Data: {args.csv_path}")
    print(f"🎯 Features: {', '.join(args.feature_cols)}")
    print(f"🏗️  Model: {args.resnet}")
    print(f"📦 Batch size: {args.batch_size} (auto-divided across GPUs)")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"📁 Output: {args.log_dir}")
    print(f"⚡ Optimizations: Mixed Precision + Smart Cache + Pin Memory")
    print("=" * 50)
    
    # Launch pretraining
    try:
        print("🚀 Starting pretraining...")
        subprocess.run(cmd, check=True)
        print("🎉 Pretraining completed successfully!")
        print(f"📁 Results saved in: {args.log_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pretraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("⏹️  Pretraining interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()

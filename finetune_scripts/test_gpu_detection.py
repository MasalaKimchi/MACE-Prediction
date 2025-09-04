#!/usr/bin/env python3
"""
Test script to verify GPU auto-detection and multi-GPU setup.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from distributed_utils import get_distributed_info, get_optimal_batch_size_per_gpu, get_optimal_num_workers


def test_gpu_detection():
    """Test GPU detection and configuration."""
    print("🔍 GPU Detection Test")
    print("=" * 40)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Get GPU count
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    
    # Display GPU information
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Test distributed info
    dist_info = get_distributed_info()
    print(f"\n📊 Distributed Info: {dist_info}")
    
    # Test batch size calculation
    print(f"\n📦 Batch Size Calculation:")
    for gpu_count in [1, 2, 4, 8]:
        if gpu_count <= torch.cuda.device_count():
            batch_size = get_optimal_batch_size_per_gpu(32, gpu_count)
            print(f"   {gpu_count} GPU(s): {batch_size} per GPU (total: {batch_size * gpu_count})")
    
    # Test worker calculation
    print(f"\n👥 Worker Calculation:")
    for gpu_count in [1, 2, 4, 8]:
        if gpu_count <= torch.cuda.device_count():
            workers = get_optimal_num_workers(gpu_count)
            print(f"   {gpu_count} GPU(s): {workers} workers per GPU")
    
    return True


def main():
    """Run GPU detection test."""
    print("🚀 Multi-GPU Auto-Detection Test")
    print("=" * 50)
    
    success = test_gpu_detection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 GPU detection test passed!")
        print("\n✅ Your system is ready for multi-GPU training!")
        print("\n🚀 Quick Start Commands:")
        print("   # Super simple (auto-detects everything)")
        print("   python quick_start_multi_gpu.py --train_csv data/train.csv --val_csv data/val.csv")
        print("\n   # With custom settings")
        print("   python launch_distributed_training.py --train_csv data/train.csv --val_csv data/val.csv --amp --use_smart_cache")
    else:
        print("❌ GPU detection test failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure NVIDIA drivers are installed")
        print("   2. Check CUDA installation: nvidia-smi")
        print("   3. Verify PyTorch CUDA support: python -c 'import torch; print(torch.cuda.is_available())'")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

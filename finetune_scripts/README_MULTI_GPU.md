# Multi-GPU Training Guide

This guide explains how to utilize multiple GPUs for training your MACE prediction model using MONAI optimizations and Distributed Data Parallel (DDP).

## Overview

The multi-GPU implementation includes:

- **Distributed Data Parallel (DDP)**: Efficient multi-GPU training with PyTorch's native DDP
- **MONAI SmartCacheDataset**: Intelligent caching and data replacement for optimal memory usage
- **GPU-accelerated transforms**: Data preprocessing directly on GPU using `ToDeviced`
- **Optimized DataLoaders**: Distributed sampling and efficient worker management
- **Automatic mixed precision (AMP)**: Reduced memory usage and faster training

## Quick Start

### 1. Super Simple (Recommended)

```bash
# Just specify your data - everything else is auto-detected!
python quick_start_multi_gpu.py \
    --csv_path data/dataset.csv
```

### 2. Basic Multi-GPU Training

```bash
# Launch distributed training (auto-detects all GPUs)
python launch_distributed_training.py \
    --csv_path data/dataset.csv \
    --batch_size 32 \
    --epochs 50 \
    --amp \
    --use_smart_cache
```

### 3. Advanced Configuration

```bash
# Custom configuration with specific settings
python launch_distributed_training.py \
    --csv_path data/dataset.csv \
    --resnet resnet50 \
    --image_size 256 256 64 \
    --batch_size 48 \
    --epochs 100 \
    --lr 2e-4 \
    --world_size 4 \
    --amp \
    --compile \
    --use_smart_cache \
    --cache_rate 0.7 \
    --replace_rate 0.3 \
    --num_workers 6
```

## Key Features

### 1. Distributed Data Parallel (DDP)

- **🔄 Automatic GPU detection**: Automatically uses all available GPUs
- **⚡ Efficient communication**: Uses NCCL backend for optimal GPU-to-GPU communication
- **🔄 Gradient synchronization**: Automatic gradient averaging across GPUs
- **💾 Memory optimization**: Each GPU processes a subset of the data
- **🎯 Zero configuration**: Works out of the box with any number of GPUs

### 2. MONAI SmartCacheDataset

- **Intelligent caching**: Caches a fraction of data in memory for faster access
- **Dynamic replacement**: Replaces cached data during training to maximize variety
- **Memory efficiency**: Configurable cache and replace rates
- **Multi-worker support**: Parallel data loading and caching

### 3. GPU-Accelerated Data Pipeline

- **ToDeviced transform**: Moves data to GPU early in the pipeline
- **Non-blocking transfers**: Asynchronous data movement for better performance
- **Optimized transforms**: All image preprocessing on GPU when possible

### 4. Optimized DataLoaders

- **Distributed sampling**: Each GPU sees a different subset of data
- **Persistent workers**: Reduces worker startup overhead
- **Prefetching**: Overlaps data loading with training
- **Automatic worker scaling**: Optimal number of workers based on GPU count

## Configuration Options

### Multi-GPU Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--world_size` | Auto | Number of GPUs to use |
| `--backend` | `nccl` | Distributed backend (`nccl` for GPU, `gloo` for CPU) |
| `--use_smart_cache` | False | Enable MONAI SmartCacheDataset |
| `--cache_rate` | 0.5 | Fraction of data to cache initially |
| `--replace_rate` | 0.5 | Fraction of cached data to replace each epoch |
| `--num_workers` | Auto | Number of DataLoader workers per GPU |

### Performance Optimization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--amp` | False | Enable automatic mixed precision |
| `--compile` | False | Enable torch.compile optimization |
| `--pin_memory` | False | Pin memory for faster GPU transfers |
| `--batch_size` | 24 | Total batch size across all GPUs |

## Memory Management

### SmartCacheDataset Configuration

```python
# Conservative memory usage (suitable for limited GPU memory)
cache_rate = 0.3
replace_rate = 0.2

# Aggressive caching (suitable for high-memory GPUs)
cache_rate = 0.8
replace_rate = 0.4
```

### Batch Size Guidelines

- **Total batch size**: Should be divisible by the number of GPUs
- **Per-GPU batch size**: `total_batch_size / num_gpus`
- **Memory scaling**: Larger models may require smaller per-GPU batch sizes

Example for 4 GPUs:
- Total batch size: 32 → Per-GPU batch size: 8
- Total batch size: 48 → Per-GPU batch size: 12

## Performance Tips

### 1. Optimal Worker Configuration

```python
# Rule of thumb: 2-4 workers per GPU
num_workers = min(4 * num_gpus, cpu_count)
```

### 2. Memory Optimization

- Use `--amp` for mixed precision training
- Enable `--pin_memory` for faster data transfers
- Adjust `cache_rate` based on available GPU memory
- Use `--compile` for PyTorch 2.0+ optimization

### 3. Data Loading Optimization

- Use `--use_smart_cache` for large datasets
- Set `replace_rate` to balance memory usage and data variety
- Ensure sufficient CPU cores for DataLoader workers

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size per GPU
   - Lower cache rate
   - Enable mixed precision (`--amp`)

2. **Slow Data Loading**
   - Increase number of workers
   - Enable smart caching
   - Use faster storage (NVMe SSD)

3. **Poor GPU Utilization**
   - Check DataLoader worker count
   - Ensure sufficient batch size
   - Monitor with `nvidia-smi`

### Debugging

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check distributed training status
export NCCL_DEBUG=INFO
python finetune_survival_distributed.py [args]
```

## Example Training Scripts

### Single Node, 4 GPUs

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python finetune_survival_distributed.py \
    --csv_path data/dataset.csv \
    --resnet resnet50 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --amp \
    --use_smart_cache \
    --cache_rate 0.6 \
    --replace_rate 0.4 \
    --log_dir logs/resnet50_4gpu
```

### Multi-Node Training (Advanced)

```bash
# Node 0
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12355 \
    finetune_survival_distributed.py [args]

# Node 1
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=12355 \
    finetune_survival_distributed.py [args]
```

## Performance Benchmarks

Expected speedup with multi-GPU training:

| GPUs | Expected Speedup | Memory Efficiency |
|------|------------------|-------------------|
| 1    | 1.0x (baseline)  | 100% |
| 2    | 1.8x            | 95% |
| 4    | 3.5x            | 90% |
| 8    | 6.5x            | 85% |

*Note: Actual performance depends on model size, data loading, and hardware configuration.*

## Integration with Existing Code

The multi-GPU implementation is designed to be backward compatible:

- Existing single-GPU scripts continue to work
- New `MultiGPUSurvivalDataset` can replace `MonaiSurvivalDataset`
- Distributed utilities can be imported and used in custom training loops

```python
from dataloaders import MultiGPUSurvivalDataset
from distributed_utils import setup_distributed, create_distributed_dataloader

# Use in your custom training script
dataset = MultiGPUSurvivalDataset(csv_path, use_smart_cache=True)
dataloader = create_distributed_dataloader(dataset, batch_size=8)
```

## Next Steps

1. **Start with 2 GPUs**: Test the setup with a small number of GPUs first
2. **Monitor performance**: Use `nvidia-smi` and training logs to optimize
3. **Scale gradually**: Increase GPU count and batch size as you gain experience
4. **Experiment with caching**: Find optimal cache and replace rates for your data
5. **Profile bottlenecks**: Use PyTorch profiler to identify performance issues

For more advanced usage and customization, refer to the source code in:
- `finetune_scripts/distributed_utils.py`
- `finetune_scripts/finetune_survival_distributed.py`
- `dataloaders/survival_dataset.py` (MultiGPUSurvivalDataset class)

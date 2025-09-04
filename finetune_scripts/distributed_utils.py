"""
Distributed training utilities for multi-GPU support.
Implements Distributed Data Parallel (DDP) and MONAI-optimized data loading.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from monai.data import SmartCacheDataset, CacheDataset
from monai.transforms import ToDeviced
from typing import Optional, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_distributed_info():
    """Get distributed training information."""
    if dist.is_initialized():
        return {
            'rank': dist.get_rank(),
            'world_size': dist.get_world_size(),
            'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
            'is_distributed': True
        }
    else:
        return {
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'is_distributed': False
        }


def setup_ddp_model(model: torch.nn.Module, device: torch.device, 
                   find_unused_parameters: bool = False) -> torch.nn.Module:
    """
    Setup model for Distributed Data Parallel training.
    
    Args:
        model: Model to wrap with DDP
        device: Device to place model on
        find_unused_parameters: Whether to find unused parameters (useful for some models)
    
    Returns:
        DDP-wrapped model
    """
    if dist.is_initialized():
        model = model.to(device)
        model = DDP(model, device_ids=[device], find_unused_parameters=find_unused_parameters)
        logger.info(f"Model wrapped with DDP on device {device}")
    else:
        model = model.to(device)
        logger.info(f"Model placed on device {device} (single GPU)")
    
    return model


def create_distributed_dataloader(dataset, batch_size: int, num_workers: int = 4,
                                pin_memory: bool = True, shuffle: bool = True,
                                drop_last: bool = True, **kwargs) -> DataLoader:
    """
    Create a DataLoader with distributed sampling support.
    
    Args:
        dataset: Dataset to create DataLoader for
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader with distributed sampling
    """
    # Create distributed sampler if running in distributed mode
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset, 
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Don't shuffle when using DistributedSampler
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        **kwargs
    )
    
    return dataloader


def create_smart_cache_dataset(data: List[dict], transforms, cache_rate: float = 0.5,
                              replace_rate: float = 0.5, num_init_workers: int = 4,
                              num_replace_workers: int = 4) -> SmartCacheDataset:
    """
    Create a MONAI SmartCacheDataset for efficient multi-GPU data loading.
    
    Args:
        data: List of data dictionaries
        transforms: MONAI transforms to apply
        cache_rate: Fraction of data to cache initially
        replace_rate: Fraction of cached data to replace each epoch
        num_init_workers: Number of workers for initial caching
        num_replace_workers: Number of workers for data replacement
    
    Returns:
        SmartCacheDataset instance
    """
    return SmartCacheDataset(
        data=data,
        transform=transforms,
        cache_rate=cache_rate,
        replace_rate=replace_rate,
        num_init_workers=num_init_workers,
        num_replace_workers=num_replace_workers,
        progress=True
    )


def create_gpu_optimized_transforms(base_transforms, device: torch.device) -> Any:
    """
    Add GPU-optimized transforms to the transform pipeline.
    
    Args:
        base_transforms: Base MONAI transforms
        device: Target device for GPU transforms
    
    Returns:
        Enhanced transforms with GPU optimization
    """
    # Add ToDeviced transform to move data to GPU early
    gpu_transforms = [
        ToDeviced(keys=["image"], device=device, non_blocking=True)
    ]
    
    # Combine with base transforms
    if hasattr(base_transforms, 'transforms'):
        # If it's a Compose object, get the transforms
        all_transforms = base_transforms.transforms + gpu_transforms
    else:
        # If it's a list, combine directly
        all_transforms = base_transforms + gpu_transforms
    
    from monai.transforms import Compose
    return Compose(all_transforms)


def synchronize_distributed():
    """Synchronize all processes in distributed training."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
    """
    Reduce tensor across all processes in distributed training.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('mean', 'sum', 'max', 'min')
    
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Convert operation string to dist.ReduceOp
    op_map = {
        'mean': dist.ReduceOp.SUM,
        'sum': dist.ReduceOp.SUM,
        'max': dist.ReduceOp.MAX,
        'min': dist.ReduceOp.MIN
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    # Reduce the tensor
    dist.all_reduce(tensor, op=op_map[op])
    
    # For mean, divide by world size
    if op == 'mean':
        tensor = tensor / dist.get_world_size()
    
    return tensor


def save_checkpoint_distributed(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                               epoch: int, loss: float, filepath: str, is_best: bool = False):
    """
    Save checkpoint in distributed training (only on rank 0).
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # Only save on rank 0
    
    # Get the actual model (unwrap DDP)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'is_best': is_best
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint_distributed(filepath: str, model: torch.nn.Module, 
                               optimizer: torch.optim.Optimizer = None) -> dict:
    """
    Load checkpoint in distributed training.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def get_optimal_batch_size_per_gpu(base_batch_size: int, num_gpus: int) -> int:
    """
    Calculate optimal batch size per GPU for distributed training.
    
    Args:
        base_batch_size: Total batch size for single GPU
        num_gpus: Number of GPUs
    
    Returns:
        Batch size per GPU
    """
    # For distributed training, we typically want to maintain the same effective batch size
    # So we divide by the number of GPUs
    batch_size_per_gpu = max(1, base_batch_size // num_gpus)
    
    # Ensure batch size is reasonable (not too small)
    if batch_size_per_gpu < 2:
        logger.warning(f"Batch size per GPU ({batch_size_per_gpu}) is very small. "
                      f"Consider increasing total batch size or reducing number of GPUs.")
    
    return batch_size_per_gpu


def get_optimal_num_workers(num_gpus: int, cpu_count: int = None) -> int:
    """
    Calculate optimal number of workers for DataLoader based on available resources.
    
    Args:
        num_gpus: Number of GPUs
        cpu_count: Number of CPU cores (auto-detected if None)
    
    Returns:
        Optimal number of workers
    """
    if cpu_count is None:
        cpu_count = os.cpu_count()
    
    # Rule of thumb: 2-4 workers per GPU, but don't exceed CPU count
    workers_per_gpu = 4
    total_workers = num_gpus * workers_per_gpu
    
    # Don't exceed available CPU cores
    optimal_workers = min(total_workers, cpu_count)
    
    # Ensure we have at least 1 worker
    optimal_workers = max(1, optimal_workers)
    
    return optimal_workers

"""Distributed helpers for loss computation."""
import torch
import torch.distributed as dist


def gather_tensor(t: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all processes and concatenate.

    Parameters
    ----------
    t : torch.Tensor
        Local tensor.

    Returns
    -------
    torch.Tensor
        Concatenated tensor across all processes. If distributed training is
        not initialized, the input tensor is returned unchanged.
    """
    if not dist.is_available() or not dist.is_initialized():
        return t
    tensors = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, t)
    return torch.cat(tensors, dim=0)

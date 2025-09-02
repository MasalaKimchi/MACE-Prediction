import torch.optim as optim
from typing import Dict, Any, Optional


def create_optimizer(
    model_params,
    optimizer_name: str = 'adamw',
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> optim.Optimizer:
    """
    Factory function to create optimizers for survival analysis training.
    
    Args:
        model_params: Model parameters to optimize
        optimizer_name: Name of the optimizer ('adamw', 'adam', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adamw':
        return optim.AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'adam':
        return optim.Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

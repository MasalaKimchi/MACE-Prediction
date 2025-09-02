import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
from typing import Dict, Any, Optional, Tuple


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


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 100,
    **kwargs
) -> Optional[object]:
    """
    Create learning rate scheduler for survival analysis training.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of the scheduler ('cosine', 'cosine_warm_restarts', 'onecycle', None)
        epochs: Total number of epochs
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        Configured scheduler or None
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    elif scheduler_name == 'cosine_warm_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', epochs // 4),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    elif scheduler_name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10),
            epochs=epochs,
            steps_per_epoch=kwargs.get('steps_per_epoch', 1),
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def create_optimizer_and_scheduler(
    model_params,
    optimizer_name: str = 'adamw',
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_name: str = 'cosine',
    epochs: int = 100,
    **kwargs
) -> Tuple[optim.Optimizer, Optional[object]]:
    """
    Create both optimizer and scheduler for survival analysis training.
    
    Args:
        model_params: Model parameters to optimize
        optimizer_name: Name of the optimizer
        lr: Learning rate
        weight_decay: Weight decay
        scheduler_name: Name of the scheduler
        epochs: Total number of epochs
        **kwargs: Additional arguments for optimizer and scheduler
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = create_optimizer(
        model_params,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        **kwargs
    )
    
    return optimizer, scheduler

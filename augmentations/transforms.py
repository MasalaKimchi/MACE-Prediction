from typing import Tuple, Optional, Callable, Any
from monai.transforms import (
    LoadImage, Compose, Resize, RandFlip, RandRotate90, ScaleIntensityRange, EnsureType,
    LoadImaged, ScaleIntensityRanged, Resized, RandFlipd, RandRotate90d, EnsureTyped
)


def create_image_transforms(
    image_size: Tuple[int, int, int] = (256, 256, 64),
    augment: bool = True,
    intensity_range: Tuple[float, float] = (-1000, 1000),
    output_range: Tuple[float, float] = (0.0, 1.0)
) -> Callable[[Any], Any]:
    """
    Create image transforms for survival analysis.
    
    Args:
        image_size: Target image size (D, H, W)
        augment: Whether to apply data augmentation
        intensity_range: Input intensity range for normalization
        output_range: Output intensity range for normalization
    
    Returns:
        Composed transform pipeline
    """
    transforms = [
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensityRange(
            a_min=intensity_range[0], 
            a_max=intensity_range[1], 
            b_min=output_range[0], 
            b_max=output_range[1], 
            clip=True
        ),
        Resize(image_size),
        EnsureType(data_type='tensor'),
    ]
    
    if augment:
        transforms += [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate90(prob=0.5, max_k=3),
        ]
    
    return Compose(transforms)


def create_augmentation_pipeline(
    image_size: Tuple[int, int, int] = (256, 256, 64),
    augment: bool = True,
    intensity_range: Tuple[float, float] = (-1000, 1000),
    output_range: Tuple[float, float] = (0.0, 1.0)
) -> Callable[[Any], Any]:
    """
    Create MONAI-style augmentation pipeline for survival analysis.
    
    Args:
        image_size: Target image size (D, H, W)
        augment: Whether to apply data augmentation
        intensity_range: Input intensity range for normalization
        output_range: Output intensity range for normalization
    
    Returns:
        Composed transform pipeline for MONAI datasets
    """
    transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=intensity_range[0], 
            a_max=intensity_range[1], 
            b_min=output_range[0], 
            b_max=output_range[1], 
            clip=True
        ),
        Resized(keys=["image"], spatial_size=image_size),
        EnsureTyped(keys=["image"], data_type='tensor'),
    ]
    
    if augment:
        transforms += [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        ]
    
    return Compose(transforms)

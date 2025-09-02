import polars as pl
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Callable, Any
from data.preprocessing import load_survival_dataset_from_csv, preprocess_features
from monai.transforms import (
    LoadImage, Compose, Resize, RandFlip, RandRotate90, ScaleIntensityRange, EnsureType,
    LoadImaged, ScaleIntensityRanged, Resized, RandFlipd, RandRotate90d, EnsureTyped
)
from monai.data import CacheDataset, Dataset as MonaiDataset

def get_survival_dataset(
    csv_path: str,
    dicom_col: str = 'Dicom path',
    time_col: str = 'time',
    event_col: str = 'event',
    feature_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, pl.Series, pl.Series, pl.Series]:
    """
    Loads and preprocesses the survival dataset from CSV.
    Returns preprocessed features, time, event, and dicom_paths.
    """
    features, time, event, dicom_paths = load_survival_dataset_from_csv(
        csv_path,
        dicom_col=dicom_col,
        time_col=time_col,
        event_col=event_col,
        feature_cols=feature_cols
    )
    features = preprocess_features(features, categorical_cols, numerical_cols)
    return features, time, event, dicom_paths

class SurvivalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dicom_col: str = 'Dicom path',
        time_col: str = 'time',
        event_col: str = 'event',
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        image_size: Tuple[int, int, int] = (256, 256, 64),
        augment: bool = True,
        image_transform: Optional[Callable[[Any], Any]] = None
    ):
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path,
            dicom_col=dicom_col,
            time_col=time_col,
            event_col=event_col,
            feature_cols=feature_cols
        )
        features = preprocess_features(features, categorical_cols, numerical_cols)
        # Store as float32 numpy for zero-copy as_tensor in __getitem__
        self.features = features.to_numpy().astype(np.float32, copy=False)
        self.time = time.to_numpy().astype(np.float32, copy=False)
        self.event = event.to_numpy().astype(np.float32, copy=False)
        self.dicom_paths = dicom_paths.to_numpy()

        # Compose default transforms if not provided
        if image_transform is not None:
            self.image_transform = image_transform
        else:
            transforms = [
                LoadImage(image_only=True, ensure_channel_first=True),
                ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                Resize(image_size),
                EnsureType(data_type='tensor'),
            ]
            if augment:
                transforms += [
                    RandFlip(prob=0.5, spatial_axis=0),
                    RandFlip(prob=0.5, spatial_axis=1),
                    RandRotate90(prob=0.5, max_k=3),
                ]
            self.image_transform = Compose(transforms)

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.features[idx])
        t = torch.as_tensor(self.time[idx])
        e = torch.as_tensor(self.event[idx])
        dicom_path = self.dicom_paths[idx]
        # Load and preprocess DICOM image
        image = self.image_transform(dicom_path)
        return x, t, e, image, dicom_path

class MonaiSurvivalDataset:
    def __init__(
        self,
        csv_path: str,
        dicom_col: str = 'Dicom path',
        time_col: str = 'time',
        event_col: str = 'event',
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        image_size: Tuple[int, int, int] = (256, 256, 64),
        augment: bool = True,
        image_transform: Optional[Callable[[Any], Any]] = None,
        use_cache: bool = True,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        fitted_scaler: Optional[object] = None
    ):
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path,
            dicom_col=dicom_col,
            time_col=time_col,
            event_col=event_col,
            feature_cols=feature_cols
        )
        features, self.feature_scaler = preprocess_features(
            features, categorical_cols, numerical_cols, 
            fitted_scaler=fitted_scaler, return_scaler=True
        )
        features_arr = features.to_numpy()
        time_arr = time.to_numpy()
        event_arr = event.to_numpy()
        dicom_paths_arr = dicom_paths.to_numpy()

        # Prepare MONAI-style data list
        self.data = [
            {
                "image": dicom_paths_arr[i],
                "features": features_arr[i],
                "time": time_arr[i],
                "event": event_arr[i],
            }
            for i in range(len(dicom_paths_arr))
        ]

        # Compose default transforms if not provided
        if image_transform is not None:
            self.transforms = image_transform
        else:
            transforms = [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                Resized(keys=["image"], spatial_size=image_size),
                EnsureTyped(keys=["image"], data_type='tensor'),
            ]
            if augment:
                transforms += [
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
                ]
            self.transforms = Compose(transforms)

        # Choose between CacheDataset and Dataset
        if use_cache:
            self.dataset = CacheDataset(data=self.data, transform=self.transforms, cache_rate=cache_rate, num_workers=num_workers)
        else:
            self.dataset = MonaiDataset(data=self.data, transform=self.transforms)

    def __len__(self):
        return len(self.dataset)
    
    def get_feature_scaler(self):
        """Get the fitted feature scaler for saving/loading."""
        return self.feature_scaler

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Output: features, time, event, image, dicom_path
        x = torch.as_tensor(item["features"], dtype=torch.float32)
        t = torch.as_tensor(item["time"], dtype=torch.float32)
        e = torch.as_tensor(item["event"], dtype=torch.float32)
        image = item["image"]  # EnsureTyped already yields torch.Tensor
        dicom_path = self.data[idx]["image"]
        return x, t, e, image, dicom_path

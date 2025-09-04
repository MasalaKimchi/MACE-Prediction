import polars as pl
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Callable, Any
from data.preprocessing import load_survival_dataset_from_csv, preprocess_features
from monai.transforms import (
    LoadImage, Compose, Resize, RandFlip, RandRotate90, ScaleIntensityRange, EnsureType,
    LoadImaged, ScaleIntensityRanged, Resized, RandFlipd, RandRotate90d, EnsureTyped,
    ToDeviced
)
from monai.data import CacheDataset, Dataset as MonaiDataset, SmartCacheDataset

def get_survival_dataset(
    csv_path: str,
    nifti_col: str = 'NIFTI Path',
    time_col: str = 'time',
    event_col: str = 'event',
    feature_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, pl.Series, pl.Series, pl.Series]:
    """
    Loads and preprocesses the survival dataset from CSV.
    Returns preprocessed features, time, event, and nifti_paths.
    """
    features, time, event, nifti_paths = load_survival_dataset_from_csv(
        csv_path,
        nifti_col=nifti_col,
        time_col=time_col,
        event_col=event_col,
        feature_cols=feature_cols
    )
    features = preprocess_features(features, categorical_cols, numerical_cols)
    return features, time, event, nifti_paths

class SurvivalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        nifti_col: str = 'NIFTI Path',
        time_col: str = 'time',
        event_col: str = 'event',
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        image_size: Tuple[int, int, int] = (256, 256, 64),
        augment: bool = True,
        image_transform: Optional[Callable[[Any], Any]] = None
    ):
        features, time, event, nifti_paths = load_survival_dataset_from_csv(
            csv_path,
            nifti_col=nifti_col,
            time_col=time_col,
            event_col=event_col,
            feature_cols=feature_cols
        )
        features = preprocess_features(features, categorical_cols, numerical_cols)
        # Store as float32 numpy for zero-copy as_tensor in __getitem__
        self.features = features.to_numpy().astype(np.float32, copy=False)
        self.time = time.to_numpy().astype(np.float32, copy=False)
        self.event = event.to_numpy().astype(np.float32, copy=False)
        self.nifti_paths = nifti_paths.to_numpy()

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
                ]
            self.image_transform = Compose(transforms)

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.features[idx])
        t = torch.as_tensor(self.time[idx])
        e = torch.as_tensor(self.event[idx])
        nifti_path = self.nifti_paths[idx]
        # Load and preprocess NIFTI image
        image = self.image_transform(nifti_path)
        return x, t, e, image, nifti_path

class MonaiSurvivalDataset:
    def __init__(
        self,
        csv_path: str,
        nifti_col: str = 'NIFTI Path',
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
        fitted_scaler: Optional[object] = None,
        fold_column: Optional[str] = None,
        fold_value: Optional[str] = None
    ):
        features, time, event, nifti_paths = load_survival_dataset_from_csv(
            csv_path,
            nifti_col=nifti_col,
            time_col=time_col,
            event_col=event_col,
            feature_cols=feature_cols,
            fold_column=fold_column,
            fold_value=fold_value
        )
        features, self.feature_scaler = preprocess_features(
            features, categorical_cols, numerical_cols, 
            fitted_scaler=fitted_scaler, return_scaler=True
        )
        features_arr = features.to_numpy()
        time_arr = time.to_numpy()
        event_arr = event.to_numpy()
        nifti_paths_arr = nifti_paths.to_numpy()

        # Prepare MONAI-style data list
        self.data = [
            {
                "image": nifti_paths_arr[i],
                "features": features_arr[i],
                "time": time_arr[i],
                "event": event_arr[i],
            }
            for i in range(len(nifti_paths_arr))
        ]

        # Compose default transforms if not provided
        if image_transform is not None:
            self.transforms = image_transform
        else:
            transforms = [
                LoadImaged(keys=["image"], ensure_channel_first=True, image_only=True),
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
        # Output: features, time, event, image, nifti_path
        x = torch.as_tensor(item["features"], dtype=torch.float32)
        t = torch.as_tensor(item["time"], dtype=torch.float32)
        e = torch.as_tensor(item["event"], dtype=torch.float32)
        image = item["image"]  # EnsureTyped already yields torch.Tensor
        nifti_path = self.data[idx]["image"]
        return x, t, e, image, nifti_path


class MultiGPUSurvivalDataset:
    """
    Multi-GPU optimized survival dataset using MONAI SmartCacheDataset.
    Provides efficient data loading and caching for distributed training.
    """
    def __init__(
        self,
        csv_path: str,
        nifti_col: str = 'NIFTI Path',
        time_col: str = 'time',
        event_col: str = 'event',
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        image_size: Tuple[int, int, int] = (256, 256, 64),
        augment: bool = True,
        image_transform: Optional[Callable[[Any], Any]] = None,
        use_smart_cache: bool = True,
        cache_rate: float = 0.5,
        replace_rate: float = 0.5,
        num_init_workers: int = 4,
        num_replace_workers: int = 4,
        device: Optional[torch.device] = None,
        fitted_scaler: Optional[object] = None,
        fold_column: Optional[str] = None,
        fold_value: Optional[str] = None
    ):
        """
        Initialize multi-GPU optimized survival dataset.
        
        Args:
            csv_path: Path to CSV file with survival data
            nifti_col: Column name for NIFTI file paths
            time_col: Column name for survival times
            event_col: Column name for event indicators
            feature_cols: List of feature column names
            categorical_cols: List of categorical feature columns
            numerical_cols: List of numerical feature columns
            image_size: Target image size (D, H, W)
            augment: Whether to apply data augmentation
            image_transform: Custom image transforms
            use_smart_cache: Whether to use SmartCacheDataset
            cache_rate: Fraction of data to cache initially
            replace_rate: Fraction of cached data to replace each epoch
            num_init_workers: Number of workers for initial caching
            num_replace_workers: Number of workers for data replacement
            device: Target device for GPU transforms
            fitted_scaler: Pre-fitted feature scaler
            fold_column: Column name for fold-based splitting (e.g., 'Fold_1')
            fold_value: Value in fold_column to filter for (e.g., 'train' or 'val')
        """
        # Load and preprocess features
        features, time, event, nifti_paths = load_survival_dataset_from_csv(
            csv_path,
            nifti_col=nifti_col,
            time_col=time_col,
            event_col=event_col,
            feature_cols=feature_cols,
            fold_column=fold_column,
            fold_value=fold_value
        )
        features, self.feature_scaler = preprocess_features(
            features, categorical_cols, numerical_cols, 
            fitted_scaler=fitted_scaler, return_scaler=True
        )
        
        # Convert to numpy arrays
        features_arr = features.to_numpy()
        time_arr = time.to_numpy()
        event_arr = event.to_numpy()
        nifti_paths_arr = nifti_paths.to_numpy()

        # Prepare MONAI-style data list
        self.data = [
            {
                "image": nifti_paths_arr[i],
                "features": features_arr[i],
                "time": time_arr[i],
                "event": event_arr[i],
            }
            for i in range(len(nifti_paths_arr))
        ]

        # Set up transforms
        if image_transform is not None:
            self.transforms = image_transform
        else:
            transforms = [
                LoadImaged(keys=["image"], ensure_channel_first=True, image_only=True),
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                Resized(keys=["image"], spatial_size=image_size),
                EnsureTyped(keys=["image"], data_type='tensor'),
            ]
            
            # Add GPU optimization if device is specified
            if device is not None and device.type == 'cuda':
                transforms.append(ToDeviced(keys=["image"], device=device, non_blocking=True))
            
            if augment:
                transforms += [
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
                ]
            
            from monai.transforms import Compose
            self.transforms = Compose(transforms)

        # Choose dataset type based on configuration
        if use_smart_cache:
            self.dataset = SmartCacheDataset(
                data=self.data,
                transform=self.transforms,
                cache_rate=cache_rate,
                replace_rate=replace_rate,
                num_init_workers=num_init_workers,
                num_replace_workers=num_replace_workers,
                progress=True
            )
        else:
            # Fallback to regular CacheDataset
            self.dataset = CacheDataset(
                data=self.data,
                transform=self.transforms,
                cache_rate=cache_rate,
                num_workers=num_init_workers
            )

    def __len__(self):
        return len(self.dataset)
    
    def get_feature_scaler(self):
        """Get the fitted feature scaler for saving/loading."""
        return self.feature_scaler

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Output: features, time, event, image, nifti_path
        x = torch.as_tensor(item["features"], dtype=torch.float32)
        t = torch.as_tensor(item["time"], dtype=torch.float32)
        e = torch.as_tensor(item["event"], dtype=torch.float32)
        image = item["image"]  # Already a tensor from transforms
        nifti_path = self.data[idx]["image"]
        return x, t, e, image, nifti_path


def test_survival_dataset():
    """Test the survival dataset classes with real data."""
    import os
    import pandas as pd
    import torch
    import numpy as np
    
    print("Testing survival dataset classes with real data...")
    
    # Use the actual Excel file
    excel_path = 'data/UH_1950_calc_09x03x2025.xlsx'
    
    if not os.path.exists(excel_path):
        print(f"Excel file not found: {excel_path}")
        return
    
    print(f"Using real Excel file: {excel_path}")
    
    # Load and examine the data first
    df = pd.read_excel(excel_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Check if required columns exist
    required_cols = ['NIFTI Path', 'time', 'event']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return
    
    try:
        # Test 1: Test get_survival_dataset function with all features
        print("\n1. Testing get_survival_dataset function with all features...")
        # Get all feature columns
        feature_cols = [col for col in df.columns if col not in ['case_num', 'NIFTI Path', 'time', 'event']]
        categorical_cols = [col for col in feature_cols if col.startswith('is') or df[col].dtype == 'object']
        numerical_cols = [col for col in feature_cols if col not in categorical_cols]
        
        print(f"   Using {len(feature_cols)} total features:")
        print(f"   - Categorical: {len(categorical_cols)} features")
        print(f"   - Numerical: {len(numerical_cols)} features")
        
        features, time, event, nifti_paths = get_survival_dataset(
            excel_path,
            nifti_col='NIFTI Path',
            time_col='time',
            event_col='event',
            feature_cols=feature_cols,  # Use ALL features
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols
        )
        
        print(f"   Features shape: {features.shape}")
        print(f"   Time shape: {time.shape}")
        print(f"   Event shape: {event.shape}")
        print(f"   NIFTI paths shape: {nifti_paths.shape}")
        print(f"   First NIFTI path: {nifti_paths[0]}")
        print(f"   Time range: {time.min():.1f} - {time.max():.1f} days")
        print(f"   Event rate: {event.sum() / len(event):.3f}")
        
        # Test 2: Test SurvivalDataset class with all features
        print("\n2. Testing SurvivalDataset class with all features...")
        try:
            # Test with a subset of data to avoid loading too many large files
            subset_size = 5
            print(f"   Testing with first {subset_size} samples...")
            
            # Create a temporary CSV with subset and full paths
            subset_df = df.head(subset_size).copy()
            nifti_dir = r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned'
            subset_df['NIFTI Path'] = subset_df['NIFTI Path'].apply(lambda x: os.path.join(nifti_dir, x))
            temp_csv = 'temp_subset.csv'
            subset_df.to_csv(temp_csv, index=False)
            
            # Test SurvivalDataset with all features
            dataset = SurvivalDataset(
                csv_path=temp_csv,
                nifti_col='NIFTI Path',
                time_col='time',
                event_col='event',
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,
                numerical_cols=numerical_cols,
                image_size=(128, 128, 32),  # Smaller size for testing
                augment=False
            )
            
            print(f"   Dataset length: {len(dataset)}")
            
            # Test getting a sample
            x, t, e, image, nifti_path = dataset[0]
            print(f"   Sample - Features shape: {x.shape}, Time: {t}, Event: {e}")
            print(f"   Sample - Image shape: {image.shape}, NIFTI path: {nifti_path}")
            
            # Verify data types
            assert isinstance(x, torch.Tensor), "Features should be torch.Tensor"
            assert isinstance(t, torch.Tensor), "Time should be torch.Tensor"
            assert isinstance(e, torch.Tensor), "Event should be torch.Tensor"
            assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
            
            # Verify feature count
            assert x.shape[0] == len(feature_cols), f"Expected {len(feature_cols)} features, got {x.shape[0]}"
            
            print("   SurvivalDataset test passed with all features!")
            
            # Clean up temp file
            os.unlink(temp_csv)
            
        except Exception as e:
            print(f"   SurvivalDataset test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Test MonaiSurvivalDataset class with all features
        print("\n3. Testing MonaiSurvivalDataset class with all features...")
        try:
            # Create a temporary CSV with subset and full paths
            subset_size = 3
            subset_df = df.head(subset_size).copy()
            nifti_dir = r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned'
            subset_df['NIFTI Path'] = subset_df['NIFTI Path'].apply(lambda x: os.path.join(nifti_dir, x))
            temp_csv = 'temp_subset_monai.csv'
            subset_df.to_csv(temp_csv, index=False)
            
            # Test MonaiSurvivalDataset with all features
            monai_dataset = MonaiSurvivalDataset(
                csv_path=temp_csv,
                nifti_col='NIFTI Path',
                time_col='time',
                event_col='event',
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,
                numerical_cols=numerical_cols,
                image_size=(128, 128, 32),
                use_cache=False,  # Disable caching for testing
                augment=False
            )
            
            print(f"   MonaiSurvivalDataset length: {len(monai_dataset)}")
            
            # Test getting a sample
            x, t, e, image, nifti_path = monai_dataset[0]
            print(f"   Sample - Features shape: {x.shape}, Time: {t}, Event: {e}")
            print(f"   Sample - Image shape: {image.shape}, NIFTI path: {nifti_path}")
            
            # Verify data types
            assert isinstance(x, torch.Tensor), "Features should be torch.Tensor"
            assert isinstance(t, torch.Tensor), "Time should be torch.Tensor"
            assert isinstance(e, torch.Tensor), "Event should be torch.Tensor"
            assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
            
            # Verify feature count
            assert x.shape[0] == len(feature_cols), f"Expected {len(feature_cols)} features, got {x.shape[0]}"
            
            # Test feature scaler
            scaler = monai_dataset.get_feature_scaler()
            print(f"   Feature scaler type: {type(scaler)}")
            
            print("   MonaiSurvivalDataset test passed with all features!")
            
            # Clean up temp file
            os.unlink(temp_csv)
            
        except Exception as e:
            print(f"   MonaiSurvivalDataset test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Test with DataLoader using all features
        print("\n4. Testing with PyTorch DataLoader using all features...")
        try:
            from torch.utils.data import DataLoader
            
            # Create a temporary CSV with subset and full paths
            subset_size = 4
            subset_df = df.head(subset_size).copy()
            nifti_dir = r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned'
            subset_df['NIFTI Path'] = subset_df['NIFTI Path'].apply(lambda x: os.path.join(nifti_dir, x))
            temp_csv = 'temp_subset_dataloader.csv'
            subset_df.to_csv(temp_csv, index=False)
            
            # Create a dataset with all features
            simple_dataset = SurvivalDataset(
                csv_path=temp_csv,
                nifti_col='NIFTI Path',
                time_col='time',
                event_col='event',
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,
                numerical_cols=numerical_cols,
                image_size=(128, 128, 32),
                augment=False
            )
            
            dataloader = DataLoader(
                simple_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0  # Use 0 workers for testing
            )
            
            # Test one batch
            for batch_idx, (features, times, events, images, nifti_paths) in enumerate(dataloader):
                print(f"   Batch {batch_idx}: Features {features.shape}, Times {times.shape}, Events {events.shape}")
                print(f"   Batch {batch_idx}: Images {images.shape}, NIFTI paths {len(nifti_paths)}")
                
                # Verify batch feature count
                assert features.shape[1] == len(feature_cols), f"Expected {len(feature_cols)} features, got {features.shape[1]}"
                break  # Only test first batch
            
            print("   DataLoader test passed with all features!")
            
            # Clean up temp file
            os.unlink(temp_csv)
            
        except Exception as e:
            print(f"   DataLoader test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nAll survival dataset tests completed with all features!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_survival_dataset()

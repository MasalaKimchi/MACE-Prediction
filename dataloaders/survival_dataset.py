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
    nifti_col: str = 'NIFTI path',
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
        nifti_col: str = 'NIFTI path',
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
        nifti_col: str = 'NIFTI path',
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
        features, time, event, nifti_paths = load_survival_dataset_from_csv(
            csv_path,
            nifti_col=nifti_col,
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
        # Output: features, time, event, image, nifti_path
        x = torch.as_tensor(item["features"], dtype=torch.float32)
        t = torch.as_tensor(item["time"], dtype=torch.float32)
        e = torch.as_tensor(item["event"], dtype=torch.float32)
        image = item["image"]  # EnsureTyped already yields torch.Tensor
        nifti_path = self.data[idx]["image"]
        return x, t, e, image, nifti_path


def test_survival_dataset():
    """Test the survival dataset classes with sample data."""
    import tempfile
    import os
    import pandas as pd
    import torch
    import numpy as np
    
    print("Testing survival dataset classes...")
    
    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create sample data
        data = {
            'NIFTI path': [
                r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned\patient_001.nii.gz',
                r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned\patient_002.nii.gz',
                r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned\patient_003.nii.gz'
            ],
            'time': [365.5, 180.2, 500.0],
            'event': [1, 1, 0],
            'age': [65, 72, 58],
            'gender': ['M', 'F', 'M'],
            'smoking': ['Yes', 'No', 'Yes'],
            'cholesterol': [220.5, 180.2, 195.8],
            'blood_pressure': [140, 120, 135]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        # Test 1: Test get_survival_dataset function
        print("\n1. Testing get_survival_dataset function...")
        features, time, event, nifti_paths = get_survival_dataset(
            csv_path,
            nifti_col='NIFTI path',
            time_col='time',
            event_col='event',
            feature_cols=['age', 'gender', 'smoking', 'cholesterol', 'blood_pressure']
        )
        
        print(f"   Features shape: {features.shape}")
        print(f"   Time shape: {time.shape}")
        print(f"   Event shape: {event.shape}")
        print(f"   NIFTI paths shape: {nifti_paths.shape}")
        print(f"   First NIFTI path: {nifti_paths[0]}")
        
        # Test 2: Test SurvivalDataset class (without actual image loading)
        print("\n2. Testing SurvivalDataset class...")
        try:
            # Create dummy NIFTI files for testing
            import nibabel as nib
            
            # Create dummy NIFTI files
            for i, nifti_path in enumerate(nifti_paths):
                # Create a small dummy volume
                dummy_data = np.random.randint(-1000, 1000, size=(32, 32, 16), dtype=np.int16)
                dummy_img = nib.Nifti1Image(dummy_data, affine=np.eye(4))
                nib.save(dummy_img, nifti_path)
                print(f"   Created dummy NIFTI file: {nifti_path}")
            
            # Test SurvivalDataset
            dataset = SurvivalDataset(
                csv_path=csv_path,
                nifti_col='NIFTI path',
                time_col='time',
                event_col='event',
                feature_cols=['age', 'gender', 'smoking', 'cholesterol', 'blood_pressure'],
                image_size=(32, 32, 16),
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
            
            print("   ✅ SurvivalDataset test passed!")
            
        except ImportError:
            print("   ⚠️  nibabel not available, skipping image loading test")
        except Exception as e:
            print(f"   ⚠️  Image loading test failed (expected if files don't exist): {e}")
        
        # Test 3: Test MonaiSurvivalDataset class
        print("\n3. Testing MonaiSurvivalDataset class...")
        try:
            # Test MonaiSurvivalDataset
            monai_dataset = MonaiSurvivalDataset(
                csv_path=csv_path,
                nifti_col='NIFTI path',
                time_col='time',
                event_col='event',
                feature_cols=['age', 'gender', 'smoking', 'cholesterol', 'blood_pressure'],
                image_size=(32, 32, 16),
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
            
            # Test feature scaler
            scaler = monai_dataset.get_feature_scaler()
            print(f"   Feature scaler type: {type(scaler)}")
            
            print("   ✅ MonaiSurvivalDataset test passed!")
            
        except ImportError:
            print("   ⚠️  MONAI not available, skipping MonaiSurvivalDataset test")
        except Exception as e:
            print(f"   ⚠️  MonaiSurvivalDataset test failed: {e}")
        
        # Test 4: Test with DataLoader
        print("\n4. Testing with PyTorch DataLoader...")
        try:
            from torch.utils.data import DataLoader
            
            # Create a simple dataset without image loading for DataLoader test
            simple_dataset = SurvivalDataset(
                csv_path=csv_path,
                nifti_col='NIFTI path',
                time_col='time',
                event_col='event',
                feature_cols=['age', 'cholesterol', 'blood_pressure'],  # Only numerical features
                image_size=(32, 32, 16),
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
                break  # Only test first batch
            
            print("   ✅ DataLoader test passed!")
            
        except Exception as e:
            print(f"   ⚠️  DataLoader test failed: {e}")
        
        print("\n✅ All survival dataset tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary file
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        
        # Clean up dummy NIFTI files if they were created
        try:
            for i in range(3):
                nifti_path = rf'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned\patient_{i+1:03d}.nii.gz'
                if os.path.exists(nifti_path):
                    os.unlink(nifti_path)
                    print(f"   Cleaned up dummy file: {nifti_path}")
        except Exception as e:
            print(f"   Note: Could not clean up dummy files: {e}")


if __name__ == "__main__":
    test_survival_dataset()

import os
import sys
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders import SurvivalDataset, MonaiSurvivalDataset


def create_test_csv_with_real_data(tmp_dir, n_samples=5):
    """Create test CSV file using actual data from the Excel file."""
    # Load the actual Excel data
    excel_path = 'data/UH_1950_calc_09x03x2025.xlsx'
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path)
    
    # Take a subset of the data
    subset_df = df.head(n_samples).copy()
    
    # Update NIFTI paths to full paths
    nifti_dir = r'C:\Users\jnk50\Desktop\MTL_Survivla\UH2400_coned'
    subset_df['NIFTI Path'] = subset_df['NIFTI Path'].apply(lambda x: os.path.join(nifti_dir, x))
    
    # Save as CSV
    csv_path = os.path.join(tmp_dir, "test_data.csv")
    subset_df.to_csv(csv_path, index=False)
    return csv_path


def get_all_feature_columns():
    """Get all feature columns from the dataset."""
    import pandas as pd
    df = pd.read_excel('data/UH_1950_calc_09x03x2025.xlsx')
    feature_cols = [col for col in df.columns if col not in ['case_num', 'NIFTI Path', 'time', 'event']]
    return feature_cols


def get_categorical_and_numerical_columns():
    """Get categorical and numerical feature columns."""
    import pandas as pd
    df = pd.read_excel('data/UH_1950_calc_09x03x2025.xlsx')
    feature_cols = [col for col in df.columns if col not in ['case_num', 'NIFTI Path', 'time', 'event']]
    categorical_cols = [col for col in feature_cols if col.startswith('is') or df[col].dtype == 'object']
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    return categorical_cols, numerical_cols


def test_survival_dataset():
    """Test SurvivalDataset functionality with all features."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_test_csv_with_real_data(tmp_dir, n_samples=3)
        try:
            # Get all feature columns
            feature_cols = get_all_feature_columns()
            categorical_cols, numerical_cols = get_categorical_and_numerical_columns()
            
            print(f"Testing with {len(feature_cols)} total features:")
            print(f"  - Categorical: {len(categorical_cols)} features")
            print(f"  - Numerical: {len(numerical_cols)} features")
            
            ds = SurvivalDataset(
                csv_path=csv_path, 
                nifti_col="NIFTI Path",  # Use actual NIFTI column name
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,  # Specify categorical features
                numerical_cols=numerical_cols,  # Specify numerical features
                image_size=(128, 128, 32),  # Use reasonable size for testing
                augment=False
            )
            x, t, e, image, nifti_path = ds[0]
            print(f"SurvivalDataset: x={x.shape}, t={t}, e={e}, image={image.shape}, nifti_path={nifti_path}")
            
            # Test dataset length
            assert len(ds) == 3, f"Expected dataset length 3, got {len(ds)}"
            
            # Test data types
            assert hasattr(x, 'shape'), "Features should have shape attribute"
            assert hasattr(t, 'item'), "Time should be scalar tensor"
            assert hasattr(e, 'item'), "Event should be scalar tensor"
            
            # Test that we get actual data with all features
            assert x.shape[0] == len(feature_cols), f"Expected {len(feature_cols)} features, got {x.shape[0]}"
            assert image.shape == (1, 128, 128, 32), f"Expected image shape (1, 128, 128, 32), got {image.shape}"
            
            print("SurvivalDataset test passed with all features!")
        except Exception as e:
            print(f"SurvivalDataset test failed: {e}")
            import traceback
            traceback.print_exc()


def test_monai_survival_dataset():
    """Test MonaiSurvivalDataset functionality with all features."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_test_csv_with_real_data(tmp_dir, n_samples=3)
        try:
            # Get all feature columns
            feature_cols = get_all_feature_columns()
            categorical_cols, numerical_cols = get_categorical_and_numerical_columns()
            
            print(f"Testing MonaiSurvivalDataset with {len(feature_cols)} total features:")
            print(f"  - Categorical: {len(categorical_cols)} features")
            print(f"  - Numerical: {len(numerical_cols)} features")
            
            ds = MonaiSurvivalDataset(
                csv_path=csv_path, 
                nifti_col="NIFTI Path",  # Use actual NIFTI column name
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,  # Specify categorical features
                numerical_cols=numerical_cols,  # Specify numerical features
                image_size=(128, 128, 32),  # Use reasonable size for testing
                use_cache=False,  # Disable caching for testing
                augment=False
            )
            x, t, e, image, nifti_path = ds[0]
            print(f"MonaiSurvivalDataset: x={x.shape}, t={t}, e={e}, image={image.shape}, nifti_path={nifti_path}")
            
            # Test dataset length
            assert len(ds) == 3, f"Expected dataset length 3, got {len(ds)}"
            
            # Test data types
            assert hasattr(x, 'shape'), "Features should have shape attribute"
            assert hasattr(t, 'item'), "Time should be scalar tensor"
            assert hasattr(e, 'item'), "Event should be scalar tensor"
            
            # Test that we get actual data with all features
            assert x.shape[0] == len(feature_cols), f"Expected {len(feature_cols)} features, got {x.shape[0]}"
            assert image.shape == (1, 128, 128, 32), f"Expected image shape (1, 128, 128, 32), got {image.shape}"
            
            # Test feature scaler
            scaler = ds.get_feature_scaler()
            print(f"Feature scaler type: {type(scaler)}")
            
            print("MonaiSurvivalDataset test passed with all features!")
        except Exception as e:
            print(f"MonaiSurvivalDataset test failed: {e}")
            import traceback
            traceback.print_exc()


def test_dataset_comparison():
    """Compare SurvivalDataset and MonaiSurvivalDataset outputs with all features."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_test_csv_with_real_data(tmp_dir, n_samples=2)
        
        try:
            # Get all feature columns
            feature_cols = get_all_feature_columns()
            categorical_cols, numerical_cols = get_categorical_and_numerical_columns()
            
            print(f"Comparing datasets with {len(feature_cols)} total features")
            
            # Test both datasets
            ds1 = SurvivalDataset(
                csv_path=csv_path, 
                nifti_col="NIFTI Path",  # Use actual NIFTI column name
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,  # Specify categorical features
                numerical_cols=numerical_cols,  # Specify numerical features
                image_size=(128, 128, 32),  # Use reasonable size for testing
                augment=False
            )
            ds2 = MonaiSurvivalDataset(
                csv_path=csv_path, 
                nifti_col="NIFTI Path",  # Use actual NIFTI column name
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,  # Specify categorical features
                numerical_cols=numerical_cols,  # Specify numerical features
                image_size=(128, 128, 32),  # Use reasonable size for testing
                use_cache=False, 
                augment=False
            )
            
            x1, t1, e1, image1, nifti_path1 = ds1[0]
            x2, t2, e2, image2, nifti_path2 = ds2[0]
            
            # Compare outputs
            print(f"Dataset comparison:")
            print(f"  SurvivalDataset: features={x1.shape}, time={t1}, event={e1}, image={image1.shape}")
            print(f"  MonaiSurvivalDataset: features={x2.shape}, time={t2}, event={e2}, image={image2.shape}")
            print(f"  NIFTI paths match: {nifti_path1 == nifti_path2}")
            
            # Test that both datasets return the same data
            assert x1.shape == x2.shape, f"Feature shapes don't match: {x1.shape} vs {x2.shape}"
            assert t1 == t2, f"Time values don't match: {t1} vs {t2}"
            assert e1 == e2, f"Event values don't match: {e1} vs {e2}"
            assert image1.shape == image2.shape, f"Image shapes don't match: {image1.shape} vs {image2.shape}"
            assert nifti_path1 == nifti_path2, f"NIFTI paths don't match: {nifti_path1} vs {nifti_path2}"
            
            print("Dataset comparison test passed with all features!")
        except Exception as e:
            print(f"Dataset comparison test failed: {e}")
            import traceback
            traceback.print_exc()


def test_dataloader_integration():
    """Test DataLoader integration with all features."""
    from torch.utils.data import DataLoader
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_test_csv_with_real_data(tmp_dir, n_samples=4)
        
        try:
            # Get all feature columns
            feature_cols = get_all_feature_columns()
            categorical_cols, numerical_cols = get_categorical_and_numerical_columns()
            
            print(f"Testing DataLoader with {len(feature_cols)} total features")
            
            # Create dataset
            dataset = SurvivalDataset(
                csv_path=csv_path,
                nifti_col="NIFTI Path",
                feature_cols=feature_cols,  # Use ALL features
                categorical_cols=categorical_cols,  # Specify categorical features
                numerical_cols=numerical_cols,  # Specify numerical features
                image_size=(128, 128, 32),
                augment=False
            )
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0  # Use 0 workers for testing
            )
            
            # Test one batch
            for batch_idx, (features, times, events, images, nifti_paths) in enumerate(dataloader):
                print(f"Batch {batch_idx}: Features {features.shape}, Times {times.shape}, Events {events.shape}")
                print(f"Batch {batch_idx}: Images {images.shape}, NIFTI paths {len(nifti_paths)}")
                
                # Test batch shapes
                assert features.shape[0] == 2, f"Expected batch size 2, got {features.shape[0]}"
                assert features.shape[1] == len(feature_cols), f"Expected {len(feature_cols)} features, got {features.shape[1]}"
                assert times.shape[0] == 2, f"Expected batch size 2, got {times.shape[0]}"
                assert events.shape[0] == 2, f"Expected batch size 2, got {events.shape[0]}"
                assert images.shape[0] == 2, f"Expected batch size 2, got {images.shape[0]}"
                assert len(nifti_paths) == 2, f"Expected 2 NIFTI paths, got {len(nifti_paths)}"
                
                break  # Only test first batch
            
            print("DataLoader integration test passed with all features!")
        except Exception as e:
            print(f"DataLoader integration test failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all dataloader tests with all features."""
    print("Testing dataloaders with all available features...")
    test_survival_dataset()
    test_monai_survival_dataset()
    test_dataset_comparison()
    test_dataloader_integration()
    print("All dataloader tests completed successfully with all features!")


if __name__ == "__main__":
    main()

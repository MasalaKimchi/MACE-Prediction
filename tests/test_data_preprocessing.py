import os
import sys
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_survival_dataset_from_csv, preprocess_features


def create_dummy_csv(tmp_dir, n_samples=5):
    """Create dummy CSV file for testing."""
    df = pd.DataFrame({
        "Dicom path": [f"dicom_{i}" for i in range(n_samples)],
        "time": np.random.randint(1, 100, size=n_samples),
        "event": np.random.randint(0, 2, size=n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "categorical_feature": np.random.choice(['A', 'B', 'C'], size=n_samples),
        "numerical_feature": np.random.randint(1, 10, size=n_samples),
    })
    csv_path = os.path.join(tmp_dir, "dummy.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_survival_dataset():
    """Test loading survival dataset from CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        
        # Test with all features
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["feature1", "feature2", "categorical_feature", "numerical_feature"]
        )
        
        print(f"Loaded dataset: features={features.shape}, time={time.shape}, event={event.shape}, dicom_paths={dicom_paths.shape}")
        
        # Verify shapes
        assert len(features) == 5, f"Expected 5 samples, got {len(features)}"
        assert len(time) == 5, f"Expected 5 time values, got {len(time)}"
        assert len(event) == 5, f"Expected 5 event values, got {len(event)}"
        assert len(dicom_paths) == 5, f"Expected 5 dicom paths, got {len(dicom_paths)}"
        
        # Verify column names
        expected_cols = ["feature1", "feature2", "categorical_feature", "numerical_feature"]
        assert list(features.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(features.columns)}"


def test_load_survival_dataset_subset():
    """Test loading survival dataset with subset of features."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        
        # Test with subset of features
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["feature1", "feature2"]
        )
        
        print(f"Loaded dataset subset: features={features.shape}, time={time.shape}, event={event.shape}")
        
        # Verify shapes and columns
        assert features.shape == (5, 2), f"Expected (5, 2), got {features.shape}"
        assert list(features.columns) == ["feature1", "feature2"]


def test_preprocess_features_numerical_only():
    """Test preprocessing with numerical features only."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["feature1", "feature2", "numerical_feature"]
        )
        
        # Preprocess numerical features
        processed_features = preprocess_features(
            features, 
            categorical_cols=None, 
            numerical_cols=["feature1", "feature2", "numerical_feature"]
        )
        
        print(f"Preprocessed numerical features: {processed_features.shape}")
        print(f"  Original columns: {list(features.columns)}")
        print(f"  Processed columns: {list(processed_features.columns)}")
        
        # Verify shape and columns
        assert processed_features.shape == (5, 3), f"Expected (5, 3), got {processed_features.shape}"
        assert list(processed_features.columns) == ["feature1", "feature2", "numerical_feature"]


def test_preprocess_features_categorical_only():
    """Test preprocessing with categorical features only."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["categorical_feature"]
        )
        
        # Preprocess categorical features
        processed_features = preprocess_features(
            features, 
            categorical_cols=["categorical_feature"], 
            numerical_cols=None
        )
        
        print(f"Preprocessed categorical features: {processed_features.shape}")
        print(f"  Original columns: {list(features.columns)}")
        print(f"  Processed columns: {list(processed_features.columns)}")
        
        # Verify one-hot encoding (check that we have categorical columns)
        assert len(processed_features.columns) >= 2, f"Expected at least 2 categorical columns, got {len(processed_features.columns)}"
        assert all(col.startswith("categorical_feature_") for col in processed_features.columns), f"Expected categorical columns, got {list(processed_features.columns)}"


def test_preprocess_features_mixed():
    """Test preprocessing with both categorical and numerical features."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["feature1", "feature2", "categorical_feature", "numerical_feature"]
        )
        
        # Preprocess mixed features
        processed_features = preprocess_features(
            features, 
            categorical_cols=["categorical_feature"], 
            numerical_cols=["feature1", "feature2", "numerical_feature"]
        )
        
        print(f"Preprocessed mixed features: {processed_features.shape}")
        print(f"  Original columns: {list(features.columns)}")
        print(f"  Processed columns: {list(processed_features.columns)}")
        
        # Verify combined preprocessing (order may vary due to one-hot encoding)
        expected_numerical_cols = ["feature1", "feature2", "numerical_feature"]
        categorical_cols = [col for col in processed_features.columns if col.startswith("categorical_feature_")]
        
        assert len(processed_features.columns) >= 5, f"Expected at least 5 columns, got {len(processed_features.columns)}"
        assert all(col in processed_features.columns for col in expected_numerical_cols), f"Missing numerical columns in {list(processed_features.columns)}"
        assert len(categorical_cols) >= 2, f"Expected at least 2 categorical columns, got {categorical_cols}"


def test_preprocess_features_auto_inference():
    """Test preprocessing with automatic column type inference."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["feature1", "feature2", "categorical_feature", "numerical_feature"]
        )
        
        # Preprocess with automatic inference
        processed_features = preprocess_features(features)
        
        print(f"Auto-inferred preprocessing: {processed_features.shape}")
        print(f"  Original columns: {list(features.columns)}")
        print(f"  Processed columns: {list(processed_features.columns)}")
        
        # Should have numerical features + one-hot encoded categorical features
        assert len(processed_features.columns) >= 3, "Should have at least 3 processed columns"


def test_end_to_end_preprocessing():
    """Test complete preprocessing pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        
        # Load and preprocess
        features, time, event, dicom_paths = load_survival_dataset_from_csv(
            csv_path, 
            feature_cols=["feature1", "feature2", "categorical_feature", "numerical_feature"]
        )
        processed_features = preprocess_features(
            features, 
            categorical_cols=["categorical_feature"], 
            numerical_cols=["feature1", "feature2", "numerical_feature"]
        )
        
        print(f"End-to-end preprocessing:")
        print(f"  Features: {processed_features.shape}")
        print(f"  Time: {time.shape}")
        print(f"  Event: {event.shape}")
        print(f"  DICOM paths: {dicom_paths.shape}")
        
        # Verify all components
        assert processed_features.shape[0] == len(time) == len(event) == len(dicom_paths), "All components should have same length"


def main():
    """Run all data preprocessing tests."""
    print("Testing data preprocessing...")
    test_load_survival_dataset()
    test_load_survival_dataset_subset()
    test_preprocess_features_numerical_only()
    test_preprocess_features_categorical_only()
    test_preprocess_features_mixed()
    test_preprocess_features_auto_inference()
    test_end_to_end_preprocessing()
    print("All data preprocessing tests completed successfully!")


if __name__ == "__main__":
    main()

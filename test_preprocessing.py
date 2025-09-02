import os
import numpy as np
import pandas as pd
import tempfile
from preprocessing import load_survival_dataset_from_csv, preprocess_features

def create_dummy_csv(tmp_dir, n_samples=2):
    df = pd.DataFrame({
        "Dicom path": [f"dicom_{i}" for i in range(n_samples)],
        "time": np.random.randint(1, 100, size=n_samples),
        "event": np.random.randint(0, 2, size=n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
    })
    csv_path = os.path.join(tmp_dir, "dummy.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def test_preprocessing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        features, time, event, dicom_paths = load_survival_dataset_from_csv(csv_path, feature_cols=["feature1", "feature2"])
        features = preprocess_features(features, categorical_cols=None, numerical_cols=["feature1", "feature2"])
        print(f"features: {features.shape}, time: {time.shape}, event: {event.shape}, dicom_paths: {dicom_paths.shape}")

def main():
    test_preprocessing()

if __name__ == "__main__":
    main() 
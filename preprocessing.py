import polars as pl
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

# Optional: import pydicom if you want to process DICOM images
# import pydicom


def load_survival_dataset_from_csv(
    csv_path: str,
    dicom_col: str = 'Dicom path',
    time_col: str = 'time',
    event_col: str = 'event',
    feature_cols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, pl.Series, pl.Series, pl.Series]:
    """
    Loads a survival analysis dataset from a CSV file using polars.
    Args:
        csv_path: Path to the CSV file.
        dicom_col: Name of the column containing DICOM image paths.
        time_col: Name of the column containing time-to-event.
        event_col: Name of the column containing event indicator (1=event, 0=censored).
        feature_cols: List of feature columns to use. If None, use all except dicom_col, time_col, event_col.
    Returns:
        features: DataFrame of features (excluding DICOM path, time, event).
        time: Series of time-to-event.
        event: Series of event indicators.
        dicom_paths: Series of DICOM image paths.
    """
    df = pl.read_csv(csv_path)
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [dicom_col, time_col, event_col]]
    
    features = df.select(feature_cols)
    time = df[time_col]
    event = df[event_col]
    dicom_paths = df[dicom_col]
    
    return features, time, event, dicom_paths

def preprocess_features(
    features: pl.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Applies one-hot encoding to categorical columns and standard scaling to numerical columns.
    Args:
        features: Input features DataFrame.
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
    Returns:
        Preprocessed features DataFrame.
    """
    df = features.clone()

    # If not specified, infer categorical and numerical columns
    if categorical_cols is None:
        categorical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    if numerical_cols is None:
        numerical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]

    # Keep logs minimal for large-scale runs

    # One-hot encode categorical columns
    if categorical_cols:
        df = df.to_dummies(columns=categorical_cols)

    # Standard scale numerical columns using scikit-learn
    if numerical_cols:
        scaler = StandardScaler()
        # Use to_numpy(copy=False) where possible
        scaled = scaler.fit_transform(df.select(numerical_cols).to_numpy())
        for i, col in enumerate(numerical_cols):
            df = df.with_columns(pl.Series(col, scaled[:, i]))

    return df

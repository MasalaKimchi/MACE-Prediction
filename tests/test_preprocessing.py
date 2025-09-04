#!/usr/bin/env python3
"""
Full dataset preprocessing test script.

This script analyzes ALL features in the dataset and provides a comprehensive report.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.preprocessing import (
    load_survival_dataset_from_csv, 
    preprocess_features, 
    validate_feature_scaling
)


def analyze_all_features(excel_path):
    """
    Analyze all features in the dataset.
    
    Args:
        excel_path: Path to the Excel file
    
    Returns:
        dict: Complete analysis report
    """
    print("=" * 100)
    print("FULL DATASET PREPROCESSING ANALYSIS")
    print("=" * 100)
    
    # Load raw data
    df_raw = pd.read_excel(excel_path)
    print(f"📁 Dataset: {excel_path}")
    print(f"📊 Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print()
    
    # Get all feature columns (exclude metadata columns)
    metadata_cols = ['case_num', 'NIFTI Path', 'time', 'event']
    all_feature_cols = [col for col in df_raw.columns if col not in metadata_cols]
    
    print(f"🔍 Analyzing {len(all_feature_cols)} features...")
    print()
    
    # Load with all features
    features, time, event, nifti_paths = load_survival_dataset_from_csv(
        excel_path,
        nifti_col='NIFTI Path',
        time_col='time',
        event_col='event',
        feature_cols=all_feature_cols
    )
    
    print(f"✅ Loaded {features.shape[1]} features for analysis")
    print()
    
    # Categorize features
    categorical_features = []
    numerical_features = []
    
    print("📋 FEATURE CATEGORIZATION:")
    print("-" * 50)
    
    for col in features.columns:
        dtype = str(features[col].dtype)
        n_unique = features[col].nunique() if hasattr(features[col], 'nunique') else features[col].n_unique()
        n_total = len(features[col])
        unique_ratio = n_unique / n_total
        
        # Determine if numeric
        is_numeric = dtype in [
            'int64', 'float64', 'int32', 'float32',
            'Int64', 'Float64', 'Int32', 'Float32'
        ]
        
        # Determine if categorical
        is_categorical = (
            not is_numeric or
            n_unique <= 5 or 
            (unique_ratio < 0.02 and n_unique <= 20)
        )
        
        if is_categorical:
            categorical_features.append(col)
            category = "🏷️  CATEGORICAL"
        else:
            numerical_features.append(col)
            category = "🔢 NUMERICAL"
        
        print(f"   {category}: {col}")
        print(f"      Type: {dtype}, Unique: {n_unique:,} ({unique_ratio:.3f})")
        
        if is_numeric:
            min_val = features[col].min()
            max_val = features[col].max()
            mean_val = features[col].mean()
            std_val = features[col].std()
            print(f"      Range: {min_val:.2f} - {max_val:.2f}")
            print(f"      Mean ± Std: {mean_val:.2f} ± {std_val:.2f}")
        print()
    
    print(f"📊 SUMMARY:")
    print(f"   Total features: {len(features.columns)}")
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    print()
    
    # Test preprocessing with all features
    print("🔧 PREPROCESSING TEST WITH ALL FEATURES:")
    print("-" * 50)
    
    try:
        processed_features, scaler = preprocess_features(features, return_scaler=True)
        print(f"✅ Preprocessing successful")
        print(f"📊 Processed shape: {processed_features.shape}")
        
        if scaler is not None:
            print(f"🔧 Scaler type: {type(scaler)}")
            print(f"📈 Scaled features: {len(scaler.feature_names_in_)}")
            print(f"   {list(scaler.feature_names_in_)}")
        
        # Validate scaling
        is_valid = validate_feature_scaling(features, processed_features, numerical_features)
        print(f"✅ Scaling validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Show scaling results for first few numerical features
        print(f"\n📊 SCALING RESULTS (first 5 numerical features):")
        for i, col in enumerate(numerical_features[:5]):
            if col in processed_features.columns:
                mean_val = processed_features[col].mean()
                std_val = processed_features[col].std()
                print(f"   {col}: mean={mean_val:.6f}, std={std_val:.6f}")
        
        print()
        
        return {
            'total_features': len(features.columns),
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'preprocessing_success': True,
            'scaling_validation': is_valid,
            'processed_shape': processed_features.shape
        }
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return {
            'total_features': len(features.columns),
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'preprocessing_success': False,
            'scaling_validation': False,
            'error': str(e)
        }


def main():
    """Main function."""
    excel_path = 'data/UH_1950_calc_09x03x2025.xlsx'
    
    if not os.path.exists(excel_path):
        print(f"❌ File not found: {excel_path}")
        return
    
    report = analyze_all_features(excel_path)
    
    print("=" * 100)
    print("FINAL REPORT")
    print("=" * 100)
    print(f"📊 Total features analyzed: {report['total_features']}")
    print(f"🔢 Numerical features: {len(report['numerical_features'])}")
    print(f"🏷️  Categorical features: {len(report['categorical_features'])}")
    print(f"✅ Preprocessing: {'SUCCESS' if report['preprocessing_success'] else 'FAILED'}")
    print(f"✅ Scaling validation: {'PASSED' if report['scaling_validation'] else 'FAILED'}")
    
    if report['preprocessing_success']:
        print(f"📊 Final processed shape: {report['processed_shape']}")
    
    print("\n📋 NUMERICAL FEATURES LIST:")
    for i, feature in enumerate(report['numerical_features'], 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n📋 CATEGORICAL FEATURES LIST:")
    for i, feature in enumerate(report['categorical_features'], 1):
        print(f"   {i:2d}. {feature}")
    
    print("=" * 100)


if __name__ == "__main__":
    main()

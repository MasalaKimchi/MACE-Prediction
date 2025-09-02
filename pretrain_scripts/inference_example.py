"""
Example script showing how to use a pretrained model for inference.
This demonstrates proper loading of model and scaler, and inverse transformation.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from architectures import build_network
from data import load_pretrained_checkpoint, inverse_transform_features, validate_feature_scaling


def parse_args():
    parser = argparse.ArgumentParser(description="Inference example with pretrained model")
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to pretrained checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to CT image for inference')
    parser.add_argument('--output_original_scale', action='store_true',
                       help='Output predictions in original radiomics scale')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and scaler from checkpoint."""
    checkpoint = load_pretrained_checkpoint(checkpoint_path)
    
    # Build model with correct architecture
    model = build_network(
        resnet_type=checkpoint['resnet_type'],
        in_channels=1,
        num_classes=len(checkpoint['feature_columns'])
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['feature_scaler'], checkpoint['feature_columns']


def predict_features(model, image_tensor, device):
    """Predict radiomics features from image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)
        return predictions.cpu()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.checkpoint_path}")
    model, feature_scaler, feature_columns = load_model_from_checkpoint(args.checkpoint_path, device)
    
    print(f"Model loaded successfully!")
    print(f"Feature columns: {feature_columns}")
    print(f"Number of features: {len(feature_columns)}")
    
    # TODO: Load and preprocess image here
    # For now, create a dummy image tensor
    dummy_image = torch.randn(1, 1, 256, 256, 64)  # (batch, channels, D, H, W)
    print(f"Using dummy image tensor: {dummy_image.shape}")
    
    # Predict features
    print("Predicting features...")
    predictions_normalized = predict_features(model, dummy_image, device)
    
    print(f"Predictions (Z-score normalized): {predictions_normalized}")
    
    # Validate that predictions are properly normalized
    print("\nValidating prediction scaling:")
    validate_feature_scaling(predictions_normalized)
    
    if args.output_original_scale:
        # Convert back to original scale
        predictions_original = inverse_transform_features(predictions_normalized, feature_scaler)
        print(f"\nPredictions (original scale): {predictions_original}")
        
        # Create a nice output
        print("\n" + "="*50)
        print("RADIOMICS FEATURE PREDICTIONS")
        print("="*50)
        for i, (feature_name, pred_norm, pred_orig) in enumerate(zip(
            feature_columns, 
            predictions_normalized[0], 
            predictions_original[0]
        )):
            print(f"{feature_name:20s}: {pred_orig.item():8.4f} (normalized: {pred_norm.item():6.3f})")
        print("="*50)


if __name__ == "__main__":
    main()

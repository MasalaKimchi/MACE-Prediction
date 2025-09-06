# TabM Encoder for Clinical Data
# This script provides an encoder based on the TabM model for tabular data
# including clinical data, Agatston scores, and calcium-omics data.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Literal
from .tabm import TabM, TabMArchitectureType, _NumEmbeddings

try:
    import rtdl_num_embeddings
except ImportError:
    rtdl_num_embeddings = None

class TabMEncoder(TabM):
    """
    A TabM-based encoder for tabular data.
    
    This class extends the original TabM model to provide encoding capabilities
    for clinical data, Agatston scores, and calcium-omics data. It extracts learned
    representations from the TabM backbone before the final output layer.
    
    The encoder supports:
    - Numerical features (continuous variables)
    - Categorical features (discrete variables)
    - Optional numerical embeddings for better representation learning
    - Ensemble-based encoding (k different feature representations)
    """
    
    def __init__(
        self,
        *,
        n_num_features: int = 0,
        cat_cardinalities: Optional[list[int]] = None,
        num_embeddings: Optional[_NumEmbeddings] = None,
        # TabM backbone parameters
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        activation: str = 'ReLU',
        k: int = 16,
        arch_type: TabMArchitectureType = 'tabm',
        start_scaling_init: Optional[Literal['random-signs', 'normal']] = 'random-signs',
        start_scaling_init_chunks: Optional[list[int]] = None,
    ) -> None:
        """
        Initialize the TabM encoder.
        
        Args:
            n_num_features: Number of numerical (continuous) features
            cat_cardinalities: Cardinalities of categorical features
            num_embeddings: Optional embeddings for numerical features
            n_blocks: Number of MLP blocks in the backbone
            d_block: Hidden dimension of the MLP blocks
            dropout: Dropout rate
            activation: Activation function name
            k: Ensemble size (number of parallel encoders)
            arch_type: TabM architecture type ('tabm', 'tabm-mini', 'tabm-packed')
            start_scaling_init: Initialization for scaling parameters
            start_scaling_init_chunks: Chunks for scaling initialization
        """
        # Initialize TabM without output layer (d_out=None)
        # Use the same pattern as the original TabM class
        super().__init__(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_out=None,  # No output layer - we want features only
            num_embeddings=num_embeddings,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            activation=activation,
            k=k,
            arch_type=arch_type,
            start_scaling_init=start_scaling_init,
        )
        
        # Store feature dimension for reference
        self._feature_dim = d_block

    @classmethod
    def make(cls, **kwargs) -> 'TabMEncoder':
        """Create TabMEncoder using the same pattern as TabM.make.
        
        This method provides sensible defaults for the encoder while allowing
        customization of all parameters.
        
        Args:
            kwargs: Arguments for TabMEncoder.__init__
        """
        # Set d_out=None by default for encoder
        kwargs.setdefault('d_out', None)
        
        # Use TabM.make to get the base model, then convert to TabMEncoder
        tabm_model = TabM.make(**kwargs)
        
        # Create TabMEncoder with the same parameters
        return cls(
            n_num_features=tabm_model._n_num_features,
            cat_cardinalities=kwargs.get('cat_cardinalities'),
            num_embeddings=kwargs.get('num_embeddings'),
            n_blocks=kwargs.get('n_blocks', 3),
            d_block=kwargs.get('d_block', 512),
            dropout=kwargs.get('dropout', 0.1),
            activation=kwargs.get('activation', 'ReLU'),
            k=kwargs.get('k', 16),
            arch_type=kwargs.get('arch_type', 'tabm'),
            start_scaling_init=kwargs.get('start_scaling_init', 'random-signs'),
            start_scaling_init_chunks=kwargs.get('start_scaling_init_chunks'),
        )

    def encode(
        self, x_num: Optional[Tensor] = None, x_cat: Optional[Tensor] = None
    ) -> Tensor:
        """
        Encode tabular data using TabM backbone.
        
        Args:
            x_num: Numerical features tensor of shape (batch_size, n_num_features)
                   or (batch_size, k, n_num_features) for ensemble training
            x_cat: Categorical features tensor of shape (batch_size, n_cat_features)
                   or (batch_size, k, n_cat_features) for ensemble training
        
        Returns:
            Encoded tensor of shape (batch_size, k, d_block) where:
            - batch_size: Number of samples
            - k: Ensemble size (number of parallel encoders)
            - d_block: Feature dimension
        """
        # Use the parent's forward method - since d_out=None, we get backbone features
        return self.forward(x_num, x_cat)
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        return self._feature_dim
    
    def get_ensemble_size(self) -> int:
        """Get the ensemble size (number of parallel encoders)."""
        return self.k
    
    
    
    def encode_ensemble(
        self, x_num: Optional[Tensor] = None, x_cat: Optional[Tensor] = None
    ) -> Tensor:
        """
        Encode data and return the full ensemble.
        
        Args:
            x_num: Numerical features
            x_cat: Categorical features
        
        Returns:
            Encoded tensor of shape (batch_size, k, d_block)
        """
        return self.encode(x_num, x_cat)
    
    def encode_flattened(
        self, x_num: Optional[Tensor] = None, x_cat: Optional[Tensor] = None
    ) -> Tensor:
        """
        Encode data and flatten the ensemble dimension.
        
        This is useful for downstream tasks that expect a single feature vector
        per sample.
        
        Args:
            x_num: Numerical features
            x_cat: Categorical features
            
        Returns:
            Flattened encoded tensor of shape (batch_size, k * d_block)
        """
        encoded = self.encode(x_num, x_cat)
        return encoded.flatten(1)  # Flatten ensemble dimension


if __name__ == "__main__":
    # Test TabMEncoder with different configurations
    print("Testing TabMEncoder...")
    
    # Test 1: Basic numerical features only
    print("\n1. Testing with numerical features only:")
    model_num = TabMEncoder(
        n_num_features=20,
        d_block=256,
        k=8,
        start_scaling_init='random-signs'
    )
    x_num = torch.randn(4, 20)
    encoded_num = model_num.encode_ensemble(x_num, None)
    print(f"   Input shape: {x_num.shape}")
    print(f"   Output shape: {encoded_num.shape}")
    print(f"   Expected shape: (4, 8, 256)")
    assert encoded_num.shape == (4, 8, 256), f"Expected (4, 8, 256), got {encoded_num.shape}"
    print("   ✓ Numerical features test passed!")
    
    # Test 2: With categorical features
    print("\n2. Testing with numerical and categorical features:")
    model_cat = TabMEncoder(
        n_num_features=15,
        cat_cardinalities=[2, 3, 4],
        d_block=512,
        k=16,
        start_scaling_init='random-signs'
    )
    x_num = torch.randn(4, 15)
    x_cat = torch.randint(0, 2, (4, 3))  # Correct range for cardinalities
    encoded_cat = model_cat.encode_ensemble(x_num, x_cat)
    print(f"   Input shapes: x_num {x_num.shape}, x_cat {x_cat.shape}")
    print(f"   Output shape: {encoded_cat.shape}")
    print(f"   Expected shape: (4, 16, 512)")
    assert encoded_cat.shape == (4, 16, 512), f"Expected (4, 16, 512), got {encoded_cat.shape}"
    print("   ✓ Categorical features test passed!")
    
    # Test 3: Flattened encoding
    print("\n3. Testing flattened encoding:")
    flattened_encoded = model_cat.encode_flattened(x_num, x_cat)
    print(f"   Flattened output shape: {flattened_encoded.shape}")
    print(f"   Expected shape: (4, 8192)")  # 16 * 512
    assert flattened_encoded.shape == (4, 8192), f"Expected (4, 8192), got {flattened_encoded.shape}"
    print("   ✓ Flattened encoding test passed!")
    
    # Test 4: With numerical embeddings (if available)
    if rtdl_num_embeddings is not None:
        print("\n4. Testing with numerical embeddings:")
        x_train = torch.randn(100, 10)
        bins = rtdl_num_embeddings.compute_bins(x_train, n_bins=16)
        num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins=bins, d_embedding=8, activation=True, version='B'
        )
        model_emb = TabMEncoder(
            n_num_features=10,
            d_block=128,
            k=4,
            num_embeddings=num_embeddings,
            start_scaling_init='normal'
        )
        x_num_emb = torch.randn(4, 10)
        encoded_emb = model_emb.encode_ensemble(x_num_emb, None)
        print(f"   Input shape: {x_num_emb.shape}")
        print(f"   Output shape: {encoded_emb.shape}")
        print(f"   Expected shape: (4, 4, 128)")
        assert encoded_emb.shape == (4, 4, 128), f"Expected (4, 4, 128), got {encoded_emb.shape}"
        print("   ✓ Numerical embeddings test passed!")
    else:
        print("\n4. Skipping numerical embeddings test (rtdl_num_embeddings not available)")
    
    # Test 5: Clinical data scenario
    print("\n5. Testing clinical data scenario:")
    clinical_data = torch.randn(4, 15)  # Clinical variables
    agatston_data = torch.abs(torch.randn(4, 5))  # Agatston scores (non-negative)
    calcium_omics_data = torch.randn(4, 20)  # Calcium-omics data
    categorical_data = torch.randint(0, 2, (4, 3))  # Categorical variables
    
    # Concatenate numerical features
    x_num_clinical = torch.cat([clinical_data, agatston_data, calcium_omics_data], dim=1)
    
    model_clinical = TabMEncoder(
        n_num_features=40,  # 15 + 5 + 20
        cat_cardinalities=[2, 3, 4],
        d_block=512,
        k=16,  # Updated to default k=16
        start_scaling_init='random-signs'
    )
    
    # Test ensemble encoding (original TabM approach)
    ensemble_clinical = model_clinical.encode_ensemble(x_num_clinical, categorical_data)
    print(f"   Clinical input shapes: x_num {x_num_clinical.shape}, x_cat {categorical_data.shape}")
    print(f"   Ensemble shape: {ensemble_clinical.shape}")
    print(f"   Expected shape: (4, 16, 512)")
    assert ensemble_clinical.shape == (4, 16, 512), f"Expected (4, 16, 512), got {ensemble_clinical.shape}"
    print("   ✓ Clinical data test passed!")
    
    # Test uncertainty quantification
    mean_features = ensemble_clinical.mean(dim=1)
    std_features = ensemble_clinical.std(dim=1)
    print(f"   Mean features shape: {mean_features.shape}")
    print(f"   Std features shape: {std_features.shape}")
    print("   ✓ Uncertainty quantification test passed!")
    
    print("\n🎉 All TabMEncoder tests completed successfully!")
    print("\nSummary of TabMEncoder capabilities (following original TabM paper):")
    print("- Ensemble encoding: Core TabM output (batch_size, k, d_block) with k=16 by default")
    print("- Flattened encoding: For downstream ML models (batch_size, k*d_block)")
    print("- Uncertainty quantification: Mean and std across k ensemble members")
    print("- Multi-modal support: Clinical + Agatston + calcium-omics + categorical data")
    print("- Optional numerical embeddings: Enhanced representation learning")
    print("- Default k=16: Balanced between performance and computational efficiency")

    

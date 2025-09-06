from __future__ import annotations

import torch
import torch.nn as nn

try:
    from monai.networks.nets.swin_unetr import SwinTransformer
except ImportError:  # pragma: no cover - handled in runtime
    SwinTransformer = None


class Swin3DEncoder(nn.Module):
    """Swin-Transformer based encoder for 3D volumes.

    This directly uses :class:`monai.networks.nets.swin_unetr.SwinTransformer` 
    for efficient encoding without the heavy decoder components. Supports both
    SwinUNETR and SwinUNETRv2 architectures via the use_v2 parameter. The input 
    volume is converted into a global embedding vector by using only the SwinTransformer 
    backbone and applying global average pooling. Uses original MONAI SwinUNETR 
    feature dimensions without projection.
    
    Parameters
    ----------
    in_ch : int, default=1
        Number of input channels.
    feature_size : int, default=48
        Base feature size (increased from MONAI default for better performance). 
        Final output dimension will be feature_size * 2^num_stages.
    window_size : int, default=7
        Window size for Swin Transformer attention.
    depths : tuple, default=(2, 2, 2, 2)
        Number of Swin Transformer blocks in each stage.
    num_heads : tuple, default=(3, 6, 12, 24)
        Number of attention heads in each stage.
    dropout_path_rate : float, default=0.1
        Dropout rate for stochastic depth.
    use_v2 : bool, default=True
        If True, uses SwinUNETRv2 architecture with additional residual convolution
        blocks at the beginning of each Swin stage for improved performance.
    use_checkpoint : bool, default=True
        If True, uses gradient checkpointing to reduce memory usage during training
        at the cost of increased computation time. Useful for large 3D volumes or
        when training with limited GPU memory.
        
    Note
    ----
    Output feature dimensions:
    - Default config (feature_size=48, depths=(2,2,2,2)): 768 features
    - Custom configs: feature_size * 2^num_stages
    """

    def __init__(
        self,
        in_ch: int = 1,
        feature_size: int = 48,  # Increased from MONAI default for better performance
        window_size: int = 7,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        dropout_path_rate: float = 0.1,
        use_v2: bool = True,  # Enable SwinUNETRv2 by default
        use_checkpoint: bool = True,  # Enable gradient checkpointing by default
    ) -> None:
        super().__init__()
        if SwinTransformer is None:  # pragma: no cover
            raise ImportError("MONAI is required for Swin3DEncoder. Install with 'pip install monai'.")

        # Use SwinTransformer directly instead of the full SwinUNETR
        self.backbone = SwinTransformer(
            in_chans=in_ch,
            embed_dim=feature_size,
            window_size=(window_size, window_size, window_size),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            use_v2=use_v2,
        )
        # The SwinTransformer will output features with dimension feature_size * 2^num_stages
        # For default config: 24 * 2^4 = 384 features
        self.embed_dim = feature_size * (2 ** len(depths))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input volume to a global embedding.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, H, W, D)``.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        # Get the output from the SwinTransformer (returns list of features from each stage)
        features = self.backbone(x, normalize=True)
        
        # Use the deepest feature map (last stage output)
        # features[-1] has shape (B, final_feature_size, H', W', D')
        deep_features = features[-1]
        
        # Apply global average pooling across spatial dimensions
        gap = deep_features.mean(dim=(2, 3, 4))  # (B, final_feature_size)
        
        # Return original SwinTransformer feature dimensions (no projection)
        return gap

if __name__ == "__main__":
    # Define an input tensor of shape (Batch, Channels, Depth, Height, Width)
    # The new dimensions (256, 256, 64) are chosen for a medical imaging example.
    input_volume = torch.randn(1, 1, 256, 256, 64)
    print(f"Input volume shape: {input_volume.shape}")

    # Test default configuration - should produce 768 features (48 * 2^4)
    print("Testing default configuration (use_v2=True, use_checkpoint=True, feature_size=48):")
    model_default = Swin3DEncoder()
    print(f"Model initialized with embed_dim: {model_default.embed_dim}")

    # Pass the input volume through the encoder
    output_embedding_default = model_default(input_volume)
    print(f"Output embedding shape: {output_embedding_default.shape}")
    
    # Assert that the output shape is as expected (768 features for default config)
    expected_dim = 48 * (2 ** 4)  # feature_size * 2^num_stages
    assert output_embedding_default.shape == (1, expected_dim), f"Output shape does not match expected (1, {expected_dim})"
    print(f"Success: Default encoder produced an output with the correct shape (1, {expected_dim}).")
    
    print("\nAll Swin3DEncoder tests completed successfully!")
    print("\nSummary of SwinTransformer feature dimensions:")
    print("- Default config (feature_size=48, use_v2=True, use_checkpoint=True): 768 features")
    print("- Custom configs: feature_size * 2^num_stages")

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
    backbone and applying global average pooling followed by projection.
    
    Parameters
    ----------
    use_v2 : bool, default=False
        If True, uses SwinUNETRv2 architecture with additional residual convolution
        blocks at the beginning of each Swin stage for improved performance.
    use_checkpoint : bool, default=False
        If True, uses gradient checkpointing to reduce memory usage during training
        at the cost of increased computation time. Useful for large 3D volumes or
        when training with limited GPU memory.
    """

    def __init__(
        self,
        in_ch: int = 1,
        embed_dim: int = 256,
        feature_size: int = 48,
        window_size: int = 7,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        dropout_path_rate: float = 0.1,
        use_v2: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if SwinTransformer is None:  # pragma: no cover
            raise ImportError("MONAI is required for Swin3DEncoder. Install with 'pip install monai'.")

        self.embed_dim = embed_dim
        
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
        # The SwinTransformer will output features with dimension embed_dim * 2^(num_stages-1)
        # We'll dynamically determine this in the forward pass
        self.proj = None  # Will be initialized in first forward pass

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
        
        # Initialize projection layer if not done yet
        if self.proj is None:
            final_feature_size = gap.shape[1]
            self.proj = nn.Linear(final_feature_size, self.embed_dim).to(gap.device)
        
        # Project to desired embedding dimension
        return self.proj(gap)

if __name__ == "__main__":
    # The following test code requires the MONAI library to be installed.
    # To install it, run: 'pip install monai'

    # Define an input tensor of shape (Batch, Channels, Depth, Height, Width)
    # The new dimensions (256, 256, 64) are chosen for a medical imaging example.
    input_volume = torch.randn(1, 1, 256, 256, 64)
    print(f"Input volume shape: {input_volume.shape}")

    # Test SwinUNETR (default)
    print("Testing SwinUNETR (use_v2=False):")
    model = Swin3DEncoder(embed_dim=256, use_v2=False)
    print("Model initialized.")

    # Pass the input volume through the encoder
    output_embedding = model(input_volume)
    print(f"Output embedding shape: {output_embedding.shape}")
    
    # Assert that the output shape is as expected
    assert output_embedding.shape == (1, 256), "Output shape does not match expected (1, 256)"
    print("Success: SwinUNETR encoder produced an output with the correct shape.")
    
    # Test SwinUNETRv2
    print("\nTesting SwinUNETRv2 (use_v2=True):")
    model_v2 = Swin3DEncoder(embed_dim=256, use_v2=True)
    print("Model initialized.")

    # Pass the input volume through the encoder
    output_embedding_v2 = model_v2(input_volume)
    print(f"Output embedding shape: {output_embedding_v2.shape}")
    
    # Assert that the output shape is as expected
    assert output_embedding_v2.shape == (1, 256), "Output shape does not match expected (1, 256)"
    print("Success: SwinUNETRv2 encoder produced an output with the correct shape.")
    
    # Test with gradient checkpointing enabled
    print("\nTesting with gradient checkpointing (use_checkpoint=True):")
    model_checkpoint = Swin3DEncoder(embed_dim=256, use_checkpoint=True)
    print("Model initialized.")

    # Pass the input volume through the encoder
    output_embedding_checkpoint = model_checkpoint(input_volume)
    print(f"Output embedding shape: {output_embedding_checkpoint.shape}")
    
    # Assert that the output shape is as expected
    assert output_embedding_checkpoint.shape == (1, 256), "Output shape does not match expected (1, 256)"
    print("Success: Encoder with checkpointing produced an output with the correct shape.")

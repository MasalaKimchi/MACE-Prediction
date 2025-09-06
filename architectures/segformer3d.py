import torch
import torch.nn as nn
import math
from functools import partial

###################################################################################
# Helper Functions and Modules
###################################################################################
def get_3d_dims(n: int) -> tuple[int, int, int]:
    """
    Computes the 3D dimensions from the total number of patches.
    This is used to convert the 1D patch sequence length back to 3D spatial dimensions.
    
    For non-perfect cubes, we try to find the closest factorization.

    Parameters
    ----------
    n: int
        The number of patches in the sequence.

    Returns
    -------
    tuple[int, int, int]
        The 3D dimensions (d, h, w) that best approximate the cube root.
    """
    # Try to find the closest cube root
    cube_root = round(math.pow(n, (1 / 3)))
    
    # If it's a perfect cube, return it
    if cube_root ** 3 == n:
        return (cube_root, cube_root, cube_root)
    
    # Try to find factors that multiply to n, preferring balanced factorizations
    best_factorization = None
    best_score = float('inf')
    
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            remaining = n // i
            for j in range(1, int(math.sqrt(remaining)) + 1):
                if remaining % j == 0:
                    k = remaining // j
                    if i * j * k == n:
                        # Score based on how close to a cube the factorization is
                        factors = sorted([i, j, k])
                        score = max(factors) - min(factors)  # Lower is better
                        if score < best_score:
                            best_score = score
                            best_factorization = tuple(factors)
    
    if best_factorization is not None:
        return best_factorization
    
    # If no exact factorization found, use the cube root approximation
    # and adjust the last dimension to make the product correct
    d = h = w = cube_root
    while d * h * w < n:
        w += 1
    while d * h * w > n and w > 1:
        w -= 1
    
    return (d, h, w)

class DWConv(nn.Module):
    """
    Depth-wise 3D Convolutional layer used within the MLP block.
    This module applies a depth-wise convolution and a batch normalization layer.

    Parameters
    ----------
    dim: int
        The number of input and output channels for the convolution.
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        # A 3x3x3 depth-wise convolution
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DWConv layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, N, C)``, where B is batch size,
            N is sequence length (number of patches), and C is channels.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``.
        """
        B, N, C = x.shape
        # Get the 3D dimensions from the number of patches
        d, h, w = get_3d_dims(N)
        
        # Reshape from (B, N, C) to (B, C, D, H, W) for 3D convolution
        x = x.transpose(1, 2).view(B, C, d, h, w)
        
        # Apply the depth-wise convolution
        x = self.dwconv(x)
        x = self.bn(x)
        
        # Flatten back to (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        return x

class _MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block with a Mix-FFN structure.
    This block is composed of two linear layers with a depth-wise convolution
    in between, which helps capture local spatial information.

    Parameters
    ----------
    in_feature: int
        Input feature dimension.
    mlp_ratio: int
        Ratio to increase the hidden dimension.
    dropout: float
        Dropout rate for the linear layers.
    """
    def __init__(self, in_feature: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP block.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``.
        """
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class SelfAttention(nn.Module):
    """
    Self-Attention block with a spatial reduction mechanism.
    This attention mechanism reduces the sequence length of the key and value
    tensors to make the computation more efficient, which is crucial for
    high-resolution 3D data.

    Parameters
    ----------
    embed_dim: int
        Input and output embedding dimension.
    num_heads: int
        Number of attention heads.
    sr_ratio: int
        The spatial reduction ratio for key and value tensors. A ratio > 1
        means the sequence length is reduced.
    qkv_bias: bool
        Whether to include a bias in the QKV linear layers.
    attn_dropout: float
        Dropout rate for the attention weights.
    proj_dropout: float
        Dropout rate for the final projection layer.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # Spatial reduction convolution
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Self-Attention block.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``.
        """
        B, N, C = x.shape

        # Compute query (Q) tensor
        q = self.query(x).reshape(B, N, self.num_heads, self.attention_head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            d, h, w = get_3d_dims(N)
            # Reshape input for 3D convolution
            x_ = x.permute(0, 2, 1).reshape(B, C, d, h, w)
            
            # Apply spatial reduction convolution
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            
            # Compute key (K) and value (V) tensors from the reduced sequence
            kv = self.key_value(x_).reshape(B, -1, 2, self.num_heads, self.attention_head_dim).permute(2, 0, 3, 1, 4)
        else:
            # Compute key (K) and value (V) tensors from the full sequence
            kv = self.key_value(x).reshape(B, -1, 2, self.num_heads, self.attention_head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.attention_head_dim)
        attnention_prob = attention_score.softmax(dim=-1)
        attnention_prob = self.attn_dropout(attnention_prob)
        out = (attnention_prob @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    """
    The core transformer block for the MixVisionTransformer.
    It consists of a self-attention layer followed by a Mix-FFN (MLP) block.

    Parameters
    ----------
    embed_dim: int
        The embedding dimension.
    mlp_ratio: int
        The ratio for the hidden dimension of the MLP.
    num_heads: int
        Number of attention heads.
    sr_ratio: int
        Spatial reduction ratio for attention.
    qkv_bias: bool
        Whether to include bias in QKV projections.
    attn_dropout: float
        Dropout rate for attention.
    proj_dropout: float
        Dropout rate for final projection.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``.
        """
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """
    Overlapping Patch Embedding with 3D Convolution.
    This module converts a 3D volume into a sequence of embedded patches.

    Parameters
    ----------
    in_channel: int
        Number of input channels.
    embed_dim: int
        The embedding dimension of each patch.
    kernel_size: int
        Kernel size for the convolution.
    stride: int
        Stride for the convolution.
    padding: int
        Padding for the convolution.
    """
    def __init__(
        self,
        in_channel: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PatchEmbedding.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``, where N is the
            number of patches and C is the embedding dimension.
        """
        # Apply convolution to get patch embeddings
        patches = self.patch_embeddings(x)
        
        # Flatten the spatial dimensions and permute to get sequence of patches
        # (B, C, D', H', W') -> (B, C, N) -> (B, N, C)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches


class MixVisionTransformer(nn.Module):
    """
    MixVisionTransformer (MiT) backbone for 3D data.
    This is the core encoder of SegFormer3D, generating multi-scale feature maps.

    Parameters
    ----------
    in_channels: int
        Number of input channels for the initial volume.
    sr_ratios: list
        List of spatial reduction ratios for each transformer stage.
    embed_dims: list
        List of embedding dimensions for each stage.
    patch_kernel_size: list
        List of kernel sizes for patch embedding at each stage.
    patch_stride: list
        List of strides for patch embedding at each stage.
    patch_padding: list
        List of padding for patch embedding at each stage.
    mlp_ratios: list
        List of MLP ratios for each stage.
    num_heads: list
        List of attention heads for each stage.
    depths: list
        List of the number of transformer blocks in each stage.
    """
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [8, 4, 2, 1],
        embed_dims: list = [64, 128, 320, 512],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
    ):
        super().__init__()

        # Define patch embeddings for each pyramid level
        self.embed_1 = PatchEmbedding(in_channel=in_channels, embed_dim=embed_dims[0], kernel_size=patch_kernel_size[0], stride=patch_stride[0], padding=patch_padding[0])
        self.embed_2 = PatchEmbedding(in_channel=embed_dims[0], embed_dim=embed_dims[1], kernel_size=patch_kernel_size[1], stride=patch_stride[1], padding=patch_padding[1])
        self.embed_3 = PatchEmbedding(in_channel=embed_dims[1], embed_dim=embed_dims[2], kernel_size=patch_kernel_size[2], stride=patch_stride[2], padding=patch_padding[2])
        self.embed_4 = PatchEmbedding(in_channel=embed_dims[2], embed_dim=embed_dims[3], kernel_size=patch_kernel_size[3], stride=patch_stride[3], padding=patch_padding[3])

        # Define the transformer blocks for each stage
        self.tf_block1 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0], qkv_bias=True) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.tf_block2 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1], qkv_bias=True) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.tf_block3 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2], qkv_bias=True) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.tf_block4 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3], qkv_bias=True) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass for the MixVisionTransformer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        list[torch.Tensor]
            A list of 4 tensors, each representing the multi-scale feature
            map from a different stage of the encoder. The shapes are
            ``(B, C_i, D_i, H_i, W_i)``, where ``C_i`` and spatial dimensions
            change at each stage.
        """
        out = []
        
        # Stage 1
        x = self.embed_1(x)
        B, N, C = x.shape
        d, h, w = get_3d_dims(N)
        for i, blk in enumerate(self.tf_block1): x = blk(x)
        x = self.norm1(x)
        x = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        
        # Stage 2
        x = self.embed_2(x)
        B, N, C = x.shape
        d, h, w = get_3d_dims(N)
        for i, blk in enumerate(self.tf_block2): x = blk(x)
        x = self.norm2(x)
        x = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        
        # Stage 3
        x = self.embed_3(x)
        B, N, C = x.shape
        d, h, w = get_3d_dims(N)
        for i, blk in enumerate(self.tf_block3): x = blk(x)
        x = self.norm3(x)
        x = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        
        # Stage 4
        x = self.embed_4(x)
        B, N, C = x.shape
        d, h, w = get_3d_dims(N)
        for i, blk in enumerate(self.tf_block4): x = blk(x)
        x = self.norm4(x)
        x = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        
        return out

###################################################################################
# The Final SegFormer3D Encoder Class
###################################################################################
class SegFormer3DEncoder(nn.Module):
    """
    SegFormer3D-based encoder for 3D volumes.

    This class provides a complete encoder for feature extraction.
    It utilizes the MixVisionTransformer (MiT) to process a 3D input volume
    and extracts a single, global feature vector from the deepest encoder layer.

    Parameters
    ----------
    in_channels: int
        Number of input channels for the initial volume.
    embed_dim: int
        The dimension of the final output embedding. This is the size of the
        final feature vector.
    sr_ratios: list
        List of spatial reduction ratios for each transformer stage.
    embed_dims: list
        List of embedding dimensions for each stage.
    patch_kernel_size: list
        List of kernel sizes for patch embedding at each stage.
    patch_stride: list
        List of strides for patch embedding at each stage.
    patch_padding: list
        List of padding for patch embedding at each stage.
    mlp_ratios: list
        List of MLP ratios for each stage.
    num_heads: list
        List of attention heads for each stage.
    depths: list
        List of the number of transformer blocks in each stage.
    """
    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [4, 4, 4, 4],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
    ) -> None:
        super().__init__()

        # Use the MixVisionTransformer as the backbone for feature extraction
        self.backbone = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )

        # The output dimension of the last encoder stage is given by embed_dims[-1]
        final_encoder_dim = embed_dims[-1]

        # A linear projection head to convert the pooled features to the desired embedding size
        self.proj = nn.Linear(final_encoder_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input volume into a single, global embedding vector.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, D, H, W)``.

        Returns
        -------
        torch.Tensor
            A global embedding vector of shape ``(B, embed_dim)``.
        """
        # Run the input through the MixVisionTransformer encoder to get multi-scale features
        multiscale_features = self.backbone(x)
        
        # Select the deepest feature map from the encoder output. This is a 5D tensor.
        deepest_feature_map = multiscale_features[-1]
        
        # Perform Global Average Pooling across the spatial dimensions (D, H, W)
        # This reduces the 5D tensor to a 2D tensor of shape (B, C)
        gap = deepest_feature_map.mean(dim=(2, 3, 4))
        
        # Project the pooled features to the final embedding size
        return self.proj(gap)


if __name__ == "__main__":
    # Example usage of the SegFormer3DEncoder
    
    # Test with single channel input (typical for medical imaging)
    print("Testing SegFormer3D with single channel input:")
    input_volume_1ch = torch.randn(1, 1, 256, 256, 64)
    print(f"Input volume shape: {input_volume_1ch.shape}")

    # Initialize the SegFormer3DEncoder with single channel input
    model_1ch = SegFormer3DEncoder(in_channels=1, embed_dim=256)
    print("Model initialized.")

    # Pass the input volume through the encoder
    output_embedding_1ch = model_1ch(input_volume_1ch)
    print(f"Output embedding shape: {output_embedding_1ch.shape}")
    
    # Assert that the output shape is as expected
    assert output_embedding_1ch.shape == (1, 256), "Output shape does not match expected (1, 256)"
    print("Success: Single channel encoder produced an output with the correct shape.")
    
    # Test with multi-channel input (e.g., multi-modal medical imaging)
    print("\nTesting SegFormer3D with multi-channel input:")
    input_volume_4ch = torch.randn(1, 4, 128, 128, 64)
    print(f"Input volume shape: {input_volume_4ch.shape}")

    # Initialize the SegFormer3DEncoder with multi-channel input
    model_4ch = SegFormer3DEncoder(in_channels=4, embed_dim=256)
    print("Model initialized.")

    # Pass the input volume through the encoder
    output_embedding_4ch = model_4ch(input_volume_4ch)
    print(f"Output embedding shape: {output_embedding_4ch.shape}")
    
    # Assert that the output shape is as expected
    assert output_embedding_4ch.shape == (1, 256), "Output shape does not match expected (1, 256)"
    print("Success: Multi-channel encoder produced an output with the correct shape.")
    
    # Test with different embedding dimensions
    print("\nTesting SegFormer3D with different embedding dimensions:")
    model_512 = SegFormer3DEncoder(in_channels=1, embed_dim=512)
    output_embedding_512 = model_512(input_volume_1ch)
    print(f"Output embedding shape (512-dim): {output_embedding_512.shape}")
    assert output_embedding_512.shape == (1, 512), "Output shape does not match expected (1, 512)"
    print("Success: 512-dimensional encoder produced an output with the correct shape.")

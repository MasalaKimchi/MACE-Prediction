"""Multimodal survival network with cross-attention + FiLM fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal

from .fusion import CrossAttentionFiLMFusion
from .resnet3d import build_network


class MultimodalSurvivalNet(nn.Module):
    """Single-stage multimodal survival model with pluggable encoders.
    
    This model supports various image encoders (ResNet3D, Swin3D, SegFormer3D) and
    tabular encoders (TabM, MLP, FTTransformer) with robust modality-dropout handling.
    
    Parameters
    ----------
    image_encoder : nn.Module
        Image encoder that outputs features of shape (B, embed_dim) or (B, T, embed_dim).
    tabular_encoder : nn.Module, optional
        Tabular encoder that outputs features of shape (B, embed_dim) or (B, T, embed_dim).
    embed_dim : int
        Embedding dimension for both modalities.
    fusion_config : dict, optional
        Configuration for the fusion module. See CrossAttentionFiLMFusion for details.
    risk_head_config : dict, optional
        Configuration for the risk prediction head.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        tabular_encoder: Optional[nn.Module],
        embed_dim: int,
        fusion_config: Optional[Dict[str, Any]] = None,
        risk_head_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.embed_dim = embed_dim
        
        # Default fusion configuration - both modalities use same dimension
        default_fusion_config = {
            'img_dim': embed_dim,
            'tab_dim': embed_dim,
            'hidden_dim': embed_dim,
            'n_heads': 4,
            'use_cross_attn': True,
            'use_film': True,
            'token_wise_film': False,
            'dropout': 0.1,
            'pool': 'mean'
        }
        if fusion_config:
            default_fusion_config.update(fusion_config)
        
        self.fusion = CrossAttentionFiLMFusion(**default_fusion_config)
        
        # Default risk head configuration
        default_risk_config = {
            'in_features': embed_dim,
            'out_features': 1,
            'bias': True
        }
        if risk_head_config:
            default_risk_config.update(risk_head_config)
        
        self.risk_head = nn.Linear(**default_risk_config)

    def forward(
        self,
        image: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        tab_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional auxiliary outputs.
        
        Parameters
        ----------
        image : torch.Tensor
            Image tensor of shape (B, C, D, H, W) or (B, C, H, W).
        tabular : torch.Tensor, optional
            Tabular tensor. Can be:
            - (B, D) for simple tabular data
            - (B, D) for TabM single output
            - (B, K, D) for TabM ensemble output
            - (x_num, x_cat) tuple for FTTransformer
        tab_mask : torch.Tensor, optional
            Boolean mask of shape (B,) indicating presence of tabular features.
        return_aux : bool, default=False
            Whether to return auxiliary outputs for debugging/analysis.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'log_risk': Risk predictions (B,)
            - 'image_embed': Image embeddings (B, embed_dim) or (B, T, embed_dim)
            - 'tab_embed': Tabular embeddings (B, embed_dim) or (B, T, embed_dim) or None
            - 'fused': Fused embeddings (B, embed_dim)
            - 'aux': Auxiliary outputs (only if return_aux=True)
        """
        # Encode image
        img_feat = self.image_encoder(image)
        
        # Encode tabular data
        tab_feat = None
        if tabular is not None and self.tabular_encoder is not None:
            tab_feat = self._encode_tabular(tabular)
        
        # Fuse modalities
        if return_aux:
            fused, aux = self.fusion.forward_with_aux(img_feat, tab_feat, tab_mask)
        else:
            fused = self.fusion(img_feat, tab_feat, tab_mask)
            aux = None
        
        # Predict risk
        log_risk = self.risk_head(fused).squeeze(-1)
        
        result = {
            "log_risk": log_risk,
            "image_embed": img_feat,
            "tab_embed": tab_feat,
            "fused": fused,
        }
        
        if return_aux and aux is not None:
            result["aux"] = aux
            
        return result

    def _encode_tabular(self, tabular: torch.Tensor) -> torch.Tensor:
        """Encode tabular data with proper handling for different encoder types."""
        try:
            # Try direct encoding first (for MLP, TabM single output)
            if isinstance(tabular, torch.Tensor):
                return self.tabular_encoder(tabular)
            else:
                # Handle tuple inputs (e.g., FTTransformer expects x_num, x_cat)
                return self.tabular_encoder(*tabular)
        except TypeError as e:
            # Fallback for encoders expecting specific input format
            if "missing" in str(e).lower() or "positional" in str(e).lower():
                # Try with None for categorical features
                if isinstance(tabular, torch.Tensor):
                    return self.tabular_encoder(tabular, None)
                else:
                    return self.tabular_encoder(tabular[0], None)
            else:
                raise e

    def get_attention_weights(
        self,
        image: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        tab_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Get attention weights for interpretability.
        
        Parameters
        ----------
        image : torch.Tensor
            Image tensor.
        tabular : torch.Tensor, optional
            Tabular tensor.
        tab_mask : torch.Tensor, optional
            Tabular mask.
            
        Returns
        -------
        torch.Tensor or None
            Attention weights if available, None otherwise.
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(image, tabular, tab_mask, return_aux=True)
            return result.get("aux", {}).get("attn_weights")


def build_multimodal_network(
    img_encoder: str = "resnet3d",
    tab_encoder: str = "mlp",
    resnet_type: str = "resnet18",
    in_channels: int = 1,
    tabular_dim: int = 16,
    embed_dim: int = None,  # None = preserve encoder-specific dimensions
    img_kwargs: dict | None = None,
    tab_kwargs: dict | None = None,
    fusion_config: dict | None = None,
    risk_head_config: dict | None = None,
) -> MultimodalSurvivalNet:
    """Factory for :class:`MultimodalSurvivalNet` with configurable encoders.
    
    Parameters
    ----------
    img_encoder : str, default="resnet3d"
        Image encoder type. Options: "resnet3d", "swin3d", "segformer3d".
    tab_encoder : str, default="mlp"
        Tabular encoder type. Options: "mlp", "ft_transformer", "tabm".
    resnet_type : str, default="resnet18"
        ResNet variant for ResNet3D encoder.
    in_channels : int, default=1
        Number of input channels for image encoder.
    tabular_dim : int, default=16
        Dimension of tabular input features.
    embed_dim : int, default=128
        Embedding dimension for both modalities.
    img_kwargs : dict, optional
        Additional arguments for image encoder.
    tab_kwargs : dict, optional
        Additional arguments for tabular encoder.
    fusion_config : dict, optional
        Configuration for fusion module.
    risk_head_config : dict, optional
        Configuration for risk prediction head.
        
    Returns
    -------
    MultimodalSurvivalNet
        Configured multimodal survival network.
    """

    img_kwargs = img_kwargs or {}
    tab_kwargs = tab_kwargs or {}

    # Build image encoder with preserved dimensions
    if img_encoder == "resnet3d":
        # ResNet3D: 512 (ResNet18/34) or 2048 (ResNet50+)
        image_model = build_network(
            resnet_type=resnet_type, in_channels=in_channels, num_classes=1
        )
        # Remove the final classification layer to get features
        if hasattr(image_model, 'fc'):
            img_dim = image_model.fc.in_features
            image_model.fc = nn.Identity()
        else:
            raise ValueError("Could not find final classification layer in ResNet")
            
    elif img_encoder == "swin3d":
        from .swin3d import Swin3DEncoder
        # Swin3D: 768 features (default: 48 * 2^4)
        image_model = Swin3DEncoder(
            in_ch=in_channels, **img_kwargs
        )
        img_dim = image_model.embed_dim
        
    elif img_encoder == "segformer3d":
        from .segformer3d import SegFormer3DEncoder
        # SegFormer3D: 256 features (default embed_dims[-1])
        image_model = SegFormer3DEncoder(
            in_channels=in_channels, **img_kwargs
        )
        # Get the final embedding dimension from the projection layer
        img_dim = image_model.proj.out_features
        
    else:  # pragma: no cover
        raise ValueError(f"Unsupported image encoder: {img_encoder}")
    
    # Use encoder-specific dimension if embed_dim not specified
    if embed_dim is None:
        embed_dim = img_dim

    # Build tabular encoder to match image dimension
    if tab_encoder == "mlp":
        tab_model = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
   
    elif tab_encoder == "tabm":
        from .tabm_encoder import TabMEncoder
        # TabM encoder configuration
        n_num_features = tab_kwargs.get("n_num_features", tabular_dim)
        cat_cardinalities = tab_kwargs.get("cat_cardinalities")
        num_embeddings = tab_kwargs.get("num_embeddings")
        
        # Create TabM encoder with ensemble output matching image dimension
        tab_model = TabMEncoder.make(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            num_embeddings=num_embeddings,
            d_block=embed_dim,  # Match image encoder dimension
            k=tab_kwargs.get("k", 16),
            n_blocks=tab_kwargs.get("n_blocks", 3),
            dropout=tab_kwargs.get("dropout", 0.1),
            arch_type=tab_kwargs.get("arch_type", "tabm"),
            start_scaling_init=tab_kwargs.get("start_scaling_init", "random-signs"),
        )
        
        # Add projection layer to convert TabM ensemble output to single vector
        # TabM outputs (B, k, d_block), we need (B, embed_dim)
        if tab_kwargs.get("use_ensemble_mean", True):
            # Use mean across ensemble
            class EnsembleMean(nn.Module):
                def forward(self, x):
                    return x.mean(dim=1) if x.dim() == 3 else x
            
            tab_model = nn.Sequential(tab_model, EnsembleMean())
        else:
            # Use flattened ensemble
            class EnsembleFlatten(nn.Module):
                def forward(self, x):
                    return x.flatten(1) if x.dim() == 3 else x
            
            tab_model = nn.Sequential(tab_model, EnsembleFlatten())
    else:  # pragma: no cover
        raise ValueError(f"Unsupported tabular encoder: {tab_encoder}")

    return MultimodalSurvivalNet(
        image_model, 
        tab_model, 
        embed_dim,
        fusion_config=fusion_config,
        risk_head_config=risk_head_config
    )


def test_multimodal_integration():
    """Test the multimodal survival network with different encoder combinations."""
    print("Testing MultimodalSurvivalNet integration...")
    
    # Test data
    batch_size = 4
    image = torch.randn(batch_size, 1, 64, 64, 32)  # 3D medical image
    tabular = torch.randn(batch_size, 20)  # Tabular features
    tab_mask = torch.ones(batch_size, dtype=torch.bool)  # All samples have tabular data
    
    # Test 1: ResNet3D + MLP
    print("\n1. Testing ResNet3D + MLP:")
    model1 = build_multimodal_network(
        img_encoder="resnet3d",
        tab_encoder="mlp",
        tabular_dim=20,
        embed_dim=128
    )
    result1 = model1(image, tabular, tab_mask)
    print(f"   Output shapes: {[(k, v.shape) for k, v in result1.items() if isinstance(v, torch.Tensor)]}")
    
    # Test 2: Swin3D + TabM
    print("\n2. Testing Swin3D + TabM:")
    model2 = build_multimodal_network(
        img_encoder="swin3d",
        tab_encoder="tabm",
        tabular_dim=20,
        embed_dim=256,
        tab_kwargs={
            "n_num_features": 20,
            "k": 8,
            "use_ensemble_mean": True
        }
    )
    result2 = model2(image, tabular, tab_mask)
    print(f"   Output shapes: {[(k, v.shape) for k, v in result2.items() if isinstance(v, torch.Tensor)]}")
    
    # Test 3: SegFormer3D + TabM with modality dropout
    print("\n3. Testing SegFormer3D + TabM with modality dropout:")
    model3 = build_multimodal_network(
        img_encoder="segformer3d",
        tab_encoder="tabm",
        tabular_dim=20,
        embed_dim=256,
        tab_kwargs={
            "n_num_features": 20,
            "k": 16,
            "use_ensemble_mean": True
        }
    )
    # Test with some missing tabular data
    tab_mask_partial = torch.tensor([True, True, False, False], dtype=torch.bool)
    result3 = model3(image, tabular, tab_mask_partial)
    print(f"   Output shapes: {[(k, v.shape) for k, v in result3.items() if isinstance(v, torch.Tensor)]}")
    
    # Test 4: Attention weights
    print("\n4. Testing attention weights:")
    attn_weights = model3.get_attention_weights(image, tabular, tab_mask)
    if attn_weights is not None:
        print(f"   Attention weights shape: {attn_weights.shape}")
    else:
        print("   No attention weights available")
    
    print("\n✅ All multimodal integration tests passed!")
    return True


if __name__ == "__main__":
    test_multimodal_integration()

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class CrossAttentionFiLMFusion(nn.Module):
    """
    Fusion: (optional) Cross-Attention then (optional) FiLM.
    Handles:
      - image as [B,C] or [B,T_img,C]
      - tabular as [B,C] or [B,T_tab,C]
      - modality masks: tab_mask: Optional[BoolTensor[B]] (1=present, 0=missing)
    
    Args:
      img_dim: int
      tab_dim: int
      hidden_dim: int
      n_heads: int = 4
      use_cross_attn: bool = True
      use_film: bool = True
      token_wise_film: bool = False   # if True, gamma/beta are [B,T,C] else [B,1,C]
      dropout: float = 0.1
      pool: str = 'mean'  # 'mean' or 'attention'
    
    Returns:
      fused_vec: Tensor[B,C]
      aux: dict with 'img_tokens','tab_tokens','attn_weights' (optional)
    """

    def __init__(
        self, 
        img_dim: int, 
        tab_dim: int, 
        hidden_dim: int, 
        n_heads: int = 4,
        use_cross_attn: bool = True,
        use_film: bool = True,
        token_wise_film: bool = False,
        dropout: float = 0.1,
        pool: str = 'mean'
    ) -> None:
        super().__init__()
        
        # Store configuration
        self.img_dim = img_dim
        self.tab_dim = tab_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_cross_attn = use_cross_attn
        self.use_film = use_film
        self.token_wise_film = token_wise_film
        self.pool = pool
        
        # Cross-attention components
        if use_cross_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=img_dim, 
                num_heads=n_heads, 
                batch_first=True,
                dropout=dropout
            )
            self.tab_proj = nn.Linear(tab_dim, img_dim)
            self.attn_dropout = nn.Dropout(dropout)
        
        # FiLM components
        if use_film:
            self.ln = nn.LayerNorm(img_dim)
            
            # FiLM MLP for gamma and beta
            if token_wise_film:
                # Output 2*embed_dim for gamma and beta per token
                self.film_mlp = nn.Sequential(
                    nn.Linear(img_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2 * img_dim)
                )
            else:
                # Output 2*embed_dim for gamma and beta (broadcasted)
                self.film_mlp = nn.Sequential(
                    nn.Linear(img_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2 * img_dim)
                )
        
        # Pooling for attention-based pooling
        if pool == 'attention':
            self.attention_pool = nn.Linear(img_dim, 1)
        
        # Backward compatibility: keep old gamma/beta for BC
        self.gamma_old = nn.Linear(tab_dim, img_dim)
        self.beta_old = nn.Linear(tab_dim, img_dim)

    def forward(
        self,
        img_feats: torch.Tensor,
        tab_feats: Optional[torch.Tensor],
        tab_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse modalities with optional cross-attention and FiLM.

        Parameters
        ----------
        img_feats : torch.Tensor
            Image feature tensor of shape ``(B, C)`` or ``(B, T_img, C)``.
        tab_feats : torch.Tensor, optional
            Tabular feature tensor of shape ``(B, C)`` or ``(B, T_tab, C)``.
        tab_mask : torch.Tensor, optional
            Boolean mask of shape ``(B,)`` indicating presence of tabular
            features (1=present, 0=missing).

        Returns
        -------
        torch.Tensor
            Fused feature representation ``(B, C)``.
        """
        # Input normalization & tokenization
        img_tokens = self._normalize_to_tokens(img_feats, "img")
        
        # Fast path: no tabular features or all masked
        if tab_feats is None:
            return self._pool_tokens(img_tokens)
        
        if tab_mask is not None and tab_mask.sum() == 0:
            return self._pool_tokens(img_tokens)
        
        tab_tokens = self._normalize_to_tokens(tab_feats, "tab")
        
        # Apply modality mask
        if tab_mask is not None:
            tab_tokens = tab_tokens * tab_mask.unsqueeze(-1).unsqueeze(-1)
        
        # Cross-attention block
        if self.use_cross_attn:
            img_tokens = self._cross_attention(img_tokens, tab_tokens)
        
        # FiLM block
        if self.use_film:
            img_tokens = self._film_modulation(img_tokens, tab_tokens, tab_mask)
        
        # Pooling to vector
        fused_vec = self._pool_tokens(img_tokens)
        
        return fused_vec

    def _normalize_to_tokens(self, feats: torch.Tensor, modality: str) -> torch.Tensor:
        """Convert features to token format [B, T, C]."""
        if feats.dim() == 2:
            # [B, C] -> [B, 1, C]
            return feats.unsqueeze(1)
        elif feats.dim() == 3:
            # [B, T, C] -> [B, T, C]
            return feats
        else:
            raise ValueError(f"Invalid {modality} feature dimensions: {feats.shape}")

    def _cross_attention(self, img_tokens: torch.Tensor, tab_tokens: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention: q=img_tokens, k=v=tab_tokens."""
        # Project tabular tokens to image dimension
        tab_proj = self.tab_proj(tab_tokens)  # [B, T_tab, C]
        
        # Cross-attention: image queries attend to tabular keys/values
        attn_out, attn_weights = self.attn(
            query=img_tokens,
            key=tab_proj,
            value=tab_proj
        )
        
        # Residual connection
        img_tokens = img_tokens + self.attn_dropout(attn_out)
        
        return img_tokens

    def _film_modulation(self, img_tokens: torch.Tensor, tab_tokens: torch.Tensor, tab_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply FiLM modulation with pre-LN and configurable token-wise processing."""
        # Pre-LN
        x = self.ln(img_tokens)  # [B, T_img, C]
        
        # Compute tabular vector (mean over tokens)
        tab_vec = tab_tokens.mean(dim=1)  # [B, C]
        
        # Compute gamma and beta
        film_params = self.film_mlp(tab_vec)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=-1)  # [B, C] each
        
        # Handle modality mask
        if tab_mask is not None:
            # For masked samples, use identity (gamma=1, beta=0)
            gamma = torch.where(tab_mask.unsqueeze(-1), gamma, torch.ones_like(gamma))
            beta = torch.where(tab_mask.unsqueeze(-1), beta, torch.zeros_like(beta))
        
        if self.token_wise_film:
            # Broadcast to all tokens
            gamma = gamma.unsqueeze(1)  # [B, 1, C]
            beta = beta.unsqueeze(1)    # [B, 1, C]
        else:
            # Already [B, C], broadcast to tokens
            gamma = gamma.unsqueeze(1)  # [B, 1, C]
            beta = beta.unsqueeze(1)    # [B, 1, C]
        
        # Apply FiLM: gamma * x + beta
        modulated = gamma * x + beta
        
        # Residual connection
        img_tokens = img_tokens + modulated
        
        return img_tokens

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool tokens to vector representation."""
        if self.pool == 'mean':
            return tokens.mean(dim=1)  # [B, T, C] -> [B, C]
        elif self.pool == 'attention':
            # Attention pooling
            attn_weights = self.attention_pool(tokens)  # [B, T, 1]
            attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]
            pooled = (tokens * attn_weights).sum(dim=1)  # [B, C]
            return pooled
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")

    def forward_with_aux(
        self,
        img_feats: torch.Tensor,
        tab_feats: Optional[torch.Tensor],
        tab_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with auxiliary outputs for debugging/analysis."""
        # Input normalization & tokenization
        img_tokens = self._normalize_to_tokens(img_feats, "img")
        
        aux = {
            'img_tokens': img_tokens,
            'tab_tokens': None,
            'attn_weights': None
        }
        
        # Fast path: no tabular features or all masked
        if tab_feats is None:
            return self._pool_tokens(img_tokens), aux
        
        if tab_mask is not None and tab_mask.sum() == 0:
            return self._pool_tokens(img_tokens), aux
        
        tab_tokens = self._normalize_to_tokens(tab_feats, "tab")
        aux['tab_tokens'] = tab_tokens
        
        # Apply modality mask
        if tab_mask is not None:
            tab_tokens = tab_tokens * tab_mask.unsqueeze(-1).unsqueeze(-1)
        
        # Cross-attention block
        if self.use_cross_attn:
            img_tokens, attn_weights = self._cross_attention_with_weights(img_tokens, tab_tokens)
            if not self.training:
                aux['attn_weights'] = attn_weights
        
        # FiLM block
        if self.use_film:
            img_tokens = self._film_modulation(img_tokens, tab_tokens, tab_mask)
        
        # Pooling to vector
        fused_vec = self._pool_tokens(img_tokens)
        
        return fused_vec, aux

    def _cross_attention_with_weights(self, img_tokens: torch.Tensor, tab_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention that returns attention weights."""
        # Project tabular tokens to image dimension
        tab_proj = self.tab_proj(tab_tokens)  # [B, T_tab, C]
        
        # Cross-attention: image queries attend to tabular keys/values
        attn_out, attn_weights = self.attn(
            query=img_tokens,
            key=tab_proj,
            value=tab_proj
        )
        
        # Residual connection
        img_tokens = img_tokens + self.attn_dropout(attn_out)
        
        return img_tokens, attn_weights

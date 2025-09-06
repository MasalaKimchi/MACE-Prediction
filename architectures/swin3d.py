from __future__ import annotations

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
except ImportError:  # pragma: no cover - handled in runtime
    SwinUNETR = None


class Swin3DEncoder(nn.Module):
    """Swin-Transformer based encoder for 3D volumes.

    This wraps :class:`monai.networks.nets.SwinUNETR` and exposes a simple
    interface that converts an input volume into a global embedding vector.
    Only the encoder path of ``SwinUNETR`` is utilised. The deepest feature
    map is globally averaged and projected to the desired embedding size.
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
    ) -> None:
        super().__init__()
        if SwinUNETR is None:  # pragma: no cover
            raise ImportError("MONAI is required for Swin3DEncoder. Install with 'pip install monai'.")

        self.backbone = SwinUNETR(
            in_channels=in_ch,
            out_channels=feature_size,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            dropout_path_rate=dropout_path_rate,
            spatial_dims=3,
        )
        self.proj = nn.Linear(feature_size * 16, embed_dim)

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
        hidden = self.backbone.swinViT(x, self.backbone.normalize)
        deep = self.backbone.encoder10(hidden[4])
        gap = deep.mean(dim=(2, 3, 4))
        return self.proj(gap)

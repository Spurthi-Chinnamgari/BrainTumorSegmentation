"""
3D Attention U-Net Model Architecture for Volumetric Medical Image Segmentation.
Ref: Oktay et al., 'Attention U-Net: Learning Where to Look for the Pancreas', MIDL 2018.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from .building_blocks import DoubleConv3D, UpBlock3D


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for Volumetric Multimodal Brain Tumor Segmentation (BraTS 2023).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: List[int] = None,
        dropout: float = 0.1,
        use_transpose: bool = True,
    ):
        """
        Args:
            in_channels: Number of input MRI modalities (default: 4 for T1, T1ce, T2, FLAIR)
            out_channels: Number of segmentation classes (default: 4 for BG, NCR, ED, ET)
            features: List of channel dimensions at each level (default: [32, 64, 128, 256, 512])
            dropout: Dropout probability in double conv blocks
            use_transpose: If True, uses ConvTranspose3d; otherwise Trilinear upsampling
        """
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        prev_channels = in_channels
        for feature in features[:-1]:
            self.encoder_blocks.append(
                DoubleConv3D(in_channels=prev_channels, out_channels=feature, dropout=dropout)
            )
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(
            in_channels=features[-2], out_channels=features[-1], dropout=dropout
        )

        # Decoder path with 3D Attention Gates
        self.decoder_blocks = nn.ModuleList()
        rev_features = list(reversed(features))

        for i in range(len(rev_features) - 1):
            in_ch = rev_features[i]
            out_ch = rev_features[i + 1]
            self.decoder_blocks.append(
                UpBlock3D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    use_transpose=use_transpose,
                    dropout=dropout,
                )
            )

        # Final 1x1x1 output convolution layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volumetric input tensor of shape (B, in_channels, D, H, W)
        Returns:
            Volumetric logits tensor of shape (B, out_channels, D, H, W)
        """
        skip_connections = []

        # Encoder path
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = list(reversed(skip_connections))

        # Decoder path
        for i, up_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i]
            x = up_block(g=x, x=skip)

        logits = self.final_conv(x)
        return logits

"""
Building Blocks for 2D and 3D Attention U-Net Architectures.
Includes Double Convolution Blocks, Attention Gates, Downsampling, and Upsampling layers.
"""

from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """
    2D Double Convolution Block:
    (Conv2D -> BatchNorm2D -> ReLU) x 2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConv3D(nn.Module):
    """
    3D Double Convolution Block:
    (Conv3D -> BatchNorm3D -> ReLU) x 2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = [
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout3d(dropout))

        layers.extend([
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate2D(nn.Module):
    """
    2D Additive Attention Gate as proposed in Oktay et al., 'Attention U-Net'.
    Filters encoder skip connections using gating signals from deeper decoder features.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature maps in gating signal g (from decoder)
            F_l: Number of feature maps in skip connection x (from encoder)
            F_int: Number of intermediate channels for attention mechanism
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal tensor of shape (B, F_g, H_g, W_g)
            x: Skip connection tensor of shape (B, F_l, H_x, W_x)
        Returns:
            Attention-weighted feature map of shape (B, F_l, H_x, W_x)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample gating signal if spatial dimensions differ from skip connection
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class AttentionGate3D(nn.Module):
    """
    3D Additive Attention Gate for volumetric medical image segmentation.
    Filters encoder skip connections using gating signals from deeper decoder features.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature channels in gating signal g
            F_l: Number of feature channels in skip connection x
            F_int: Number of intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal tensor of shape (B, F_g, D_g, H_g, W_g)
            x: Skip connection tensor of shape (B, F_l, D_x, H_x, W_x)
        Returns:
            Attention-weighted feature map of shape (B, F_l, D_x, H_x, W_x)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class UpBlock2D(nn.Module):
    """
    2D Decoder Upsampling Block:
    Upsample (ConvTranspose / Interpolate) -> AttentionGate -> Concatenate -> DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_transpose: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_transpose = use_transpose
        if use_transpose:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        feature_channel = in_channels // 2 if use_transpose else in_channels
        self.attention = AttentionGate2D(
            F_g=feature_channel, F_l=out_channels, F_int=out_channels // 2
        )
        self.conv = DoubleConv2D(
            in_channels=feature_channel + out_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        up_g = self.up(g)

        # Pad spatial dimensions if necessary due to odd input shapes
        diff_y = x.size()[2] - up_g.size()[2]
        diff_x = x.size()[3] - up_g.size()[3]
        if diff_y != 0 or diff_x != 0:
            up_g = F.pad(
                up_g,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        attn_x = self.attention(g=up_g, x=x)
        concat = torch.cat([attn_x, up_g], dim=1)
        return self.conv(concat)


class UpBlock3D(nn.Module):
    """
    3D Decoder Upsampling Block:
    Upsample (ConvTranspose / Interpolate) -> AttentionGate -> Concatenate -> DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_transpose: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_transpose = use_transpose
        if use_transpose:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        feature_channel = in_channels // 2 if use_transpose else in_channels
        self.attention = AttentionGate3D(
            F_g=feature_channel, F_l=out_channels, F_int=out_channels // 2
        )
        self.conv = DoubleConv3D(
            in_channels=feature_channel + out_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        up_g = self.up(g)

        diff_z = x.size()[2] - up_g.size()[2]
        diff_y = x.size()[3] - up_g.size()[3]
        diff_x = x.size()[4] - up_g.size()[4]

        if diff_z != 0 or diff_y != 0 or diff_x != 0:
            up_g = F.pad(
                up_g,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                    diff_z // 2,
                    diff_z - diff_z // 2,
                ],
            )

        attn_x = self.attention(g=up_g, x=x)
        concat = torch.cat([attn_x, up_g], dim=1)
        return self.conv(concat)

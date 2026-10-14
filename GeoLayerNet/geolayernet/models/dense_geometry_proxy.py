"""Module B: dense geometry proxy.

A small U-Net that turns RGB + sparse depth + valid mask into a coarse but
COMPLETE dense depth, preserving observed measurements. This is the strongest
branch: with ``output_mode="proxy_only"`` the proxy IS the final output (adopted
best). Submodule layer names match the original checkpoint (``enc0`` ...).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, ResidualBlock, upsample_block


class DenseGeometryProxy(nn.Module):
    def __init__(self, rgb_channels=3, depth_channels=1, base_channels=32, norm="batch",
                 extra_in_channels=0):
        super().__init__()
        # extra_in_channels carries optional conditioning (e.g. coarse depth +
        # confidence from CoarseDensifier). 0 == r11 baseline (checkpoint-compatible).
        self.extra_in_channels = int(extra_in_channels)
        in_channels = rgb_channels + depth_channels + 1 + self.extra_in_channels
        self.enc0 = ConvBlock(in_channels, base_channels, norm=norm)
        self.enc1 = ConvBlock(base_channels, base_channels * 2, stride=2, norm=norm)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, norm=norm)
        self.body = nn.Sequential(
            ResidualBlock(base_channels * 4, norm=norm),
            ResidualBlock(base_channels * 4, norm=norm),
        )
        self.up1 = upsample_block(base_channels * 4, base_channels * 2, norm=norm)
        self.refine1 = ConvBlock(base_channels * 4, base_channels * 2, norm=norm)
        self.up0 = upsample_block(base_channels * 2, base_channels, norm=norm)
        self.refine0 = ConvBlock(base_channels * 2, base_channels, norm=norm)
        self.out = nn.Conv2d(base_channels, depth_channels, kernel_size=3, padding=1)

    def forward(self, rgb, sparse_depth, valid_mask, coarse=None):
        parts = [rgb, sparse_depth, valid_mask]
        if coarse is not None:
            parts.append(coarse)
        x = torch.cat(parts, dim=1)
        skip0 = self.enc0(x)
        skip1 = self.enc1(skip0)
        feat = self.body(self.enc2(skip1))
        x = self.up1(feat)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine1(torch.cat([x, skip1], dim=1))
        x = self.up0(x)
        if x.shape[-2:] != skip0.shape[-2:]:
            x = F.interpolate(x, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine0(torch.cat([x, skip0], dim=1))
        predicted = torch.sigmoid(self.out(x))
        return valid_mask * sparse_depth + (1.0 - valid_mask) * predicted

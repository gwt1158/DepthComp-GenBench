"""Per-layer geometry-aware decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, upsample_block


class LayerDecoder(nn.Module):
    """Decode a layer feature into a layer-wise depth (residual) map."""

    def __init__(self, feature_channels, skip_channels, out_channels, norm="batch",
                 out_activation="sigmoid"):
        super().__init__()
        self.up1 = upsample_block(feature_channels, feature_channels // 2, norm=norm)
        self.refine1 = ConvBlock(feature_channels // 2 + skip_channels * 2, feature_channels // 2, norm=norm)
        self.up2 = upsample_block(feature_channels // 2, feature_channels // 4, norm=norm)
        self.refine2 = ConvBlock(feature_channels // 4 + skip_channels, feature_channels // 4, norm=norm)
        self.out = nn.Conv2d(feature_channels // 4, out_channels, kernel_size=3, padding=1)
        self.out_activation = out_activation

    def forward(self, feat, skip0, skip1):
        x = self.up1(feat)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine1(torch.cat([x, skip1], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != skip0.shape[-2:]:
            x = F.interpolate(x, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine2(torch.cat([x, skip0], dim=1))
        x = self.out(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.out_activation == "tanh":
            return torch.tanh(x)
        return x

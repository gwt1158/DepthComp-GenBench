"""Shared convolution blocks."""

import torch.nn as nn


def norm_layer(channels, norm="batch"):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)
    if norm == "none":
        return nn.Identity()
    raise NotImplementedError("normalization [%s] is not supported" % norm)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm="batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=(norm == "none")),
            norm_layer(out_channels, norm), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=(norm == "none")),
            norm_layer(out_channels, norm), nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=(norm == "none")),
            norm_layer(channels, norm), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=(norm == "none")),
            norm_layer(channels, norm),
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(x + self.block(x))


def upsample_block(in_channels, out_channels, norm="batch"):
    """Checkpoint-compatible 2x transposed-convolution upsampling."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
        norm_layer(out_channels, norm), nn.ReLU(True),
    )

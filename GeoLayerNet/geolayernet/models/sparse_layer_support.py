"""Module A: sparse-depth layer support.

From the sparse depth valid points, build K per-layer depth-quantile seeds and
expand each locally with a normalized Gaussian kernel modulated by RGB bilateral
affinity (``gaussian_bilateral``). Support is LOCAL evidence (never sums to 1
over the image); pixels with no nearby support fall into an explicit ``unknown``
region left to the dense proxy. Parameter-free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDepthLayerSupport(nn.Module):
    def __init__(self, num_layers=4, expand_radius=11, sigma_xy=3.0,
                 rgb_guided=True, rgb_sigma=0.10, eps=1e-6,
                 partition_mode="quantile_ordered", partition_seed=42):
        super().__init__()
        self.num_layers = int(num_layers)
        self.radius = int(expand_radius)
        self.sigma_xy = float(sigma_xy)
        self.rgb_guided = bool(rgb_guided)
        self.rgb_sigma = float(rgb_sigma)
        self.eps = float(eps)
        self.partition_mode = str(partition_mode)
        self.partition_seed = int(partition_seed)
        if self.partition_mode not in {
                "quantile_ordered", "uniform_normalized",
                "random_equal_count"}:
            raise ValueError(
                f"unsupported sparse layer partition: {self.partition_mode}")

    @torch.no_grad()
    def _seeds(self, sparse_depth, valid_mask):
        b, _, h, w = sparse_depth.shape
        k = self.num_layers
        seeds = sparse_depth.new_zeros(b, k, 1, h, w)
        for i in range(b):
            valid = valid_mask[i, 0] > 0.5
            values = sparse_depth[i, 0][valid]
            if values.numel() == 0:
                continue
            if self.partition_mode == "random_equal_count":
                flat_valid = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
                # A fixed spatial hash makes the random control reproducible for
                # a sample across epochs.  It intentionally does not use depth,
                # so it cannot leak the order that this control must destroy.
                modulus = 2147483647
                keys = ((flat_valid.to(torch.int64) + self.partition_seed * 104729)
                        * 1103515245 + 12345) % modulus
                order = flat_valid[torch.argsort(keys)]
                # rank*k//n yields equal-count groups whose sizes differ by at
                # most one, while deliberately destroying depth order.
                layer_ids = (torch.arange(
                    order.numel(), device=order.device) * k
                    // order.numel()).clamp_max(k - 1)
                flat_seeds = seeds[i, :, 0].view(k, -1)
                flat_seeds[layer_ids, order] = 1.0
                continue

            if torch.isclose(values.min(), values.max()):
                seeds[i, 0, 0][valid] = 1.0
                continue

            if self.partition_mode == "uniform_normalized":
                vmin, vmax = values.min(), values.max()
                d = (sparse_depth[i, 0] - vmin) / (vmax - vmin).clamp_min(
                    self.eps)
                bounds = torch.linspace(
                    0.0, 1.0, k + 1,
                    device=values.device, dtype=values.dtype)
            else:
                d = sparse_depth[i, 0]
                qs = torch.linspace(
                    0.0, 1.0, k + 1,
                    device=values.device, dtype=values.dtype)
                bounds = torch.quantile(values, qs)
            for layer in range(k):
                lo, hi = bounds[layer], bounds[layer + 1]
                mask = ((d >= lo) if layer == 0 else (d > lo)) & (d <= hi) & valid
                seeds[i, layer, 0] = mask.float()
        return seeds

    def _support_at(self, seeds, rgb, radius, b, k, h, w):
        """Expand per-layer seeds with a Gaussian (+ optional RGB bilateral) kernel
        at one radius. Returns support of shape [b, k, 1, h, w]."""
        radius = max(int(radius), 1)
        ksize = 2 * radius + 1
        coords = torch.arange(ksize, device=seeds.device, dtype=seeds.dtype) - radius
        g1 = torch.exp(-coords.pow(2) / (2.0 * max(self.sigma_xy, 1e-3) ** 2))
        g1 = g1 / g1.max()
        kernel = (g1[:, None] * g1[None, :]).view(1, 1, ksize, ksize)

        flat_seeds = seeds.view(b * k, 1, h, w)
        spatial = F.conv2d(flat_seeds, kernel, padding=radius).clamp(0.0, 1.0)

        if self.rgb_guided and rgb is not None:
            rgb01 = (rgb + 1.0) * 0.5
            denom = F.conv2d(flat_seeds, kernel, padding=radius)
            rgb_rep = rgb01.unsqueeze(1).expand(b, k, 3, h, w).reshape(b * k, 3, h, w)
            kernel3 = kernel.expand(3, 1, ksize, ksize)
            seed_rgb = F.conv2d(rgb_rep * flat_seeds, kernel3, padding=radius, groups=3) / denom.clamp_min(self.eps)
            affinity = torch.exp(-(rgb_rep - seed_rgb).pow(2).sum(dim=1, keepdim=True) / max(self.rgb_sigma, 1e-3) ** 2)
            affinity = torch.where(denom > self.eps, affinity, torch.zeros_like(affinity))
            support = spatial * affinity
        else:
            support = spatial

        support = support.view(b, k, 1, h, w)
        return torch.maximum(support, seeds)

    def forward(self, sparse_depth, valid_mask, rgb=None):
        b, _, h, w = sparse_depth.shape
        k = self.num_layers
        seeds = self._seeds(sparse_depth, valid_mask)

        support = self._support_at(seeds, rgb, self.radius, b, k, h, w)
        support_total = support.sum(dim=1).clamp(0.0, 1.0)
        unknown = (1.0 - support_total).clamp(0.0, 1.0)
        return {"seeds": seeds, "support": support,
                "support_total": support_total, "unknown_support": unknown}

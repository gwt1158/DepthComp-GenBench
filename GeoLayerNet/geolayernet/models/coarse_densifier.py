"""Non-CNN coarse depth densifier (training-free, parameter-free).

Turns the sparse depth + valid mask into a COMPLETE coarse depth plus a
per-pixel confidence, used as an extra conditioning input to the dense geometry
proxy (Phase 1 of the geometry-pyramid enhancement). It learns nothing — it is a
pure multi-radius masked-averaging fill, implemented with ``F.avg_pool2d`` so it
runs batched on CPU or GPU tensors with no scipy / cv2 dependency.

Design notes:
  * ``coarse_depth`` is guaranteed dense (every pixel filled): nearer (smaller)
    radii win where they have support, larger radii fill big holes, and a final
    per-image valid-mean fills anything still unsupported.
  * ``coarse_confidence`` reflects how much local sparse evidence backs a pixel:
    the valid-point density in the smallest window, optionally down-weighted by
    RGB bilateral disagreement. It is meant for the proxy / loss to learn where
    the coarse fill is trustworthy.
  * Observed pixels keep their measured depth exactly and get confidence 1.

This module never sees ``target_depth``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _box(x, radius):
    """Mean over a (2r+1)^2 window, stride 1, same spatial size."""
    k = 2 * int(radius) + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=int(radius), count_include_pad=True)


class CoarseDensifier(nn.Module):
    def __init__(self, radii=(5, 11), rgb_guided=False, rgb_sigma=0.10, eps=1e-6):
        super().__init__()
        radii = tuple(sorted(int(r) for r in radii if int(r) >= 1)) or (5,)
        self.radii = radii
        self.rgb_guided = bool(rgb_guided)
        self.rgb_sigma = float(rgb_sigma)
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, sparse_depth, valid_mask, rgb=None):
        mask = (valid_mask > 0.5).to(sparse_depth.dtype)
        value = sparse_depth * mask

        # progressive multi-radius masked-average fill (small radius preferred) #
        filled = value
        fmask = mask
        smallest_den = None
        for r in self.radii:
            num = _box(value, r)
            den = _box(mask, r)
            avg_r = num / den.clamp_min(self.eps)
            have_r = (den > self.eps).to(sparse_depth.dtype)
            if smallest_den is None:
                smallest_den = den  # window valid-mass at the finest scale
            take = have_r * (1.0 - fmask)            # fill only not-yet-filled pixels
            filled = filled + take * avg_r
            fmask = (fmask + take).clamp(0.0, 1.0)

        # fallback: anything still unfilled -> per-image valid mean (keeps it dense)
        b = sparse_depth.size(0)
        flat_v = value.view(b, -1)
        flat_m = mask.view(b, -1)
        img_mean = (flat_v.sum(dim=1) / flat_m.sum(dim=1).clamp_min(self.eps)).view(b, 1, 1, 1)
        filled = torch.where(fmask > 0.5, filled, img_mean.expand_as(filled))

        # confidence: local valid-density in the finest window, in [0, 1]
        if smallest_den is None:
            smallest_den = _box(mask, self.radii[0])
        confidence = smallest_den.clamp(0.0, 1.0)

        if self.rgb_guided and rgb is not None:
            rgb01 = (rgb + 1.0) * 0.5
            r0 = self.radii[0]
            den0 = _box(mask, r0).clamp_min(self.eps)
            local_rgb = _box(rgb01 * mask, r0) / den0           # mean RGB of valid in window
            affinity = torch.exp(-(rgb01 - local_rgb).pow(2).sum(dim=1, keepdim=True)
                                 / max(self.rgb_sigma, 1e-3) ** 2)
            confidence = confidence * affinity

        # observed pixels are exact and fully trusted
        coarse_depth = mask * sparse_depth + (1.0 - mask) * filled
        coarse_confidence = (mask + (1.0 - mask) * confidence).clamp(0.0, 1.0)
        return coarse_depth, coarse_confidence

"""GeoLayerNet — assembly of the three modules.

  A. SparseDepthLayerSupport  (support)
  B. DenseGeometryProxy       (proxy)   -> the strongest branch (proxy_only)
  C. LayerGuidedCompletion    (completion)

Forward output follows the final checkpoint-compatible inference graph. The
only runtime knob is ``output_mode``.
"""

import torch
import torch.nn as nn

from ..config import GeoLayerConfig
from .sparse_layer_support import SparseDepthLayerSupport
from .dense_geometry_proxy import DenseGeometryProxy
from .layer_guided_completion import LayerGuidedCompletion
from .coarse_densifier import CoarseDensifier

# public alias requested by the project (LayerGuidedCompletion == the completion head)
GeoLayerNetCompletion = LayerGuidedCompletion


class GeoLayerNet(nn.Module):
    def __init__(self, cfg: GeoLayerConfig = None, output_mode=None):
        super().__init__()
        cfg = cfg or GeoLayerConfig()
        if output_mode is not None:
            cfg.output_mode = output_mode
        self.cfg = cfg
        self.output_mode = cfg.output_mode
        self.residual_warmup = 1.0

        # Phase 1 — optional training-free coarse densifier. Built only when ON
        # so the default module set stays byte-identical to r11. It is
        # parameter-free, so it adds no weights to the checkpoint.
        self.use_coarse = bool(getattr(cfg, "use_coarse_densifier", False))
        self.proxy_use_coarse = bool(getattr(cfg, "proxy_use_coarse", False))
        self.densifier = None
        if self.use_coarse:
            self.densifier = CoarseDensifier(
                radii=getattr(cfg, "coarse_densifier_radii", (5, 11)),
                rgb_guided=getattr(cfg, "coarse_densifier_rgb_guided", False),
                rgb_sigma=cfg.sparse_layer_rgb_sigma)
        extra_proxy_ch = 2 if (self.use_coarse and self.proxy_use_coarse) else 0

        proxy_channels = max(8, cfg.base_channels // 2)
        self.proxy = DenseGeometryProxy(cfg.rgb_channels, cfg.depth_channels,
                                        base_channels=proxy_channels, norm=cfg.norm,
                                        extra_in_channels=extra_proxy_ch)
        self.support = SparseDepthLayerSupport(
            num_layers=cfg.num_layers, expand_radius=cfg.sparse_layer_expand_radius,
            sigma_xy=cfg.sparse_layer_sigma_xy, rgb_guided=cfg.sparse_layer_rgb_guided,
            rgb_sigma=cfg.sparse_layer_rgb_sigma,
            partition_mode=getattr(cfg, "layer_partition_mode", "quantile_ordered"),
            partition_seed=getattr(cfg, "layer_partition_seed", 42))
        self.completion = LayerGuidedCompletion(cfg)

    def set_residual_warmup(self, factor):
        self.residual_warmup = float(min(max(factor, 0.0), 1.0))

    def forward(self, rgb, depth, valid_mask=None, output_mode=None):
        if valid_mask is None:
            valid_mask = (depth > 0).float()
        mode = output_mode or self.output_mode

        # Phase 1: coarse densification (training-free; sparse + mask + rgb only).
        coarse_depth = coarse_conf = None
        coarse_for_proxy = None
        if self.densifier is not None:
            coarse_depth, coarse_conf = self.densifier(depth, valid_mask, rgb)
            if self.proxy_use_coarse:
                coarse_for_proxy = torch.cat([coarse_depth, coarse_conf], dim=1)

        dense_geometry = self.proxy(rgb, depth, valid_mask, coarse=coarse_for_proxy)
        support = self.support(depth, valid_mask, rgb)
        out = self.completion(rgb, depth, valid_mask, dense_geometry, support,
                              output_mode=mode, residual_warmup=self.residual_warmup)
        out["coarse_depth"] = coarse_depth
        out["coarse_confidence"] = coarse_conf
        return out


# Training-facing alias (same module; kept for naming clarity).
GeoLayerNetModel = GeoLayerNet

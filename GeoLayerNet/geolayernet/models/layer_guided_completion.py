"""Module C: layer-guided completion.

Uses the sparse layer support to extract per-layer ACM RGB-D features, propagates
them with depth-ordered cross-layer relation attention, decodes per-layer
residuals, and applies an identity-safe gated residual on the dense proxy anchor.
Produces raw / safe / proxy / final outputs; the adopted best output is
``proxy_only``. Submodule names (encoder/fusion/depth_decoder/refine_gate_head/
uncertainty_head/layer_uncertainty/assignment) match the original checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, ResidualBlock
from .decoders import LayerDecoder


# --------------------------------------------------------------------------- #
class AcmLayerEncoder(nn.Module):
    """ACMNet-style dual RGB/depth encoder applied inside each geometry layer."""

    def __init__(self, rgb_channels=3, depth_channels=4, base_channels=64, norm="batch"):
        super().__init__()
        self.d_stem = ConvBlock(depth_channels, base_channels, stride=1, norm=norm)
        self.d_down1 = ConvBlock(base_channels, base_channels * 2, stride=2, norm=norm)
        self.d_down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, norm=norm)
        self.d_body = nn.Sequential(ResidualBlock(base_channels * 4, norm=norm), ResidualBlock(base_channels * 4, norm=norm))
        self.r_stem = ConvBlock(rgb_channels, base_channels, stride=1, norm=norm)
        self.r_down1 = ConvBlock(base_channels, base_channels * 2, stride=2, norm=norm)
        self.r_down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, norm=norm)
        self.r_body = nn.Sequential(ResidualBlock(base_channels * 4, norm=norm), ResidualBlock(base_channels * 4, norm=norm))
        self.d_gate0 = nn.Conv2d(base_channels, base_channels, 1)
        self.r_gate0 = nn.Conv2d(base_channels, base_channels, 1)
        self.fuse0 = ConvBlock(base_channels * 4, base_channels, norm=norm)
        self.d_gate1 = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.r_gate1 = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fuse1 = ConvBlock(base_channels * 8, base_channels * 2, norm=norm)
        self.d_gate2 = nn.Conv2d(base_channels * 4, base_channels * 4, 1)
        self.r_gate2 = nn.Conv2d(base_channels * 4, base_channels * 4, 1)
        self.fuse2 = ConvBlock(base_channels * 16, base_channels * 4, norm=norm)

    @staticmethod
    def _fuse(depth_feat, rgb_feat, depth_gate, rgb_gate, fuse):
        depth_guided_rgb = torch.sigmoid(depth_gate(depth_feat)) * rgb_feat
        rgb_guided_depth = torch.sigmoid(rgb_gate(rgb_feat)) * depth_feat
        return fuse(torch.cat([depth_feat, rgb_feat, depth_guided_rgb, rgb_guided_depth], dim=1))

    def forward(self, rgb, depth_guidance):
        d0 = self.d_stem(depth_guidance); r0 = self.r_stem(rgb)
        skip0 = self._fuse(d0, r0, self.d_gate0, self.r_gate0, self.fuse0)
        d1 = self.d_down1(d0); r1 = self.r_down1(r0)
        skip1 = self._fuse(d1, r1, self.d_gate1, self.r_gate1, self.fuse1)
        d2 = self.d_body(self.d_down2(d1)); r2 = self.r_body(self.r_down2(r1))
        feat = self._fuse(d2, r2, self.d_gate2, self.r_gate2, self.fuse2)
        return feat, skip0, skip1


# --------------------------------------------------------------------------- #
class DepthOrderedRelationPropagation(nn.Module):
    """Depth-ordered cross-layer relation propagation + gated fusion (global_gated)."""

    def __init__(self, channels, num_layers, heads=4, dropout=0.0, fusion_type="global_gated",
                 relation_steps=2, use_order_cues=True):
        super().__init__()
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.relation_steps = max(int(relation_steps), 1)
        self.use_order_cues = bool(use_order_cues)
        self.layer_embedding = nn.Parameter(torch.randn(num_layers, channels) * 0.02)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.value = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(True), nn.Linear(channels, channels))
        self.gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(True), nn.Linear(channels, channels), nn.Sigmoid())
        self.proj = nn.Conv2d(channels, channels, 1)
        self.adjacent = nn.Conv1d(channels, channels, 3, padding=1, groups=1)
        self.saliency = nn.Conv2d(channels, 1, 1)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.relative_layer_bias = nn.Parameter(torch.zeros(num_layers * 2 - 1))
        self.front_to_back_bias = nn.Parameter(torch.zeros(1))
        self.back_to_front_bias = nn.Parameter(torch.zeros(1))
        self.depth_distance_scale = nn.Parameter(torch.tensor(0.1))
        gate_hidden = max(channels // 2, 16)
        self.occlusion_pair_gate = nn.Sequential(
            nn.Linear(channels * 2 + 2, gate_hidden), nn.ReLU(True), nn.Linear(gate_hidden, 1), nn.Sigmoid())

    def forward(self, layer_features, layer_weights, layer_uncertainty, dense_geometry):
        b, k, c, h, w = layer_features.shape
        if self.fusion_type == "none" or k == 1:
            return layer_features
        tokens = layer_features.view(b * k, c, h, w).mean(dim=(-2, -1)).view(b, k, c)
        if self.use_order_cues:
            tokens = tokens + self.layer_embedding[:k].unsqueeze(0)
        coverage, reliability, depth_centers = self._stats(layer_weights, layer_uncertainty, dense_geometry, b, k, tokens.dtype, tokens.device)
        tokens = tokens * (0.5 + coverage)
        tokens = tokens * reliability.clamp(0.05, 1.0)
        context = self._relation(tokens, coverage, reliability, depth_centers)
        value = self.value(context)
        gate = self.gate(torch.cat([tokens, context], dim=-1))
        update = (gate * value).view(b, k, c, 1, 1)
        return layer_features + self.proj(update.flatten(0, 1)).view(b, k, c, 1, 1)

    def _relation(self, tokens, coverage, reliability, depth_centers):
        state = tokens
        for _ in range(self.relation_steps):
            context = self._attn_step(state, coverage, reliability, depth_centers)
            step_gate = self.gate(torch.cat([state, context], dim=-1))
            state = state + step_gate * self.value(context)
        return state

    def _attn_step(self, tokens, coverage, reliability, depth_centers):
        b, k, c = tokens.shape
        q = self.q_proj(tokens); key = self.k_proj(tokens); value = self.v_proj(tokens)
        logits = torch.einsum("bic,bjc->bij", q, key) / (c ** 0.5)
        layer_ids = torch.arange(k, device=tokens.device)
        rel = layer_ids.view(k, 1) - layer_ids.view(1, k)
        if self.use_order_cues:
            bias = self.relative_layer_bias[rel + self.num_layers - 1]
            logits = logits + bias.to(tokens.dtype).unsqueeze(0)
            logits = logits - F.softplus(self.depth_distance_scale) * self._depth_dist(
                rel, depth_centers, tokens.dtype, b)
            direction = torch.zeros_like(bias, dtype=tokens.dtype)
            direction = torch.where(
                rel > 0, self.front_to_back_bias.to(tokens.dtype), direction)
            direction = torch.where(
                rel < 0, self.back_to_front_bias.to(tokens.dtype), direction)
            logits = logits + direction.unsqueeze(0)
        cov_m = self._cov_matrix(coverage)
        rel_m = self._src_matrix(reliability)
        logits = logits + torch.log(cov_m.clamp_min(1e-4)) + torch.log(rel_m.clamp_min(1e-4))
        attn = torch.softmax(logits, dim=-1)
        qi = tokens.unsqueeze(2).expand(-1, -1, k, -1)
        kj = tokens.unsqueeze(1).expand(-1, k, -1, -1)
        if self.use_order_cues:
            rel_norm = rel.to(tokens.dtype) / max(float(k - 1), 1.0)
        else:
            # Preserve the same module graph and parameter count while removing
            # every explicit layer-position/direction/depth-gap cue.
            rel_norm = torch.zeros_like(rel, dtype=tokens.dtype)
        pair_geom = torch.stack([rel_norm, rel_norm.abs()], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        occ = self.occlusion_pair_gate(torch.cat([qi, kj, pair_geom], dim=-1)).squeeze(-1)
        attn = attn * occ
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bij,bjc->bic", attn, value)

    def _stats(self, layer_weights, layer_uncertainty, dense_geometry, b, k, dtype, device):
        coverage = F.adaptive_avg_pool2d(layer_weights.flatten(0, 1), 1).view(b, k, 1)
        uncertainty = F.adaptive_avg_pool2d(layer_uncertainty.flatten(0, 1), 1).view(b, k, 1)
        reliability = 1.0 - uncertainty.clamp(0.0, 1.0)
        depth = dense_geometry
        if depth.shape[-2:] != layer_weights.shape[-2:]:
            depth = F.interpolate(depth, size=layer_weights.shape[-2:], mode="bilinear", align_corners=False)
        weights = layer_weights.to(dtype)
        num = (weights * depth.unsqueeze(1).to(dtype)).sum(dim=(-2, -1)).view(b, k)
        den = weights.sum(dim=(-2, -1)).view(b, k).clamp_min(1e-6)
        return coverage.to(dtype), reliability.to(dtype), num / den

    def _depth_dist(self, rel, depth_centers, dtype, batch_size):
        dist = (depth_centers.unsqueeze(2) - depth_centers.unsqueeze(1)).abs().to(dtype)
        scale = dist.detach().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        normalized = dist / scale
        if normalized.size(0) != batch_size:
            normalized = normalized.expand(batch_size, -1, -1)
        return normalized

    def _src_matrix(self, values):
        b, k, _ = values.shape
        return values.view(b, 1, k).expand(-1, k, -1)

    def _cov_matrix(self, coverage):
        source = self._src_matrix(coverage)
        target = coverage.view(coverage.size(0), coverage.size(1), 1)
        return torch.sqrt((target * source).clamp_min(1e-8))


# --------------------------------------------------------------------------- #
class LayerGuidedCompletion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nl, base, norm = cfg.num_layers, cfg.base_channels, cfg.norm
        rgb_c, depth_c = cfg.rgb_channels, cfg.depth_channels
        feat_c = base * 4
        layer_depth_in = depth_c * 2 + 2
        self.num_layers = nl
        self.global_pool = cfg.global_pool
        self.layer_drop_prob = cfg.layer_drop_prob
        self.residual_scale = float(cfg.residual_scale)
        self.dorp_use_order_cues = bool(
            getattr(cfg, "dorp_use_order_cues", True))

        # ---- enhancement switches (default OFF == r11 baseline) ------------- #
        self.selective_residual = bool(getattr(cfg, "selective_residual", False))
        # fixed Sobel kernels for the parameter-free RGB edge gate. Non-persistent
        # buffers -> they never enter the checkpoint, so this stays compatible.
        sob = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        self.register_buffer("_sobel_x", sob.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("_sobel_y", sob.t().contiguous().view(1, 1, 3, 3), persistent=False)

        # retained for checkpoint compatibility; UNUSED on the sparse-depth path
        self.assignment = nn.Sequential(ConvBlock(rgb_c + depth_c * 2 + 1, base // 2, norm=norm),
                                        nn.Conv2d(base // 2, nl, kernel_size=1))
        self.encoder = AcmLayerEncoder(rgb_channels=rgb_c, depth_channels=layer_depth_in, base_channels=base, norm=norm)
        self.fusion = DepthOrderedRelationPropagation(feat_c, nl, heads=cfg.attention_heads,
                                                      fusion_type=cfg.fusion_type,
                                                      relation_steps=cfg.relation_steps,
                                                      use_order_cues=self.dorp_use_order_cues)
        self.layer_uncertainty = nn.Sequential(nn.Conv2d(feat_c, base, 3, padding=1), nn.ReLU(True), nn.Conv2d(base, 1, 1))
        self.depth_decoder = LayerDecoder(feat_c, base, depth_c, norm=norm, out_activation="none")
        nn.init.zeros_(self.depth_decoder.out.weight)
        if self.depth_decoder.out.bias is not None:
            nn.init.zeros_(self.depth_decoder.out.bias)
        self.refine_gate_head = nn.Sequential(nn.Conv2d(rgb_c + depth_c + 1, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 1, 3, padding=1))
        nn.init.constant_(self.refine_gate_head[-1].bias, float(cfg.gate_bias_init))
        self.uncertainty_head = nn.Sequential(nn.Conv2d(feat_c, base, 3, padding=1), nn.ReLU(True), nn.Conv2d(base, 1, 1))

    def _make_drop(self, rgb):
        if (not self.training) or self.layer_drop_prob <= 0.0:
            return None
        prob = min(max(float(self.layer_drop_prob), 0.0), 1.0)
        drop = (torch.rand(rgb.size(0), self.num_layers, 1, 1, 1, device=rgb.device, dtype=rgb.dtype) < prob).float()
        if self.num_layers > 1:
            all_dropped = drop.view(rgb.size(0), self.num_layers).sum(dim=1) >= self.num_layers
            if all_dropped.any():
                idx = torch.nonzero(all_dropped, as_tuple=False).flatten()
                keep = torch.randint(self.num_layers, (idx.numel(),), device=rgb.device)
                drop[idx] = 1.0
                drop[idx, keep, :, :, :] = 0.0
        return drop

    def _pool_global(self, fused, layer_weights):
        if self.global_pool == "mean":
            return fused.mean(dim=1)
        weight_sum = layer_weights.sum(dim=1)
        weighted = (fused * layer_weights).sum(dim=1) / weight_sum.clamp_min(1e-6)
        return torch.where(weight_sum > 1e-6, weighted, fused.mean(dim=1))

    def _edge_gate(self, rgb):
        """Parameter-free RGB edge magnitude in [0, 1] (per-image normalized)."""
        gray = rgb.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self._sobel_x, padding=1)
        gy = F.conv2d(gray, self._sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        scale = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / scale).clamp(0.0, 1.0)

    def forward(self, rgb, depth, valid_mask, dense_geometry, support, output_mode="safe", residual_warmup=1.0):
        masks = support["support"]
        anchor = dense_geometry
        if self.dorp_use_order_cues:
            layer_pos = torch.linspace(
                0.0, 1.0, self.num_layers,
                device=rgb.device, dtype=rgb.dtype)
        else:
            layer_pos = torch.zeros(
                self.num_layers, device=rgb.device, dtype=rgb.dtype)
        layer_pos = layer_pos.view(1, self.num_layers, 1, 1, 1).expand(rgb.size(0), -1, -1, rgb.size(2), rgb.size(3))
        drop = self._make_drop(rgb)

        feats, skips0, skips1 = [], [], []
        observed = depth * valid_mask
        for layer in range(self.num_layers):
            mask = masks[:, layer]
            guidance = torch.cat([observed * mask, anchor * mask, mask, layer_pos[:, layer]], dim=1)
            feat, s0, s1 = self.encoder(rgb * mask, guidance)
            feats.append(feat); skips0.append(s0); skips1.append(s1)

        layer_feats = torch.stack(feats, dim=1)
        if drop is not None:
            layer_feats = layer_feats * (1.0 - drop)
        layer_weights = F.interpolate(masks.flatten(0, 1), size=layer_feats.shape[-2:], mode="bilinear", align_corners=False)
        layer_weights = layer_weights.view(rgb.size(0), self.num_layers, 1, layer_feats.size(-2), layer_feats.size(-1))
        layer_unc = torch.sigmoid(self.layer_uncertainty(layer_feats.flatten(0, 1)).view(
            rgb.size(0), self.num_layers, 1, layer_feats.size(-2), layer_feats.size(-1)))
        fused = self.fusion(layer_feats, layer_weights, layer_unc, anchor)

        decoded = []
        for layer in range(self.num_layers):
            s0, s1 = skips0[layer], skips1[layer]
            if drop is not None:
                keep = 1.0 - drop[:, layer]
                s0 = s0 * keep; s1 = s1 * keep
            decoded.append(self.depth_decoder(fused[:, layer], s0, s1))
        decoded = torch.stack(decoded, dim=1)

        # Uncertainty is needed BEFORE forming the selective gate, so compute it
        # here (depends only on fused + layer_weights).
        global_feat = self._pool_global(fused, layer_weights)
        unc_logvar = F.interpolate(self.uncertainty_head(global_feat), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        uncertainty = torch.sigmoid(unc_logvar)

        support_sum = masks.sum(dim=1, keepdim=True)
        fusion_masks = masks / support_sum.clamp_min(1e-6)
        presence = support_sum.squeeze(1).clamp(0.0, 1.0)
        tanh_res = torch.tanh(decoded)
        fused_residual = (tanh_res * fusion_masks).sum(dim=1) * presence

        gate_in = torch.cat([rgb, anchor, fused_residual], dim=1)
        refine_gate = torch.sigmoid(self.refine_gate_head(gate_in))

        # Phase 2: selective reliability gate. The learned refine_gate is further
        # concentrated onto pixels that are (a) well supported by sparse depth,
        # (b) on a depth/RGB edge, or (c) high proxy uncertainty — a soft-OR.
        # Elsewhere the gate -> 0, so the proxy anchor is preserved untouched.
        edge_gate = self._edge_gate(rgb)
        if self.selective_residual:
            support_sig = presence.clamp(0.0, 1.0)
            reliability = 1.0 - (1.0 - support_sig) * (1.0 - edge_gate) * (1.0 - uncertainty)
            applied_gate = refine_gate * reliability
        else:
            applied_gate = refine_gate

        scale = self.residual_scale * (residual_warmup if self.training else 1.0)
        raw = (anchor + applied_gate * scale * fused_residual).clamp(0.0, 1.0)
        # The final checkpoint-compatible path applies the learned gate twice.
        safe = (anchor + applied_gate * (raw - anchor)).clamp(0.0, 1.0)
        per_layer = (anchor.unsqueeze(1) + applied_gate.unsqueeze(1) * scale * tanh_res).clamp(0.0, 1.0)

        if output_mode == "proxy_only":
            selected = dense_geometry
        elif output_mode in ("raw", "final"):
            selected = raw
        else:
            selected = safe
        completed = valid_mask * depth + (1.0 - valid_mask) * selected

        return {
            "completed_depth": completed,
            "raw_completed_depth": raw,
            "safe_completed_depth": safe,
            "dense_geometry": dense_geometry,
            "proxy_depth": dense_geometry,         # alias for visualization/diagnosis
            "refine_gate": refine_gate,
            "selective_gate": applied_gate,        # gate actually applied to the residual
            "candidate_weight": (reliability if self.selective_residual
                                  else torch.ones_like(refine_gate)),
            "edge_gate": edge_gate,
            "residual_map": raw - anchor,
            "residual": raw - anchor,              # alias
            "layer_depths": per_layer,
            # r11 parity: the loss consumes the SUPPORT-NORMALIZED fusion masks
            # (old network output "layer_masks": fusion_masks), not raw support.
            "layer_masks": fusion_masks,
            "hard_layer_masks": (masks > 0.5).float(),
            "layer_support": masks,
            "layer_seed_masks": support["seeds"],
            "support_total": support["support_total"],
            "unknown_support": support["unknown_support"],
            "layer_drop_mask": drop,
            "uncertainty": uncertainty,
            "uncertainty_logvar": unc_logvar,
            "refined_geometry": None, "completed_rgb": None, "layer_rgbs": None, "seg_logits": None,
        }

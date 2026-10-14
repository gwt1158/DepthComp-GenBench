"""GeoLayerNet training objectives (canonical implementation).

``GeoLayerNetLoss`` combines missing-region reconstruction, observed-depth
preservation, boundary-aware gradient consistency, layer ordinal consistency,
surface-normal consistency, and self-supervised layer reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_aware_smoothness_loss(depth, rgb, mask=None):
    dx_depth = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy_depth = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    dx_rgb = torch.mean(torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1]), dim=1, keepdim=True)
    dy_rgb = torch.mean(torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]), dim=1, keepdim=True)
    loss_x = dx_depth * torch.exp(-dx_rgb)
    loss_y = dy_depth * torch.exp(-dy_rgb)
    if mask is not None:
        loss_x = loss_x * mask[:, :, :, 1:]
        loss_y = loss_y * mask[:, :, 1:, :]
        return (loss_x.sum() + loss_y.sum()) / (mask.sum().clamp_min(1.0))
    return loss_x.mean() + loss_y.mean()


def _masked_mean_loss(loss_map, mask):
    mask = mask.float()
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def _resize_depth_like(depth, reference, mode="bilinear"):
    if depth.shape[-2:] == reference.shape[-2:]:
        return depth
    return F.interpolate(depth.float(), size=reference.shape[-2:], mode=mode, align_corners=False if mode == "bilinear" else None)


def masked_mse_depth_loss(pred_depth, target_depth, target_mask):
    pred_depth = _resize_depth_like(pred_depth, target_depth, mode="bilinear")
    target_mask = _resize_depth_like(target_mask.float(), target_depth, mode="nearest")
    sq = (pred_depth - target_depth).pow(2)
    return _masked_mean_loss(sq, target_mask)


def acm_auxiliary_depth_loss(outputs, target_depth, target_mask):
    """ACMNet-style auxiliary MSE supervision for intermediate depth outputs."""
    candidates = [
        outputs.get("raw_completed_depth"),
        outputs.get("dense_geometry"),
        outputs.get("refined_geometry"),
    ]
    losses = [masked_mse_depth_loss(pred, target_depth, target_mask) for pred in candidates if pred is not None]
    if not losses:
        return target_depth.new_tensor(0.0)
    return torch.stack(losses).mean()


def layer_depth_auxiliary_loss(outputs, target_depth, target_mask):
    layer_depths = outputs.get("layer_depths")
    layer_masks = outputs.get("hard_layer_masks", outputs.get("layer_masks"))
    if layer_depths is None or layer_masks is None:
        return target_depth.new_tensor(0.0)
    if layer_depths.shape[-2:] != target_depth.shape[-2:]:
        layer_depths = F.interpolate(
            layer_depths.flatten(0, 1).float(),
            size=target_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view(layer_depths.size(0), layer_depths.size(1), layer_depths.size(2), target_depth.size(-2), target_depth.size(-1))
    if layer_masks.shape[-2:] != target_depth.shape[-2:]:
        layer_masks = F.interpolate(
            layer_masks.flatten(0, 1).float(),
            size=target_depth.shape[-2:],
            mode="nearest",
        ).view(layer_masks.size(0), layer_masks.size(1), layer_masks.size(2), target_depth.size(-2), target_depth.size(-1))
    weights = layer_masks.detach().float() * target_mask.unsqueeze(1).float()
    sq = (layer_depths - target_depth.unsqueeze(1)).pow(2)
    return (sq * weights).sum() / weights.sum().clamp_min(1.0)


def ordinal_layer_loss(layer_depths, layer_masks, margin=0.01):
    loss = layer_depths.new_tensor(0.0)
    count = 0
    for layer in range(layer_depths.size(1) - 1):
        near = layer_depths[:, layer]
        far = layer_depths[:, layer + 1]
        overlap = torch.minimum(layer_masks[:, layer], layer_masks[:, layer + 1])
        penalty = F.relu(near - far + margin) * overlap
        loss = loss + penalty.sum() / overlap.sum().clamp_min(1.0)
        count += 1
    return loss / max(count, 1)


def strong_layer_ordinal_loss(layer_depths, layer_masks, target_mask=None, margin=0.01):
    if target_mask is None:
        target_mask = torch.ones_like(layer_depths[:, 0])
    if target_mask.shape[-2:] != layer_depths.shape[-2:]:
        target_mask = F.interpolate(target_mask.float(), size=layer_depths.shape[-2:], mode="nearest")
    if layer_masks.shape[-2:] != layer_depths.shape[-2:]:
        layer_masks = F.interpolate(layer_masks.flatten(0, 1).float(), size=layer_depths.shape[-2:], mode="nearest").view_as(layer_depths)

    weights = layer_masks.float() * target_mask.unsqueeze(1).float()
    denom = weights.sum(dim=(2, 3, 4)).clamp_min(1.0)
    means = (layer_depths * weights).sum(dim=(2, 3, 4)) / denom
    valid = denom > 1.0
    loss = layer_depths.new_tensor(0.0)
    count = 0
    for layer in range(layer_depths.size(1) - 1):
        pair_valid = valid[:, layer] & valid[:, layer + 1]
        if pair_valid.any():
            penalty = F.relu(means[:, layer] - means[:, layer + 1] + margin)
            loss = loss + penalty[pair_valid].mean()
            count += 1
    return loss / max(count, 1)


def depth_gradient_magnitude(depth):
    grad = torch.zeros_like(depth)
    grad[:, :, :, 1:] = torch.maximum(grad[:, :, :, 1:], (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs())
    grad[:, :, 1:, :] = torch.maximum(grad[:, :, 1:, :], (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs())
    return grad


def depth_boundary_mask(reference_depth, valid_mask=None, top_percent=10.0):
    top_percent = min(max(float(top_percent), 0.1), 100.0)
    if valid_mask is None:
        valid_mask = torch.ones_like(reference_depth)
    valid = valid_mask.bool()
    grad = depth_gradient_magnitude(reference_depth.detach())
    boundary = torch.zeros_like(valid)
    quantile = max(0.0, min(1.0, 1.0 - top_percent / 100.0))
    for idx in range(reference_depth.size(0)):
        values = grad[idx][valid[idx]]
        values = values[values > 0]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values.float(), quantile)
        boundary[idx] = valid[idx] & (grad[idx] >= threshold) & (grad[idx] > 0)
    return boundary.float()


def gradient_difference_loss(pred_depth, target_depth, valid_mask=None):
    """First-order gradient (finite-difference) L1 between pred and target, only
    where both the pixel and its neighbour are valid. Returns 0 if no valid pair."""
    if valid_mask is None:
        valid_mask = torch.ones_like(target_depth)
    vm = valid_mask.float()
    dx_p = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
    dy_p = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
    dx_t = target_depth[:, :, :, 1:] - target_depth[:, :, :, :-1]
    dy_t = target_depth[:, :, 1:, :] - target_depth[:, :, :-1, :]
    mx = (vm[:, :, :, 1:] * vm[:, :, :, :-1])
    my = (vm[:, :, 1:, :] * vm[:, :, :-1, :])
    if mx.sum() + my.sum() <= 0:
        return pred_depth.new_tensor(0.0)
    lx = ((dx_p - dx_t).abs() * mx).sum()
    ly = ((dy_p - dy_t).abs() * my).sum()
    return (lx + ly) / (mx.sum() + my.sum()).clamp_min(1.0)


def boundary_l1_loss(pred_depth, target_depth, valid_mask=None, top_percent=10.0, charbonnier=True):
    """L1 / Charbonnier between pred and target on the top-``top_percent``
    target-gradient (boundary) pixels. Returns 0 if too few boundary pixels."""
    boundary = depth_boundary_mask(target_depth, valid_mask, top_percent)
    if boundary.sum() <= 0:
        return pred_depth.new_tensor(0.0)
    diff = pred_depth - target_depth
    err = torch.sqrt(diff * diff + 1e-6) if charbonnier else diff.abs()
    return _masked_mean_loss(err, boundary)


def laplacian_loss(pred_depth, target_depth, valid_mask=None):
    """Second-order (Laplacian) consistency: |lap(pred) - lap(target)| on valid
    pixels. Laplacian = 4*center - 4-neighbours (via avg-pool). Returns 0 if empty."""
    if valid_mask is None:
        valid_mask = torch.ones_like(target_depth)
    vm = valid_mask.float()
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=pred_depth.device, dtype=pred_depth.dtype).view(1, 1, 3, 3)
    lap_p = F.conv2d(pred_depth, k, padding=1)
    lap_t = F.conv2d(target_depth, k, padding=1)
    if vm.sum() <= 0:
        return pred_depth.new_tensor(0.0)
    return _masked_mean_loss((lap_p - lap_t).abs(), vm)


def boundary_gradient_consistency_loss(pred_depth, target_depth, valid_mask=None, top_percent=10.0):
    boundary = depth_boundary_mask(target_depth, valid_mask, top_percent)
    dx_pred = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
    dx_target = target_depth[:, :, :, 1:] - target_depth[:, :, :, :-1]
    dy_pred = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
    dy_target = target_depth[:, :, 1:, :] - target_depth[:, :, :-1, :]
    mask_x = torch.maximum(boundary[:, :, :, 1:], boundary[:, :, :, :-1])
    mask_y = torch.maximum(boundary[:, :, 1:, :], boundary[:, :, :-1, :])
    if mask_x.sum() + mask_y.sum() <= 0:
        return pred_depth.new_tensor(0.0)
    loss_x = (dx_pred - dx_target).abs() * mask_x
    loss_y = (dy_pred - dy_target).abs() * mask_y
    return (loss_x.sum() + loss_y.sum()) / (mask_x.sum() + mask_y.sum()).clamp_min(1.0)


def nonboundary_smoothness_loss(depth, reference_depth, valid_mask=None, top_percent=10.0):
    if valid_mask is None:
        valid_mask = torch.ones_like(reference_depth)
    if valid_mask.shape[-2:] != depth.shape[-2:]:
        valid_mask = F.interpolate(valid_mask.float(), size=depth.shape[-2:], mode="nearest")
    if reference_depth.shape[-2:] != depth.shape[-2:]:
        reference_depth = F.interpolate(reference_depth, size=depth.shape[-2:], mode="bilinear", align_corners=False)
    boundary = depth_boundary_mask(reference_depth, valid_mask, top_percent)
    nonboundary = valid_mask.float() * (1.0 - boundary.float())
    dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    mask_x = torch.minimum(nonboundary[:, :, :, 1:], nonboundary[:, :, :, :-1])
    mask_y = torch.minimum(nonboundary[:, :, 1:, :], nonboundary[:, :, :-1, :])
    if mask_x.sum() + mask_y.sum() <= 0:
        return depth.new_tensor(0.0)
    return (dx * mask_x).sum().add((dy * mask_y).sum()) / (mask_x.sum() + mask_y.sum()).clamp_min(1.0)


def layer_self_reconstruction_loss(outputs, target_depth, target_rgb=None, target_mask=None):
    layer_drop_mask = outputs.get("layer_drop_mask")
    layer_depths = outputs.get("layer_depths")
    layer_masks = outputs.get("layer_masks")
    if layer_drop_mask is None or layer_depths is None or layer_masks is None:
        zero = target_depth.new_tensor(0.0)
        return zero, zero
    if target_mask is None:
        target_mask = torch.ones_like(target_depth)
    if layer_masks.shape[-2:] != layer_depths.shape[-2:]:
        layer_masks = F.interpolate(layer_masks.flatten(0, 1).float(), size=layer_depths.shape[-2:], mode="nearest").view_as(layer_depths)
    if target_depth.shape[-2:] != layer_depths.shape[-2:]:
        target_depth = F.interpolate(target_depth, size=layer_depths.shape[-2:], mode="bilinear", align_corners=False)
    if target_mask.shape[-2:] != layer_depths.shape[-2:]:
        target_mask = F.interpolate(target_mask.float(), size=layer_depths.shape[-2:], mode="nearest")

    region = layer_drop_mask.to(layer_depths.dtype) * layer_masks.float() * target_mask.unsqueeze(1).float()
    if region.sum() <= 0:
        zero = target_depth.new_tensor(0.0)
        return zero, zero
    depth_target = target_depth.unsqueeze(1)
    loss_depth = (torch.abs(layer_depths - depth_target) * region).sum() / region.sum().clamp_min(1.0)

    loss_rgb = target_depth.new_tensor(0.0)
    layer_rgbs = outputs.get("layer_rgbs")
    if target_rgb is not None and layer_rgbs is not None:
        if target_rgb.shape[-2:] != layer_rgbs.shape[-2:]:
            target_rgb = F.interpolate(target_rgb, size=layer_rgbs.shape[-2:], mode="bilinear", align_corners=False)
        rgb_region = region
        if rgb_region.shape[-2:] != layer_rgbs.shape[-2:]:
            rgb_region = F.interpolate(rgb_region.flatten(0, 1).float(), size=layer_rgbs.shape[-2:], mode="nearest").view(layer_rgbs.size(0), layer_rgbs.size(1), 1, layer_rgbs.size(-2), layer_rgbs.size(-1))
        rgb_error = torch.abs(layer_rgbs - target_rgb.unsqueeze(1)) * rgb_region
        loss_rgb = rgb_error.sum() / (rgb_region.sum().clamp_min(1.0) * layer_rgbs.size(2))
    return loss_depth, loss_rgb


def surface_normal_loss(pred_depth, target_depth, mask=None, eps=1e-6):
    pred_normal = depth_to_normals(pred_depth, eps)
    target_normal = depth_to_normals(target_depth, eps)
    cosine = (pred_normal * target_normal).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss = 1.0 - cosine
    if mask is not None:
        valid = mask[:, :, 1:-1, 1:-1]
        loss = loss * valid
        return loss.sum() / valid.sum().clamp_min(1.0)
    return loss.mean()


def dilate_mask(mask, radius):
    """Binary/float mask dilation via max-pooling."""
    if radius <= 0:
        return mask
    return F.max_pool2d(mask.float(), kernel_size=2 * int(radius) + 1, stride=1, padding=int(radius))


def depth_to_normals(depth, eps=1e-6):
    dzdx = depth[:, :, 1:-1, 2:] - depth[:, :, 1:-1, :-2]
    dzdy = depth[:, :, 2:, 1:-1] - depth[:, :, :-2, 1:-1]
    nx = -dzdx
    ny = -dzdy
    nz = torch.ones_like(nx)
    normal = torch.cat([nx, ny, nz], dim=1)
    return F.normalize(normal, dim=1, eps=eps)


def _high_pass(depth, kernel=5):
    """High-frequency part of a depth map: x - lowpass(x) (avg-pool low-pass)."""
    k = int(kernel) | 1
    low = F.avg_pool2d(depth, kernel_size=k, stride=1, padding=k // 2)
    return depth - low


def align_scale_shift(src, ref, mask, eps=1e-6):
    """Per-image affine alignment of ``src`` to ``ref`` (least squares over mask).

    Used to remove a foundation-depth teacher's global scale/shift ambiguity so
    that only its STRUCTURE (high-freq / gradient / normal) is distilled, never
    its absolute depth. Returns ``s*src + t`` (caller decides whether to detach).
    """
    b = src.size(0)
    m = mask.float()
    msum = m.flatten(1).sum(1).clamp_min(1.0)
    sx = (src * m).flatten(1).sum(1) / msum
    sy = (ref * m).flatten(1).sum(1) / msum
    sxx = (src * src * m).flatten(1).sum(1) / msum
    sxy = (src * ref * m).flatten(1).sum(1) / msum
    var = (sxx - sx * sx).clamp_min(eps)
    s = ((sxy - sx * sy) / var).view(b, 1, 1, 1)
    t = (sy - s.view(b) * sx).view(b, 1, 1, 1)
    return s * src + t


def teacher_distillation_losses(pred_depth, teacher_depth, teacher_mask, hf_kernel=5):
    """High-frequency / gradient / surface-normal distillation from a (scale/shift
    aligned, detached) foundation-model teacher. Returns (hf, grad, normal); all
    zero when the teacher mask is empty. ``teacher_depth`` is used only here, never
    in the model forward."""
    zero = pred_depth.new_tensor(0.0)
    if teacher_depth is None:
        return zero, zero, zero
    if teacher_depth.shape[-2:] != pred_depth.shape[-2:]:
        teacher_depth = F.interpolate(teacher_depth, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
    if teacher_mask is None:
        teacher_mask = (teacher_depth > 0).float()
    elif teacher_mask.shape[-2:] != pred_depth.shape[-2:]:
        teacher_mask = F.interpolate(teacher_mask.float(), size=pred_depth.shape[-2:], mode="nearest")
    if teacher_mask.sum() <= 0:
        return zero, zero, zero
    aligned = align_scale_shift(teacher_depth, pred_depth.detach(), teacher_mask).detach()

    hf = _masked_mean_loss((_high_pass(pred_depth, hf_kernel) - _high_pass(aligned, hf_kernel)).abs(), teacher_mask)
    grad = _masked_mean_loss((depth_gradient_magnitude(pred_depth) - depth_gradient_magnitude(aligned)).abs(), teacher_mask)
    normal = surface_normal_loss(pred_depth, aligned, teacher_mask)
    return hf, grad, normal


class GeoLayerNetLoss(nn.Module):
    def __init__(
        self,
        lambda_depth=100.0,
        lambda_final_depth=None,
        lambda_raw_depth=100.0,
        lambda_improve=20.0,
        improve_margin=0.0,
        lambda_residual_reg=10.0,
        lambda_gate_sparsity=0.01,
        lambda_layer_balance=0.01,
        lambda_edge_raw=5.0,
        lambda_sparse_layer_local_smooth=0.1,
        lambda_sparse_layer_area=0.01,
        lambda_residual_nonedge_tv=0.05,
        residual_tv_boundary_dilate=3,
        lambda_gate_nonedge=0.01,
        lambda_edge_band_raw=5.0,
        edge_band_radius=3,
        lambda_residual_nonedge_laplacian=0.0,
        lambda_proxy_nonedge_tv=0.0,
        lambda_depth_l2=0.0,
        lambda_relative_depth=0.0,
        lambda_rgb=10.0,
        lambda_smooth=1.0,
        lambda_nonboundary_smooth=0.0,
        lambda_ordinal=1.0,
        lambda_uncertainty=1.0,
        lambda_task=0.0,
        lambda_normal=0.2,
        lambda_proxy=5.0,
        lambda_proxy_nonboundary_smooth=0.0,
        lambda_geometry_refine_smooth=0.0,
        lambda_observed=10.0,
        lambda_boundary=1.0,
        lambda_missing=1.0,
        lambda_observed_region=0.2,
        lambda_layer_ordinal_strong=0.5,
        lambda_layer_reconstruction=1.0,
        lambda_layer_rgb_reconstruction=0.0,
        lambda_acm_aux=0.0,
        lambda_layer_depth_aux=0.0,
        lambda_teacher_hf=0.0,
        lambda_teacher_grad=0.0,
        lambda_teacher_normal=0.0,
        teacher_hf_kernel=5,
        # optional sharpness supervision on the FINAL completed depth (default OFF).
        grad_loss_weight=0.0,
        boundary_l1_weight=0.0,
        lap_loss_weight=0.0,
        use_uncertainty_depth_loss=False,
        boundary_top_percent=10.0,
        ignore_index=255,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        # lambda_final_depth weights the completed-depth loss; falls back to the
        # legacy lambda_depth when not given.
        self.lambda_final_depth = lambda_depth if lambda_final_depth is None else lambda_final_depth
        self.lambda_raw_depth = lambda_raw_depth
        self.lambda_improve = lambda_improve
        self.improve_margin = improve_margin
        self.lambda_residual_reg = lambda_residual_reg
        self.lambda_gate_sparsity = lambda_gate_sparsity
        self.lambda_layer_balance = lambda_layer_balance
        self.lambda_edge_raw = lambda_edge_raw
        self.lambda_sparse_layer_local_smooth = lambda_sparse_layer_local_smooth
        self.lambda_sparse_layer_area = lambda_sparse_layer_area
        self.lambda_residual_nonedge_tv = lambda_residual_nonedge_tv
        self.residual_tv_boundary_dilate = residual_tv_boundary_dilate
        self.lambda_gate_nonedge = lambda_gate_nonedge
        self.lambda_edge_band_raw = lambda_edge_band_raw
        self.edge_band_radius = edge_band_radius
        self.lambda_residual_nonedge_laplacian = lambda_residual_nonedge_laplacian
        self.lambda_proxy_nonedge_tv = lambda_proxy_nonedge_tv
        self.lambda_depth_l2 = lambda_depth_l2
        self.lambda_relative_depth = lambda_relative_depth
        self.lambda_rgb = lambda_rgb
        self.lambda_smooth = lambda_smooth
        self.lambda_nonboundary_smooth = lambda_nonboundary_smooth
        self.lambda_ordinal = lambda_ordinal
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_task = lambda_task
        self.lambda_normal = lambda_normal
        self.lambda_proxy = lambda_proxy
        self.lambda_proxy_nonboundary_smooth = lambda_proxy_nonboundary_smooth
        self.lambda_geometry_refine_smooth = lambda_geometry_refine_smooth
        self.lambda_observed = lambda_observed
        self.lambda_boundary = lambda_boundary
        self.lambda_missing = lambda_missing
        self.lambda_observed_region = lambda_observed_region
        self.lambda_layer_ordinal_strong = lambda_layer_ordinal_strong
        self.lambda_layer_reconstruction = lambda_layer_reconstruction
        self.lambda_layer_rgb_reconstruction = lambda_layer_rgb_reconstruction
        self.lambda_acm_aux = lambda_acm_aux
        self.lambda_layer_depth_aux = lambda_layer_depth_aux
        self.lambda_teacher_hf = lambda_teacher_hf
        self.lambda_teacher_grad = lambda_teacher_grad
        self.lambda_teacher_normal = lambda_teacher_normal
        self.teacher_hf_kernel = teacher_hf_kernel
        self.grad_loss_weight = grad_loss_weight
        self.boundary_l1_weight = boundary_l1_weight
        self.lap_loss_weight = lap_loss_weight
        self.use_uncertainty_depth_loss = use_uncertainty_depth_loss
        self.boundary_top_percent = boundary_top_percent
        self.ignore_index = ignore_index

    def forward(self, outputs, rgb, sparse_depth, target_depth, target_rgb=None, valid_mask=None, target_mask=None, semantic_label=None,
                teacher_depth=None, teacher_mask=None):
        if target_mask is None:
            target_mask = torch.ones_like(target_depth)
        if valid_mask is None:
            valid_mask = (sparse_depth > 0).float()
        missing_mask = (1.0 - valid_mask).clamp(0, 1)
        missing_supervise = missing_mask * target_mask
        if missing_supervise.sum() <= 0:
            missing_supervise = target_mask
        observed_supervise = valid_mask * target_mask
        depth_abs = torch.abs(outputs["completed_depth"] - target_depth)
        loss_missing = _masked_mean_loss(depth_abs, missing_supervise)
        loss_observed_region = _masked_mean_loss(depth_abs, observed_supervise) if observed_supervise.sum() > 0 else rgb.new_tensor(0.0)
        supervise_mask = self.lambda_missing * missing_supervise + self.lambda_observed_region * observed_supervise
        if supervise_mask.sum() <= 0:
            supervise_mask = missing_supervise
        depth_l1 = depth_abs * supervise_mask
        loss_depth = depth_l1.sum() / supervise_mask.sum().clamp_min(1.0)
        depth_sq = depth_abs.pow(2)
        loss_depth_l2 = (depth_sq * supervise_mask).sum() / supervise_mask.sum().clamp_min(1.0)
        rel_map = depth_abs / target_depth.clamp_min(1e-3)
        loss_relative_depth = (rel_map * supervise_mask).sum() / supervise_mask.sum().clamp_min(1.0)
        loss_smooth = edge_aware_smoothness_loss(outputs["completed_depth"], rgb, target_mask) if self.lambda_smooth > 0 else rgb.new_tensor(0.0)
        loss_nonboundary_smooth = nonboundary_smoothness_loss(outputs["completed_depth"], target_depth, target_mask, self.boundary_top_percent) if self.lambda_nonboundary_smooth > 0 else rgb.new_tensor(0.0)
        loss_ordinal = ordinal_layer_loss(outputs["layer_depths"], outputs["layer_masks"]) if self.lambda_ordinal > 0 else rgb.new_tensor(0.0)
        loss_layer_ordinal_strong = strong_layer_ordinal_loss(outputs["layer_depths"], outputs["layer_masks"], target_mask) if self.lambda_layer_ordinal_strong > 0 else rgb.new_tensor(0.0)
        loss_boundary = boundary_gradient_consistency_loss(outputs["completed_depth"], target_depth, target_mask, self.boundary_top_percent) if self.lambda_boundary > 0 else rgb.new_tensor(0.0)
        loss_normal = surface_normal_loss(outputs["completed_depth"], target_depth, target_mask) if self.lambda_normal > 0 else rgb.new_tensor(0.0)
        if self.lambda_layer_reconstruction > 0 or self.lambda_layer_rgb_reconstruction > 0:
            loss_layer_reconstruction, loss_layer_rgb_reconstruction = layer_self_reconstruction_loss(outputs, target_depth, target_rgb, target_mask)
        else:
            loss_layer_reconstruction = rgb.new_tensor(0.0)
            loss_layer_rgb_reconstruction = rgb.new_tensor(0.0)
        loss_acm_aux = acm_auxiliary_depth_loss(outputs, target_depth, target_mask) if self.lambda_acm_aux > 0 else rgb.new_tensor(0.0)
        loss_layer_depth_aux = layer_depth_auxiliary_loss(outputs, target_depth, target_mask) if self.lambda_layer_depth_aux > 0 else rgb.new_tensor(0.0)
        dense_geometry = outputs.get("dense_geometry")
        loss_proxy = rgb.new_tensor(0.0)
        loss_proxy_nonboundary_smooth = rgb.new_tensor(0.0)
        loss_geometry_refine_smooth = rgb.new_tensor(0.0)
        if self.lambda_proxy > 0 and dense_geometry is not None:
            proxy_l1 = torch.abs(dense_geometry - target_depth) * target_mask
            loss_proxy = proxy_l1.sum() / target_mask.sum().clamp_min(1.0)
        if self.lambda_proxy_nonboundary_smooth > 0 and dense_geometry is not None:
            loss_proxy_nonboundary_smooth = nonboundary_smoothness_loss(dense_geometry, target_depth, target_mask, self.boundary_top_percent)
        refined_geometry = outputs.get("refined_geometry")
        if self.lambda_geometry_refine_smooth > 0 and refined_geometry is not None:
            loss_geometry_refine_smooth = nonboundary_smoothness_loss(refined_geometry, target_depth, target_mask, self.boundary_top_percent)
        raw_completed = outputs.get("raw_completed_depth")
        proxy_depth = outputs.get("dense_geometry")
        # Dense supervision of the raw prediction itself (missing-weighted, same
        # mask as the main depth loss). The main loss on completed_depth is
        # vacuous on observed pixels when preserve_observed_depth copies them
        # back, so the raw prediction needs its own dense target supervision.
        loss_raw_depth = rgb.new_tensor(0.0)
        raw_abs = None
        if raw_completed is not None:
            raw_abs = torch.abs(raw_completed - target_depth)
            if self.lambda_raw_depth > 0:
                loss_raw_depth = (raw_abs * supervise_mask).sum() / supervise_mask.sum().clamp_min(1.0)

        # ── refinement-safety terms ──────────────────────────────────────────
        # improve hinge: raw must not be worse than the (detached) proxy on the
        # missing region; the proxy error is detached so the hinge can only pull
        # raw toward improvement, never degrade the proxy to "win".
        loss_improve = rgb.new_tensor(0.0)
        loss_residual_reg = rgb.new_tensor(0.0)
        raw_minus_proxy = rgb.new_tensor(0.0)
        if raw_completed is not None and proxy_depth is not None:
            proxy_abs = torch.abs(proxy_depth - target_depth).detach()
            if self.lambda_improve > 0:
                hinge = torch.relu(raw_abs - proxy_abs + self.improve_margin) * missing_supervise
                loss_improve = hinge.sum() / missing_supervise.sum().clamp_min(1.0)
            # Charbonnier residual regularization: raw must stay near the
            # (detached) proxy unless supervision clearly favors a deviation.
            residual = raw_completed - proxy_depth.detach()
            if self.lambda_residual_reg > 0:
                charbonnier = torch.sqrt(residual.pow(2) + 1e-6) * missing_supervise
                loss_residual_reg = charbonnier.sum() / missing_supervise.sum().clamp_min(1.0)
            raw_minus_proxy = (residual.abs() * missing_supervise).sum() / missing_supervise.sum().clamp_min(1.0)

        # gate sparsity: keep refinement localized instead of repainting the image.
        refine_gate = outputs.get("refine_gate")
        loss_gate_sparsity = rgb.new_tensor(0.0)
        mean_gate = rgb.new_tensor(0.0)
        if refine_gate is not None:
            mean_gate = refine_gate.mean()
            if self.lambda_gate_sparsity > 0:
                loss_gate_sparsity = mean_gate

        # layer balance: KL(coverage || uniform) discourages layer collapse.
        loss_layer_balance = rgb.new_tensor(0.0)
        layer_masks_for_balance = outputs.get("layer_masks")
        if self.lambda_layer_balance > 0 and layer_masks_for_balance is not None and layer_masks_for_balance.size(1) > 1:
            coverage = layer_masks_for_balance.mean(dim=(2, 3, 4)).clamp_min(1e-8)  # B x K
            coverage = coverage / coverage.sum(dim=1, keepdim=True)
            k = coverage.size(1)
            loss_layer_balance = (coverage * (coverage * k).log()).sum(dim=1).mean()

        # Boundary-aware terms share one target-depth boundary mask.
        need_boundary = (self.lambda_edge_raw > 0 or self.lambda_edge_band_raw > 0
                         or self.lambda_residual_nonedge_tv > 0 or self.lambda_gate_nonedge > 0
                         or self.lambda_residual_nonedge_laplacian > 0 or self.lambda_proxy_nonedge_tv > 0)
        boundary_f = None
        if need_boundary:
            boundary_f = depth_boundary_mask(target_depth, target_mask, self.boundary_top_percent).float()

        # edge-aware raw loss: upweight raw supervision near target-depth edges.
        loss_edge_raw = rgb.new_tensor(0.0)
        if self.lambda_edge_raw > 0 and raw_completed is not None and boundary_f is not None:
            edge_mask = boundary_f * target_mask
            if edge_mask.sum() > 0:
                loss_edge_raw = (raw_abs * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)

        # dilated boundary-band raw loss: the thin boundary mask misses the
        # transition pixels where averaging artifacts live, so supervise raw on
        # a dilated band as well.
        loss_edge_band_raw = rgb.new_tensor(0.0)
        if self.lambda_edge_band_raw > 0 and raw_completed is not None and boundary_f is not None:
            edge_band = dilate_mask(boundary_f, self.edge_band_radius) * target_mask
            if edge_band.sum() > 0:
                loss_edge_band_raw = (raw_abs * edge_band).sum() / edge_band.sum().clamp_min(1.0)

        # non-edge residual TV: speckle/texture-leak suppression — the residual
        # must be piecewise-flat AWAY from depth boundaries (weighted by
        # support_total when available); boundaries remain free to change.
        loss_nonedge_tv = rgb.new_tensor(0.0)
        residual_for_tv = outputs.get("residual_map")
        nonedge = None
        if boundary_f is not None and (self.lambda_residual_nonedge_tv > 0 or self.lambda_gate_nonedge > 0
                                       or self.lambda_residual_nonedge_laplacian > 0 or self.lambda_proxy_nonedge_tv > 0):
            nonedge = (1.0 - dilate_mask(boundary_f, self.residual_tv_boundary_dilate)).clamp(0.0, 1.0) * target_mask
        if self.lambda_residual_nonedge_tv > 0 and residual_for_tv is not None and nonedge is not None:
            tv_weight = nonedge
            support_total_for_tv = outputs.get("support_total")
            if support_total_for_tv is not None:
                tv_weight = tv_weight * support_total_for_tv
            dx_tv = (residual_for_tv[:, :, :, 1:] - residual_for_tv[:, :, :, :-1]).abs()
            dy_tv = (residual_for_tv[:, :, 1:, :] - residual_for_tv[:, :, :-1, :]).abs()
            wx_tv = tv_weight[:, :, :, 1:]
            wy_tv = tv_weight[:, :, 1:, :]
            loss_nonedge_tv = ((dx_tv * wx_tv).sum() + (dy_tv * wy_tv).sum()) / (wx_tv.sum() + wy_tv.sum()).clamp_min(1.0)

        # non-edge gate sparsity: the gate must not light up across large smooth
        # regions; corrections belong near boundaries / supported evidence.
        loss_gate_nonedge = rgb.new_tensor(0.0)
        if self.lambda_gate_nonedge > 0 and refine_gate is not None and nonedge is not None and nonedge.sum() > 0:
            loss_gate_nonedge = (refine_gate * nonedge).sum() / nonedge.sum().clamp_min(1.0)

        # non-edge residual Laplacian: second-order penalty |R - avgpool3(R)|
        # kills checkerboard patterns that first-order TV tolerates.
        loss_nonedge_lap = rgb.new_tensor(0.0)
        if self.lambda_residual_nonedge_laplacian > 0 and residual_for_tv is not None and nonedge is not None and nonedge.sum() > 0:
            local_mean = F.avg_pool2d(residual_for_tv, kernel_size=3, stride=1, padding=1)
            lap = (residual_for_tv - local_mean).abs()
            loss_nonedge_lap = (lap * nonedge).sum() / nonedge.sum().clamp_min(1.0)

        # proxy non-edge TV: the dense geometry proxy itself must be smooth away
        # from boundaries so checkerboard cannot enter through the anchor.
        loss_proxy_nonedge_tv = rgb.new_tensor(0.0)
        if self.lambda_proxy_nonedge_tv > 0 and proxy_depth is not None and nonedge is not None:
            pdx = (proxy_depth[:, :, :, 1:] - proxy_depth[:, :, :, :-1]).abs()
            pdy = (proxy_depth[:, :, 1:, :] - proxy_depth[:, :, :-1, :]).abs()
            pwx = nonedge[:, :, :, 1:]
            pwy = nonedge[:, :, 1:, :]
            loss_proxy_nonedge_tv = ((pdx * pwx).sum() + (pdy * pwy).sum()) / (pwx.sum() + pwy.sum()).clamp_min(1.0)

        # sparse-layer local consistency: residual should be smooth INSIDE
        # supported regions but free to change across RGB edges.
        loss_sparse_local_smooth = rgb.new_tensor(0.0)
        loss_sparse_area = rgb.new_tensor(0.0)
        support_total = outputs.get("support_total")
        residual_for_smooth = outputs.get("residual_map")
        if self.lambda_sparse_layer_local_smooth > 0 and support_total is not None and residual_for_smooth is not None:
            dx_r = (residual_for_smooth[:, :, :, 1:] - residual_for_smooth[:, :, :, :-1]).abs()
            dy_r = (residual_for_smooth[:, :, 1:, :] - residual_for_smooth[:, :, :-1, :]).abs()
            dx_rgb = torch.mean((rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs(), dim=1, keepdim=True)
            dy_rgb = torch.mean((rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs(), dim=1, keepdim=True)
            wx = support_total[:, :, :, 1:] * torch.exp(-dx_rgb * 10.0)
            wy = support_total[:, :, 1:, :] * torch.exp(-dy_rgb * 10.0)
            loss_sparse_local_smooth = ((dx_r * wx).sum() + (dy_r * wy).sum()) / (wx.sum() + wy.sum()).clamp_min(1.0)
        # area/confinement: penalize the refinement gate firing OUTSIDE the
        # supported regions (the expansion itself is parameter-free, so the
        # learnable confinement target is the gate).
        unknown_support = outputs.get("unknown_support")
        if self.lambda_sparse_layer_area > 0 and unknown_support is not None and refine_gate is not None:
            loss_sparse_area = (refine_gate * unknown_support).sum() / unknown_support.sum().clamp_min(1.0)

        loss_observed = rgb.new_tensor(0.0)
        if self.lambda_observed > 0 and raw_completed is not None:
            raw_observed_l1 = torch.abs(raw_completed - sparse_depth) * valid_mask
            final_observed_l1 = torch.abs(outputs["completed_depth"] - sparse_depth) * valid_mask
            loss_observed = (raw_observed_l1.sum() + final_observed_l1.sum()) / valid_mask.sum().clamp_min(1.0)
        loss_rgb = rgb.new_tensor(0.0)
        if self.lambda_rgb > 0 and target_rgb is not None and outputs.get("completed_rgb") is not None:
            loss_rgb = torch.mean(torch.abs(outputs["completed_rgb"] - target_rgb))
        loss_unc = rgb.new_tensor(0.0)
        uncertainty_logvar = outputs.get("uncertainty_logvar")
        if uncertainty_logvar is not None and self.use_uncertainty_depth_loss:
            if uncertainty_logvar.shape[-2:] != target_depth.shape[-2:]:
                uncertainty_logvar = F.interpolate(uncertainty_logvar, size=target_depth.shape[-2:], mode="bilinear", align_corners=False)
            uncertainty_logvar = uncertainty_logvar.clamp(-10.0, 10.0)
            valid = supervise_mask.clamp_min(0.0)
            kendall_depth = (depth_l1 * torch.exp(-uncertainty_logvar) + self.lambda_uncertainty * uncertainty_logvar * valid)
            loss_depth = kendall_depth.sum() / valid.sum().clamp_min(1.0)
            loss_unc = (uncertainty_logvar * valid).sum() / valid.sum().clamp_min(1.0)
        loss_task = rgb.new_tensor(0.0)
        if semantic_label is not None and outputs.get("seg_logits") is not None:
            loss_task = F.cross_entropy(outputs["seg_logits"], semantic_label.long(), ignore_index=self.ignore_index)

        # ── high-frequency teacher distillation (Phase 5) ────────────────────
        # teacher_depth is a foundation-model depth prior used ONLY here. It is
        # scale/shift-aligned then detached, and only its structure is distilled
        # into the completed depth — never absolute depth, never the forward pass.
        loss_teacher_hf = rgb.new_tensor(0.0)
        loss_teacher_grad = rgb.new_tensor(0.0)
        loss_teacher_normal = rgb.new_tensor(0.0)
        if teacher_depth is not None and (self.lambda_teacher_hf > 0 or self.lambda_teacher_grad > 0 or self.lambda_teacher_normal > 0):
            loss_teacher_hf, loss_teacher_grad, loss_teacher_normal = teacher_distillation_losses(
                outputs["completed_depth"], teacher_depth, teacher_mask, self.teacher_hf_kernel)

        # ── optional sharpness supervision on the FINAL completed depth ───────
        # (default weights 0 -> fully backward compatible; proxy loss untouched).
        final_for_sharp = outputs["completed_depth"]
        loss_grad = rgb.new_tensor(0.0)
        loss_boundary_l1 = rgb.new_tensor(0.0)
        loss_lap = rgb.new_tensor(0.0)
        if self.grad_loss_weight > 0:
            loss_grad = gradient_difference_loss(final_for_sharp, target_depth, target_mask)
        if self.boundary_l1_weight > 0:
            loss_boundary_l1 = boundary_l1_loss(final_for_sharp, target_depth, target_mask, self.boundary_top_percent)
        if self.lap_loss_weight > 0:
            loss_lap = laplacian_loss(final_for_sharp, target_depth, target_mask)

        total = (
            self.lambda_final_depth * loss_depth
            + self.lambda_raw_depth * loss_raw_depth
            + self.lambda_improve * loss_improve
            + self.lambda_residual_reg * loss_residual_reg
            + self.lambda_gate_sparsity * loss_gate_sparsity
            + self.lambda_layer_balance * loss_layer_balance
            + self.lambda_edge_raw * loss_edge_raw
            + self.lambda_sparse_layer_local_smooth * loss_sparse_local_smooth
            + self.lambda_sparse_layer_area * loss_sparse_area
            + self.lambda_residual_nonedge_tv * loss_nonedge_tv
            + self.lambda_gate_nonedge * loss_gate_nonedge
            + self.lambda_edge_band_raw * loss_edge_band_raw
            + self.lambda_residual_nonedge_laplacian * loss_nonedge_lap
            + self.lambda_proxy_nonedge_tv * loss_proxy_nonedge_tv
            + self.lambda_depth_l2 * loss_depth_l2
            + self.lambda_relative_depth * loss_relative_depth
            + self.lambda_rgb * loss_rgb
            + self.lambda_smooth * loss_smooth
            + self.lambda_nonboundary_smooth * loss_nonboundary_smooth
            + self.lambda_ordinal * loss_ordinal
            + self.lambda_task * loss_task
            + self.lambda_normal * loss_normal
            + self.lambda_proxy * loss_proxy
            + self.lambda_proxy_nonboundary_smooth * loss_proxy_nonboundary_smooth
            + self.lambda_geometry_refine_smooth * loss_geometry_refine_smooth
            + self.lambda_observed * loss_observed
            + self.lambda_boundary * loss_boundary
            + self.lambda_layer_ordinal_strong * loss_layer_ordinal_strong
            + self.lambda_layer_reconstruction * loss_layer_reconstruction
            + self.lambda_layer_rgb_reconstruction * loss_layer_rgb_reconstruction
            + self.lambda_acm_aux * loss_acm_aux
            + self.lambda_layer_depth_aux * loss_layer_depth_aux
            + self.lambda_teacher_hf * loss_teacher_hf
            + self.lambda_teacher_grad * loss_teacher_grad
            + self.lambda_teacher_normal * loss_teacher_normal
            + self.grad_loss_weight * loss_grad
            + self.boundary_l1_weight * loss_boundary_l1
            + self.lap_loss_weight * loss_lap
        )
        return {
            "total": total,
            "depth": loss_depth,
            "final": loss_depth,
            "raw_depth": loss_raw_depth,
            "improve": loss_improve,
            "residual_reg": loss_residual_reg,
            "gate_sparsity": loss_gate_sparsity,
            "layer_balance": loss_layer_balance,
            "edge_raw": loss_edge_raw,
            "sparse_local_smooth": loss_sparse_local_smooth,
            "sparse_area": loss_sparse_area,
            "nonedge_tv": loss_nonedge_tv,
            "gate_nonedge": loss_gate_nonedge,
            "edge_band_raw": loss_edge_band_raw,
            "nonedge_lap": loss_nonedge_lap,
            "proxy_nonedge_tv": loss_proxy_nonedge_tv,
            "raw_minus_proxy": raw_minus_proxy,
            "mean_gate": mean_gate,
            "missing": loss_missing,
            "observed_region": loss_observed_region,
            "depth_l2": loss_depth_l2,
            "relative_depth": loss_relative_depth,
            "rgb": loss_rgb,
            "smooth": loss_smooth,
            "nonboundary_smooth": loss_nonboundary_smooth,
            "ordinal": loss_ordinal,
            "layer_ordinal_strong": loss_layer_ordinal_strong,
            "layer_reconstruction": loss_layer_reconstruction,
            "layer_rgb_reconstruction": loss_layer_rgb_reconstruction,
            "acm_aux": loss_acm_aux,
            "layer_depth_aux": loss_layer_depth_aux,
            "teacher_hf": loss_teacher_hf,
            "teacher_grad": loss_teacher_grad,
            "teacher_normal": loss_teacher_normal,
            "grad_loss": loss_grad,
            "boundary_l1": loss_boundary_l1,
            "lap_loss": loss_lap,
            "boundary": loss_boundary,
            "unc": loss_unc,
            "task": loss_task,
            "normal": loss_normal,
            "proxy": loss_proxy,
            "proxy_nonboundary_smooth": loss_proxy_nonboundary_smooth,
            "geometry_refine_smooth": loss_geometry_refine_smooth,
            "observed": loss_observed,
        }


# backward-compat alias

"""Depth metrics (identical computation to the original eval) + model stats.

RunningMetric is copied verbatim so numbers match. DepthEvaluator aggregates the
global + region (missing/observed) + diagnostic (proxy/raw/safe) metrics and
applies the NYU meter scaling. model_stats() reports Params / FLOPs / FPS.
"""

import time
from collections import defaultdict

import torch
import torch.nn.functional as F


def _metric_suffix(value):
    text = ("%f" % float(value)).rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


class RunningMetric:
    def __init__(self, eps=1e-6, bad_thresholds=None):
        self.eps = eps
        self.bad_thresholds = bad_thresholds or []
        self.sums = defaultdict(float)
        self.count = 0.0

    def update_depth(self, pred, target, mask):
        mask = mask.bool()
        if mask.sum() == 0:
            return
        p = pred[mask].clamp_min(self.eps)
        t = target[mask].clamp_min(self.eps)
        diff = p - t
        abs_diff = diff.abs()
        sq = diff.pow(2)
        inv_diff = (1.0 / p - 1.0 / t)
        log_diff = torch.log(p) - torch.log(t)
        ratio = torch.maximum(p / t, t / p)
        n = float(mask.sum().item())
        self.sums["rmse_sq"] += float(sq.sum().item())
        self.sums["mae"] += float(abs_diff.sum().item())
        self.sums["irmse_sq"] += float(inv_diff.pow(2).sum().item())
        self.sums["imae"] += float(inv_diff.abs().sum().item())
        self.sums["absrel"] += float((abs_diff / t).sum().item())
        self.sums["sqrel"] += float((sq / t).sum().item())
        self.sums["log_sq"] += float(log_diff.pow(2).sum().item())
        self.sums["log"] += float(log_diff.sum().item())
        self.sums["log10"] += float((torch.log10(p) - torch.log10(t)).abs().sum().item())
        self.sums["bias"] += float(diff.sum().item())
        self.sums["delta1"] += float((ratio < 1.25).float().sum().item())
        self.sums["delta2"] += float((ratio < 1.25 ** 2).float().sum().item())
        self.sums["delta3"] += float((ratio < 1.25 ** 3).float().sum().item())
        self.count += n

    def compute(self):
        if self.count <= 0:
            return {k: float("nan") for k in ("RMSE", "MAE", "AbsRel", "delta1")} | {"pixels": 0.0}
        d = max(self.count, 1.0)
        return {
            "RMSE": (self.sums["rmse_sq"] / d) ** 0.5, "MAE": self.sums["mae"] / d,
            "AbsRel": self.sums["absrel"] / d, "Log10": self.sums["log10"] / d,
            "iRMSE": (self.sums["irmse_sq"] / d) ** 0.5, "iMAE": self.sums["imae"] / d,
            "delta1": self.sums["delta1"] / d, "delta2": self.sums["delta2"] / d,
            "delta3": self.sums["delta3"] / d, "pixels": self.count,
        }


def _resize_like(x, ref):
    if x is None or x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x.float(), size=ref.shape[-2:], mode="nearest")


def _bool_mask_like(mask, ref, name):
    """Move a mask to ``ref`` and allow only explicit, safe broadcasting."""
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=ref.device)
    else:
        mask = mask.to(device=ref.device)
    if mask.ndim == ref.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != ref.ndim:
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} is incompatible with "
            f"target shape {tuple(ref.shape)}")
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError(f"{name} must contain only finite values")
    try:
        mask = torch.broadcast_to(mask, ref.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} is incompatible with "
            f"target shape {tuple(ref.shape)}") from exc
    return mask.bool()


class DepthEvaluator:
    def __init__(self, depth_unit_scale=10.0, eval_clip_prediction=True,
                 eval_min_depth=1e-3, eval_max_depth=1.0):
        self.scale = float(depth_unit_scale)
        self.clip = bool(eval_clip_prediction)
        self.lo, self.hi = float(eval_min_depth), float(eval_max_depth)
        self.n_images = 0
        self.m = {k: RunningMetric() for k in (
            "global", "missing", "observed",
            "proxy_global", "proxy_missing", "raw_global", "raw_missing",
            "safe_global", "safe_missing")}

    def _clip(self, x):
        return x.clamp(self.lo, self.hi) if (x is not None and self.clip) else x

    @torch.no_grad()
    def update(self, outputs, target, valid_mask, target_mask=None):
        pred = outputs["completed_depth"]
        pred_finite = torch.isfinite(pred)
        pred = self._clip(pred)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
            pred_finite = F.interpolate(
                pred_finite.float(), size=target.shape[-2:], mode="nearest").bool()
        original_gt_mask = (target > 0) & (target >= self.lo) & (target <= self.hi)
        if target_mask is not None:
            original_gt_mask = original_gt_mask & _bool_mask_like(
                target_mask, target, "target_mask")
        gt_metric_mask = original_gt_mask & torch.isfinite(target)
        metric_mask = gt_metric_mask & pred_finite & torch.isfinite(pred)
        valid_mask = _resize_like(valid_mask.float().to(target.device), target)
        missing = metric_mask & (valid_mask <= 0.5)
        observed = metric_mask & (valid_mask > 0.5)
        self.m["global"].update_depth(pred, target, metric_mask)
        self.m["missing"].update_depth(pred, target, missing)
        self.m["observed"].update_depth(pred, target, observed)
        for key, pfx in (("raw_completed_depth", "raw"), ("dense_geometry", "proxy"), ("safe_completed_depth", "safe")):
            src = outputs.get(key)
            if src is None:
                continue
            src_finite = torch.isfinite(src)
            src = self._clip(src)
            if src.shape[-2:] != target.shape[-2:]:
                src = F.interpolate(src, size=target.shape[-2:], mode="bilinear", align_corners=False)
                src_finite = F.interpolate(
                    src_finite.float(), size=target.shape[-2:], mode="nearest").bool()
            src_metric_mask = gt_metric_mask & src_finite & torch.isfinite(src)
            src_missing = src_metric_mask & (valid_mask <= 0.5)
            self.m[f"{pfx}_global"].update_depth(src, target, src_metric_mask)
            self.m[f"{pfx}_missing"].update_depth(src, target, src_missing)
        self.n_images += target.size(0)

    def result(self):
        s = self.scale
        g = self.m["global"].compute()
        return {
            "images": self.n_images,
            "RMSE_m": g["RMSE"] * s, "MAE_m": g["MAE"] * s, "REL": g["AbsRel"],
            "Log10": g["Log10"], "iRMSE": g["iRMSE"], "iMAE": g["iMAE"],
            "delta1(%)": g["delta1"] * 100.0, "delta2(%)": g["delta2"] * 100.0, "delta3(%)": g["delta3"] * 100.0,
            "FinalMissing_RMSE_m": self.m["missing"].compute()["RMSE"] * s,
            "ProxyMissing_RMSE_m": self.m["proxy_missing"].compute()["RMSE"] * s,
            "RawMissing_RMSE_m": self.m["raw_missing"].compute()["RMSE"] * s,
            "SafeMissing_RMSE_m": self.m["safe_missing"].compute()["RMSE"] * s,
            "Proxy_RMSE_m": self.m["proxy_global"].compute()["RMSE"] * s,
            "Observed_RMSE_m": self.m["observed"].compute()["RMSE"] * s,
        }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, sample):
    """Return GFLOPs for one forward over `sample`=(rgb,depth,valid) or None."""
    rgb, depth, valid = sample
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(rgb, depth, valid), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            return float(FlopCountAnalysis(model, (rgb, depth, valid)).total()) * 2 / 1e9
        except Exception:
            return None


@torch.no_grad()
def measure_fps(model, sample, iters=30, warmup=5):
    rgb, depth, valid = sample
    device = rgb.device
    for _ in range(warmup):
        model(rgb, depth, valid)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(rgb, depth, valid)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / max(iters, 1)
    return (rgb.size(0) / dt) if dt > 0 else float("nan")


def model_stats(model, sample, measure_speed=True):
    total, trainable = count_params(model)
    stats = {"Params(M)": total / 1e6, "Trainable(M)": trainable / 1e6,
             "GFLOPs": estimate_flops(model, sample), "FPS": None}
    if measure_speed:
        try:
            stats["FPS"] = measure_fps(model, sample)
        except Exception:
            stats["FPS"] = None
    return stats

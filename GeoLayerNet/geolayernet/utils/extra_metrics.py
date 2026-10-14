"""Comprehensive paper-grade depth-completion metrics (per-image + aggregate).

Single source of truth for eval numbers. A ``MetricBag`` accumulates raw pixel
sums per region (global / missing / observed / boundary / non-boundary, plus the
proxy / safe / raw diagnostic branches), and per-image normal / gradient /
distance-to-sparse accumulators. The exact same code path produces:

  * the aggregate metrics.csv (pixel-pooled over the whole test set), via
    ``MetricBag.merge`` of per-image bags, and
  * per_image_metrics.csv (one row per image), by finalizing a per-image bag.

Everything degrades to NaN (never crashes) when a region is empty or an optional
dependency (scipy) is missing. ``extra=False`` skips the more expensive boundary
/ normal / gradient / distance-transform metrics (their fields are still emitted
as NaN so the CSV schema is stable).

Legacy field names emitted by the previous evaluator are preserved verbatim
(``RMSE_m``, ``FinalMissing_RMSE_m``, ``ProxyMissing_RMSE_m`` ...) so metrics.csv
stays backward compatible; new paper fields are appended after them.
"""

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as _ndimage
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_EPS = 1e-6
_NAN = float("nan")
_BAD_CM = (1, 2, 5, 10)            # bad-pixel thresholds in centimetres


# --------------------------------------------------------------------------- #
# low-level: raw pixel sums for one region (masked 1-D pixel stats)
# --------------------------------------------------------------------------- #
def region_sums(pred, target, mask, scale):
    """Return a dict of raw sums + pixel count ``n`` for the masked pixels.
    pred / target are in normalized [0,1]; ``scale`` maps to metres."""
    m = mask.bool()
    n = int(m.sum().item())
    s = defaultdict(float)
    s["n"] = float(n)
    if n == 0:
        return s
    p = pred[m].clamp_min(_EPS).double()
    t = target[m].clamp_min(_EPS).double()
    diff = p - t
    ad = diff.abs()
    sq = diff.pow(2)
    logd = torch.log(p) - torch.log(t)
    ratio = torch.maximum(p / t, t / p)
    err_m = ad * scale
    pm = p * scale
    tm = t * scale
    inv_norm = 1.0 / p - 1.0 / t                       # legacy (normalized inverse)
    inv_km = 1000.0 / pm - 1000.0 / tm                 # KITTI inverse depth, 1/km
    s["sq"] = float(sq.sum())
    s["abs"] = float(ad.sum())
    s["absrel"] = float((ad / t).sum())
    s["sqrel"] = float((sq / t).sum())
    s["logsq"] = float(logd.pow(2).sum())
    s["log"] = float(logd.sum())
    s["log10"] = float((torch.log10(p) - torch.log10(t)).abs().sum())
    s["invsq"] = float(inv_norm.pow(2).sum())
    s["invabs"] = float(inv_norm.abs().sum())
    s["inv_km_sq"] = float(inv_km.pow(2).sum())
    s["inv_km_abs"] = float(inv_km.abs().sum())
    s["d1"] = float((ratio < 1.25).sum())
    s["d2"] = float((ratio < 1.25 ** 2).sum())
    s["d3"] = float((ratio < 1.25 ** 3).sum())
    for cm in _BAD_CM:
        s[f"bad{cm}"] = float((err_m > (cm / 100.0)).sum())
    return s


def finalize_region(s, scale):
    """Turn raw sums into final (metre-unit) metrics; NaN when the region empty."""
    n = s.get("n", 0.0)
    if n <= 0:
        keys = ["RMSE_m", "MAE_m", "MSE_m2", "RMSE_mm", "MAE_mm", "AbsRel", "REL",
                "SqRel", "Log10", "LogRMSE", "SILog", "iRMSE", "iMAE",
                "KITTI_iRMSE", "KITTI_iMAE", "delta1", "delta2", "delta3"]
        keys += [f"BadPix_{cm}cm" for cm in _BAD_CM] + ["BadPix_1.25", "n"]
        return {k: _NAN for k in keys} | {"n": 0.0}
    rmse_m = math.sqrt(s["sq"] / n) * scale
    mae_m = (s["abs"] / n) * scale
    d1 = s["d1"] / n * 100.0
    out = {
        "RMSE_m": rmse_m, "MAE_m": mae_m,
        "MSE_m2": (s["sq"] / n) * scale * scale,
        "RMSE_mm": rmse_m * 1000.0, "MAE_mm": mae_m * 1000.0,
        "AbsRel": s["absrel"] / n, "REL": s["absrel"] / n,
        "SqRel": (s["sqrel"] / n) * scale,                 # metre-unit SqRel
        "Log10": s["log10"] / n,
        "LogRMSE": math.sqrt(max(s["logsq"] / n, 0.0)),    # scale-invariant
        "SILog": math.sqrt(max(s["logsq"] / n - (s["log"] / n) ** 2, 0.0)) * 100.0,
        "iRMSE": math.sqrt(s["invsq"] / n), "iMAE": s["invabs"] / n,
        "KITTI_iRMSE": math.sqrt(s["inv_km_sq"] / n), "KITTI_iMAE": s["inv_km_abs"] / n,
        "delta1": d1, "delta2": s["d2"] / n * 100.0, "delta3": s["d3"] / n * 100.0,
        "BadPix_1.25": 100.0 - d1, "n": n,
    }
    for cm in _BAD_CM:
        out[f"BadPix_{cm}cm"] = s[f"bad{cm}"] / n * 100.0
    return out


# --------------------------------------------------------------------------- #
# 2-D helpers: boundary mask, gradients, normals, distance transform
# --------------------------------------------------------------------------- #
def _grad_mag(depth):
    """Sobel-free finite-difference gradient magnitude, same HxW (pad-extend)."""
    dx = torch.zeros_like(depth)
    dy = torch.zeros_like(depth)
    dx[..., :, 1:] = depth[..., :, 1:] - depth[..., :, :-1]
    dy[..., 1:, :] = depth[..., 1:, :] - depth[..., :-1, :]
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def boundary_mask(target, valid, top_percent=10.0):
    """Top-``top_percent`` gradient-magnitude pixels among valid target as the
    boundary. Returns bool [1,H,W]; empty when too few valid pixels."""
    grad = _grad_mag(target)
    v = valid.bool()
    vals = grad[v]
    vals = vals[vals > 0]
    if vals.numel() < 16:
        return torch.zeros_like(v)
    thr = torch.quantile(vals.float(), max(0.0, 1.0 - top_percent / 100.0))
    return v & (grad >= thr) & (grad > 0)


def _depth_to_normal(depth):
    dzdx = torch.zeros_like(depth)
    dzdy = torch.zeros_like(depth)
    dzdx[..., :, 1:-1] = (depth[..., :, 2:] - depth[..., :, :-2]) * 0.5
    dzdy[..., 1:-1, :] = (depth[..., 2:, :] - depth[..., :-2, :]) * 0.5
    nx, ny, nz = -dzdx, -dzdy, torch.ones_like(depth)
    n = torch.cat([nx, ny, nz], dim=0 if depth.dim() == 3 else 1)
    return F.normalize(n, dim=0 if depth.dim() == 3 else 1, eps=1e-6)


def normal_angle_sums(pred, target, mask):
    """Return (sum_deg, sumsq_deg, count) of per-pixel normal angular error."""
    pn = _depth_to_normal(pred)
    tn = _depth_to_normal(target)
    cos = (pn * tn).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    m = mask.bool()
    if m.sum() == 0:
        return 0.0, 0.0, 0.0
    a = ang[m]
    return float(a.sum().item()), float((a * a).sum().item()), float(m.sum().item())


def distance_to_sparse(valid_1hw):
    """Euclidean distance (px) to nearest sparse valid pixel. None if no scipy
    or no valid points."""
    if not _HAS_SCIPY:
        return None
    v = valid_1hw[0].detach().cpu().numpy() > 0.5
    if v.sum() == 0:
        return None
    return _ndimage.distance_transform_edt(~v)


_DIST_BINS = [("0_5px", 0, 5), ("5_15px", 5, 15), ("15_30px", 15, 30), ("30px_plus", 30, 1e9)]


# --------------------------------------------------------------------------- #
# MetricBag: accumulate everything for one image (or many, via merge)
# --------------------------------------------------------------------------- #
class MetricBag:
    def __init__(self, scale=10.0, extra=True, clip=True, lo=1e-3, hi=1.0,
                 boundary_top_percent=10.0):
        self.scale = float(scale)
        self.extra = bool(extra)
        self.clip = bool(clip)
        self.lo, self.hi = float(lo), float(hi)
        self.btp = float(boundary_top_percent)
        self.regions = defaultdict(lambda: defaultdict(float))
        self.aux = defaultdict(float)        # normal/gradient/distance sums
        self.n_images = 0

    def _clip(self, x):
        return x.clamp(self.lo, self.hi) if (x is not None and self.clip) else x

    def _add(self, region, pred, target, mask):
        s = region_sums(pred, target, mask, self.scale)
        for k, v in s.items():
            self.regions[region][k] += v

    @torch.no_grad()
    def update(self, outputs, target, valid, sparse=None):
        """Accumulate one image. Tensors are [1,H,W] (single image) or [B,1,H,W];
        callers pass single-image slices for the per-image path."""
        if target.dim() == 4:  # collapse a leading batch of 1
            target, valid = target[0], valid[0]
            sparse = sparse[0] if sparse is not None else None
            out = {k: (v[0] if torch.is_tensor(v) and v.dim() == 4 else v) for k, v in outputs.items()}
        else:
            out = outputs
        self.n_images += 1

        pred = self._clip(out["completed_depth"])
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False)[0]
        proxy = self._fit(out.get("dense_geometry"), target)
        safe = self._fit(out.get("safe_completed_depth"), target)
        raw = self._fit(out.get("raw_completed_depth"), target)
        valid = valid.float()
        if valid.shape[-2:] != target.shape[-2:]:
            valid = F.interpolate(valid.unsqueeze(0), size=target.shape[-2:], mode="nearest")[0]
        if sparse is not None and sparse.shape[-2:] != target.shape[-2:]:
            sparse = F.interpolate(sparse.unsqueeze(0), size=target.shape[-2:], mode="nearest")[0]

        metric_mask = (target > 0) & (target >= self.lo) & (target <= self.hi)
        missing = metric_mask & (valid <= 0.5)
        observed = metric_mask & (valid > 0.5)

        self._add("global", pred, target, metric_mask)
        self._add("missing", pred, target, missing)
        self._add("observed", pred, target, observed)
        if proxy is not None:
            self._add("proxy_global", proxy, target, metric_mask)
            self._add("proxy_missing", proxy, target, missing)
            self._add("proxy_observed", proxy, target, observed)
        if safe is not None:
            self._add("safe_global", safe, target, metric_mask)
            self._add("safe_missing", safe, target, missing)
        if raw is not None:
            self._add("raw_global", raw, target, metric_mask)
            self._add("raw_missing", raw, target, missing)
        # sparse preservation: final vs the input sparse depth on valid points
        if sparse is not None:
            sp_mask = (valid > 0.5) & (sparse > 0)
            self._add("sparse_preservation", pred, sparse, sp_mask)

        # ---- extra (boundary / normal / gradient / distance) ----
        if self.extra:
            bmask = boundary_mask(target, metric_mask.float(), self.btp)
            nonb = metric_mask & (~bmask)
            self._add("boundary", pred, target, bmask)
            self._add("nonboundary", pred, target, nonb)
            # gradient consistency on the metric region
            gp, gt = _grad_mag(pred), _grad_mag(target)
            self._add("gradient", gp, gt, metric_mask)          # reuse RMSE/MAE on grad maps
            # normals (degrees)
            sd, ssd, nc = normal_angle_sums(pred, target, metric_mask)
            self.aux["normal_sum"] += sd
            self.aux["normal_sqsum"] += ssd
            self.aux["normal_n"] += nc
            # distance-to-sparse bins (RMSE per bin), on the metric region
            dist = distance_to_sparse(valid) if sparse is None else distance_to_sparse((sparse > 0).float())
            if dist is not None:
                dist_t = torch.from_numpy(dist).to(target.device).unsqueeze(0)
                for name, lo, hi in _DIST_BINS:
                    bin_mask = metric_mask & (dist_t >= lo) & (dist_t < hi)
                    sums = region_sums(pred, target, bin_mask, self.scale)
                    self.aux[f"dist_{name}_sq"] += sums["sq"]
                    self.aux[f"dist_{name}_n"] += sums["n"]
                self.aux["dist_available"] = 1.0

    def _fit(self, x, target):
        if x is None:
            return None
        x = self._clip(x)
        if x.dim() == 4:
            x = x[0]
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False)[0]
        return x

    def merge(self, other):
        for region, sums in other.regions.items():
            for k, v in sums.items():
                self.regions[region][k] += v
        for k, v in other.aux.items():
            if k == "dist_available":
                self.aux[k] = max(self.aux.get(k, 0.0), v)
            else:
                self.aux[k] += v
        self.n_images += other.n_images
        return self

    # ------------------------------------------------------------------ #
    def finalize(self):
        s = self.scale
        g = finalize_region(self.regions["global"], s)
        miss = finalize_region(self.regions["missing"], s)
        obs = finalize_region(self.regions["observed"], s)
        pg = finalize_region(self.regions["proxy_global"], s)
        pm = finalize_region(self.regions["proxy_missing"], s)
        po = finalize_region(self.regions["proxy_observed"], s)
        sm = finalize_region(self.regions["safe_missing"], s)
        rm = finalize_region(self.regions["raw_missing"], s)
        sp = finalize_region(self.regions["sparse_preservation"], s)
        bnd = finalize_region(self.regions["boundary"], s)
        nbn = finalize_region(self.regions["nonboundary"], s)
        grad = finalize_region(self.regions["gradient"], s)

        out = {}
        # ---- legacy fields (keep names + order for backward compatibility) ----
        out["images"] = self.n_images
        out["RMSE_m"] = g["RMSE_m"]; out["MAE_m"] = g["MAE_m"]; out["REL"] = g["REL"]
        out["Log10"] = g["Log10"]; out["iRMSE"] = g["iRMSE"]; out["iMAE"] = g["iMAE"]
        out["delta1(%)"] = g["delta1"]; out["delta2(%)"] = g["delta2"]; out["delta3(%)"] = g["delta3"]
        out["FinalMissing_RMSE_m"] = miss["RMSE_m"]
        out["ProxyMissing_RMSE_m"] = pm["RMSE_m"]
        out["RawMissing_RMSE_m"] = rm["RMSE_m"]
        out["SafeMissing_RMSE_m"] = sm["RMSE_m"]
        out["Proxy_RMSE_m"] = pg["RMSE_m"]
        out["Observed_RMSE_m"] = obs["RMSE_m"]

        # ---- B1 NYU / indoor ----
        out["MSE_m2"] = g["MSE_m2"]; out["RMSE_mm"] = g["RMSE_mm"]; out["MAE_mm"] = g["MAE_mm"]
        out["AbsRel"] = g["AbsRel"]; out["SqRel"] = g["SqRel"]; out["LogRMSE"] = g["LogRMSE"]
        out["SILog"] = g["SILog"]
        out["delta1"] = g["delta1"]; out["delta2"] = g["delta2"]; out["delta3"] = g["delta3"]
        for cm in _BAD_CM:
            out[f"BadPix_{cm}cm(%)"] = g[f"BadPix_{cm}cm"]
        out["BadPix_1.25(%)"] = g["BadPix_1.25"]

        # ---- B2 KITTI ----
        out["KITTI_RMSE_mm"] = g["RMSE_mm"]; out["KITTI_MAE_mm"] = g["MAE_mm"]
        out["KITTI_iRMSE"] = g["KITTI_iRMSE"]; out["KITTI_iMAE"] = g["KITTI_iMAE"]

        # ---- B3 missing / observed / preservation ----
        out["Final_Missing_RMSE_m"] = miss["RMSE_m"]; out["Final_Missing_MAE_m"] = miss["MAE_m"]; out["Final_Missing_REL"] = miss["REL"]
        out["Proxy_Missing_RMSE_m"] = pm["RMSE_m"]; out["Proxy_Missing_MAE_m"] = pm["MAE_m"]; out["Proxy_Missing_REL"] = pm["REL"]
        out["Safe_Missing_RMSE_m"] = sm["RMSE_m"]; out["Safe_Missing_MAE_m"] = sm["MAE_m"]; out["Safe_Missing_REL"] = sm["REL"]
        out["Proxy_Observed_RMSE_m"] = po["RMSE_m"]; out["Proxy_Observed_MAE_m"] = po["MAE_m"]
        out["Final_Observed_RMSE_m"] = obs["RMSE_m"]; out["Final_Observed_MAE_m"] = obs["MAE_m"]
        out["Sparse_Preservation_RMSE_m"] = sp["RMSE_m"]; out["Sparse_Preservation_MAE_m"] = sp["MAE_m"]

        # ---- B4 boundary / geometry ----
        out["Boundary_RMSE_m"] = bnd["RMSE_m"]; out["Boundary_MAE_m"] = bnd["MAE_m"]
        out["NonBoundary_RMSE_m"] = nbn["RMSE_m"]; out["NonBoundary_MAE_m"] = nbn["MAE_m"]
        out["Gradient_MAE_m"] = grad["MAE_m"]; out["Gradient_RMSE_m"] = grad["RMSE_m"]
        nn = self.aux.get("normal_n", 0.0)
        out["Normal_MAE_deg"] = (self.aux["normal_sum"] / nn) if nn > 0 else _NAN
        out["Normal_RMSE_deg"] = math.sqrt(self.aux["normal_sqsum"] / nn) if nn > 0 else _NAN
        out["Edge_AbsRel"] = bnd["AbsRel"]

        # ---- B5 distance-to-sparse bins ----
        for name, _lo, _hi in _DIST_BINS:
            dn = self.aux.get(f"dist_{name}_n", 0.0)
            key = f"DistToSparse_RMSE_bin_{name}"
            out[key] = (math.sqrt(self.aux[f"dist_{name}_sq"] / dn) * s) if dn > 0 else _NAN

        # ---- per-image bookkeeping fields ----
        out["SparseInput_Proxy_RMSE_m"] = po["RMSE_m"]
        out["SparseInput_Final_RMSE_m"] = obs["RMSE_m"]
        out["ValidPixels"] = obs["n"]
        out["MissingPixels"] = miss["n"]
        out["BoundaryPixels"] = bnd["n"]
        return out


# canonical per-image CSV column order (task A2)
PER_IMAGE_FIELDS = [
    "sample_id", "RMSE_m", "MAE_m", "REL", "Log10", "iRMSE", "iMAE",
    "delta1", "delta2", "delta3", "FinalMissing_RMSE_m", "ProxyMissing_RMSE_m",
    "SafeMissing_RMSE_m", "RawMissing_RMSE_m", "Boundary_RMSE_m", "Boundary_MAE_m",
    "NonBoundary_RMSE_m", "Gradient_MAE_m", "Normal_MAE_deg",
    "SparseInput_Proxy_RMSE_m", "SparseInput_Final_RMSE_m",
    "ValidPixels", "MissingPixels", "BoundaryPixels",
]

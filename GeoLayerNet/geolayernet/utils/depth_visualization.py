"""Lightweight depth/mask visualization (numpy + PIL, no matplotlib needed).

Saves the baseline GeoLayerNet visualizations: RGB, sparse-on-RGB, target/proxy/
raw/safe/final false-color (auto = per-image p1/p99, shared = target p1/p99),
error maps, layer seed/support/support_total/unknown, residual_map, refine_gate.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# compact "turbo"-like colormap control points (R,G,B in 0..1)
_TURBO = np.array([
    [0.19, 0.07, 0.23], [0.27, 0.31, 0.84], [0.10, 0.65, 0.93],
    [0.16, 0.89, 0.66], [0.62, 0.99, 0.23], [0.98, 0.81, 0.19],
    [0.97, 0.43, 0.13], [0.73, 0.13, 0.02], [0.48, 0.01, 0.01],
], dtype=np.float32)


def _colormap(x01):
    x = np.clip(x01, 0.0, 1.0)
    idx = x * (len(_TURBO) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(_TURBO) - 1)
    t = (idx - lo)[..., None]
    rgb = _TURBO[lo] * (1 - t) + _TURBO[hi] * t
    return (rgb * 255).astype(np.uint8)


def _to_np(t):
    return t.detach().float().cpu().numpy()


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def save_rgb(path, rgb_chw_m1p1):
    arr = _to_np(rgb_chw_m1p1)
    arr = ((arr.transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(_ensure_parent(path))


def save_false_color(path, depth_1hw, vmin=None, vmax=None):
    d = _to_np(depth_1hw)[0]
    if vmin is None:
        vmin = np.percentile(d, 1)
    if vmax is None:
        vmax = np.percentile(d, 99)
    rng = max(vmax - vmin, 1e-6)
    Image.fromarray(_colormap((d - vmin) / rng)).save(_ensure_parent(path))
    return float(vmin), float(vmax)


def save_error_map(path, pred_1hw, target_1hw, err_max=0.05):
    e = np.abs(_to_np(pred_1hw)[0] - _to_np(target_1hw)[0])
    Image.fromarray(_colormap(e / max(err_max, 1e-6))).save(_ensure_parent(path))


def save_mask(path, mask_1hw):
    m = _to_np(mask_1hw)[0]
    m = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(m).save(_ensure_parent(path))


def save_sparse_on_rgb(path, rgb_chw_m1p1, valid_1hw):
    rgb = ((_to_np(rgb_chw_m1p1).transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8).copy()
    v = _to_np(valid_1hw)[0] > 0.5
    rgb[v] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(rgb).save(_ensure_parent(path))


def save_sample_visuals(out_root, name, batch, outputs, idx, err_max=0.05,
                        vis_depth_mode="both"):
    """Save the baseline visual set for one sample (index idx in the batch)."""
    def p(sub):
        d = os.path.join(out_root, sub)
        os.makedirs(d, exist_ok=True)
        return _ensure_parent(os.path.join(d, name))

    rgb = batch["rgb"][idx]
    target = batch["target_depth"][idx]
    valid = batch["valid_mask"][idx]
    tvmin = float(np.percentile(_to_np(target)[0], 1))
    tvmax = float(np.percentile(_to_np(target)[0], 99))

    save_rgb(p("rgb") + "_rgb.png", rgb)
    save_sparse_on_rgb(p("sparse_on_rgb") + "_sparse.png", rgb, valid)

    depth_keys = [("target_depth", target), ("dense_geometry", outputs["dense_geometry"][idx]),
                  ("raw_completed_depth", outputs.get("raw_completed_depth")),
                  ("safe_completed_depth", outputs.get("safe_completed_depth")),
                  ("coarse_depth", outputs.get("coarse_depth")),
                  ("completed_depth", outputs["completed_depth"][idx])]
    for key, t in depth_keys:
        if t is None:
            continue
        t = t[idx] if t.dim() == 4 else t
        if vis_depth_mode in ("auto", "both"):
            save_false_color(p(key) + "_auto.png", t)
        if vis_depth_mode in ("shared", "both"):
            save_false_color(p(key) + "_shared.png", t, tvmin, tvmax)

    save_error_map(p("error_maps") + "_final_error.png", outputs["completed_depth"][idx], target, err_max)
    save_error_map(p("error_maps") + "_proxy_error.png", outputs["dense_geometry"][idx], target, err_max)

    # diagnostic gates/maps (incl. the enhancement outputs used to explain why
    # full does or doesn't beat proxy: criterion C).
    for key in ("support_total", "unknown_support", "refine_gate", "residual_map",
                "selective_gate", "edge_gate", "highfreq_gate", "uncertainty",
                "coarse_confidence"):
        t = outputs.get(key)
        if t is not None:
            save_mask(p("layer_maps") + f"_{key}.png", t[idx])
    # per-layer seed/support
    for lk in ("layer_seed_masks", "layer_support"):
        t = outputs.get(lk)
        if t is None:
            continue
        for L in range(t.size(1)):
            save_mask(p("layer_maps") + f"_{lk}_L{L}.png", t[idx, L])


def _colorize_masked(depth_hw, mask_hw, vmin, vmax):
    """False-color a depth map, invalid pixels -> black. Returns uint8 HxWx3."""
    rng = max(vmax - vmin, 1e-6)
    rgb = _colormap((depth_hw - vmin) / rng)
    rgb[~(mask_hw > 0.5)] = 0
    return rgb


def save_panel_materials(out_root, name, batch, outputs, idx, output_mode="safe",
                         scale=1.0, err_pct=95.0):
    """Save standardized per-sample material files (filenames contain the
    sample id) PLUS a raw-array .npz, so the rich contact-sheet builder can
    re-render every run with a SHARED colormap/error range.

    Materials: rgb, sparse_depth (color-on-black) + sparse_on_rgb, gt_depth,
    pred_final (= completed_depth for this run's output mode), pred_proxy
    (= dense_geometry), abs_error_final, abs_error_proxy. Error = |pred - gt| on
    the gt-valid region only (invalid -> black). Does NOT touch metrics/outputs.
    """
    def pj(sub, suffix):
        d = os.path.join(out_root, sub)
        os.makedirs(d, exist_ok=True)
        return _ensure_parent(os.path.join(d, f"{name}{suffix}"))

    rgb_t = batch["rgb"][idx]
    sparse_t = batch["sparse_depth"][idx]
    valid_t = batch["valid_mask"][idx]
    gt_t = batch["target_depth"][idx]
    pred_final_t = outputs["completed_depth"][idx]
    pred_proxy_t = outputs.get("dense_geometry")
    pred_proxy_t = pred_proxy_t[idx] if pred_proxy_t is not None else pred_final_t

    rgb = ((_to_np(rgb_t).transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    sparse = _to_np(sparse_t)[0]
    valid = _to_np(valid_t)[0]
    gt = _to_np(gt_t)[0]
    pred_final = _to_np(pred_final_t)[0]
    pred_proxy = _to_np(pred_proxy_t)[0]
    gt_valid = gt > 0

    # RGB (denormalized) + sparse visualizations
    Image.fromarray(rgb).save(pj("panel", "_rgb.png"))
    save_sparse_on_rgb(pj("panel", "_sparse_on_rgb.png"), rgb_t, valid_t)
    # sparse depth as colored valid points on black
    if gt_valid.any():
        svmin, svmax = float(np.percentile(gt[gt_valid], 1)), float(np.percentile(gt[gt_valid], 99))
    else:
        svmin, svmax = 0.0, 1.0
    Image.fromarray(_colorize_masked(sparse, (sparse > 0).astype(np.float32), svmin, svmax)).save(
        pj("panel", "_sparse_depth.png"))

    # depth materials share the gt p1/p99 range (per sample)
    for arr, suffix in ((gt, "_gt_depth.png"), (pred_final, "_pred_final.png"), (pred_proxy, "_pred_proxy.png")):
        Image.fromarray(_colorize_masked(arr, gt_valid.astype(np.float32), svmin, svmax)).save(pj("panel", suffix))

    # abs error materials share one error range (0 .. p95 of max error on valid)
    err_f = np.abs(pred_final - gt)
    err_p = np.abs(pred_proxy - gt)
    if gt_valid.any():
        emax = float(np.percentile(np.maximum(err_f, err_p)[gt_valid], err_pct))
    else:
        emax = 1.0
    emax = max(emax, 1e-6)
    for err, suffix in ((err_f, "_abs_error_final.png"), (err_p, "_abs_error_proxy.png")):
        e = _colormap(np.clip(err / emax, 0, 1))
        e[~gt_valid] = 0
        Image.fromarray(e).save(pj("panel", suffix))

    # raw arrays for shared-range rich rendering downstream
    npz_dir = os.path.join(out_root, "panel_data")
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(
        _ensure_parent(os.path.join(npz_dir, f"{name}.npz")),
        rgb=rgb.astype(np.uint8), sparse=sparse.astype(np.float16),
        valid=(valid > 0.5).astype(np.uint8), gt=gt.astype(np.float16),
        pred_final=pred_final.astype(np.float16), pred_proxy=pred_proxy.astype(np.float16),
        scale=np.float32(scale), output_mode=np.array(str(output_mode)),
    )

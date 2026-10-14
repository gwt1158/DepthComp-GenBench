"""Fixed-range visualization protocol derived from GeoLayerNet's real code.

Pixel rendering deliberately mirrors
``GeoLayerNet/geolayernet/utils/depth_visualization.py``:

* the exact nine control-point compact turbo-like colormap;
* RGB inverse normalization ``((x + 1) * .5 * 255).clip(...).astype(uint8)``;
* invalid pixels use an unambiguous dark gray;
* PNGs are raw PIL images with no title, font, colorbar, margin or resampling;
* non-road output is a five-panel zero-gap concatenation; registered road
  datasets add the fixed 0..30 m near-depth panel and are six-panel.

The reference implementation's historical default derives p1/p99 per image.
That cannot provide a fair cross-model color scale.  The comparison extension
therefore fixes display depth to the GeoLayerNet YAML normalization interval
(``0 .. eval.max_depth * depth_unit_scale``) and error to
``0 .. eval.error_max * depth_unit_scale``.  It never derives a range from a
prediction or from the current image.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


REFERENCE_FILE = "GeoLayerNet/geolayernet/utils/depth_visualization.py"
REFERENCE_FUNCTIONS = ["_colormap", "_colorize_masked", "save_panel_materials"]
DEPTH_COLORMAP = "GeoLayerNet compact turbo-like (9 control points)"
ERROR_COLORMAP = DEPTH_COLORMAP
INVALID_RGB = np.array([32, 32, 32], dtype=np.uint8)
INVALID_PIXEL_POLICY = (
    "dark gray RGB=(32,32,32); invalid GT is never rendered as zero error")
RGB_INVERSE_NORMALIZATION = (
    "((rgb_chw_m1p1.transpose(1,2,0)+1.0)*0.5*255)"
    ".clip(0,255).astype(uint8)"
)
CONCAT_LAYOUT = (
    "RGB | Sparse Overlay | Completed Full | Completed Near | GT Overlay | "
    "Valid Error Overlay; publication image has no text")

OOD_ROAD_DATASETS = frozenset({
    "drivingstereo", "argoverse", "cityscapes", "diodeoutdoor",
    "vkitti2", "kitti360",
})

DATASET_VISUAL_REGISTRY = {
    "kitti": {"road": True},
    "drivingstereo": {"road": True},
    "argoverse": {"road": True},
    "cityscapes": {"road": True},
    "cityscape": {"road": True},
    "diodeoutdoor": {"road": True},
    "vkitti2": {"road": True},
    "oodvkitti2": {"road": True},
    "kitti360": {"road": True},
}
ROAD_DATASETS = {
    name for name, policy in DATASET_VISUAL_REGISTRY.items()
    if policy.get("road")
}

_TURBO = np.array([
    [0.19, 0.07, 0.23], [0.27, 0.31, 0.84], [0.10, 0.65, 0.93],
    [0.16, 0.89, 0.66], [0.62, 0.99, 0.23], [0.98, 0.81, 0.19],
    [0.97, 0.43, 0.13], [0.73, 0.13, 0.02], [0.48, 0.01, 0.01],
], dtype=np.float32)


def _colormap(x01):
    """Byte-identical implementation of GeoLayerNet's private colormap."""
    x = np.clip(x01, 0.0, 1.0)
    idx = x * (len(_TURBO) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(_TURBO) - 1)
    t = (idx - lo)[..., None]
    rgb = _TURBO[lo] * (1 - t) + _TURBO[hi] * t
    return (rgb * 255).astype(np.uint8)


def _to_np(value):
    if isinstance(value, np.ndarray):
        return value
    return value.detach().float().cpu().numpy()


def _mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def safe_sample_stem(sample_id):
    return str(sample_id).replace("\\", "/").strip("/").replace("/", "__")


def sha256_file(path, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_uint8(rgb_chw_m1p1):
    arr = _to_np(rgb_chw_m1p1)
    return ((arr.transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(
        0, 255).astype(np.uint8)


def resolve_visualization_protocol(
        *, dataset, dataset_config, depth_unit_scale,
        max_depth_normalized=1.0, error_max_normalized=0.05,
        depth_vmin_m=None, depth_vmax_m=None, error_vmax_m=None,
        visualization_depth_unit_scale=None):
    """Resolve one fixed dataset-wide visualization contract.

    ``max_depth_normalized`` and ``error_max_normalized`` are read from the
    selected GeoLayerNet YAML by the caller.  The minimum is fixed at zero so
    NYU/KITTI display intervals are exactly 0..10 m and 0..80 m; invalid zero
    pixels remain black and are never interpreted as measurements.
    """
    scale = float(depth_unit_scale)
    range_scale = float(
        scale if visualization_depth_unit_scale is None
        else visualization_depth_unit_scale)
    maximum = float(max_depth_normalized)
    error_maximum = float(error_max_normalized)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_unit_scale must be positive and finite")
    if not np.isfinite(range_scale) or range_scale <= 0:
        raise ValueError(
            "visualization_depth_unit_scale must be positive and finite")
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_depth_normalized must be positive and finite")
    if not np.isfinite(error_maximum) or error_maximum <= 0:
        raise ValueError("error_max_normalized must be positive and finite")
    normalized_dataset = str(dataset).lower().replace("_", "").replace("-", "")
    road = normalized_dataset in ROAD_DATASETS
    if road:
        registered_vmax_m, registered_error_vmax_m = 80.0, 4.0
        registered_range_source = "road dataset visualization registry"
    elif normalized_dataset in ("nyu", "nyuv2", "nyuv2official654"):
        registered_vmax_m, registered_error_vmax_m = 10.0, 1.0
        registered_range_source = "NYU visualization registry"
    else:
        registered_vmax_m = maximum * range_scale
        registered_error_vmax_m = error_maximum * range_scale
        registered_range_source = "selected GeoLayerNet dataset YAML"
    vmin_m = 0.0 if depth_vmin_m is None else float(depth_vmin_m)
    vmax_m = (registered_vmax_m if depth_vmax_m is None
              else float(depth_vmax_m))
    resolved_error_vmax_m = (
        registered_error_vmax_m if error_vmax_m is None
        else float(error_vmax_m))
    if not np.isfinite(vmin_m + vmax_m) or vmax_m <= vmin_m:
        raise ValueError("fixed depth display range is invalid")
    if (not np.isfinite(resolved_error_vmax_m)
            or resolved_error_vmax_m <= 0):
        raise ValueError("fixed error display range is invalid")
    return {
        "schema": "comparison-protocol-v2-visualization",
        "dataset": str(dataset),
        "dataset_config": (str(Path(dataset_config).expanduser().resolve())
                           if dataset_config else None),
        "depth_colormap": DEPTH_COLORMAP,
        "depth_colormap_control_points_rgb01": _TURBO.tolist(),
        "colormap_interpolation": (
            "linear between nine RGB control points, multiply by 255 then uint8"),
        "depth_vmin_m": vmin_m,
        "depth_vmax_m": vmax_m,
        "error_colormap": ERROR_COLORMAP,
        "error_vmin_m": 0.0,
        "error_vmax_m": resolved_error_vmax_m,
        "road_dataset": road,
        "near_depth_vmin_m": 0.0,
        "near_depth_vmax_m": 30.0 if road else vmax_m,
        "relative_error_vmin": 0.0,
        "relative_error_vmax": 0.2,
        "point_overlay": {
            "display_only": True,
            "point_radius_px": 2,
            "alpha": 0.85,
            "source": "raw_sparse_or_valid_target",
            "overlap_policy": "nearer positive depth wins",
        },
        "depth_unit_scale": scale,
        "visualization_depth_unit_scale": range_scale,
        "raw_input_depth_unit": (
            "normalized depth; metres=value*depth_unit_scale"),
        "raw_prediction_unit": "normalized depth; metres=value*depth_unit_scale",
        "raw_error_unit": "metres",
        "absolute_error_definition": (
            "abs(prediction_normalized-target_normalized)*depth_unit_scale"),
        "fixed_range_source": (
            registered_range_source
            + " unless explicit metre overrides are supplied"),
        "invalid_pixel_policy": INVALID_PIXEL_POLICY,
        "RGB inverse normalization": RGB_INVERSE_NORMALIZATION,
        "GeoLayerNet reference file/function": {
            "file": REFERENCE_FILE, "functions": REFERENCE_FUNCTIONS},
        "reference_historical_range_policy": (
            "GeoLayerNet save_panel_materials used per-image valid-GT p1/p99; "
            "comparison v1 replaces only range selection with the fixed YAML "
            "normalization interval required for cross-model comparability"),
        "concat_layout": CONCAT_LAYOUT,
        "font": None,
        "titles": None,
        "colorbar": None,
        "margin_px": 0,
        "dpi": None,
        "png_format": "PIL uint8 RGB; raw depth companion PNG is uint16",
        "no_prediction_smoothing": True,
        "no_sparse_interpolation": True,
    }


def _colorize_masked(depth_m, mask_hw, vmin_m, vmax_m):
    rng = max(float(vmax_m) - float(vmin_m), 1e-6)
    rgb = _colormap((depth_m - float(vmin_m)) / rng)
    rgb[~(mask_hw > 0.5)] = INVALID_RGB
    return rgb


def _absolute_error_m(prediction_normalized, target_normalized, scale):
    return np.abs(
        prediction_normalized.astype(np.float32, copy=False)
        - target_normalized.astype(np.float32, copy=False)
    ) * np.float32(scale)


def _error_color(error_m, valid_hw, error_vmax_m):
    rgb = _colormap(error_m / max(float(error_vmax_m), 1e-6))
    rgb[~(valid_hw > 0.5)] = INVALID_RGB
    return rgb


def _splat_color(color, valid, priority, radius=2, lower_wins=True):
    """Display-only disk splat with deterministic overlap priority."""
    height, width = valid.shape
    out = np.zeros((height, width, 3), dtype=np.uint8)
    best = np.full((height, width), np.inf if lower_wins else -np.inf,
                   dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            sy0, sy1 = max(0, -dy), min(height, height - dy)
            sx0, sx1 = max(0, -dx), min(width, width - dx)
            dy0, dy1 = sy0 + dy, sy1 + dy
            dx0, dx1 = sx0 + dx, sx1 + dx
            src_valid = valid[sy0:sy1, sx0:sx1]
            src_priority = priority[sy0:sy1, sx0:sx1]
            dst_best = best[dy0:dy1, dx0:dx1]
            better = src_valid & ((src_priority < dst_best) if lower_wins
                                  else (src_priority > dst_best))
            dst_best[better] = src_priority[better]
            dst_color = out[dy0:dy1, dx0:dx1]
            src_color = color[sy0:sy1, sx0:sx1]
            dst_color[better] = src_color[better]
    return out, np.isfinite(best) if lower_wins else (best > -np.inf)


def _overlay(rgb, splat, mask, alpha=0.85):
    result = rgb.copy()
    blended = np.rint(
        (1.0 - float(alpha)) * rgb.astype(np.float32)
        + float(alpha) * splat.astype(np.float32)).clip(0, 255).astype(np.uint8)
    result[mask] = blended[mask]
    return result


def _save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
    return path


def _save_depth_png16(path, normalized_depth, valid_mask):
    array = np.clip(normalized_depth, 0.0, 1.0)
    encoded = np.rint(array * 65535.0).astype(np.uint16)
    encoded[~(valid_mask > 0.5)] = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded).save(path)


def _prediction_array(outputs, idx):
    value = _to_np(outputs["completed_depth"][idx])
    return np.ascontiguousarray(value[0], dtype=np.float32)


def save_sample_visuals(
        out_root, name, batch, outputs, idx, *, protocol,
        prediction_npy_path=None, prediction_npy_unit="normalized_depth"):
    """Write Protocol V2 raw, overlay, fixed-range and concat materials.

    The supplied prediction and input tensors are copied to CPU arrays.  This
    function never mutates them and never returns an array used by metrics.
    """
    root = Path(out_root)
    stem = safe_sample_stem(name)
    scale = float(protocol["depth_unit_scale"])
    rgb = rgb_uint8(batch["rgb"][idx])
    sparse = np.ascontiguousarray(
        _to_np(batch["sparse_depth"][idx])[0], dtype=np.float32)
    target = np.ascontiguousarray(
        _to_np(batch["target_depth"][idx])[0], dtype=np.float32)
    prediction = _prediction_array(outputs, idx)
    if prediction.shape != target.shape:
        raise ValueError(
            f"visual prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    sparse_valid = np.isfinite(sparse) & (sparse > 0)
    target_valid = np.isfinite(target) & (target > 0)
    if "target_mask" in batch:
        registered_target_mask = np.ascontiguousarray(
            _to_np(batch["target_mask"][idx])[0]) > 0.5
        target_valid &= registered_target_mask
    error_m = _absolute_error_m(prediction, target, scale)
    error_m[~target_valid] = 0.0
    prediction_m = prediction * scale
    sparse_m = sparse * scale
    target_m = target * scale
    prediction_valid = np.isfinite(prediction_m) & (prediction_m > 0)
    relative_error = np.zeros_like(error_m, dtype=np.float32)
    relative_error[target_valid] = (
        error_m[target_valid] / np.maximum(target_m[target_valid], 1e-6))
    road = bool(protocol.get("road_dataset", False))
    if prediction_npy_unit not in ("normalized_depth", "metres"):
        raise RuntimeError(
            f"unsupported prediction NPY unit: {prediction_npy_unit}")

    paths = {
        "rgb": root / "rgb" / f"{stem}.png",
        "sparse_raw_npy": root / "sparse_depth_raw" / f"{stem}.npy",
        "sparse_raw_png": root / "sparse_depth_raw" / f"{stem}.png",
        "sparse_color": root / "sparse_depth_color_raw" / f"{stem}.png",
        "sparse_overlay": root / "sparse_depth_overlay" / f"{stem}.png",
        "target_raw_npy": root / "target_depth_raw" / f"{stem}.npy",
        "target_raw_png": root / "target_depth_raw" / f"{stem}.png",
        "target_color": root / "target_depth_color_raw" / f"{stem}.png",
        "target_overlay": root / "target_depth_overlay" / f"{stem}.png",
        "completed_raw_npy": root / "completed_depth_raw" / f"{stem}.npy",
        "completed_raw_png": root / "completed_depth_raw" / f"{stem}.png",
        "completed_color": root / "completed_depth_color_full" / f"{stem}.png",
        "completed_near": root / "completed_depth_color_near" / f"{stem}.png",
        "error_raw": root / "error_maps_raw" / f"{stem}.npy",
        "error_color": root / "error_valid_color" / f"{stem}.png",
        "error_overlay": root / "error_valid_overlay" / f"{stem}.png",
        "absolute_error_near": root / "absolute_error_near_color" / f"{stem}.png",
        "relative_error": root / "relative_error_color" / f"{stem}.png",
        "concat": root / "concat" / f"{stem}.png",
        "concat_audit_text": root / "concat_audit_text" / f"{stem}.png",
        "metadata": root / "metadata" / f"{stem}.json",
    }
    road_only_paths = {
        "completed_near", "absolute_error_near", "relative_error"}
    for key, path in paths.items():
        if key in road_only_paths and not road:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(paths["rgb"])
    _save_npy(paths["sparse_raw_npy"], sparse_m)
    _save_depth_png16(paths["sparse_raw_png"], sparse, sparse_valid)
    _save_npy(paths["target_raw_npy"], target_m)
    _save_depth_png16(paths["target_raw_png"], target, target_valid)
    if prediction_npy_path:
        source = Path(prediction_npy_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if np.load(source, allow_pickle=False).dtype != np.float32:
            raise RuntimeError("prediction NPY must be float32")
        shutil.copyfile(source, paths["completed_raw_npy"])
    else:
        raw_prediction = (
            prediction_m if prediction_npy_unit == "metres" else prediction)
        _save_npy(paths["completed_raw_npy"], raw_prediction)
    _save_depth_png16(paths["completed_raw_png"], prediction, target_valid)
    _save_npy(paths["error_raw"], error_m)

    sparse_color = _colorize_masked(
        sparse_m, sparse_valid,
        protocol["depth_vmin_m"], protocol["depth_vmax_m"])
    target_color = _colorize_masked(
        target_m, target_valid,
        protocol["depth_vmin_m"], protocol["depth_vmax_m"])
    completed_color = _colorize_masked(
        prediction_m, prediction_valid,
        protocol["depth_vmin_m"], protocol["depth_vmax_m"])
    error_color = _error_color(
        error_m, target_valid, protocol["error_vmax_m"])

    radius = int(protocol.get("point_overlay", {}).get("point_radius_px", 2))
    alpha = float(protocol.get("point_overlay", {}).get("alpha", 0.85))
    sparse_splat, sparse_splat_mask = _splat_color(
        sparse_color, sparse_valid, sparse_m, radius=radius, lower_wins=True)
    target_splat, target_splat_mask = _splat_color(
        target_color, target_valid, target_m, radius=radius, lower_wins=True)
    error_splat, error_splat_mask = _splat_color(
        error_color, target_valid, error_m, radius=radius, lower_wins=False)
    sparse_overlay = _overlay(rgb, sparse_splat, sparse_splat_mask, alpha)
    target_overlay = _overlay(rgb, target_splat, target_splat_mask, alpha)
    error_overlay = _overlay(rgb, error_splat, error_splat_mask, alpha)

    Image.fromarray(sparse_color).save(paths["sparse_color"])
    Image.fromarray(sparse_overlay).save(paths["sparse_overlay"])
    Image.fromarray(target_color).save(paths["target_color"])
    Image.fromarray(target_overlay).save(paths["target_overlay"])
    Image.fromarray(completed_color).save(paths["completed_color"])
    Image.fromarray(error_color).save(paths["error_color"])
    Image.fromarray(error_overlay).save(paths["error_overlay"])
    near_max = float(protocol.get("near_depth_vmax_m", protocol["depth_vmax_m"]))
    if road:
        completed_near = _colorize_masked(
            prediction_m, prediction_valid, 0.0, near_max)
        near_valid = target_valid & (target_m <= near_max)
        absolute_error_near = _error_color(
            error_m, near_valid, protocol["error_vmax_m"])
        relative_color = _colormap(
            relative_error / max(float(
                protocol.get("relative_error_vmax", 0.2)), 1e-6))
        relative_color[~target_valid] = INVALID_RGB
        Image.fromarray(completed_near).save(paths["completed_near"])
        Image.fromarray(absolute_error_near).save(paths["absolute_error_near"])
        Image.fromarray(relative_color).save(paths["relative_error"])

    concat_panels = [rgb, sparse_overlay, completed_color]
    if road:
        concat_panels.append(completed_near)
    concat_panels.extend([target_overlay, error_overlay])
    concat = np.concatenate(concat_panels, axis=1)
    Image.fromarray(concat).save(paths["concat"])

    sparse_ratio = float(sparse_valid.mean())
    target_ratio = float(target_valid.mean())
    metadata = {
        "sample_id": str(name),
        "sparse_valid_ratio": sparse_ratio,
        "target_valid_ratio": target_ratio,
        "method": protocol.get("method"),
        "training_domain": protocol.get("training_domain"),
        "seed": protocol.get("seed"),
        "depth_ranges_m": {
            "full": [protocol["depth_vmin_m"], protocol["depth_vmax_m"]],
            "near": [0.0, near_max],
            "absolute_error": [0.0, protocol["error_vmax_m"]],
            "relative_error": [0.0, protocol.get("relative_error_vmax", 0.2)],
        },
        "display_only": True,
        "point_radius_px": radius,
        "point_alpha": alpha,
        "raw_sparse_modified": False,
        "raw_target_modified": False,
        "raw_sparse_npy_unit": "metres",
        "raw_target_npy_unit": "metres",
        "raw_prediction_npy_unit": prediction_npy_unit,
        "per_image_normalization": False,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    from PIL import ImageDraw
    audit = Image.new("RGB", (concat.shape[1], concat.shape[0] + 28), (24, 24, 24))
    audit.paste(Image.fromarray(concat), (0, 28))
    ImageDraw.Draw(audit).text(
        (5, 7),
        (f"{name}  sparse={sparse_ratio:.5f} target={target_ratio:.5f} "
         f"domain={protocol.get('training_domain')} seed={protocol.get('seed')} "
         f"radius={radius}px display-only"),
        fill=(235, 235, 235))
    audit.save(paths["concat_audit_text"])

    return {
        "sample_id": str(name),
        "filename_stem": stem,
        "completed_depth_raw": str(paths["completed_raw_npy"].resolve()),
        "completed_depth_raw_sha256": sha256_file(paths["completed_raw_npy"]),
        "concat": str(paths["concat"].resolve()),
        "concat_audit_text": str(paths["concat_audit_text"].resolve()),
        "metadata": str(paths["metadata"].resolve()),
    }


def write_visualization_protocol(
        output_dir, protocol, sample_ids, *, missing_sample_ids=None,
        selection_source=None):
    payload = dict(protocol)
    payload.update({
        "sample_ids": list(sample_ids),
        "missing_sample_ids": list(missing_sample_ids or []),
        "selection_source": selection_source,
    })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path = output / "visual_protocol.json"
    path.write_text(encoded, encoding="utf-8")
    return path


def write_prediction_index(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# Compatibility name retained for older callers. New comparison evaluation
# writes the standardized tree above and does not create legacy panel files.
def save_panel_materials(*_args, **_kwargs):
    return None

"""GeoLayerNet data protocol and dataset-leakage audit for comparison methods.

This module mirrors the reference GeoLayerNet NYU directory reader and OOD npz
reader while making split/config selection explicit. It never creates a split,
never samples sparse depth from GT, and never falls back to a hard-coded dataset.
A split is read from one user-selected dataroot plus an optional ordered sample
list. A GeoLayerNet YAML may supply data-only defaults; command-line paths win.
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .seed import seed_worker

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_HASH_CHUNK = 1024 * 1024
_OOD_PROTOCOL_HW = (228, 304)

# Camera calibration used by the formal comparison loaders.  These values are
# metadata, not learned parameters.  NYU48K stores the standard NYUv2 640x480
# camera after the official half-resolution + 8/6 centre crop (304x228).
_NYU_RAW_K = np.array([
    [518.8579011745019, 0.0, 325.58244941119034],
    [0.0, 519.4696111212749, 253.73616633400465],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
_NYU_CROPPED_HW = (228, 304)
_NYU_CROPPED_K = _NYU_RAW_K.copy()
_NYU_CROPPED_K[0, :] /= 2.0
_NYU_CROPPED_K[1, :] /= 2.0
_NYU_CROPPED_K[0, 2] -= 8.0
_NYU_CROPPED_K[1, 2] -= 6.0

# Exact per-date KITTI calibration used by the official PSD loader.
_KITTI_K_BY_DATE = {
    "2011_09_26": (721.5377, 721.5377, 609.5593, 172.8540),
    "2011_09_28": (707.0493, 707.0493, 604.0814, 180.5066),
    "2011_09_29": (718.3351, 718.3351, 600.3891, 181.5122),
    "2011_09_30": (707.0912, 707.0912, 601.8873, 183.1104),
    "2011_10_03": (718.8560, 718.8560, 607.1928, 185.2157),
}


def _scale_intrinsics(K, source_hw, target_hw):
    source_h, source_w = (float(source_hw[0]), float(source_hw[1]))
    target_h, target_w = (float(target_hw[0]), float(target_hw[1]))
    out = np.asarray(K, dtype=np.float32).copy()
    sx, sy = target_w / source_w, target_h / source_h
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def _nyu_intrinsics(original_hw, preprocess, load_size, crop_size, flip):
    """Return K in the exact tensor coordinates emitted by Nyu500Dataset."""
    original_hw = tuple(int(v) for v in original_hw)
    if original_hw == (480, 640):
        K = _NYU_RAW_K.copy()
    elif original_hw == (240, 320):
        K = _scale_intrinsics(_NYU_RAW_K, (480, 640), original_hw)
    else:
        # Formal NYU48K is 228x304.  Scaling this calibrated cropped camera also
        # keeps explicitly resized exports geometrically consistent.
        K = _scale_intrinsics(_NYU_CROPPED_K, _NYU_CROPPED_HW, original_hw)
    height, width = original_hw
    if "resize" in preprocess:
        K = _scale_intrinsics(K, (height, width), (load_size, load_size))
        height = width = int(load_size)
    if "crop" in preprocess:
        crop_h = min(int(crop_size), height)
        crop_w = min(int(crop_size), width)
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        K[0, 2] -= left
        K[1, 2] -= top
        height, width = crop_h, crop_w
    if flip:
        K[0, 2] = (width - 1) - K[0, 2]
    return torch.from_numpy(K.astype(np.float32, copy=False))


def _kitti_intrinsics(date, crop_box, resize_size):
    date = str(date or "")
    if date not in _KITTI_K_BY_DATE:
        raise RuntimeError(
            f"KITTI manifest date {date!r} has no calibrated intrinsics; refusing to guess")
    fx, fy, cx, cy = _KITTI_K_BY_DATE[date]
    top, left, crop_h, crop_w = (float(v) for v in crop_box)
    out_h, out_w = (float(v) for v in resize_size)
    sx, sy = out_w / crop_w, out_h / crop_h
    return torch.tensor([
        [fx * sx, 0.0, (cx - left) * sx],
        [0.0, fy * sy, (cy - top) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def _resolve_config_path(config_path, value):
    """Resolve a GeoLayerNet data path without importing its code or changing it."""
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    config = Path(config_path).expanduser().resolve()
    configs_root = next((parent for parent in (config.parent, *config.parents)
                         if parent.name == "configs"), None)
    repo_root = configs_root.parent if configs_root is not None else config.parent.parent
    return str((repo_root / path).resolve())


def load_dataset_config(config_path):
    """Read only dataset/eval fields from a GeoLayerNet YAML configuration.

    The returned object intentionally does not expose model fields to adapters.
    Relative GeoLayerNet paths are resolved against the reference repository root
    (the parent of ``configs``), matching the reference command-line behavior.
    """
    if not config_path:
        return {
            "path": None, "dataset": {}, "train": {}, "eval": {},
            "dataset_type": "nyu", "dataroot": None, "preprocess": "none",
            "depth_unit_scale": None, "min_depth": None, "max_depth": None,
            "clip_prediction": None, "error_max": None,
        }
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("--dataset-config requires PyYAML") from exc
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"dataset config does not exist: {config_path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"dataset config must contain a YAML mapping: {config_path}")
    dataset = dict(raw.get("dataset") or {})
    train_cfg = dict(raw.get("train") or {})
    eval_cfg = dict(raw.get("eval") or {})
    dataset_type = str(dataset.get("type") or "nyu").lower()
    if dataset_type in ("rgbd_completion", "rgbd_completion_nyu"):
        dataset_type = "nyu"
    dataroot = dataset.get("dataroot") or dataset.get("data_root")
    return {
        "path": str(path),
        "dataset": dataset,
        "train": train_cfg,
        "eval": eval_cfg,
        "dataset_type": dataset_type,
        "dataroot": _resolve_config_path(path, dataroot),
        "dataset_name": dataset.get("dataset_tag"),
        "preprocess": dataset.get("preprocess"),
        "depth_unit_scale": dataset.get("depth_unit_scale", eval_cfg.get("depth_unit_scale")),
        "min_depth": eval_cfg.get("min_depth"),
        "max_depth": eval_cfg.get("max_depth"),
        "clip_prediction": eval_cfg.get("clip_prediction"),
        "error_max": eval_cfg.get("error_max"),
        "val_phase": train_cfg.get("val_phase"),
        "eval_phase": eval_cfg.get("phase"),
    }


def _list(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in _EXTS}


def _clean_line(raw):
    line = raw.split("#", 1)[0].strip()
    if not line:
        return None
    return [p for p in re.split(r"[\s,]+", line) if p]


def read_sample_list(path):
    """Return ordered list entries from a sample list.

    The first token is treated as the stable sample id. Extra columns are kept so
    future list formats can carry paths, but the current GeoLayerNet directory
    protocol resolves files by that id/stem under rgb/sparse_depth/target_depth.
    """
    if not path:
        return None
    entries = []
    with open(path) as fh:
        for line_no, raw in enumerate(fh, 1):
            parts = _clean_line(raw)
            if not parts:
                continue
            sample_id = _normalize_sample_id(parts[0])
            entries.append({"sample_id": sample_id, "tokens": parts, "line": line_no})
    if not entries:
        raise RuntimeError(f"sample list is empty: {path}")
    return entries


def _normalize_sample_id(value):
    text = str(value).strip().replace("\\", "/")
    if not text:
        return text
    text = text.lstrip("./")
    suffix = Path(text).suffix
    if suffix.lower() in _EXTS:
        text = text[: -len(suffix)]
    parts = [p for p in text.split("/") if p and p not in (".",)]
    if parts and parts[0] in ("rgb", "sparse_depth", "target_depth", "valid_mask"):
        parts = parts[1:]
    return "/".join(parts)


def _dataset_namespace(dataroot, split_root):
    """Stable display namespace; never used as a leakage identity."""
    candidates = [Path(dataroot).expanduser(), Path(split_root)]
    for candidate in candidates:
        for part in reversed(candidate.parts):
            if part.lower().startswith("nyu48k"):
                return "NYU48K"
    root = Path(split_root)
    base = root.parent.name if root.name.lower() in ("train", "val", "test") else root.name
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return clean.upper() or "DATASET"


def _bookkeeping_sample_id(namespace, phase, local_sample_id):
    local = _normalize_sample_id(local_sample_id)
    return "/".join((str(namespace), str(phase), local))


def _manifest_metadata(tokens):
    """Read only explicit source/scene identities; never guess from a flat stem."""
    out = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        value = value.strip().replace("\\", "/")
        if key in ("canonical_source_id", "source_id", "source_identity") and value:
            out["canonical_source_id"] = value
        elif key in ("scene_id", "scene_identity") and value:
            out["explicit_scene_id"] = value
    return out


def _key_variants(path, split_root):
    rel = path.relative_to(split_root).as_posix()
    no_suffix = rel[: -len(path.suffix)] if path.suffix else rel
    stem = path.stem
    variants = {_normalize_sample_id(no_suffix), _normalize_sample_id(stem)}
    parts = no_suffix.split("/")
    if parts and parts[0] in ("rgb", "sparse_depth", "target_depth", "valid_mask"):
        variants.add(_normalize_sample_id("/".join(parts[1:])))
    return {v for v in variants if v}


def resolve_split_root(dataroot, phase):
    if not dataroot:
        raise RuntimeError("missing required dataset path")
    root = Path(dataroot).expanduser()
    direct = root
    phased = root / phase if phase else root
    if (phased / "rgb").is_dir():
        return phased
    if (direct / "rgb").is_dir():
        return direct
    raise RuntimeError(
        "dataset path does not contain an rgb directory: "
        f"dataroot={dataroot!r}, phase={phase!r}, checked={phased / 'rgb'} and {direct / 'rgb'}")


def _index_by_id(paths, split_root):
    out = {}
    for p in paths.values():
        for key in _key_variants(p, split_root):
            out.setdefault(key, p)
    return out


def _rel_no_suffix(path, split_root):
    rel = path.relative_to(split_root).as_posix()
    if path.suffix:
        rel = rel[: -len(path.suffix)]
    if rel.startswith("rgb/"):
        rel = rel[4:]
    return rel


def _safe_resolve(path):
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def _relative_identity(path, split_root):
    try:
        rel = Path(path).absolute().relative_to(Path(split_root).absolute()).as_posix()
    except Exception:
        rel = Path(path).as_posix()
    return _normalize_sample_id(rel)


def _infer_scene_id(sample):
    explicit = sample.get("explicit_scene_id")
    if explicit:
        return str(explicit), "metadata"
    sid = sample.get("local_sample_id", sample.get("sample_id", "")).replace("\\", "/")
    parts = [p for p in sid.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[:-1]), "directory"
    return None, "unresolved"


def _canonical_source_id(sample):
    explicit = sample.get("canonical_source_id")
    if explicit:
        return str(explicit)
    local = sample.get("local_sample_id", "").replace("\\", "/")
    # A nested exported path contains source context; a flat re-numbered stem
    # such as 00000000 deliberately has no canonical source identity.
    return local if "/" in local else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _content_id(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    stat = p.stat()
    return {
        "path": _safe_resolve(p),
        "size": int(stat.st_size),
        "sha256": _sha256(p),
    }


class Nyu500Dataset(torch.utils.data.Dataset):
    """GeoLayerNet directory dataset with explicit split/list selection."""

    def __init__(self, dataroot, phase="test", sample_list=None, preprocess="none",
                 load_size=286, crop_size=256, no_flip=True, depth_scale=0.0,
                 dynamic_sparse=False, sparse_points=None, sparse_pattern=None,
                 depth_noise_std=0.0, missing_block_prob=0.0, config_hint=None):
        if dynamic_sparse or sparse_points or sparse_pattern or depth_noise_std or missing_block_prob:
            raise RuntimeError(
                "dynamic sparse generation is disabled for comparison protocol: "
                "use pre-generated sparse_depth files selected by --*-dataroot/--*-list")
        self.config_hint = config_hint
        self.dataroot = str(dataroot)
        self.root = resolve_split_root(dataroot, phase)
        self.phase = phase
        self.dataset_namespace = _dataset_namespace(dataroot, self.root)
        self.sample_list = sample_list
        self.preprocess = preprocess
        self.load_size = load_size
        self.crop_size = crop_size
        self.no_flip = no_flip
        self.depth_scale = depth_scale

        rgb = _list(self.root / "rgb")
        sparse = _list(self.root / "sparse_depth")
        target = _list(self.root / "target_depth")
        vmask = _list(self.root / "valid_mask")
        if not rgb:
            print("[data][ERROR] no RGB images found -- check the dataset path.")
            print(f"  configured dataroot : {dataroot}")
            print(f"  resolved data dir   : {self.root.resolve()}")
            print(f"  expected rgb dir    : {(self.root / 'rgb').resolve()}")
            print(f"  config              : {self.config_hint or '(not provided)'}")
            print(f"  current working dir : {Path.cwd()}")
            raise RuntimeError(f"no rgb images under {self.root / 'rgb'}")
        if not sparse:
            raise RuntimeError(f"no sparse_depth under {self.root / 'sparse_depth'}")
        if not target:
            raise RuntimeError(f"no target_depth under {self.root / 'target_depth'}")

        rgb_idx = _index_by_id(rgb, self.root)
        sparse_idx = _index_by_id(sparse, self.root)
        target_idx = _index_by_id(target, self.root)
        vmask_idx = _index_by_id(vmask, self.root)
        entries = read_sample_list(sample_list)
        if entries is None:
            entries = [
                {"sample_id": _normalize_sample_id(_rel_no_suffix(p, self.root)),
                 "tokens": [], "line": None}
                for _k, p in sorted(rgb.items(), key=lambda kv: kv[0])
            ]

        self.samples = []
        missing = []
        seen = set()
        for entry in entries:
            local_sample_id = entry["sample_id"]
            if local_sample_id in seen:
                raise RuntimeError(f"duplicate sample_id within list/split: {local_sample_id}")
            seen.add(local_sample_id)
            rp = rgb_idx.get(local_sample_id) or rgb_idx.get(Path(local_sample_id).stem)
            sp = sparse_idx.get(local_sample_id) or sparse_idx.get(Path(local_sample_id).stem)
            tp = target_idx.get(local_sample_id) or target_idx.get(Path(local_sample_id).stem)
            vp = vmask_idx.get(local_sample_id) or vmask_idx.get(Path(local_sample_id).stem)
            if not (rp and sp and tp):
                missing.append({
                    "sample_id": local_sample_id, "line": entry.get("line"),
                    "rgb": bool(rp), "sparse_depth": bool(sp), "target_depth": bool(tp),
                })
                continue
            metadata = _manifest_metadata(entry.get("tokens") or [])
            # The split-local numbering collision is a NYU48K export property.
            # Preserve legacy IDs for unrelated directory datasets while making
            # NYU48K result/log IDs globally unique.
            sample_id = (_bookkeeping_sample_id(
                self.dataset_namespace, phase, local_sample_id)
                if self.dataset_namespace == "NYU48K" else local_sample_id)
            self.samples.append({
                "sample_id": sample_id,
                "local_sample_id": local_sample_id,
                "name": sample_id,
                "rgb": rp,
                "sparse": sp,
                "target": tp,
                "valid": vp,
                **metadata,
            })
        if missing:
            preview = ", ".join(m["sample_id"] for m in missing[:8])
            raise RuntimeError(
                f"{len(missing)} listed samples missing required files under {self.root}; "
                f"first missing: {preview}")
        if not self.samples:
            raise RuntimeError(f"no matched samples under {self.root}")

    def __len__(self):
        return len(self.samples)

    def _geom(self, img, flip, interp):
        if "resize" in self.preprocess:
            img = TF.resize(img, [self.load_size, self.load_size], interpolation=interp)
        if "crop" in self.preprocess:
            img = TF.center_crop(img, [self.crop_size, self.crop_size])
        if flip:
            img = TF.hflip(img)
        return img

    def _rgb(self, path, flip):
        img = self._geom(Image.open(path).convert("RGB"), flip, TF.InterpolationMode.BICUBIC)
        return TF.normalize(TF.to_tensor(img), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _depth(self, path, flip):
        img = self._geom(Image.open(path), flip, TF.InterpolationMode.NEAREST)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        scale = self.depth_scale
        if scale <= 0:
            scale = 65535.0 if arr.max(initial=0) > 255.0 else 255.0
        arr = np.clip(arr / max(scale, 1.0), 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, i):
        s = self.samples[i]
        flip = (self.phase == "train") and (not self.no_flip) and (np.random.rand() > 0.5)
        with Image.open(s["rgb"]) as source_rgb:
            original_hw = (source_rgb.height, source_rgb.width)
        rgb = self._rgb(s["rgb"], flip)
        sparse = self._depth(s["sparse"], flip)
        target = self._depth(s["target"], flip)
        valid = (sparse > 0).float()
        return {
            "rgb": rgb,
            "sparse_depth": sparse,
            "target_depth": target,
            "valid_mask": valid,
            "target_mask": (target > 0).float(),
            "intrinsics": _nyu_intrinsics(
                original_hw, self.preprocess, self.load_size, self.crop_size, flip),
            "intrinsics_source": "NYUv2 official calibration transformed with loader geometry",
            # Geometry metadata only: this reports the already-sampled common
            # loader transform so external dense priors can stay aligned. It
            # does not add another random draw or change any tensor content.
            "horizontal_flip": bool(flip),
            "name": s["sample_id"],
            "sample_id": s["sample_id"],
            "rgb_path": str(s["rgb"]),
            "sparse_path": str(s["sparse"]),
            "target_path": str(s["target"]),
        }


def _read_kitti_depth_m(path, png_scale):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32) / float(png_scale)


class KittiManifestDataset(torch.utils.data.Dataset):
    """Read the formal GeoLayerNet KITTI JSONL protocol without scanning data.

    ``sample_list`` is deliberately mandatory: train90, dev10 and val1000 are
    protocol decisions, not directory names that a comparison method may infer.
    The tensor preparation mirrors GeoLayerNet's read-only KittiDataset.
    """

    def __init__(self, dataroot, phase="val1000", sample_list=None,
                 max_depth=80.0, depth_png_scale=256.0,
                 kitti_layout="manifest", kitti_crop=(256, 1216),
                 train_crop=(256, 832), resize_to=None, color_jitter=True,
                 source_root=None, config_hint=None, expected_val1000_samples=None,
                 expected_train_full_samples=None, **_ignored):
        if str(kitti_layout).lower() != "manifest":
            raise RuntimeError(
                "formal comparison KITTI protocol requires kitti_layout=manifest; "
                "automatic/preprocessed directory scanning is disabled")
        if not sample_list:
            raise RuntimeError(
                "KITTI requires an explicit JSONL manifest: use --train-list "
                "train90.jsonl, --val-list dev10.jsonl, or --test-list val1000.jsonl")
        self.dataroot = str(dataroot)
        self.root = Path(source_root).expanduser() if source_root else Path(dataroot).expanduser()
        self.phase = str(phase)
        self.max_depth = float(max_depth)
        self.png_scale = float(depth_png_scale)
        self.kitti_crop = tuple(int(v) for v in kitti_crop)
        self.train_crop = tuple(int(v) for v in train_crop)
        self.resize_to = tuple(int(v) for v in resize_to) if resize_to else None
        self.color_jitter = bool(color_jitter)
        self.config_hint = config_hint
        self.sample_list = str(sample_list)
        self._jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.05)
        manifest = Path(sample_list).expanduser()
        if not manifest.is_absolute():
            manifest = Path.cwd() / manifest
        if not manifest.is_file():
            raise RuntimeError(f"KITTI manifest does not exist: {manifest}")
        self.manifest = manifest.resolve()
        self.samples = self._read_manifest()
        expected = None
        if self.phase == "val1000" and expected_val1000_samples is not None:
            expected = int(expected_val1000_samples)
        elif self.phase == "train_full" and expected_train_full_samples is not None:
            expected = int(expected_train_full_samples)
        if expected is not None and len(self.samples) != expected:
            raise RuntimeError(
                f"KITTI {self.phase} manifest count mismatch: expected {expected}, "
                f"got {len(self.samples)} ({self.manifest})")
        if not self.samples:
            raise RuntimeError(f"KITTI manifest is empty: {self.manifest}")

    def _read_manifest(self):
        samples = []
        seen = set()
        with self.manifest.open() as fh:
            for line_no, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"invalid KITTI JSONL at {self.manifest}:{line_no}: {exc}") from exc
                required = ("sample_id", "rgb", "sparse_depth", "target_depth")
                missing = [key for key in required if key not in record]
                if missing:
                    raise RuntimeError(
                        f"KITTI manifest {self.manifest}:{line_no} missing keys {missing}")
                local_id = _normalize_sample_id(record["sample_id"])
                if local_id in seen:
                    raise RuntimeError(f"duplicate KITTI sample_id in manifest: {local_id}")
                seen.add(local_id)

                def source_path(key):
                    value = Path(str(record[key])).expanduser()
                    path = value if value.is_absolute() else self.root / value
                    if not path.is_file():
                        raise RuntimeError(
                            f"KITTI manifest {self.manifest}:{line_no} missing {key}: {path}")
                    resolved_path = path.resolve()
                    resolved_root = self.root.resolve()
                    try:
                        resolved_path.relative_to(resolved_root)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"KITTI manifest {self.manifest}:{line_no} {key} "
                            f"escapes registered source root {resolved_root}: "
                            f"{resolved_path}") from exc
                    return resolved_path

                date = record.get("date")
                drive = record.get("drive")
                camera = record.get("camera")
                frame = record.get("frame")
                canonical = None
                scene = None
                if date and drive:
                    scene = f"{date}/{drive}"
                if date and drive and camera and frame:
                    canonical = "/".join(map(str, (date, drive, camera, frame)))
                samples.append({
                    "sample_id": _bookkeeping_sample_id("KITTI", self.phase, local_id),
                    "local_sample_id": local_id,
                    "name": _bookkeeping_sample_id("KITTI", self.phase, local_id),
                    "rgb": source_path("rgb"),
                    "sparse": source_path("sparse_depth"),
                    "target": source_path("target_depth"),
                    "valid": None,
                    "date": date, "drive": drive, "camera": camera, "frame": frame,
                    "explicit_scene_id": scene,
                    "canonical_source_id": canonical,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _crop_box(self, height, width):
        crop_h, crop_w = self.train_crop if self.phase == "train" else self.kitti_crop
        crop_h, crop_w = min(crop_h, height), min(crop_w, width)
        top = height - crop_h
        if self.phase == "train":
            low = max(height - self.kitti_crop[0], 0)
            top = np.random.randint(low, height - crop_h + 1) if height - crop_h > 0 else 0
            left = np.random.randint(0, width - crop_w + 1) if width - crop_w > 0 else 0
        else:
            left = (width - crop_w) // 2
        return top, left, crop_h, crop_w

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb = Image.open(sample["rgb"]).convert("RGB")
        sparse_m = _read_kitti_depth_m(sample["sparse"], self.png_scale)
        target_m = _read_kitti_depth_m(sample["target"], self.png_scale)
        rgb_np = np.asarray(rgb)
        height, width = target_m.shape
        if rgb_np.shape[:2] != (height, width):
            rgb = rgb.resize((width, height), Image.BILINEAR)
            rgb_np = np.asarray(rgb)
        top, left, crop_h, crop_w = self._crop_box(height, width)
        source_h, source_w = height, width
        rgb_np = np.ascontiguousarray(rgb_np[top:top + crop_h, left:left + crop_w])
        sparse_m = np.ascontiguousarray(sparse_m[top:top + crop_h, left:left + crop_w])
        target_m = np.ascontiguousarray(target_m[top:top + crop_h, left:left + crop_w])
        rgb = Image.fromarray(rgb_np)
        if self.phase == "train" and self.color_jitter:
            rgb = self._jitter(rgb)
        if self.resize_to is not None:
            out_h, out_w = self.resize_to
            rgb = rgb.resize((out_w, out_h), Image.BILINEAR)
            sparse_m = np.asarray(Image.fromarray(sparse_m, mode="F").resize(
                (out_w, out_h), Image.NEAREST)).copy()
            target_m = np.asarray(Image.fromarray(target_m, mode="F").resize(
                (out_w, out_h), Image.NEAREST)).copy()
        rgb_t = TF.normalize(TF.to_tensor(rgb), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        sparse_t = torch.from_numpy(np.clip(
            sparse_m / self.max_depth, 0.0, 1.0)).unsqueeze(0).float()
        target_t = torch.from_numpy(np.clip(
            target_m / self.max_depth, 0.0, 1.0)).unsqueeze(0).float()
        valid_t = torch.from_numpy(sparse_m).unsqueeze(0).gt(0).float()
        target_mask = (torch.from_numpy(target_m).unsqueeze(0).gt(0) &
                       torch.from_numpy(target_m).unsqueeze(0).le(self.max_depth)).float()
        return {
            "rgb": rgb_t, "sparse_depth": sparse_t, "target_depth": target_t,
            "valid_mask": valid_t, "target_mask": target_mask,
            "name": sample["sample_id"], "sample_id": sample["sample_id"],
            "rgb_path": str(sample["rgb"]), "sparse_path": str(sample["sparse"]),
            "target_path": str(sample["target"]),
            # Explicit geometry/manifest metadata for methods (such as PSD)
            # that require calibrated camera intrinsics. These describe the
            # exact crop already applied above; no frame identity is inferred.
            "source_size": torch.tensor([source_h, source_w], dtype=torch.int64),
            "crop_box": torch.tensor([top, left, crop_h, crop_w], dtype=torch.int64),
            "resize_size": torch.tensor(list(self.resize_to or (crop_h, crop_w)),
                                        dtype=torch.int64),
            "intrinsics": _kitti_intrinsics(
                sample.get("date"), (top, left, crop_h, crop_w),
                self.resize_to or (crop_h, crop_w)),
            "intrinsics_source": "KITTI official per-date calibration transformed with loader geometry",
            "kitti_date": str(sample.get("date") or ""),
            "kitti_drive": str(sample.get("drive") or ""),
            "kitti_camera": str(sample.get("camera") or ""),
            "kitti_frame": str(sample.get("frame") or ""),
        }


def _as_chw_rgb(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"OOD rgb must be 3-D, got shape {arr.shape}")
    if arr.shape[0] == 3:
        out = arr
    elif arr.shape[-1] == 3:
        out = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"OOD rgb must have 3 channels, got shape {arr.shape}")
    out = out.astype(np.float32, copy=False)
    if np.nanmax(out) > 1.5:
        out = out / 255.0
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0))


def _as_1hw(arr, key):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"OOD {key} must be 2-D or single-channel 3-D, got {arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False)[None, ...])


def _npz_name(value, fallback):
    if value is None:
        return fallback
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()
    else:
        value = arr.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _validate_ood_k_matrix(arr, path, key):
    """Accept only floating (3,3) camera matrices; never invent or reshape junk."""
    arr = np.asarray(arr)
    if arr.dtype.kind not in "f":
        raise ValueError(
            f"OOD {key} must be floating dtype in {path}, got {arr.dtype}")
    if arr.shape != (3, 3):
        raise ValueError(
            f"OOD {key} must be float32 (3,3) in {path}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"OOD {key} has non-finite values in {path}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _optional_npz_intrinsics(archive, path):
    """Load per-sample K from NPZ ``K`` (preferred) or legacy ``intrinsics``.

    Injected GeoLayerNet OOD NPZs store ``K`` at the NPZ RGB H×W.  This loader
    never rescales with ``intrinsics_source_size`` (provenance only).
    """
    has_k = "K" in archive.files
    has_intrinsics = "intrinsics" in archive.files
    if not has_k and not has_intrinsics:
        return None
    if has_k and has_intrinsics:
        k_mat = _validate_ood_k_matrix(archive["K"], path, "K")
        legacy = _validate_ood_k_matrix(archive["intrinsics"], path, "intrinsics")
        if not np.array_equal(k_mat, legacy):
            raise ValueError(
                f"OOD NPZ {path} has disagreeing K and intrinsics matrices")
        return k_mat
    key = "K" if has_k else "intrinsics"
    return _validate_ood_k_matrix(archive[key], path, key)


def _attach_ood_intrinsics_provenance(archive, path, out):
    """Pass through calibration provenance; never use it for numeric transforms."""
    if "camera_model" in archive.files:
        out["camera_model"] = _npz_name(archive["camera_model"], "")
    if "intrinsics_source" in archive.files:
        out["intrinsics_source"] = _npz_name(archive["intrinsics_source"], "")
    if "psd_manifest_source" in archive.files:
        out["psd_manifest_source"] = _npz_name(archive["psd_manifest_source"], "")
    if "psd_manifest_line_number" in archive.files:
        line = np.asarray(archive["psd_manifest_line_number"]).reshape(-1)
        if line.size != 1:
            raise ValueError(
                f"OOD psd_manifest_line_number must be scalar in {path}, "
                f"got shape {np.asarray(archive['psd_manifest_line_number']).shape}")
        out["psd_manifest_line_number"] = int(line[0])
    if "intrinsics_source_size" in archive.files:
        size = np.asarray(archive["intrinsics_source_size"])
        if size.shape != (2,):
            raise ValueError(
                f"OOD intrinsics_source_size must be int32 (2,) in {path}, "
                f"got shape {size.shape}")
        out["intrinsics_source_size"] = torch.from_numpy(
            np.ascontiguousarray(size.astype(np.int32, copy=False)))


class OodNpzDataset(torch.utils.data.Dataset):
    """Reader for GeoLayerNet's prepared OOD ``npz`` samples.

    GeoLayerNet stores OOD RGB in [0,1] and depth in metres. Every OOD domain is
    mapped once at this shared boundary to the frozen 228x304 geometry: RGB uses
    bilinear interpolation; sparse/target depth use nearest. Intrinsics stored at
    the NPZ RGB size are scaled exactly once here. No sparse samples are generated
    and no GT-derived input is made.
    """

    def __init__(self, dataroot, phase="test", sample_list=None, depth_scale=10.0,
                 config_hint=None, dataset_cfg=None, eval_cfg=None,
                 protocol_depth_unit_scale=None, **_ignored):
        self.dataroot = Path(dataroot).expanduser()
        self.root = self.dataroot
        self.phase = phase
        self.sample_list = sample_list
        self.config_hint = config_hint
        dataset_cfg = dict(dataset_cfg or {})
        eval_cfg = dict(eval_cfg or {})
        configured_scale = dataset_cfg.get(
            "depth_unit_scale", eval_cfg.get("depth_unit_scale"))
        selected_scale = (
            protocol_depth_unit_scale
            if protocol_depth_unit_scale is not None else configured_scale)
        self.depth_unit_scale = float(selected_scale or depth_scale or 10.0)
        if self.depth_unit_scale <= 0:
            raise RuntimeError(f"OOD depth_unit_scale must be > 0, got {self.depth_unit_scale}")
        self.samples = self._index()
        if not self.samples:
            raise RuntimeError(
                f"no GeoLayerNet OOD npz samples under {self.dataroot}; "
                f"config={config_hint or '(not provided)'}")

    def _candidate_files(self):
        manifest = self.dataroot / f"{self.phase}.txt"
        if not manifest.is_file() and self.phase != "test":
            manifest = self.dataroot / "test.txt"
        if manifest.is_file():
            files = []
            for raw in manifest.read_text().splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                path = Path(line).expanduser()
                if not path.is_absolute():
                    path = self.dataroot / path
                if path.suffix.lower() != ".npz":
                    path = path.with_suffix(".npz")
                if not path.is_file():
                    raise RuntimeError(f"OOD manifest entry missing: {line} -> {path}")
                resolved_path = path.resolve()
                resolved_root = self.dataroot.resolve()
                try:
                    resolved_path.relative_to(resolved_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"OOD manifest entry escapes registered domain root "
                        f"{resolved_root}: {resolved_path}") from exc
                files.append(resolved_path)
            return files
        raise RuntimeError(
            f"OOD domain requires an explicit {manifest} manifest; "
            "directory scanning is forbidden by GeoLayerNet Comparison Protocol V2")

    def _index(self):
        files = self._candidate_files()
        by_id = {}
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as z:
                    sample_id = _normalize_sample_id(_npz_name(z["name"] if "name" in z.files else None, path.stem))
            except Exception as exc:
                raise RuntimeError(f"failed to inspect OOD sample {path}: {exc}") from exc
            if sample_id in by_id:
                raise RuntimeError(f"duplicate OOD sample_id: {sample_id}")
            by_id[sample_id] = path
        if self.sample_list:
            entries = read_sample_list(self.sample_list)
            selected = []
            missing = []
            for entry in entries:
                sid = entry["sample_id"]
                path = by_id.get(sid) or by_id.get(Path(sid).stem)
                if not path:
                    missing.append(sid)
                else:
                    selected.append((sid, path))
            if missing:
                raise RuntimeError(f"OOD sample list entries missing: {', '.join(missing[:8])}")
        else:
            selected = sorted(by_id.items(), key=lambda item: item[0])
        return [{"sample_id": sid, "name": sid, "npz": path,
                 "rgb": path, "sparse": path, "target": path, "valid": None}
                for sid, path in selected]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        path = sample["npz"]
        with np.load(path, allow_pickle=False) as z:
            missing = [key for key in ("rgb", "sparse_depth", "target_depth") if key not in z.files]
            if missing:
                raise KeyError(f"{path} missing required OOD npz keys: {missing}")
            rgb01 = _as_chw_rgb(z["rgb"])
            sparse_m = _as_1hw(z["sparse_depth"], "sparse_depth")
            target_m = _as_1hw(z["target_depth"], "target_depth")
            original_h, original_w = (int(rgb01.shape[-2]), int(rgb01.shape[-1]))
            if sparse_m.shape[-2:] != (original_h, original_w):
                raise ValueError(
                    f"OOD sparse_depth shape {sparse_m.shape[-2:]} does not "
                    f"match RGB {(original_h, original_w)} in {path}")
            if target_m.shape[-2:] != (original_h, original_w):
                raise ValueError(
                    f"OOD target_depth shape {target_m.shape[-2:]} does not "
                    f"match RGB {(original_h, original_w)} in {path}")
            if "original_input_size" in z.files:
                recorded_size = np.asarray(z["original_input_size"])
                if recorded_size.shape != (2,) or tuple(
                        int(v) for v in recorded_size.tolist()) != (
                            original_h, original_w):
                    raise ValueError(
                        f"OOD original_input_size must equal NPZ RGB HxW "
                        f"{(original_h, original_w)} in {path}, got "
                        f"{recorded_size.tolist()}")
            intrinsics = _optional_npz_intrinsics(z, path)
            provenance = {}
            _attach_ood_intrinsics_provenance(z, path, provenance)

        output_h, output_w = _OOD_PROTOCOL_HW
        rgb_source = torch.from_numpy(rgb01)
        sparse_source = torch.from_numpy(sparse_m)
        target_source = torch.from_numpy(target_m)
        rgb_tensor = F.interpolate(
            rgb_source.unsqueeze(0),
            size=_OOD_PROTOCOL_HW, mode="bilinear", align_corners=False)[0]
        sparse_m_tensor = F.interpolate(
            sparse_source.unsqueeze(0),
            size=_OOD_PROTOCOL_HW, mode="nearest")[0]
        target_m_tensor = F.interpolate(
            target_source.unsqueeze(0),
            size=_OOD_PROTOCOL_HW, mode="nearest")[0]
        # Masks follow the same nearest geometry, then are authoritatively
        # regenerated below from the resized depths.
        valid_nearest = F.interpolate(
            (sparse_source > 0).float().unsqueeze(0),
            size=_OOD_PROTOCOL_HW, mode="nearest")[0]
        target_nearest = F.interpolate(
            (target_source > 0).float().unsqueeze(0),
            size=_OOD_PROTOCOL_HW, mode="nearest")[0]
        sparse = sparse_m_tensor / self.depth_unit_scale
        target = target_m_tensor / self.depth_unit_scale
        valid_mask = (sparse > 0).float()
        target_mask = (target > 0).float()
        if not torch.equal(valid_mask, valid_nearest):
            raise RuntimeError("OOD nearest sparse mask and regenerated mask disagree")
        if not torch.equal(target_mask, target_nearest):
            raise RuntimeError("OOD nearest target mask and regenerated mask disagree")
        out = {
            "rgb": (rgb_tensor - 0.5) / 0.5,
            "sparse_depth": sparse,
            "target_depth": target,
            "valid_mask": valid_mask,
            "target_mask": target_mask,
            "name": sample["sample_id"],
            "sample_id": sample["sample_id"],
            "rgb_path": str(path),
            "sparse_path": str(path),
            "target_path": str(path),
            "npz_path": str(path),
            "original_input_size": torch.tensor(
                [original_h, original_w], dtype=torch.int32),
            "protocol_input_size": torch.tensor(
                [output_h, output_w], dtype=torch.int32),
            "intrinsics_resize_applied": torch.tensor(True),
        }
        out.update(provenance)
        if intrinsics is not None:
            k_tensor = torch.from_numpy(
                _scale_intrinsics(
                    intrinsics, (original_h, original_w), _OOD_PROTOCOL_HW))
            # Adapter historically reads ``intrinsics``; expose ``K`` as the
            # NPZ-native name so batch fields match injected OOD payloads.
            out["K"] = k_tensor
            out["intrinsics"] = k_tensor
        return out


def make_dataset(dataroot, phase, sample_list=None, preprocess="none",
                 depth_scale=0.0, max_samples=None, config_hint=None,
                 dataset_type="nyu", dataset_opts=None):
    dtype = str(dataset_type or "nyu").lower()
    if dtype == "ood_npz":
        opts = dict(dataset_opts or {})
        ds = OodNpzDataset(dataroot, phase=phase, sample_list=sample_list,
                           depth_scale=depth_scale, config_hint=config_hint,
                           dataset_cfg=opts.get("dataset_cfg", opts),
                           eval_cfg=opts.get("eval_cfg", {}),
                           protocol_depth_unit_scale=opts.get(
                               "protocol_depth_unit_scale"))
    elif dtype == "kitti":
        opts = dict(dataset_opts or {})
        opts.pop("dataset_cfg", None)
        opts.pop("eval_cfg", None)
        train_cfg = opts.pop("train_cfg", {}) or {}
        for key in ("dataroot", "data_root", "type", "dataset_mode", "dataset_tag",
                    "preprocess", "input_nc", "depth_unit_scale"):
            opts.pop(key, None)
        opts.setdefault("expected_val1000_samples",
                        train_cfg.get("expected_val1000_samples"))
        opts.setdefault("expected_train_full_samples",
                        train_cfg.get("expected_train_full_samples"))
        ds = KittiManifestDataset(
            dataroot, phase=phase, sample_list=sample_list,
            config_hint=config_hint, **opts)
    elif dtype == "nyu":
        ds = Nyu500Dataset(dataroot, phase=phase, sample_list=sample_list,
                           preprocess=preprocess, no_flip=(phase != "train"),
                           depth_scale=depth_scale, config_hint=config_hint)
    else:
        raise RuntimeError(f"unsupported comparison dataset type: {dataset_type}")
    if max_samples is not None and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    return ds


def dataset_samples(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        return [base.samples[i] for i in dataset.indices]
    return list(dataset.samples)


def create_loader(dataroot, phase, batch_size, num_workers=4, preprocess="none",
                  shuffle=False, depth_scale=0.0, max_samples=None,
                  sample_list=None, config_hint=None, seed=42,
                  dataset_type="nyu", dataset_opts=None, **disabled_kwargs):
    if any(disabled_kwargs.get(k) for k in (
            "dynamic_sparse", "sparse_points", "sparse_pattern",
            "depth_noise_std", "missing_block_prob")):
        raise RuntimeError("sparse-depth generation/augmentation options are disabled")
    ds = make_dataset(dataroot, phase=phase, sample_list=sample_list,
                      preprocess=preprocess, depth_scale=depth_scale,
                      max_samples=max_samples, config_hint=config_hint,
                      dataset_type=dataset_type, dataset_opts=dataset_opts)
    print(f"[data] {phase}: {len(ds)} samples")
    samples = dataset_samples(ds)
    if samples:
        print("[data] first sample_ids:", ", ".join(s["sample_id"] for s in samples[:5]))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=shuffle, worker_init_fn=seed_worker, generator=generator)


def _split_record(name, dataroot, phase, sample_list, dataset):
    samples = dataset_samples(dataset)
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    root = base.root
    if isinstance(base, OodNpzDataset):
        dataset_type = "ood_npz"
    elif isinstance(base, KittiManifestDataset):
        dataset_type = "kitti"
    else:
        dataset_type = "nyu"
    records = []
    for s in samples:
        scene, rule = _infer_scene_id(s)
        primary = s.get("npz") or s.get("rgb")
        local_sample_id = s.get("local_sample_id", s["sample_id"])
        records.append({
            "split": name,
            "sample_id": s["sample_id"],
            "local_sample_id": local_sample_id,
            "basename": Path(local_sample_id).stem,
            "short_relative_path": _normalize_sample_id(local_sample_id),
            "canonical_source_id": _canonical_source_id(s),
            "scene_id": scene,
            "scene_rule": rule,
            "dataset_type": dataset_type,
            "rgb_path": _safe_resolve(s.get("rgb") or primary),
            "sparse_path": _safe_resolve(s.get("sparse") or primary),
            "target_path": _safe_resolve(s.get("target") or primary),
            "valid_path": _safe_resolve(s["valid"]) if s.get("valid") else None,
            "rgb_rel": _relative_identity(s.get("rgb") or primary, root),
            "sparse_rel": _relative_identity(s.get("sparse") or primary, root),
            "target_rel": _relative_identity(s.get("target") or primary, root),
            "rgb_hash": _content_id(s.get("rgb") or primary),
            "sparse_hash": _content_id(s.get("sparse") or primary),
            "target_hash": _content_id(s.get("target") or primary),
        })
    return {
        "name": name,
        "dataroot": str(dataroot) if dataroot else None,
        "phase": phase,
        "sample_list": str(sample_list) if sample_list else None,
        "dataset_type": dataset_type,
        "resolved_root": _safe_resolve(root),
        "count": len(records),
        "records": records,
    }


def _split_content_fingerprint(split):
    """Hash the ordered sample identities and all three modality contents."""
    digest = hashlib.sha256()
    for record in split["records"]:
        item = {"sample_id": record["sample_id"]}
        for modality in ("rgb_hash", "sparse_hash", "target_hash"):
            content = record.get(modality) or {}
            item[modality] = {
                "sha256": content.get("sha256"),
                "size": content.get("size"),
            }
        digest.update(json.dumps(
            item, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "schema": "ordered-sample-content-sha256-v1",
        "count": int(split["count"]),
        "sha256": digest.hexdigest(),
    }


def _pairwise_overlaps(splits, field):
    out = {}
    for i, a in enumerate(splits):
        av = defaultdict(list)
        for r in a["records"]:
            value = r.get(field)
            if value is not None:
                av[value].append(r["sample_id"])
        for b in splits[i + 1:]:
            bv = defaultdict(list)
            for r in b["records"]:
                value = r.get(field)
                if value is not None:
                    bv[value].append(r["sample_id"])
            common = sorted(set(av) & set(bv))
            out[f"{a['name']}/{b['name']}"] = [
                {"value": v, a["name"]: av[v], b["name"]: bv[v]} for v in common
            ]
    return out


def _hash_overlaps(splits, hash_field):
    out = {}
    for i, a in enumerate(splits):
        av = defaultdict(list)
        for record in a["records"]:
            value = record.get(hash_field)
            if value:
                av[(value["sha256"], value["size"])].append(record)
        for b in splits[i + 1:]:
            bv = defaultdict(list)
            for record in b["records"]:
                value = record.get(hash_field)
                if value:
                    bv[(value["sha256"], value["size"])].append(record)
            common = sorted(set(av) & set(bv))
            out[f"{a['name']}/{b['name']}"] = [
                {
                    "sha256": sha,
                    "size": size,
                    a["name"]: [
                        {"sample_id": r["sample_id"], "path": r[hash_field]["path"]}
                        for r in av[(sha, size)]
                    ],
                    b["name"]: [
                        {"sample_id": r["sample_id"], "path": r[hash_field]["path"]}
                        for r in bv[(sha, size)]
                    ],
                }
                for sha, size in common
            ]
    return out


def _overlap_count(overlaps, excluded_pairs=()):
    excluded = set(excluded_pairs)
    return sum(len(items) for pair, items in overlaps.items() if pair not in excluded)


def _split_count(datasets, name):
    return next((item["count"] for item in datasets if item["name"] == name), 0)


def _known_val_test_protocol(datasets, rgb_hash, target_hash, sparse_hash, enabled):
    pair = "val/test"
    if not enabled or pair not in rgb_hash:
        return None
    val_count = _split_count(datasets, "val")
    test_count = _split_count(datasets, "test")
    rgb_count = len(rgb_hash.get(pair, []))
    target_count = len(target_hash.get(pair, []))
    sparse_count = len(sparse_hash.get(pair, []))
    if (val_count > 0 and val_count == test_count and
            rgb_count == val_count and target_count == val_count and sparse_count == 0):
        return {
            "pair": pair,
            "type": "allow_val_test_shared_rgb_gt_with_distinct_sparse",
            "status": "KNOWN_PROTOCOL",
            "shared_rgb_frames": rgb_count,
            "shared_gt_frames": target_count,
            "shared_sparse_depth": sparse_count,
            "relation": "same RGB/GT, independent sparse realizations",
        }
    return None


def _reused_basename_warnings(datasets, basename_overlaps):
    by_split = {item["name"]: item for item in datasets}
    warnings = []
    for pair, overlaps in basename_overlaps.items():
        left, right = pair.split("/", 1)
        left_records = defaultdict(list)
        right_records = defaultdict(list)
        for record in by_split[left]["records"]:
            left_records[record["basename"]].append(record)
        for record in by_split[right]["records"]:
            right_records[record["basename"]].append(record)
        harmless = 0
        for overlap in overlaps:
            basename = overlap["value"]
            for a in left_records[basename]:
                for b in right_records[basename]:
                    hashes_differ = all(
                        a.get(field) and b.get(field) and
                        a[field]["sha256"] != b[field]["sha256"]
                        for field in ("rgb_hash", "target_hash", "sparse_hash"))
                    if hashes_differ:
                        harmless += 1
                        break
                else:
                    continue
                break
        if harmless:
            warnings.append(
                f"{harmless} reused basenames across {pair} with different RGB/GT/sparse content")
    return warnings


def audit_splits(split_specs, output_dir=None, preprocess="none", depth_scale=0.0,
                 max_samples=None, config_hint=None, fail_on_leak=True,
                 allow_val_test_shared_rgb_gt_with_distinct_sparse=False):
    """Build split indexes, check leakage, print and optionally save audit files."""
    datasets = []
    warnings = []
    for spec in split_specs:
        if not spec.get("dataroot"):
            warnings.append(f"{spec['name']} split not provided; cross-split leakage coverage is incomplete")
            continue
        ds = make_dataset(spec["dataroot"], spec.get("phase") or spec["name"],
                          sample_list=spec.get("sample_list"), preprocess=preprocess,
                          depth_scale=depth_scale, max_samples=max_samples,
                          config_hint=spec.get("dataset_config") or config_hint,
                          dataset_type=spec.get("dataset_type", "nyu"),
                          dataset_opts=spec.get("dataset_opts"))
        datasets.append(_split_record(spec["name"], spec["dataroot"],
                                      spec.get("phase") or spec["name"],
                                      spec.get("sample_list"), ds))

    realpath_overlaps = {
        "rgb": _pairwise_overlaps(datasets, "rgb_path"),
        "sparse": _pairwise_overlaps(datasets, "sparse_path"),
        "target": _pairwise_overlaps(datasets, "target_path"),
    }
    modality_relative_overlaps = {
        "rgb_rel": _pairwise_overlaps(datasets, "rgb_rel"),
        "sparse_rel": _pairwise_overlaps(datasets, "sparse_rel"),
        "target_rel": _pairwise_overlaps(datasets, "target_rel"),
    }
    sample_overlaps = _pairwise_overlaps(datasets, "sample_id")
    basename_overlaps = _pairwise_overlaps(datasets, "basename")
    short_relative_overlaps = _pairwise_overlaps(datasets, "short_relative_path")
    canonical_overlaps = _pairwise_overlaps(datasets, "canonical_source_id")
    scene_overlaps = _pairwise_overlaps(datasets, "scene_id")
    unresolved = sum(1 for d in datasets for r in d["records"] if r["scene_id"] is None)
    if unresolved:
        warnings.append(
            f"scene identity unavailable for {unresolved} exported samples; no identity was inferred from re-numbered stems")
    hash_rgb = _hash_overlaps(datasets, "rgb_hash")
    hash_sparse = _hash_overlaps(datasets, "sparse_hash")
    hash_target = _hash_overlaps(datasets, "target_hash")
    warnings.extend(_reused_basename_warnings(datasets, basename_overlaps))

    known = _known_val_test_protocol(
        datasets, hash_rgb, hash_target, hash_sparse,
        bool(allow_val_test_shared_rgb_gt_with_distinct_sparse))
    known_protocol_exceptions = [known] if known else []
    except_hash_pairs = {"val/test"} if known else set()

    realpath_count = sum(_overlap_count(value) for value in realpath_overlaps.values())
    leak_counts = {
        "bookkeeping_sample_id_overlap": _overlap_count(sample_overlaps),
        "basename_overlap": _overlap_count(basename_overlaps),
        "short_relative_path_overlap": _overlap_count(short_relative_overlaps),
        "canonical_source_overlap": _overlap_count(canonical_overlaps),
        "realpath_overlap": realpath_count,
        "rgb_hash_overlap": _overlap_count(hash_rgb),
        "target_hash_overlap": _overlap_count(hash_target),
        "sparse_hash_overlap": _overlap_count(hash_sparse),
        "scene_overlap": _overlap_count(scene_overlaps),
    }
    failure_counts = {
        "canonical_source_overlap": leak_counts["canonical_source_overlap"],
        "realpath_overlap": leak_counts["realpath_overlap"],
        "rgb_hash_overlap": _overlap_count(hash_rgb, except_hash_pairs),
        "target_hash_overlap": _overlap_count(hash_target, except_hash_pairs),
        "sparse_hash_overlap": leak_counts["sparse_hash_overlap"],
        "scene_overlap": leak_counts["scene_overlap"],
    }
    if any(value > 0 for value in failure_counts.values()):
        status = "FAIL"
    elif known_protocol_exceptions:
        status = "KNOWN_PROTOCOL"
    else:
        status = "PASS"
    audit = {
        "protocol": {
            "rgb": "PIL RGB -> tensor [0,1] -> normalize mean/std 0.5 => [-1,1]; resize uses bicubic",
            "depth": "PNG/TIFF depth -> float32 [0,1], scale inferred as 65535 if max>255 else 255",
            "valid_mask": "valid_mask = sparse_depth > 0, never derived from target_depth",
            "target_valid_mask": "metrics use target_depth in [1e-3, 1.0]",
            "resize": "RGB bicubic; sparse/target/mask nearest; default preprocess=none",
            "dynamic_sparse": "disabled",
            "bookkeeping_sample_id": "<dataset-namespace>/<split>/<relative-stem>; display/logging only",
            "allow_val_test_shared_rgb_gt_with_distinct_sparse": bool(
                allow_val_test_shared_rgb_gt_with_distinct_sparse),
            "ood_npz": ("GeoLayerNet OOD npz stores RGB [0,1] and metre depths; "
                        "all domains resize once to 228x304 (RGB bilinear; "
                        "sparse/target nearest), regenerate masks, and scale K "
                        "from NPZ RGB HxW exactly once"),
            "kitti": ("explicit GeoLayerNet JSONL only; uint16 depth / 256 metres / "
                      "80 normalization; train bottom-band random 256x832 crop, eval "
                      "bottom-centre 256x1216 crop, optional configured 228x304 resize"),
        },
        "splits": [{k: v for k, v in d.items() if k != "records"} for d in datasets],
        "counts": {d["name"]: d["count"] for d in datasets},
        "split_fingerprints": {
            d["name"]: _split_content_fingerprint(d) for d in datasets
        },
        "bookkeeping_sample_id_examples": {
            d["name"]: [r["sample_id"] for r in d["records"][:5]] for d in datasets
        },
        "overlaps": {
            "bookkeeping_sample_id": sample_overlaps,
            "basename": basename_overlaps,
            "short_relative_path": short_relative_overlaps,
            "modality_relative_path": modality_relative_overlaps,
            "canonical_source": canonical_overlaps,
            "realpath": realpath_overlaps,
            "scene": scene_overlaps,
            "rgb_hash": hash_rgb,
            "sparse_hash": hash_sparse,
            "target_hash": hash_target,
        },
        "leak_counts": leak_counts,
        "failure_counts": failure_counts,
        "known_protocol_exceptions": known_protocol_exceptions,
        "warnings": warnings,
        "status": status,
    }
    if any(any(record.get("dataset_type") == "ood_npz"
               for record in dataset["records"]) for dataset in datasets):
        audit["protocol"].update({
            "input_resolution": [228, 304],
            "metric_resolution": [228, 304],
            "prediction_resolution": [228, 304],
            "visualization_resolution": [228, 304],
        })
    text = format_audit_text(audit)
    print(text)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "dataset_protocol_audit.json"), "w") as fh:
            json.dump(audit, fh, indent=2, default=str)
        with open(os.path.join(output_dir, "dataset_protocol_audit.txt"), "w") as fh:
            fh.write(text + "\n")
    if status == "FAIL" and fail_on_leak:
        raise RuntimeError("dataset protocol audit failed; see dataset_protocol_audit.txt/json")
    return audit


def format_audit_text(audit):
    counts = audit.get("counts", {})
    lc = audit.get("leak_counts", {})
    lines = [
        "===== Dataset Protocol Audit =====",
        f"train samples: {counts.get('train', 'N/A')}",
        f"val samples: {counts.get('val', 'N/A')}",
        f"test samples: {counts.get('test', 'N/A')}",
        f"bookkeeping sample_id overlap: {lc.get('bookkeeping_sample_id_overlap', 0)}",
        f"basename overlap: {lc.get('basename_overlap', 0)}",
        f"short relative-path overlap: {lc.get('short_relative_path_overlap', 0)}",
        f"canonical source overlap: {lc.get('canonical_source_overlap', 0)}",
        f"realpath overlap: {lc.get('realpath_overlap', 0)}",
        f"RGB hash overlap: {lc.get('rgb_hash_overlap', 0)}",
        f"GT hash overlap: {lc.get('target_hash_overlap', 0)}",
        f"sparse hash overlap: {lc.get('sparse_hash_overlap', 0)}",
        f"scene overlap: {lc.get('scene_overlap', 0)}",
    ]
    for split, fingerprint in audit.get("split_fingerprints", {}).items():
        lines.append(
            f"{split} ordered content fingerprint: {fingerprint.get('sha256')}")
    if audit.get("known_protocol_exceptions"):
        lines.append("known protocol exceptions:")
        for item in audit["known_protocol_exceptions"]:
            lines.extend([
                f"  - {item['pair']} shared RGB frames: {item['shared_rgb_frames']}",
                f"  - {item['pair']} shared GT frames: {item['shared_gt_frames']}",
                f"  - {item['pair']} shared sparse depth: {item['shared_sparse_depth']}",
                f"  - {item['pair']} relation: {item['relation']}",
            ])
    if audit.get("warnings"):
        lines.append("warnings:")
        for w in audit["warnings"]:
            lines.append(f"  - {w}")
    lines += [
        f"protocol status: {audit.get('status', 'UNKNOWN')}",
        "==================================",
    ]
    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the GeoLayerNet comparison dataset protocol audit only.")
    parser.add_argument("--train-dataroot", default=None)
    parser.add_argument("--val-dataroot", default=None)
    parser.add_argument("--test-dataroot", default=None)
    parser.add_argument("--train-list", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--dataset-config", default=None,
                        help="optional GeoLayerNet YAML; only data/protocol fields are read")
    parser.add_argument("--train-phase", default="train")
    parser.add_argument("--val-phase", default="val")
    parser.add_argument("--test-phase", default="test")
    parser.add_argument("--dataset-name", default="comparison_dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preprocess", default="none")
    parser.add_argument("--depth-scale", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="accepted for command compatibility; audit is dataset-index based")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="accepted for command compatibility; audit does not spawn workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="accepted for command compatibility; audit is deterministic")
    parser.add_argument("--device", default="cpu",
                        help="accepted for command compatibility; audit is CPU/filesystem only")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="optional audit cap for smoke tests; omit for full audit")
    parser.add_argument("--allow-leak", action="store_true",
                        help="write FAIL audit but do not raise; for debugging only")
    parser.add_argument(
        "--allow-val-test-shared-rgb-gt-with-distinct-sparse",
        dest="allow_val_test_shared_rgb_gt_with_distinct_sparse",
        action="store_true",
        help=("explicit GeoLayerNet protocol: allow complete val/test RGB+GT "
              "identity only when every sparse realization is distinct"))
    args = parser.parse_args(argv)

    config = load_dataset_config(args.dataset_config)
    config_type = config.get("dataset_type", "nyu")
    config_opts = dict(config.get("dataset", {}) or {})
    config_opts["dataset_cfg"] = dict(config.get("dataset", {}) or {})
    config_opts["train_cfg"] = dict(config.get("train", {}) or {})
    config_opts["eval_cfg"] = dict(config.get("eval", {}) or {})
    if args.val_phase == "val" and config.get("val_phase"):
        args.val_phase = config["val_phase"]
    if args.test_phase == "test" and config.get("eval_phase"):
        args.test_phase = config["eval_phase"]
    if args.test_dataroot is None and config.get("dataroot"):
        args.test_dataroot = config["dataroot"]
    if args.dataset_config and args.dataset_name == "comparison_dataset" and config.get("dataset_name"):
        args.dataset_name = config["dataset_name"]

    specs = [
        {"name": "train", "dataroot": args.train_dataroot,
         "phase": args.train_phase, "sample_list": args.train_list,
         "dataset_type": config_type, "dataset_opts": config_opts,
         "dataset_config": args.dataset_config},
        {"name": "val", "dataroot": args.val_dataroot,
         "phase": args.val_phase, "sample_list": args.val_list,
         "dataset_type": config_type, "dataset_opts": config_opts,
         "dataset_config": args.dataset_config},
        {"name": "test", "dataroot": args.test_dataroot,
         "phase": args.test_phase, "sample_list": args.test_list,
         "dataset_type": config_type, "dataset_opts": config_opts,
         "dataset_config": args.dataset_config},
    ]
    if not any(spec.get("dataroot") for spec in specs):
        raise RuntimeError(
            "missing dataset paths: pass at least one of --train-dataroot, "
            "--val-dataroot, or --test-dataroot")
    return audit_splits(
        specs, output_dir=args.output_dir, preprocess=args.preprocess,
        depth_scale=args.depth_scale, max_samples=args.max_samples,
        config_hint=f"{args.dataset_name} audit",
        fail_on_leak=not args.allow_leak,
        allow_val_test_shared_rgb_gt_with_distinct_sparse=(
            args.allow_val_test_shared_rgb_gt_with_distinct_sparse))


if __name__ == "__main__":
    main()

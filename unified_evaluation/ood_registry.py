"""Fixed GeoLayerNet OOD-16 registry for comparison evaluation.

This registry is deliberately data-only. It reads the existing GeoLayerNet YAML
files, resolves their dataroots, and never creates, copies, or transforms data.
The formal display names are fixed so DIODE remains split and VKITTI2 has one
main entry (the config decides that its sparse input is scatter001).
"""

import json
from pathlib import Path

import numpy as np

from .geolayernet_protocol import load_dataset_config


OOD16_ORDER = [
    "VOID1500", "SUNRGBD", "TOFDC", "DIML",
    "DIODE_indoor", "Middlebury", "ETH3D", "ScanNet",
    "DrivingStereo", "Argoverse", "Cityscapes", "DIODE_outdoor",
    "Hypersim", "VKITTI2", "KITTI360", "Stanford2D3D",
]

OOD15_PARTIAL_ORDER = [name for name in OOD16_ORDER if name != "ScanNet"]
OOD15_PARTIAL_COUNTS = {
    "VOID1500": 800, "SUNRGBD": 661, "TOFDC": 560, "DIML": 503,
    "DIODE_indoor": 325, "Middlebury": 15, "ETH3D": 27,
    "DrivingStereo": 1108, "Argoverse": 1009, "Cityscapes": 500,
    "DIODE_outdoor": 446, "Hypersim": 1098, "VKITTI2": 1061,
    "KITTI360": 500, "Stanford2D3D": 275,
}
OOD15_PARTIAL_TOTAL = 8888
OOD16_COUNTS = dict(OOD15_PARTIAL_COUNTS, ScanNet=1038)
OOD16_TOTAL = 9926

GROUPS = {
    "Raw-4": ["VOID1500", "SUNRGBD", "TOFDC", "DIML"],
    "Indoor-4": ["DIODE_indoor", "Middlebury", "ETH3D", "ScanNet"],
    "Outdoor-4": ["DrivingStereo", "Argoverse", "Cityscapes", "DIODE_outdoor"],
    "Synthetic-2": ["Hypersim", "VKITTI2"],
    "Special-2": ["KITTI360", "Stanford2D3D"],
}

# These are the actual file names in GeoLayerNet/configs/ood. The aliases are
# intentionally explicit because Cityscape/Stanford/Drivingstereo differ from
# the fixed paper display names.
CONFIG_FILES = {
    "VOID1500": "void1500.yaml",
    "SUNRGBD": "sunrgbd.yaml",
    "TOFDC": "tofdc.yaml",
    "DIML": "diml.yaml",
    "DIODE_indoor": "diode_indoor.yaml",
    "Middlebury": "middlebury.yaml",
    "ETH3D": "eth3d.yaml",
    "ScanNet": "scannet.yaml",
    "DrivingStereo": "drivingstereo.yaml",
    "Argoverse": "argoverse.yaml",
    "Cityscapes": "cityscape.yaml",
    "DIODE_outdoor": "diode_outdoor.yaml",
    "Hypersim": "hypersim.yaml",
    "VKITTI2": "vkitti2.yaml",
    "KITTI360": "kitti360.yaml",
    "Stanford2D3D": "stanford.yaml",
}

DATA_DIRS = {
    "DrivingStereo": "Drivingstereo",
    "Cityscapes": "Cityscape",
    "Stanford2D3D": "Stanford",
}


def _manifest_npz_files(root):
    """Read the fixed per-domain test manifest; never scan a directory."""
    root = Path(root)
    manifest = root / "test.txt"
    if not root.is_dir() or not manifest.is_file():
        return []
    files = []
    seen = set()
    for line_no, raw in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        path = Path(value).expanduser()
        path = path if path.is_absolute() else root / path
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path = path.resolve(strict=False)
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"OOD manifest entry escapes domain root at "
                f"{manifest}:{line_no}: {path}") from exc
        if path in seen:
            raise RuntimeError(
                f"duplicate OOD manifest entry at {manifest}:{line_no}: {path}")
        if not path.is_file():
            raise RuntimeError(
                f"OOD manifest entry missing at {manifest}:{line_no}: {path}")
        seen.add(path)
        files.append(path)
    return files


def _count_npz(root):
    return len(_manifest_npz_files(root))


def npz_key_coverage(root, key):
    """Inspect one optional NPZ field without reading RGB/depth payload arrays."""
    files = _manifest_npz_files(root)
    present = 0
    invalid = 0
    errors = []
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                # Injected OOD NPZs store the matrix as ``K``; accept that as
                # satisfying a legacy ``intrinsics`` coverage requirement.
                resolved_key = key
                if key not in archive.files:
                    if key == "intrinsics" and "K" in archive.files:
                        resolved_key = "K"
                    elif key == "K" and "intrinsics" in archive.files:
                        resolved_key = "intrinsics"
                    else:
                        continue
                present += 1
                if key in ("intrinsics", "K"):
                    value = np.asarray(archive[resolved_key])
                    valid = (
                        value.shape == (3, 3) and np.isfinite(value).all()
                        and value[0, 0] > 0 and value[1, 1] > 0
                        and np.allclose(value[2], (0.0, 0.0, 1.0), atol=1e-4))
                    if not valid:
                        invalid += 1
                        if len(errors) < 8:
                            errors.append(
                                f"{path}: invalid {resolved_key} shape/value")
        except Exception as exc:
            invalid += 1
            if len(errors) < 8:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "key": str(key), "total_npz": len(files), "present": present,
        "missing": len(files) - present, "invalid": invalid,
        "complete": bool(files) and present == len(files) and invalid == 0,
        "errors": errors,
    }


def _root_override(root, display):
    if not root:
        return None
    return str((Path(root).expanduser() / DATA_DIRS.get(display, display)).resolve())


def load_ood16(config_dir, data_root=None, required_npz_keys=None,
               required_npz_keys_by_dataset=None, unsupported_datasets=None,
               not_measurable_datasets=None):
    """Return ordered data-only entries and a registry status summary."""
    required_npz_keys = tuple(dict.fromkeys(required_npz_keys or ()))
    required_npz_keys_by_dataset = {
        str(name): tuple(dict.fromkeys(keys or ()))
        for name, keys in (required_npz_keys_by_dataset or {}).items()
    }
    unsupported_datasets = {str(name): str(reason) for name, reason in
                            (unsupported_datasets or {}).items()}
    not_measurable_datasets = {str(name): str(reason) for name, reason in
                               (not_measurable_datasets or {}).items()}
    config_source = Path(config_dir).expanduser().resolve()
    template = config_source if config_source.is_file() else None
    entries = []
    for display in OOD16_ORDER:
        config_path = template or config_source / CONFIG_FILES[display]
        if not config_path.is_file() and not template:
            fallback = config_source / "ood_example.yaml"
            config_path = fallback if fallback.is_file() else config_path
        config = load_dataset_config(config_path) if config_path.is_file() else None
        dataroot = _root_override(data_root, display) or (config or {}).get("dataroot")
        sample_count = _count_npz(dataroot) if dataroot else 0
        expected = OOD16_COUNTS[display]
        status = "COMPLETE"
        reasons = []
        if config is None:
            status = "INCOMPLETE"
            reasons.append("missing YAML config")
        if not dataroot or not Path(dataroot).is_dir():
            status = "INCOMPLETE"
            reasons.append("dataroot missing")
        elif not (Path(dataroot) / "test.txt").is_file():
            status = "INCOMPLETE"
            reasons.append("missing required test.txt manifest; directory scanning is forbidden")
        elif sample_count == 0:
            status = "INCOMPLETE"
            reasons.append("no npz samples")
        if sample_count != expected:
            status = "INCOMPLETE"
            reasons.append(f"expected {expected} samples, found {sample_count}")
        if display == "VKITTI2" and dataroot and Path(dataroot).name != "VKITTI2":
            status = "INCOMPLETE"
            reasons.append("formal VKITTI2 root must be named VKITTI2")
        if display in ("DIODE_indoor", "DIODE_outdoor") and dataroot and Path(dataroot).name not in display:
            status = "INCOMPLETE"
            reasons.append(f"formal DIODE root must be named {display}")
        if display in unsupported_datasets:
            status = "BLOCKED_METHOD"
            reasons.append(unsupported_datasets[display])
        if display in not_measurable_datasets:
            status = "NOT_MEASURABLE"
            reasons.append(not_measurable_datasets[display])
        coverage = {}
        dataset_required_keys = tuple(dict.fromkeys(
            required_npz_keys + required_npz_keys_by_dataset.get(display, ())))
        if dataroot and Path(dataroot).is_dir():
            for key in dataset_required_keys:
                checked = npz_key_coverage(dataroot, key)
                coverage[key] = checked
                if sample_count and not checked["complete"]:
                    if status not in ("BLOCKED_METHOD", "NOT_MEASURABLE"):
                        status = "BLOCKED_METADATA"
                    reasons.append(
                        f"required NPZ key {key}: present={checked['present']}/"
                        f"{checked['total_npz']} missing={checked['missing']} "
                        f"invalid={checked['invalid']}")
        entries.append({
            "dataset_name": display,
            "config": str(config_path),
            "dataroot": dataroot,
            "dataset_type": (config or {}).get("dataset_type", "ood_npz"),
            "sample_count": sample_count,
            "expected_sample_count": expected,
            "status": status,
            "reasons": reasons,
            "required_npz_key_coverage": coverage,
            "required_npz_keys": list(dataset_required_keys),
        })
    complete = all(item["status"] == "COMPLETE" for item in entries)
    return {
        "schema_version": 1,
        "dataset_order": list(OOD16_ORDER),
        "groups": GROUPS,
        "result_root_template": "<output-root>/{dataset_name}",
        "excluded_legacy_names": ["VKITTI2_raw", "VKITTI2_scatter001"],
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "expected_prediction_total": OOD16_TOTAL,
        "required_npz_keys": list(required_npz_keys),
        "required_npz_keys_by_dataset": {
            name: list(keys) for name, keys in required_npz_keys_by_dataset.items()},
        "unsupported_datasets": dict(unsupported_datasets),
        "not_measurable_datasets": dict(not_measurable_datasets),
        "datasets": entries,
        "aggregation_rule": "Macro-16 is written only when all 16 formal entries are COMPLETE; skipped/incomplete domains never enter aggregation.",
    }


def write_registry(path, registry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")


def load_ood15_partial(config_dir, data_root=None, required_npz_keys=None,
                       required_npz_keys_by_dataset=None,
                       unsupported_datasets=None,
                       not_measurable_datasets=None):
    """Load the explicitly named 15-domain partial registry (ScanNet absent)."""
    full = load_ood16(
        config_dir, data_root, required_npz_keys=required_npz_keys,
        required_npz_keys_by_dataset=required_npz_keys_by_dataset,
        unsupported_datasets=unsupported_datasets,
        not_measurable_datasets=not_measurable_datasets)
    entries = []
    for item in full["datasets"]:
        if item["dataset_name"] == "ScanNet":
            continue
        row = dict(item)
        expected = OOD15_PARTIAL_COUNTS[row["dataset_name"]]
        row["expected_sample_count"] = expected
        if row["sample_count"] != expected:
            if row["status"] not in (
                    "BLOCKED_METHOD", "BLOCKED_METADATA", "NOT_MEASURABLE"):
                row["status"] = "INCOMPLETE"
            row["reasons"] = list(row.get("reasons", [])) + [
                f"expected {expected} samples, found {row['sample_count']}"]
        entries.append(row)
    complete = all(row["status"] == "COMPLETE" for row in entries)
    actual_total = sum(row["sample_count"] for row in entries)
    return {
        "schema_version": 2,
        "protocol_name": "OOD-15 partial",
        "dataset_order": list(OOD15_PARTIAL_ORDER),
        "groups": {
            key: [name for name in values if name != "ScanNet"]
            for key, values in GROUPS.items()},
        "scanNet_status": "not evaluated",
        "macro16_status": "unavailable",
        "expected_prediction_total": OOD15_PARTIAL_TOTAL,
        "sample_count_total": actual_total,
        "status": "OOD-15_PARTIAL_COMPLETE" if complete and actual_total == OOD15_PARTIAL_TOTAL else "INCOMPLETE",
        "datasets": entries,
        "aggregation_rule": (
            "Macro-15 partial is written only when all 15 registered domains pass "
            "with exactly 8888 predictions; Macro-16 is never synthesized."),
    }

"""Fail-closed GeoLayerNet Comparison Protocol V2 dataset path policy."""

from __future__ import annotations

import json
from pathlib import Path

from .ood_registry import CONFIG_FILES, DATA_DIRS


DATASET_PATH_REGISTRY = "unified_evaluation/protocols/geolayernet_dataset_paths.yaml"
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_FILE = WORKSPACE_ROOT / DATASET_PATH_REGISTRY


def load_dataset_path_registry(path=None):
    source = Path(path or _REGISTRY_FILE).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "GeoLayerNet dataset path validation requires PyYAML") from exc
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid dataset path registry: {source}")
    if payload.get("path_policy") != "workspace_relative_paths":
        raise RuntimeError("dataset path registry must use workspace_relative_paths")
    if payload.get("legacy_paths_allowed") is not False:
        raise RuntimeError("dataset path registry must reject legacy paths")
    if payload.get("directory_scanning_allowed") is not False:
        raise RuntimeError("dataset path registry must reject directory scanning")
    return payload


def _configured(value):
    return None if value in (None, "") else str(value)


def _resolved(value):
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return str(path.resolve(strict=False))


def _mismatch(field, actual, expected, status="PATH_MISMATCH"):
    raise RuntimeError(
        "PROTOCOL_PATH_MISMATCH "
        f"field={field} actual={actual or '<missing>'} "
        f"expected={expected} status={status}")


def _require_exact(field, actual, expected, configured, resolved):
    raw = _configured(actual)
    configured[field] = raw
    resolved[field] = _resolved(raw)
    if raw is None:
        _mismatch(field, raw, expected, "MISSING_REQUIRED_PATH")
    expected_resolved = _resolved(expected)
    if resolved[field] != expected_resolved:
        _mismatch(field, resolved[field], expected_resolved)


def _forbid_present(field, value):
    if value not in (None, ""):
        _mismatch(field, value, "<absent>")


def load_nyu_small_ablation_protocol(args):
    """Validate the explicit, non-main-table NYUv2-small protocol.

    This is deliberately separate from the frozen NYU48K path registry.  It
    opens only when the caller supplies both the protocol receipt and the
    registered sparse-point count; it never weakens the formal NYU48K policy.
    """
    value = getattr(args, "nyu_small_ablation_protocol", None)
    if value in (None, ""):
        return None
    source = Path(str(value)).expanduser()
    if not source.is_absolute():
        _mismatch(
            "nyu_small.protocol", value, "<absolute protocol.json>",
            "RELATIVE_PATH")
    source = source.resolve(strict=False)
    if not source.is_file():
        _mismatch(
            "nyu_small.protocol", str(source), "<existing protocol.json>",
            "MISSING_REQUIRED_PATH")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"NYU_SMALL_PROTOCOL_INVALID path={source}: {exc}") from exc
    required = {
        "schema": "geolayernet-nyu-labeled-small-ablation-v1",
        "status": "COMPLETE",
        "prepared_counts": {"train": 640, "val": 155, "test": 654},
        "output_resolution": [228, 304],
        "depth_unit": "m",
        "nested_sparse_masks": True,
        "valid_mask_semantics": "sparse_depth_gt_zero",
        "valid_mask_verified": True,
        "storage_layout": "independent_physical_dataset_per_sparse_point_count",
        "soft_links_used": False,
        "test_used_for_model_selection": False,
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise RuntimeError(
                "NYU_SMALL_PROTOCOL_INVALID "
                f"field={name} actual={payload.get(name)!r} expected={expected!r}")
    if int(payload.get("seed", -1)) != int(getattr(args, "seed", -2)):
        raise RuntimeError("NYU_SMALL_PROTOCOL_INVALID seed mismatch")
    points = getattr(args, "nyu_small_sparse_points", None)
    allowed_points = (10, 50, 100, 200, 500, 693)
    if points is None or int(points) not in allowed_points:
        raise RuntimeError(
            "--nyu-small-sparse-points must be one of 10,50,100,200,500,693")
    points = int(points)
    if points not in [int(item) for item in payload.get("sparse_point_counts", [])]:
        raise RuntimeError(
            f"NYU_SMALL_PROTOCOL_INVALID sparse point count {points} is unregistered")
    sparse_percent = payload.get("declared_input_percent")
    if points == 693:
        percentage_required = {
            "declared_input_percent": 1.0,
            "sparse_density_reference": "model_input_pixel_grid_228x304",
            "input_pixel_count": 228 * 304,
            "points_per_sample": 693,
            "sampling_population": "valid_metric_gt_locations",
            "one_percent_module_ablation_input": True,
        }
        for name, expected in percentage_required.items():
            if payload.get(name) != expected:
                raise RuntimeError(
                    "NYU_SMALL_PROTOCOL_INVALID one-percent binding "
                    f"field={name} actual={payload.get(name)!r} "
                    f"expected={expected!r}")
    elif sparse_percent is not None:
        raise RuntimeError(
            "NYU_SMALL_PROTOCOL_INVALID percentage-bound protocol must use "
            "the exact 228x304 one-percent count 693")
    expected_root = Path(str(
        (payload.get("variant_roots") or {}).get(str(points),
         source.parent / f"points_{points}"))).expanduser().resolve(strict=False)
    canonical_root = (source.parent / f"points_{points}").resolve(strict=False)
    if expected_root != canonical_root:
        raise RuntimeError(
            "NYU_SMALL_PROTOCOL_INVALID variant root is not the canonical "
            f"points_{points} child: {expected_root}")
    if not expected_root.is_dir() or expected_root.is_symlink():
        raise RuntimeError(
            "NYU_SMALL_PROTOCOL_INVALID variant root must be an existing physical "
            f"directory: {expected_root}")
    return {
        "path": str(source),
        "payload": payload,
        "points": points,
        "sparse_percent": sparse_percent,
        "root": str(expected_root),
        "counts": dict(required["prepared_counts"]),
    }


def validate_ood_wrapper_paths(config_dir, root, registry=None):
    paths = registry or load_dataset_path_registry()
    configured = {}
    resolved = {}
    _require_exact(
        "ood.config_dir", config_dir, paths["ood"]["config_dir"],
        configured, resolved)
    _require_exact(
        "ood.root", root, paths["ood"]["root"], configured, resolved)
    return {
        "dataset_path_policy": "exact_geolayernet_paths",
        "dataset_path_registry": DATASET_PATH_REGISTRY,
        "configured_paths": configured,
        "resolved_paths": resolved,
        "path_audit_status": "PASS_EXACT_PATHS",
    }


def validate_geolayernet_protocol_paths(args, mode, training_domain=None):
    """Validate the resolved source paths before any dataset is constructed.

    Values are never substituted. A missing, relative, legacy, or mismatched
    path raises ``PROTOCOL_PATH_MISMATCH`` with the exact field and expectation.
    """
    del training_domain
    paths = load_dataset_path_registry()
    configured = {}
    resolved = {}
    dataset_type = str(getattr(args, "_dataset_type", "nyu"))

    if dataset_type == "nyu":
        small = load_nyu_small_ablation_protocol(args)
        if small is not None:
            config = getattr(args, "dataset_config", None)
            if not config or not Path(str(config)).expanduser().is_absolute():
                _mismatch(
                    "nyu_small.config", config, "<absolute generated YAML>",
                    "RELATIVE_PATH")
            config_path = Path(str(config)).expanduser().resolve(strict=False)
            if not config_path.is_file():
                _mismatch(
                    "nyu_small.config", str(config_path),
                    "<existing generated YAML>", "MISSING_REQUIRED_PATH")
            try:
                import yaml
                generated = yaml.safe_load(
                    config_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                raise RuntimeError(
                    f"NYU_SMALL_PROTOCOL_INVALID config={config_path}: {exc}") from exc
            dataset_cfg = dict(generated.get("dataset") or {})
            protocol_cfg = dict(generated.get("protocol") or {})
            if _resolved(dataset_cfg.get("dataroot")) != small["root"]:
                raise RuntimeError(
                    "NYU_SMALL_PROTOCOL_INVALID generated config dataroot mismatch")
            if (_resolved(protocol_cfg.get("nyu_small_ablation_protocol")) !=
                    small["path"]):
                raise RuntimeError(
                    "NYU_SMALL_PROTOCOL_INVALID generated config protocol binding mismatch")
            if int(protocol_cfg.get("sparse_points", -1)) != int(small["points"]):
                raise RuntimeError(
                    "NYU_SMALL_PROTOCOL_INVALID generated config sparse-point mismatch")
            if small["sparse_percent"] is not None and float(
                    protocol_cfg.get("sparse_input_percent", -1.0)) != float(
                        small["sparse_percent"]):
                raise RuntimeError(
                    "NYU_SMALL_PROTOCOL_INVALID generated config sparse-percent mismatch")
            if protocol_cfg.get("formal_result") is not False:
                raise RuntimeError(
                    "NYU_SMALL_PROTOCOL_INVALID generated config must set formal_result=false")
            configured["nyu_small.config"] = str(config)
            resolved["nyu_small.config"] = str(config_path)
            expected_root = small["root"]
            if mode == "train":
                _require_exact(
                    "nyu_small.train_root", getattr(args, "train_dataroot", None),
                    expected_root, configured, resolved)
                _require_exact(
                    "nyu_small.val_root", getattr(args, "val_dataroot", None),
                    expected_root, configured, resolved)
                _forbid_present(
                    "nyu_small.train_list", getattr(args, "train_list", None))
                _forbid_present(
                    "nyu_small.val_list", getattr(args, "val_list", None))
                _forbid_present(
                    "nyu_small.training_test_root",
                    getattr(args, "test_dataroot", None))
            else:
                _require_exact(
                    "nyu_small.test_root", getattr(args, "test_dataroot", None),
                    expected_root, configured, resolved)
                _forbid_present(
                    "nyu_small.test_list", getattr(args, "test_list", None))
                _forbid_present(
                    "nyu_small.eval_train_root",
                    getattr(args, "train_dataroot", None))
                _forbid_present(
                    "nyu_small.eval_val_root", getattr(args, "val_dataroot", None))
            configured["nyu_small.protocol"] = str(
                getattr(args, "nyu_small_ablation_protocol"))
            resolved["nyu_small.protocol"] = small["path"]
            metadata = {
                "dataset_path_policy": "frozen_nyu_small_ablation_protocol",
                "dataset_path_registry": small["path"],
                "configured_paths": configured,
                "resolved_paths": resolved,
                "path_audit_status": "PASS_EXACT_NYU_SMALL_ABLATION_PATHS",
            }
            args._nyu_small_ablation_spec = small
            for name, value in metadata.items():
                setattr(args, name, value)
            return metadata
        _require_exact(
            "nyu.config", getattr(args, "dataset_config", None),
            paths["nyu"]["config"], configured, resolved)
        if mode == "train":
            _require_exact(
                "nyu.train_root", getattr(args, "train_dataroot", None),
                paths["nyu"]["root"], configured, resolved)
            _require_exact(
                "nyu.val_root", getattr(args, "val_dataroot", None),
                paths["nyu"]["root"], configured, resolved)
            _forbid_present("nyu.train_list", getattr(args, "train_list", None))
            _forbid_present("nyu.val_list", getattr(args, "val_list", None))
            _forbid_present(
                "nyu.training_test_root", getattr(args, "test_dataroot", None))
            _forbid_present(
                "nyu.training_test_list", getattr(args, "test_list", None))
        else:
            _require_exact(
                "nyu.test_root", getattr(args, "test_dataroot", None),
                paths["nyu"]["root"], configured, resolved)
            _forbid_present("nyu.test_list", getattr(args, "test_list", None))
            if getattr(args, "train_dataroot", None) not in (None, ""):
                _require_exact(
                    "nyu.audit_train_root", args.train_dataroot,
                    paths["nyu"]["root"], configured, resolved)
            if getattr(args, "val_dataroot", None) not in (None, ""):
                _require_exact(
                    "nyu.audit_val_root", args.val_dataroot,
                    paths["nyu"]["root"], configured, resolved)
            _forbid_present(
                "nyu.audit_train_list", getattr(args, "train_list", None))
            _forbid_present(
                "nyu.audit_val_list", getattr(args, "val_list", None))

    elif dataset_type == "kitti":
        _require_exact(
            "kitti.config", getattr(args, "dataset_config", None),
            paths["kitti"]["config"], configured, resolved)
        if mode == "train":
            stage = str(getattr(args, "kitti_stage", "") or "")
            if stage not in ("develop", "final-train"):
                _mismatch(
                    "kitti.stage", stage, "develop|final-train",
                    "MISSING_REQUIRED_PATH")
            _require_exact(
                "kitti.train_root", getattr(args, "train_dataroot", None),
                paths["kitti"]["root"], configured, resolved)
            manifest = "train90" if stage == "develop" else "train_full"
            _require_exact(
                f"kitti.{manifest}", getattr(args, "train_list", None),
                paths["kitti"][manifest], configured, resolved)
            if stage == "develop":
                _require_exact(
                    "kitti.dev_root", getattr(args, "val_dataroot", None),
                    paths["kitti"]["root"], configured, resolved)
                _require_exact(
                    "kitti.dev10", getattr(args, "val_list", None),
                    paths["kitti"]["dev10"], configured, resolved)
            else:
                _forbid_present(
                    "kitti.final_train_val_root",
                    getattr(args, "val_dataroot", None))
                _forbid_present(
                    "kitti.final_train_val_list", getattr(args, "val_list", None))
                _forbid_present(
                    "kitti.final_train_test_list", getattr(args, "test_list", None))
            _forbid_present(
                "kitti.training_test_root", getattr(args, "test_dataroot", None))
            _forbid_present(
                "kitti.training_test_list", getattr(args, "test_list", None))
        else:
            _forbid_present(
                "kitti.eval_train_root", getattr(args, "train_dataroot", None))
            _forbid_present(
                "kitti.eval_val_root", getattr(args, "val_dataroot", None))
            _forbid_present(
                "kitti.eval_train_list", getattr(args, "train_list", None))
            _forbid_present(
                "kitti.eval_val_list", getattr(args, "val_list", None))
            _require_exact(
                "kitti.test_root", getattr(args, "test_dataroot", None),
                paths["kitti"]["root"], configured, resolved)
            role = str(getattr(args, "kitti_eval_role", "final") or "final")
            if role not in ("develop", "final", "diagnostic-best"):
                _mismatch(
                    "kitti.eval_role", role,
                    "develop|final|diagnostic-best", "MISSING_REQUIRED_PATH")
            manifest = "dev10" if role == "develop" else "val1000"
            _require_exact(
                f"kitti.{manifest}", getattr(args, "test_list", None),
                paths["kitti"][manifest], configured, resolved)

    elif dataset_type == "ood_npz":
        for field, attribute in (
                ("ood.train_root", "train_dataroot"),
                ("ood.val_root", "val_dataroot"),
                ("ood.train_list", "train_list"),
                ("ood.val_list", "val_list"),
                ("ood.test_list", "test_list")):
            _forbid_present(field, getattr(args, attribute, None))
        wrapper = validate_ood_wrapper_paths(
            getattr(args, "protocol_ood_config_dir", None),
            getattr(args, "protocol_ood_root", None), paths)
        configured.update(wrapper["configured_paths"])
        resolved.update(wrapper["resolved_paths"])
        dataset_name = str(getattr(args, "dataset_name", "") or "")
        if dataset_name not in CONFIG_FILES:
            _mismatch(
                "ood.dataset_name", dataset_name,
                "one of " + "|".join(CONFIG_FILES), "UNREGISTERED_DATASET")
        expected_config = str(
            Path(paths["ood"]["config_dir"]) / CONFIG_FILES[dataset_name])
        expected_leaf = DATA_DIRS.get(dataset_name, dataset_name)
        expected_root = str(Path(paths["ood"]["root"]) / expected_leaf)
        _require_exact(
            "ood.dataset_config", getattr(args, "dataset_config", None),
            expected_config, configured, resolved)
        _require_exact(
            "ood.dataset_root", getattr(args, "test_dataroot", None),
            expected_root, configured, resolved)
    else:
        _mismatch("dataset.type", dataset_type, "nyu|kitti|ood_npz")

    metadata = {
        "dataset_path_policy": "exact_geolayernet_paths",
        "dataset_path_registry": DATASET_PATH_REGISTRY,
        "configured_paths": configured,
        "resolved_paths": resolved,
        "path_audit_status": "PASS_EXACT_PATHS",
    }
    for name, value in metadata.items():
        setattr(args, name, value)
    return metadata

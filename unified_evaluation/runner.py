"""Shared train/eval runner for comparison methods on the GeoLayerNet NYU
directory protocol.

Each method provides an *adapter* module exposing:

  METHOD_NAME : str
  COMPARISON_BUCKET / SUPERVISION / PRIOR : str   (recorded in run_info.txt)
  add_method_args(parser)                          (optional) -- extra CLI args
  build_model(args, device) -> torch.nn.Module     -- the method's REAL core net
  forward_completed(model, batch, device) -> Tensor[B,1,H,W]
        -- runs the method's REAL forward and converts its output to a single
           depth map roughly in [0,1] (same normalized space as target_depth).
           The method's internal forward / layers are NOT modified; only an
           external input/output wrapper is added.

The runner owns: data loading (GeoLayerNet protocol), metrics (MetricBag),
visualization, run records, and checkpoint save/load. Training uses an adapter's
registered native ``train_step`` and optimizer. Missing loss, optimizer, or
provenance-aware checkpoint hooks fail closed; no generic L1/Adam/plain-state
substitution is allowed. The network bodies remain untouched. It never hardcodes
CUDA_VISIBLE_DEVICES (the GPU is selected by the external environment).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import traceback
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import io as result_io
from . import fairness as fairness_io
from . import visualization as vis
from .bn_trace import ConfheadBNTracer, RollingReplayManager
from .dataset_paths import (load_dataset_path_registry,
                            validate_geolayernet_protocol_paths)
from .geolayernet_protocol import (audit_splits, create_loader, dataset_samples,
                                    load_dataset_config)
from .metrics import MetricBag, model_stats
from .seed import set_global_seed


_FORMAL_PATHS = load_dataset_path_registry()
_FORMAL_NYU_CONFIG = str(_FORMAL_PATHS["nyu"]["config"])
_FORMAL_NYU_ROOT = str(_FORMAL_PATHS["nyu"]["root"])
_FORMAL_NYU_COUNTS = {
    "train": int(_FORMAL_PATHS["nyu"]["expected_train"]),
    "val": int(_FORMAL_PATHS["nyu"]["expected_val"]),
    "test": int(_FORMAL_PATHS["nyu"]["expected_test"]),
}
_FORMAL_KITTI_CONFIG = str(_FORMAL_PATHS["kitti"]["config"])
_FORMAL_KITTI_ROOT = str(_FORMAL_PATHS["kitti"]["root"])
_FORMAL_KITTI_MANIFESTS = {
    name: str(_FORMAL_PATHS["kitti"][name])
    for name in ("train90", "dev10", "train_full", "val1000")
}
_FORMAL_KITTI_PROTOCOL_ROOT = str(
    os.path.commonpath(tuple(_FORMAL_KITTI_MANIFESTS.values())))
_FORMAL_KITTI_COUNTS = {
    name: int(_FORMAL_PATHS["kitti"][f"expected_{name}"])
    for name in ("train90", "dev10", "train_full", "val1000")
}
_KITTI_EVAL_ROLE_SPECS = {
    "develop": {
        "split": "dev10",
        "phase": "dev10",
        "checkpoint": "best_net_G.pth",
        "checkpoint_role": "develop_best",
        "checkpoint_stage": "develop",
    },
    "final": {
        "split": "val1000",
        "phase": "val1000",
        "checkpoint": "final_net_G.pth",
        "checkpoint_role": "final",
        "checkpoint_stage": "final-train",
    },
    "diagnostic-best": {
        "split": "val1000",
        "phase": "val1000",
        "checkpoint": "best_net_G.pth",
        "checkpoint_role": "develop_best",
        "checkpoint_stage": "develop",
        "evaluation_status": "KITTI_VAL1000_DEVELOP_BEST_DIAGNOSTIC",
        "official_final_protocol": False,
        "formal_result": False,
        "controlled_ranking": False,
        "result_eligible_for_main_table": False,
    },
}
_KITTI_DIAGNOSTIC_METADATA_FIELDS = (
    "evaluation_status",
    "checkpoint_role",
    "checkpoint_stage",
    "official_final_protocol",
    "formal_result",
    "controlled_ranking",
    "result_eligible_for_main_table",
)


# --------------------------------------------------------------------------- #
# CLI parsers (the required argument set, identical across methods)
# --------------------------------------------------------------------------- #
def _add_common(parser):
    parser.add_argument("--dataset-name", default="comparison_dataset",
                        help="label used only in logs/output metadata; never selects paths")
    parser.add_argument("--dataroot", default=None,
                        help="deprecated legacy alias; use explicit --train/--val/--test-dataroot")
    parser.add_argument("--train-dataroot", default=None,
                        help="explicit training split root or root containing train/")
    parser.add_argument("--val-dataroot", default=None,
                        help="explicit validation split root or root containing val/")
    parser.add_argument("--test-dataroot", default=None,
                        help="explicit test split root or root containing test/")
    parser.add_argument("--train-list", default=None, help="ordered training sample_id list")
    parser.add_argument("--val-list", default=None, help="ordered validation sample_id list")
    parser.add_argument("--test-list", default=None, help="ordered test sample_id list")
    parser.add_argument("--dataset-config", default=None,
                        help="optional GeoLayerNet YAML; only data/protocol fields are read")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess", default=None,
                        help="GeoLayerNet preprocessing string; default none keeps native resolution")
    parser.add_argument("--depth-scale", type=float, default=0.0,
                        help="raw depth PNG scale override; <=0 infers 65535/255 like GeoLayerNet")
    parser.add_argument("--depth-unit-scale", type=float, default=None,
                        help="normalized depth -> metres (NYU protocol = 10)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing run records / checkpoints instead of backing up")
    parser.add_argument("--audit-only", action="store_true",
                        help="run dataset protocol audit, write audit files, then exit")
    parser.add_argument(
        "--allow-val-test-shared-rgb-gt-with-distinct-sparse",
        action="store_true",
        help=("explicit GeoLayerNet-only audit exception for complete val/test "
              "RGB+GT identity with entirely distinct sparse realizations"))
    parser.add_argument("--protocol-ood-config-dir", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--protocol-ood-root", default=None,
                        help=argparse.SUPPRESS)


def build_eval_parser(adapter, description=None):
    p = argparse.ArgumentParser(description=description or f"{getattr(adapter,'METHOD_NAME','method')} eval (GeoLayerNet NYU directory protocol)")
    _add_common(p)
    p.add_argument("--phase", "--test-phase", dest="test_phase", default=None,
                   help="split phase; defaults to dataset config eval.phase or test")
    p.add_argument("--checkpoint", default=None,
                   help=("model checkpoint for eval; required unless --audit-only is used "
                         "or the adapter explicitly declares a checkpoint-free method"))
    p.add_argument("--strict-checkpoint", action="store_true",
                   help="request strict loading; V2 adapters may enforce strict loading unconditionally")
    p.add_argument("--num-test", type=int, default=100000)
    p.add_argument("--save-visuals", action="store_true")
    p.add_argument("--save-visual-limit", "--num-visuals", dest="save_visual_limit", type=int, default=50)
    p.set_defaults(clip_prediction=None, save_predictions=True)
    p.add_argument("--clip-prediction", dest="clip_prediction", action="store_true",
                   help="clip prediction to [1e-3,1.0] before metrics")
    p.add_argument("--no-clip-prediction", dest="clip_prediction", action="store_false",
                   help="disable GeoLayerNet prediction clipping before metrics")
    error_max_cli_unit = str(getattr(
        adapter, "ERROR_MAX_CLI_UNIT", "normalized_depth"))
    error_max_help = (
        "fixed absolute-error saturation in metres"
        if error_max_cli_unit == "metres" else
        "fixed abs-error saturation in normalized depth; converted to metres using the dataset scale")
    p.add_argument("--error-max", type=float, default=None,
                   help=error_max_help)
    p.add_argument("--vis-depth-mode", default=None, choices=["shared", "fixed"],
                   help="formal comparison requires fixed; shared/per-image normalization is rejected")
    p.add_argument("--vis-depth-min", type=float, default=None,
                   help="optional fixed display minimum in metres (default: 0 m)")
    p.add_argument("--vis-depth-max", type=float, default=None,
                   help="optional fixed display maximum in metres (default: YAML max_depth * scale)")
    p.add_argument("--min-depth", type=float, default=None,
                   help="minimum normalized target depth included in metrics")
    p.add_argument("--max-depth", type=float, default=None,
                   help="maximum normalized target depth included in metrics")
    p.add_argument("--visual-sample-list", default=None,
                   help="optional ordered sample_id list for visualization; missing ids are errors")
    p.add_argument("--visual-seed", type=int, default=None,
                   help="deterministic seed for selecting visual samples when no list is given")
    p.add_argument("--save-predictions", dest="save_predictions", action="store_true",
                   help="save predictions_npy and predictions_png16")
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false",
                   help="skip prediction array/image export")
    p.add_argument(
        "--kitti-eval-role", choices=tuple(_KITTI_EVAL_ROLE_SPECS), default="final",
        help=("develop evaluates best_net_G.pth on dev10 for diagnostics; "
              "final evaluates final_net_G.pth on val1000; "
              "diagnostic-best evaluates develop best_net_G.pth on val1000 "
              "as a non-main-table diagnostic"))
    if hasattr(adapter, "add_method_args"):
        adapter.add_method_args(p)
    p.set_defaults(
        batch_size=int(getattr(adapter, "EVAL_BATCH_SIZE", p.get_default("batch_size"))),
        num_workers=int(getattr(adapter, "EVAL_NUM_WORKERS", p.get_default("num_workers"))),
        _error_max_cli_unit=error_max_cli_unit)
    return p


def build_train_parser(adapter, description=None):
    p = argparse.ArgumentParser(description=description or f"{getattr(adapter,'METHOD_NAME','method')} train (GeoLayerNet NYU directory protocol)")
    _add_common(p)
    p.add_argument("--checkpoint", default=None,
                   help="optional initialization/resume checkpoint; omitted means train from adapter default init")
    p.add_argument("--strict-checkpoint", action="store_true",
                   help="fail when checkpoint loading reports any missing or unexpected keys")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-best-metric", default="RMSE_m",
                   help="validation metric used for best checkpoint selection; computed on val only")
    p.add_argument("--val-best-mode", default="min", choices=["min", "max"],
                   help="whether lower or higher --val-best-metric is better")
    p.add_argument("--val-phase", default=None,
                   help="validation phase; defaults to dataset config train.val_phase or val")
    p.add_argument("--no-val", action="store_true",
                   help="formal use is restricted to KITTI final-train after E is selected")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="micro-batches accumulated before each optimizer update")
    p.add_argument("--amp", dest="amp", action="store_true",
                   help="use CUDA automatic mixed precision during training")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--kitti-stage", choices=("develop", "final-train"), default=None,
                   help=("formal KITTI stage: develop=train90/dev10 checkpoint selection; "
                         "final-train=train_full from scratch for the selected fixed epoch count"))
    p.add_argument("--fixed-epochs", type=int, default=None,
                   help="epoch E selected by KITTI develop; required by final-train")
    if not bool(getattr(adapter, "REQUIRE_KITTI_FIXED_E_ONLY", False)):
        p.add_argument("--develop-receipt", default=None,
                       help="develop-stage kitti_develop_receipt.json required by final-train")
    p.add_argument("--val-freq", type=int, default=1)
    p.add_argument("--train-phase", default="train")
    p.add_argument("--test-phase", default=None,
                   help="optional in-domain test phase; defaults to config eval.phase or test")
    if hasattr(adapter, "add_method_args"):
        adapter.add_method_args(p)
    p.set_defaults(
        batch_size=int(getattr(adapter, "TRAIN_BATCH_SIZE", p.get_default("batch_size"))),
        num_workers=int(getattr(adapter, "TRAIN_NUM_WORKERS", p.get_default("num_workers"))),
        amp=bool(getattr(adapter, "TRAIN_AMP", False)))
    return p


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _StatsModule(nn.Module):
    """Wrap the real model + adapter forward as ``forward(rgb, sparse, valid)`` so
    the GeoLayerNet model_stats (Params/GFLOPs/FPS) helper can profile it."""

    def __init__(self, adapter, model, device):
        super().__init__()
        self.adapter = adapter
        self.model = model
        self._device = device

    def forward(self, rgb, sparse, valid):
        batch = {"rgb": rgb, "sparse_depth": sparse, "valid_mask": valid}
        return self.adapter.forward_completed(self.model, batch, self._device)


def _model_stats_enabled(adapter):
    """Whether evaluation may call the adapter an extra time for profiling.

    Stateful online/TTA methods change their parameters in ``forward_completed``.
    Profiling such a method would adapt on the first evaluation batch twice and
    therefore change the registered result.  Those adapters opt out explicitly
    with ``SKIP_MODEL_STATS = True``.
    """
    return not bool(getattr(adapter, "SKIP_MODEL_STATS", False))


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class _NonfiniteDetected(RuntimeError):
    """Structured fail-fast signal; it is never swallowed by the train loop."""

    def __init__(self, stage, tensor_name, statistics,
                 failure_loss_component=None):
        self.failure_stage = str(stage)
        self.failure_tensor = str(tensor_name)
        self.failure_loss_component = failure_loss_component
        self.statistics = statistics
        super().__init__(
            f"non-finite tensor at {self.failure_stage}: {self.failure_tensor}")


def _iter_named_tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix or "tensor", value
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(item, name)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            name = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_named_tensors(item, name)


def _is_finite_check_dtype(tensor):
    return tensor.is_floating_point() or tensor.is_complex()


def _tensor_statistics(name, tensor):
    tensor = tensor.detach()
    count = int(tensor.numel())
    floating = _is_finite_check_dtype(tensor)
    if floating:
        finite = torch.isfinite(tensor)
        nan_count = int(torch.isnan(tensor).sum().item())
        if tensor.is_complex():
            posinf_count = int(torch.isinf(tensor).sum().item())
            neginf_count = 0
            values = tensor.abs()
        else:
            posinf_count = int(torch.isposinf(tensor).sum().item())
            neginf_count = int(torch.isneginf(tensor).sum().item())
            values = tensor
    else:
        finite = torch.ones_like(tensor, dtype=torch.bool)
        nan_count = posinf_count = neginf_count = 0
        values = tensor
    finite_count = int(finite.sum().item())
    finite_values = values[finite]
    if finite_count:
        finite_double = finite_values.double()
        minimum = float(finite_double.min().item())
        maximum = float(finite_double.max().item())
        mean = float(finite_double.mean().item())
        std = float(finite_double.std(unbiased=False).item())
        max_abs = float(finite_double.abs().max().item())
        l2_norm = float(torch.linalg.vector_norm(finite_double).item())
    else:
        minimum = maximum = mean = std = max_abs = l2_norm = None
    nonzero_count = int((tensor != 0).sum().item()) if count else 0
    if tensor.is_complex():
        positive_count = None
    else:
        positive_count = int((tensor > 0).sum().item()) if count else 0
    row = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": count,
        "finite_count": finite_count,
        "finite_ratio": (float(finite_count) / count if count else 1.0),
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
        "minimum_finite_value": minimum,
        "maximum_finite_value": maximum,
        "mean_finite_value": mean,
        "std_finite_value": std,
        "maximum_absolute_finite_value": max_abs,
        "finite_l2_norm": l2_norm,
        "scalar_value": mean if count == 1 and finite_count == 1 else None,
        "valid_pixel_count": positive_count,
        "nonzero_pixel_count": nonzero_count,
    }
    lowered = str(name).lower()
    if "mask" in lowered or "valid" in lowered:
        # A mask is itself one of the tensors guarded by fail-fast.  Computing
        # its aggregate over NaN/Inf values would put NaN/Inf in the JSON event
        # and make ``allow_nan=False`` hide the original numerical failure.
        # Keep the ordinary sum/mean only when the complete mask is finite and
        # additionally expose finite-only aggregates for a corrupted mask.
        finite_mask_values = tensor[finite].double()
        finite_mask_sum = (
            float(finite_mask_values.sum().item()) if finite_count else 0.0)
        mask_sum = finite_mask_sum if finite_count == count else None
        row.update({
            "mask_sum": mask_sum,
            "mask_mean": (
                mask_sum / count if count and mask_sum is not None else None),
            "finite_mask_sum": finite_mask_sum,
            "finite_mask_mean": (
                finite_mask_sum / finite_count if finite_count else None),
            "mask_empty": (
                count == 0 or (finite_count == count and mask_sum == 0.0)),
        })
    return row


def _json_safe_value(value):
    """Return a strict-JSON representation without concealing tensor counts."""
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _collection_statistics(value, prefix=""):
    return {
        name: _tensor_statistics(name, tensor)
        for name, tensor in _iter_named_tensors(value, prefix)
    }


def _check_finite_tensor(name, tensor, stage,
                         failure_loss_component=None):
    if not torch.is_tensor(tensor) or not _is_finite_check_dtype(tensor):
        return
    if not bool(torch.isfinite(tensor).all().item()):
        raise _NonfiniteDetected(
            stage, name, _tensor_statistics(name, tensor),
            failure_loss_component=failure_loss_component)


def _check_finite_collection(value, stage, prefix=""):
    for name, tensor in _iter_named_tensors(value, prefix):
        _check_finite_tensor(name, tensor, stage)


def _check_finite_loss(loss, loss_components=None):
    if isinstance(loss_components, dict):
        for component, value in loss_components.items():
            if component == "loss":
                continue
            for name, tensor in _iter_named_tensors(
                    value, f"loss_components.{component}"):
                _check_finite_tensor(
                    name, tensor, "loss_component",
                    failure_loss_component=str(component))
    if not torch.is_tensor(loss) or loss.numel() != 1 or not loss.is_floating_point():
        raise RuntimeError(
            "numerical fail-fast requires the adapter's total loss to be one "
            "floating scalar tensor")
    _check_finite_tensor("loss.total", loss, "loss")


def _named_gradient_tensors(model):
    return [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]


def _gradient_statistics(model):
    rows = {}
    missing = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing += 1
            continue
        rows[name] = _tensor_statistics(name, parameter.grad)
    return {
        "checked_gradient_count": len(rows),
        "missing_gradient_count": missing,
        "parameters": rows,
    }


def _check_finite_gradients(model):
    gradients = _named_gradient_tensors(model)
    if not gradients:
        return
    flags = torch.stack([
        torch.isfinite(gradient).all() for _name, gradient in gradients
    ])
    if bool(flags.all().item()):
        return
    for (name, gradient), finite in zip(gradients, flags.tolist()):
        if not finite:
            raise _NonfiniteDetected(
                "backward", f"gradient.{name}",
                _tensor_statistics(name, gradient))


def _gradient_finite_report(model):
    """Return a read-only gradient report without altering gradient values."""
    gradients = _named_gradient_tensors(model)
    first_name = None
    first_statistics = None
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    for name, gradient in gradients:
        statistics = None
        if not bool(torch.isfinite(gradient).all().item()):
            statistics = _tensor_statistics(name, gradient)
            nan_count += int(statistics["nan_count"])
            posinf_count += int(statistics["positive_infinity_count"])
            neginf_count += int(statistics["negative_infinity_count"])
        if statistics is not None and first_name is None:
            first_name = name
            first_statistics = statistics
    return {
        "finite": first_name is None,
        "checked_gradient_count": len(gradients),
        "first_nonfinite_parameter": first_name,
        "first_nonfinite_statistics": first_statistics,
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
    }


def _install_readonly_backward_diagnostics(model):
    """Observe intermediate/parameter gradients without replacing tensors."""
    state = {"context": None}
    handles = []
    root = getattr(model, "net", model)

    def observe(label, gradient):
        context = state.get("context")
        if (context is None or not torch.is_tensor(gradient)
                or not _is_finite_check_dtype(gradient)
                or bool(torch.isfinite(gradient).all().item())
                or context.get("first_nonfinite_backward_tensor") is not None):
            return
        context["first_nonfinite_backward_tensor"] = str(label)
        context["first_nonfinite_backward_statistics"] = _tensor_statistics(
            str(label), gradient)

    def tensor_hook(label):
        def hook(gradient):
            observe(label, gradient)
            return None
        return hook

    def forward_hook(module_name):
        def hook(_module, _inputs, output):
            for tensor_name, tensor in _iter_named_tensors(
                    output, f"module.net.{module_name}.output"):
                if tensor.requires_grad:
                    tensor.register_hook(tensor_hook(tensor_name))
        return hook

    for module_name, module in root.named_modules():
        if module_name:
            handles.append(module.register_forward_hook(
                forward_hook(module_name)))
    for parameter_name, parameter in root.named_parameters():
        if parameter.requires_grad:
            handles.append(parameter.register_hook(tensor_hook(
                f"parameter.net.{parameter_name}")))
    return state, handles


def _remove_hook_handles(handles):
    for handle in handles:
        handle.remove()


def _trace_tensor_fields(prefix, tensor):
    if not torch.is_tensor(tensor):
        return {
            f"{prefix}_dtype": None,
            f"{prefix}_finite": None,
            f"{prefix}_value": None,
        }
    finite = bool(torch.isfinite(tensor).all().item())
    return {
        f"{prefix}_dtype": str(tensor.dtype).replace("torch.", ""),
        f"{prefix}_finite": finite,
        f"{prefix}_value": _finite_scalar(tensor),
    }


def _numerical_trace_record(context, *, epoch, iteration, global_step,
                            successful_optimizer_steps, skipped,
                            scheduler_step_count, scheduler_advanced,
                            device):
    prediction = _first_tensor(context.get("predictions") or {})
    prediction_stats = _finite_min_max_mean(prediction)
    components = context.get("loss_components") or {}
    gradient_report = (
        context.get("unscaled_gradient_report")
        or context.get("scaled_gradient_report") or {})
    gradient_stats = gradient_report.get("first_nonfinite_statistics") or {}
    record = {
        "epoch": int(epoch),
        "iteration": int(iteration),
        "global_step": int(global_step),
        "successful_optimizer_step": bool(not skipped),
        "successful_optimizer_step_count": int(successful_optimizer_steps),
        "optimizer_step_count": int(successful_optimizer_steps),
        "optimizer_step_skipped": bool(skipped),
        "scheduler_step": bool(scheduler_advanced),
        "scheduler_step_count": int(scheduler_step_count),
        "prediction_dtype": (
            str(prediction.dtype).replace("torch.", "")
            if torch.is_tensor(prediction) else None),
        "prediction_finite": (
            bool(torch.isfinite(prediction).all().item())
            if torch.is_tensor(prediction) else None),
        "prediction_min": prediction_stats["min"],
        "prediction_max": prediction_stats["max"],
        "prediction_mean": prediction_stats["mean"],
        "first_nonfinite_parameter": gradient_report.get(
            "first_nonfinite_parameter"),
        "first_bad_parameter": gradient_report.get(
            "first_nonfinite_parameter"),
        "first_nonfinite_backward_tensor": context.get(
            "first_nonfinite_backward_tensor"),
        "grad_nan_count": gradient_report.get(
            "nan_count", gradient_stats.get("nan_count", 0)),
        "grad_posinf_count": gradient_report.get(
            "positive_infinity_count",
            gradient_stats.get("positive_infinity_count", 0)),
        "grad_neginf_count": gradient_report.get(
            "negative_infinity_count",
            gradient_stats.get("negative_infinity_count", 0)),
        "max_memory_allocated": None,
        "max_memory_reserved": None,
        "scaler_before": context.get("amp_scale_before"),
        "scaler_after": context.get("amp_scale_after"),
        "scaled_gradient_finite": (
            (context.get("scaled_gradient_report") or {}).get("finite")),
        "unscaled_gradient_finite": (
            (context.get("unscaled_gradient_report") or {}).get("finite")),
    }
    for name, keys in {
            "loss_absolute": ("absolute", "loss_absolute"),
            "loss_relative": ("relative", "loss_relative"),
            "loss_gradient": ("gradient", "loss_gradient"),
            "loss_total": ("total", "loss")}.items():
        value = next((components[key] for key in keys if key in components), None)
        record.update(_trace_tensor_fields(name, value))
    # Official SPNet loss names, kept in addition to the generic runner names.
    for official_name, generic_name in {
            "weighted_data_loss": "loss_absolute",
            "standardize_data_loss": "loss_relative",
            "weighted_msgrad_loss": "loss_gradient"}.items():
        for suffix in ("dtype", "finite", "value"):
            record[f"{official_name}_{suffix}"] = record.get(
                f"{generic_name}_{suffix}")
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        record["max_memory_allocated"] = int(
            torch.cuda.max_memory_allocated(device))
        record["max_memory_reserved"] = int(
            torch.cuda.max_memory_reserved(device))
    return record


def _parameter_module(model, parameter_name):
    if not parameter_name:
        return None
    module_name = str(parameter_name).rsplit(".", 1)[0]
    module = dict(model.named_modules()).get(module_name)
    if module is None:
        return module_name
    return f"{module_name} ({type(module).__name__})"


def _named_model_state_tensors(model, include_buffers=True):
    rows = [(f"parameter.{name}", value)
            for name, value in model.named_parameters()]
    if include_buffers:
        rows.extend((f"buffer.{name}", value)
                    for name, value in model.named_buffers())
    return rows


def _parameter_statistics(model, only_nonfinite=False):
    rows = {}
    for name, tensor in _named_model_state_tensors(model):
        if (only_nonfinite and _is_finite_check_dtype(tensor)
                and bool(torch.isfinite(tensor).all().item())):
            continue
        rows[name] = _tensor_statistics(name, tensor)
    return {"state_tensors": rows}


def _check_finite_parameters(model, stage="optimizer_step"):
    tensors = [
        (name, tensor) for name, tensor in _named_model_state_tensors(model)
        if _is_finite_check_dtype(tensor)
    ]
    if not tensors:
        return
    flags = torch.stack([
        torch.isfinite(tensor).all() for _name, tensor in tensors
    ])
    if bool(flags.all().item()):
        return
    for (name, tensor), finite in zip(tensors, flags.tolist()):
        if not finite:
            raise _NonfiniteDetected(
                stage, name, _tensor_statistics(name, tensor))


def _iter_optimizer_state_tensors(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        parameter_name = names.get(id(parameter), "unnamed_parameter")
        for name, tensor in _iter_named_tensors(
                state, f"optimizer_state.{parameter_name}"):
            yield name, tensor


def _check_finite_optimizer_state(model, optimizer):
    tensors = [
        (name, tensor)
        for name, tensor in _iter_optimizer_state_tensors(model, optimizer)
        if _is_finite_check_dtype(tensor)
    ]
    if not tensors:
        return
    by_device = {}
    for item in tensors:
        by_device.setdefault(str(item[1].device), []).append(item)
    for device_tensors in by_device.values():
        flags = torch.stack([
            torch.isfinite(tensor).all()
            for _name, tensor in device_tensors
        ])
        if bool(flags.all().item()):
            continue
        for (name, tensor), finite in zip(
                device_tensors, flags.tolist()):
            if not finite:
                raise _NonfiniteDetected(
                    "optimizer_step", name,
                    _tensor_statistics(name, tensor))


def _create_grad_scaler(enabled):
    """Create a CUDA GradScaler only when loss scaling is explicitly enabled."""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _checked_optimizer_step(model, optimizer, scaler, context,
                            numerical_failfast, amp_enabled=False,
                            post_step_diagnostic=None):
    """Execute one scaler-mediated optimizer step and validate its result.

    GradScaler intentionally skips ``optimizer.step`` when any unscaled
    gradient is non-finite. A decreasing scale signals that the step was
    skipped. Transient skips are expected AMP recovery; persistent consecutive
    skips are fail-closed by the caller. Parameters and optimizer state are
    still validated afterwards under numerical fail-fast.
    """
    scaler_enabled = bool(
        amp_enabled and scaler is not None and scaler.is_enabled())
    scale_before = float(scaler.get_scale()) if scaler_enabled else None
    context["optimizer_step_attempted"] = True
    if scaler_enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scale_after = float(scaler.get_scale()) if scaler_enabled else None
    step_skipped = bool(
        scaler_enabled
        and scale_before is not None
        and scale_after is not None
        and scale_after < scale_before
    )
    context["optimizer_step_completed"] = not step_skipped
    context["amp_overflow_skipped"] = step_skipped
    context["amp_scale_before"] = scale_before
    context["amp_scale_after"] = scale_after
    if step_skipped:
        print(
            "[train][amp] non-finite unscaled gradient detected; "
            "GradScaler skipped optimizer.step and reduced scale "
            f"{scale_before:g} -> {scale_after:g}")
    if callable(post_step_diagnostic):
        post_step_diagnostic()
    if numerical_failfast:
        _check_finite_parameters(model, stage="optimizer_step")
        _check_finite_optimizer_state(model, optimizer)
    return not step_skipped


def _amp_overflow_fail_closed(
        context, *, consecutive_overflows, successful_optimizer_steps,
        maximum_allowed_consecutive_amp_overflows):
    """Raise a structured fail-fast signal for persistent AMP overflow."""
    scale_before = context.get("amp_scale_before")
    scale_after = context.get("amp_scale_after")
    statistics = {
        "scale_before": scale_before,
        "scale_after": scale_after,
        "consecutive_amp_overflows": int(consecutive_overflows),
        "successful_optimizer_steps": int(successful_optimizer_steps),
        "maximum_allowed_consecutive_amp_overflows": int(
            maximum_allowed_consecutive_amp_overflows),
    }
    context["amp_consecutive_overflows"] = int(consecutive_overflows)
    context["amp_successful_optimizer_steps"] = int(successful_optimizer_steps)
    context["maximum_allowed_consecutive_amp_overflows"] = int(
        maximum_allowed_consecutive_amp_overflows)
    raise _NonfiniteDetected(
        "amp_overflow",
        "persistent_nonfinite_unscaled_gradients",
        statistics)


def _capture_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_repro_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _cpu_repro_value(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_repro_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_repro_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _cpu_model_state(model, start_buffers=None):
    start_buffers = start_buffers or {}
    return {
        name: (
            (start_buffers[name] if name in start_buffers else value).detach().cpu()
            if torch.is_tensor(start_buffers.get(name, value))
            else _cpu_repro_value(start_buffers.get(name, value)))
        for name, value in model.state_dict().items()
    }


def _capture_start_buffers(model):
    return {name: value.detach().clone()
            for name, value in model.named_buffers()}


def _save_known_finite_reference(path, model, optimizer, epoch, global_step):
    torch.save({
        "schema": "comparison-numerical-reference-v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def _load_torch_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _source_sha256(adapter):
    paths = [os.path.realpath(__file__)]
    adapter_file = getattr(adapter, "__file__", None)
    if adapter_file:
        paths.append(os.path.realpath(adapter_file))
    if hasattr(adapter, "numerical_source_files"):
        paths.extend(os.path.realpath(str(path))
                     for path in adapter.numerical_source_files())
    rows = {}
    for path in dict.fromkeys(paths):
        if os.path.isfile(path):
            rows[path] = _sha256_file(path)
    return rows


def _sample_ids_from_batch(batch):
    for key in ("sample_id", "name"):
        value = batch.get(key) if isinstance(batch, dict) else None
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
    return []


def _loss_component_statistics(context):
    components = context.get("loss_components") or {}
    return {
        str(name): _collection_statistics(value, str(name))
        for name, value in components.items()
    }


def _finite_scalar(value):
    if not torch.is_tensor(value) or value.numel() != 1:
        return None
    result = float(value.detach().item())
    return result if math.isfinite(result) else None


def _component_scalar(context, *names):
    components = context.get("loss_components") or {}
    for name in names:
        if name in components:
            return _finite_scalar(components[name])
    return None


def _component_finite(context, *names):
    components = context.get("loss_components") or {}
    for name in names:
        value = components.get(name)
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all().item())
    return None


def _first_tensor(value):
    for _name, tensor in _iter_named_tensors(value):
        return tensor
    return None


def _finite_min_max_mean(tensor):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return {"min": None, "max": None, "mean": None}
    finite = tensor.detach()[torch.isfinite(tensor.detach())]
    if finite.numel() == 0:
        return {"min": None, "max": None, "mean": None}
    finite = finite.double()
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
    }


def _write_nonfinite_diagnostic(adapter, args, model, optimizer, batch,
                                context, error, audit, epoch, iteration,
                                global_step, rng_state, start_buffers=None,
                                finite_reference_path=None):
    """Persist the first failure atomically enough for a single training process."""
    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    diagnostic_dir = os.path.join(output_dir, "nonfinite_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    event_path = os.path.join(diagnostic_dir, "nonfinite_event.json")
    if os.path.exists(event_path):
        raise RuntimeError(
            "refusing to reuse a stale nonfinite_event.json; numerical "
            f"diagnostics require a clean output directory: {event_path}")

    if isinstance(error, _NonfiniteDetected):
        failure_stage = error.failure_stage
        failure_tensor = error.failure_tensor
        failure_component = error.failure_loss_component
    else:
        failure_stage = str(context.get("current_stage", "exception"))
        failure_tensor = str(context.get("failure_tensor", "exception"))
        failure_component = context.get("failure_loss_component")
    model._gln_numerical_valid = False
    model._gln_nonfinite_events = int(
        getattr(model, "_gln_nonfinite_events", 0)) + 1

    post_step = bool(context.get("optimizer_step_completed", False))
    validation_failure = failure_stage == "validation"
    initialization_failure = failure_stage == "initialization"
    if post_step and finite_reference_path and os.path.isfile(finite_reference_path):
        reference = _load_torch_payload(finite_reference_path)
        model_state = _cpu_repro_value(reference["model"])
        optimizer_state = _cpu_repro_value(reference["optimizer"])
        state_timing = (
            "last_persisted_confirmed_finite_reference_before_optimizer_step")
        state_reference = {
            "epoch": reference.get("epoch"),
            "global_step": reference.get("global_step"),
            "path": os.path.realpath(finite_reference_path),
        }
    else:
        model_state = _cpu_model_state(
            model, None if validation_failure or initialization_failure
            else start_buffers)
        optimizer_state = _cpu_repro_value(optimizer.state_dict())
        if validation_failure:
            state_timing = "end_of_epoch_before_checkpoint_after_finite_training_steps"
        elif initialization_failure:
            state_timing = "initialization_state_at_failure"
        else:
            state_timing = (
                "iteration_start_before_forward_reconstructed_from_unchanged_"
                "parameters_and_captured_buffers")
        state_reference = {"epoch": int(epoch), "global_step": int(global_step)}

    model_inputs = context.get("model_inputs") or {}
    predictions = context.get("predictions") or {}
    intermediates = context.get("intermediates") or {}
    base_diagnostic_only = context.get("diagnostic_only") or {}
    diagnostic_only = base_diagnostic_only
    diagnostic_event_fields = {}
    diagnostic_tensor_artifacts = {}
    enrichment_error = None
    if hasattr(adapter, "nonfinite_diagnostic_context"):
        try:
            adapter_context = dict(context)
            adapter_context.setdefault("sample_ids", _sample_ids_from_batch(batch))
            adapter_context.setdefault("runner_epoch", int(epoch))
            adapter_context.setdefault("failure_stage", failure_stage)
            adapter_context.setdefault("failure_tensor", failure_tensor)
            adapter_context.setdefault("failure_loss_component", failure_component)
            adapter_diagnostic = adapter.nonfinite_diagnostic_context(
                model, adapter_context)
            if not isinstance(adapter_diagnostic, dict):
                raise TypeError("adapter nonfinite_diagnostic_context must return a mapping")
            reserved = {
                "event_fields", "tensor_artifacts", "tensor_statistics",
            }
            if reserved.intersection(adapter_diagnostic):
                unexpected = set(adapter_diagnostic) - reserved
                if unexpected:
                    raise ValueError(
                        "reserved diagnostic response has unexpected keys: "
                        + ", ".join(sorted(unexpected)))
                event_fields = adapter_diagnostic.get("event_fields") or {}
                tensor_artifacts = adapter_diagnostic.get("tensor_artifacts") or {}
                tensor_statistics = adapter_diagnostic.get("tensor_statistics") or {}
                if not all(isinstance(value, dict) for value in (
                        event_fields, tensor_artifacts, tensor_statistics)):
                    raise TypeError(
                        "adapter diagnostic event fields, tensor artifacts, "
                        "and tensor statistics must be mappings")
                # Fail here, while the core event is still untouched, if an
                # adapter tries to place a tensor/path/custom object in JSON.
                json.dumps(
                    _json_safe_value(event_fields), allow_nan=False,
                    sort_keys=True)
                pending_diagnostic_only = dict(base_diagnostic_only)
                pending_diagnostic_only.update(tensor_statistics)
                # Commit all three channels only after the whole response has
                # passed validation.
                diagnostic_event_fields = dict(event_fields)
                diagnostic_tensor_artifacts = dict(tensor_artifacts)
                diagnostic_only = pending_diagnostic_only
            else:
                # Backward-compatible path for existing adapters whose hook
                # returns only diagnostic tensors for generic statistics.
                diagnostic_only = dict(base_diagnostic_only)
                diagnostic_only.update(adapter_diagnostic)
        except Exception as exc:  # diagnostics must not hide the first failure
            enrichment_error = f"{type(exc).__name__}: {exc}"
    input_statistics = {
        "raw_batch": _collection_statistics(batch, "batch"),
        "model_inputs": _collection_statistics(model_inputs, "model_inputs"),
    }
    prediction_statistics = {
        "predictions": _collection_statistics(predictions, "predictions"),
        "intermediates": _collection_statistics(
            intermediates, "intermediates"),
        "diagnostic_only": _collection_statistics(
            diagnostic_only, "diagnostic_only"),
    }
    gradient_statistics = _gradient_statistics(model)
    parameter_statistics = _parameter_statistics(model, only_nonfinite=True)
    device = next(model.parameters(), torch.empty(0)).device
    cuda_index = device.index if device.type == "cuda" else None
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(cuda_index)
    fingerprint = ((audit or {}).get("split_fingerprints") or {}).get("train")
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    scaled_gradient_report = context.get("scaled_gradient_report") or {}
    unscaled_gradient_report = context.get("unscaled_gradient_report") or {}
    live_gradient_report = _gradient_finite_report(model)
    first_bad_parameter = (
        unscaled_gradient_report.get("first_nonfinite_parameter")
        or scaled_gradient_report.get("first_nonfinite_parameter")
        or live_gradient_report.get("first_nonfinite_parameter"))
    first_bad_statistics = (
        unscaled_gradient_report.get("first_nonfinite_statistics")
        or scaled_gradient_report.get("first_nonfinite_statistics")
        or live_gradient_report.get("first_nonfinite_statistics") or {})
    prediction_tensor = _first_tensor(predictions)
    prediction_summary = _finite_min_max_mean(prediction_tensor)
    target_tensor = model_inputs.get("target")
    if not torch.is_tensor(target_tensor) and isinstance(batch, dict):
        target_tensor = batch.get("target_depth")
    target_summary = _finite_min_max_mean(target_tensor)
    sparse_mask = model_inputs.get("hole_raw")
    if not torch.is_tensor(sparse_mask) and isinstance(batch, dict):
        sparse_value = batch.get("sparse_depth")
        if torch.is_tensor(sparse_value):
            sparse_mask = sparse_value > 0
    target_mask = None
    if torch.is_tensor(target_tensor):
        target_mask = target_tensor > 0
    previous_losses = context.get("previous_successful_step_losses") or {}
    event = {
        "schema": "comparison-nonfinite-event-v1",
        "method": getattr(adapter, "METHOD_NAME", "method"),
        "formal_protocol_variant": getattr(
            args, "formal_protocol_variant", None),
        "psd_training_domain": getattr(args, "psd_training_domain", None),
        "dataset_name": getattr(args, "dataset_name", None),
        "output_dir": output_dir,
        "seed": int(args.seed),
        "epoch": int(epoch),
        "iteration": int(iteration),
        "global_step": int(global_step),
        "sample_ids": _sample_ids_from_batch(batch),
        "scaler_before": context.get("amp_scale_before"),
        "scaler_after": context.get("amp_scale_after"),
        "optimizer_step_skipped": bool(
            context.get("optimizer_step_skipped", False)
            or context.get("amp_overflow_skipped", False)),
        "loss_total": _component_scalar(context, "total", "loss"),
        "loss_absolute": _component_scalar(context, "absolute", "loss_absolute"),
        "loss_relative": _component_scalar(context, "relative", "loss_relative"),
        "loss_gradient": _component_scalar(context, "gradient", "loss_gradient"),
        "loss_total_finite": _component_finite(context, "total", "loss"),
        "loss_absolute_finite": _component_finite(
            context, "absolute", "loss_absolute"),
        "loss_relative_finite": _component_finite(
            context, "relative", "loss_relative"),
        "loss_gradient_finite": _component_finite(
            context, "gradient", "loss_gradient"),
        "forward_prediction_finite": (
            bool(torch.isfinite(prediction_tensor).all().item())
            if torch.is_tensor(prediction_tensor) else None),
        "prediction_min": prediction_summary["min"],
        "prediction_max": prediction_summary["max"],
        "prediction_mean": prediction_summary["mean"],
        "target_min": target_summary["min"],
        "target_max": target_summary["max"],
        "sparse_valid_count": (
            int((sparse_mask > 0).sum().item())
            if torch.is_tensor(sparse_mask) else None),
        "target_valid_count": (
            int(target_mask.sum().item())
            if torch.is_tensor(target_mask) else None),
        "first_nonfinite_parameter": first_bad_parameter,
        "first_nonfinite_backward_tensor": context.get(
            "first_nonfinite_backward_tensor"),
        "first_nonfinite_backward_statistics": context.get(
            "first_nonfinite_backward_statistics"),
        "parameter_module": _parameter_module(model, first_bad_parameter),
        "grad_nan_count": first_bad_statistics.get("nan_count", 0),
        "grad_posinf_count": first_bad_statistics.get(
            "positive_infinity_count", 0),
        "grad_neginf_count": first_bad_statistics.get(
            "negative_infinity_count", 0),
        "scaled_gradient_finite": scaled_gradient_report.get("finite"),
        "unscaled_gradient_finite": unscaled_gradient_report.get("finite"),
        "forward_compute_dtype": getattr(
            args, "forward_compute_dtype", None),
        "loss_compute_dtype": getattr(args, "loss_compute_dtype", None),
        "grad_scaler_enabled": bool(getattr(
            args, "grad_scaler_enabled", False)),
        "max_memory_allocated": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" and torch.cuda.is_available() else None),
        "max_memory_reserved": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" and torch.cuda.is_available() else None),
        "previous_successful_step_losses": previous_losses,
        "failure_stage": failure_stage,
        "failure_tensor": failure_tensor,
        "failure_loss_component": failure_component,
        "failure_tensor_statistics": (
            error.statistics if isinstance(error, _NonfiniteDetected) else None),
        "loss_components": _loss_component_statistics(context),
        "learning_rate": learning_rates[0] if len(learning_rates) == 1 else learning_rates,
        "physical_batch_size": int(getattr(
            args, "physical_batch_size", args.batch_size)),
        "gradient_accumulation": int(getattr(args, "grad_accum_steps", 1)),
        "effective_batch_size": int(getattr(
            args, "effective_batch_size",
            int(args.batch_size) * int(getattr(args, "grad_accum_steps", 1)))),
        "amp": bool(getattr(args, "amp", False)),
        "optimizer_step_attempted": bool(
            context.get("optimizer_step_attempted", False)),
        "optimizer_step_completed": bool(
            context.get("optimizer_step_completed", False)),
        "amp_overflow_skipped": bool(
            context.get("amp_overflow_skipped", False)),
        "amp_scale_before": context.get("amp_scale_before"),
        "amp_scale_after": context.get("amp_scale_after"),
        "amp_consecutive_overflows": context.get("amp_consecutive_overflows"),
        "amp_successful_optimizer_steps": context.get(
            "amp_successful_optimizer_steps"),
        "maximum_allowed_consecutive_amp_overflows": context.get(
            "maximum_allowed_consecutive_amp_overflows"),
        "scheduler_step_skipped_due_amp_overflow": bool(
            context.get("scheduler_step_skipped_due_amp_overflow", False)),
        "input_tensor_statistics": input_statistics,
        "prediction_tensor_statistics": prediction_statistics,
        "gradient_statistics": gradient_statistics,
        "parameter_statistics": parameter_statistics,
        "CUDA_device": str(device),
        "CUDA_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "GPU_name": gpu_name,
        "torch_version": torch.__version__,
        "CUDA_version": torch.version.cuda,
        "git_commit": None,
        "source_sha256": _source_sha256(adapter),
        "dataset_ordered_fingerprint": fingerprint,
        "saved_state_timing": state_timing,
        "saved_state_reference": state_reference,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
        },
        "diagnostic_enrichment_error": enrichment_error,
        "confhead_batchnorm_trace": context.get("confhead_bn_trace"),
        "confhead_batchnorm_failure_artifacts": context.get(
            "confhead_bn_failure_artifacts"),
        "optimizer_state_statistics": context.get(
            "confhead_optimizer_state_statistics"),
    }
    collisions = set(event).intersection(diagnostic_event_fields)
    if collisions:
        event["diagnostic_enrichment_error"] = (
            "adapter diagnostic event fields collide with runner fields: "
            + ", ".join(sorted(collisions)))
        diagnostic_event_fields = {}
        diagnostic_tensor_artifacts = {}
        prediction_statistics["diagnostic_only"] = _collection_statistics(
            base_diagnostic_only, "diagnostic_only")
    event.update(diagnostic_event_fields)

    torch.save({
        "schema": "comparison-nonfinite-batch-v1",
        "raw_batch": _cpu_repro_value(batch),
        "model_inputs": _cpu_repro_value(model_inputs),
    }, os.path.join(diagnostic_dir, "nonfinite_batch.pt"))
    torch.save(model_state, os.path.join(
        diagnostic_dir, "nonfinite_model_state.pt"))
    torch.save(optimizer_state, os.path.join(
        diagnostic_dir, "nonfinite_optimizer_state.pt"))
    torch.save(_cpu_repro_value(rng_state), os.path.join(
        diagnostic_dir, "nonfinite_rng_state.pt"))
    saved_tensor_artifacts = []
    artifact_records = []
    validated_artifacts = []
    try:
        seen_filenames = set()
        for raw_filename, payload in diagnostic_tensor_artifacts.items():
            filename = str(raw_filename)
            artifact_path = os.path.join(diagnostic_dir, filename)
            temporary_path = artifact_path + f".tmp.{os.getpid()}"
            if (os.path.basename(filename) != filename
                    or not filename.endswith(".pt")):
                raise ValueError(
                    "invalid diagnostic tensor artifact filename: " + filename)
            if filename in seen_filenames:
                raise ValueError(
                    "duplicate diagnostic tensor artifact filename: " + filename)
            if os.path.lexists(artifact_path) or os.path.lexists(temporary_path):
                raise FileExistsError(
                    "refusing to overwrite diagnostic tensor artifact: "
                    + artifact_path)
            seen_filenames.add(filename)
            validated_artifacts.append(
                (filename, artifact_path, temporary_path, payload))
        for filename, artifact_path, temporary_path, payload in validated_artifacts:
            torch.save(_cpu_repro_value(payload), temporary_path)
            os.replace(temporary_path, artifact_path)
            saved_tensor_artifacts.append(filename)
            artifact_records.append({
                "filename": filename,
                "size_bytes": int(os.path.getsize(artifact_path)),
                "sha256": _sha256_file(artifact_path),
            })
    except Exception as exc:  # preserve the original non-finite event
        event["diagnostic_enrichment_error"] = (
            f"{type(exc).__name__} while saving diagnostic tensor artifacts: {exc}")
        for _filename, artifact_path, temporary_path, _payload in validated_artifacts:
            for cleanup_path in (temporary_path, artifact_path):
                if os.path.lexists(cleanup_path):
                    try:
                        os.unlink(cleanup_path)
                    except OSError:
                        pass
        saved_tensor_artifacts = []
        artifact_records = []
    event["diagnostic_tensor_artifacts"] = saved_tensor_artifacts
    event["diagnostic_tensor_artifact_records"] = artifact_records
    trace = traceback.format_exc()
    if not trace or trace.strip() == "NoneType: None":
        trace = "".join(traceback.format_exception(
            type(error), error, error.__traceback__))
    with open(os.path.join(diagnostic_dir, "nonfinite_traceback.txt"), "w") as stream:
        stream.write(trace)
    with open(event_path, "w") as stream:
        json.dump(_json_safe_value(event), stream, indent=2,
                  sort_keys=True, allow_nan=False)
    return event_path


def _requires_numerical_valid_checkpoint(adapter):
    return bool(getattr(adapter, "REQUIRE_NUMERICAL_VALID_CHECKPOINT", False))


def _validate_numerical_validity_record(record, label="checkpoint"):
    if record.get("numerical_valid") is not True:
        raise RuntimeError(
            f"{label} numerical_valid=true binding is missing or false")
    nonfinite_events = record.get("nonfinite_events")
    if (type(nonfinite_events) is not int or nonfinite_events != 0):
        raise RuntimeError(f"{label} must bind nonfinite_events=0")


def _first_invalid_validation_metric(metrics, required_fields=("RMSE_m",)):
    for name in dict.fromkeys(required_fields):
        value = metrics.get(name)
        if (not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))):
            return name, value
    return None


def _validation_metrics_finite(metrics, required_fields=("RMSE_m",)):
    """Check only metrics that the validation protocol requires to exist.

    MetricBag intentionally emits NaN placeholders for unavailable optional
    branches such as proxy/raw/safe outputs.  Those neutral schema values must
    not invalidate a completed-depth method.  The raw completed prediction is
    checked separately, before MetricBag can clip it.
    """
    return _first_invalid_validation_metric(
        metrics, required_fields=required_fields) is None


def _numerical_validity_fields(adapter):
    if not _requires_numerical_valid_checkpoint(adapter):
        return {}
    return {"numerical_valid": True, "nonfinite_events": 0}


def _enforce_clean_numerical_output_dir(output_dir):
    conflicts = (
        "latest_net_G.pth", "best_net_G.pth", "final_net_G.pth",
        "diagnostic_run.json", ".numerical_finite_reference.pt",
        os.path.join("nonfinite_diagnostics", "nonfinite_event.json"),
        "bn_trace",
        os.path.join("bn_trace", "first_warning.json"),
        os.path.join("bn_trace", "failure_trace.json"),
        os.path.join("bn_trace", "failure_replay_bundle.pt"),
        os.path.join("bn_trace", "rolling"),
    )
    present = [name for name in conflicts
               if os.path.exists(os.path.join(output_dir, name))]
    if present:
        raise RuntimeError(
            "PSD numerical fail-fast requires a clean output directory; "
            "refusing stale formal/diagnostic artifacts: " + ", ".join(present))


def _prep_outdir(output_dir, overwrite):
    os.makedirs(output_dir, exist_ok=True)


def _autocast_context(device, enabled, dtype=None):
    """Return a version-safe autocast context without changing FP32 execution.

    PyTorch 1.10 validates ``dtype`` even when autocast is disabled, so passing
    ``dtype=None, enabled=False`` fails before the forward pass.  Avoid entering
    autocast entirely for FP32 and make CUDA's FP16 default explicit otherwise.
    """
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=dtype if dtype is not None else torch.float16,
        enabled=True)


def _require_path(args, attr, flag):
    value = getattr(args, attr, None)
    if value:
        return value
    legacy = getattr(args, "dataroot", None)
    if legacy:
        raise RuntimeError(
            f"missing required {flag}; --dataroot is deprecated for this strict protocol. "
            f"Pass {flag} explicitly.")
    raise RuntimeError(f"missing required {flag}")


def _require_checkpoint(args):
    if getattr(args, "checkpoint", None):
        return args.checkpoint
    raise RuntimeError("missing required --checkpoint")


def _sha256_file(path, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_name(path):
    return os.path.basename(os.path.realpath(os.path.expanduser(str(path or ""))))


def _kitti_eval_spec(role):
    role = str(role or "final")
    if role not in _KITTI_EVAL_ROLE_SPECS:
        raise RuntimeError(
            "formal KITTI evaluation role must be one of "
            + ", ".join(_KITTI_EVAL_ROLE_SPECS))
    return _KITTI_EVAL_ROLE_SPECS[role]


def _kitti_diagnostic_metadata(args):
    dataset_type = getattr(args, "_dataset_type", None)
    role = str(getattr(args, "kitti_eval_role", "final"))
    if dataset_type == "kitti" and role == "diagnostic-best":
        spec = _kitti_eval_spec("diagnostic-best")
        return {
            "evaluation_status": spec["evaluation_status"],
            "checkpoint_role": spec["checkpoint_role"],
            "checkpoint_stage": spec["checkpoint_stage"],
            "official_final_protocol": spec["official_final_protocol"],
            "formal_result": spec["formal_result"],
            "controlled_ranking": spec["controlled_ranking"],
            "result_eligible_for_main_table": spec[
                "result_eligible_for_main_table"],
        }
    if (dataset_type == "ood_npz" and role == "develop"
            and _source_training_domain(args) == "K"):
        return {
            "evaluation_status": "OOD_DEVELOP_BEST_DIAGNOSTIC",
            "checkpoint_role": "develop_best",
            "checkpoint_stage": "develop",
            "official_final_protocol": False,
            "formal_result": False,
            "controlled_ranking": False,
            "result_eligible_for_main_table": False,
        }
    return {}


def _apply_kitti_eval_role_metadata(args):
    dataset_type = getattr(args, "_dataset_type", None)
    role = str(getattr(args, "kitti_eval_role", "final") or "final")
    if dataset_type == "ood_npz":
        if role == "develop" and "develop_diagnostic" not in os.path.normpath(
                os.path.expanduser(str(args.output_dir))):
            raise RuntimeError(
                "OOD evaluation with a K develop checkpoint requires an output "
                "path containing develop_diagnostic")
        for key, value in _kitti_diagnostic_metadata(args).items():
            setattr(args, key, value)
        return
    if dataset_type != "kitti":
        return
    spec = _kitti_eval_spec(role)
    args.kitti_eval_role = role
    args.kitti_eval_split = spec["split"]
    args.kitti_eval_expected_phase = spec["phase"]
    args.kitti_eval_expected_checkpoint = spec["checkpoint"]
    if role == "diagnostic-best":
        leaf = os.path.basename(os.path.normpath(
            os.path.expanduser(str(args.output_dir))))
        if "best_develop_diagnostic" not in leaf:
            raise RuntimeError(
                "KITTI diagnostic-best output directory basename must contain "
                "best_develop_diagnostic")
        for key, value in _kitti_diagnostic_metadata(args).items():
            setattr(args, key, value)


def _enforce_kitti_diagnostic_best_scope(adapter, args):
    if (getattr(args, "_dataset_type", None) != "kitti"
            or str(getattr(args, "kitti_eval_role", "final")) != "diagnostic-best"):
        return
    method = getattr(adapter, "METHOD_NAME", None)
    if method not in ("SGNet", "GeoLayerNet"):
        raise RuntimeError(
            "KITTI diagnostic-best is available only for registered "
            "SGNet-K or GeoLayerNet-K develop checkpoints")
    if _source_training_domain(args) != "K":
        raise RuntimeError("KITTI diagnostic-best requires a K checkpoint")


def _checkpoint_protocol_stage(checkpoint):
    path = os.path.realpath(os.path.expanduser(str(checkpoint or "")))
    if not os.path.isfile(path):
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        return None
    stage = protocol.get("kitti_stage")
    return str(stage) if stage else None


def _formal_kitti_stage_enabled(adapter, args):
    return (
        bool(getattr(adapter, "REQUIRE_KITTI_TWO_STAGE", False))
        and getattr(args, "_dataset_type", None) == "kitti"
    )


def _resolved(path):
    return os.path.realpath(os.path.expanduser(str(path or "")))


def _nyu_small_spec(args):
    del args
    return None


def _apply_nyu_small_scope(args):
    del args


def _enforce_formal_nyu_source(args, mode):
    """Lock formal N to the exact GeoLayerNet NYU48K export and split roles."""
    if getattr(args, "_dataset_type", None) != "nyu":
        return
    small = _nyu_small_spec(args)
    if small is not None:
        if float(args.depth_unit_scale) != 10.0:
            raise RuntimeError("NYUv2-small ablation requires depth_unit_scale=10 metres")
        if str(getattr(args, "preprocess", "none") or "none").lower() != "none":
            raise RuntimeError("NYUv2-small ablation requires preprocess=none")
        if mode == "train":
            if str(args.train_phase) != "train" or str(args.val_phase) != "val":
                raise RuntimeError(
                    "NYUv2-small ablation requires train_phase=train and val_phase=val")
            if getattr(args, "train_list", None) or getattr(args, "val_list", None):
                raise RuntimeError(
                    "NYUv2-small ablation uses complete physical split directories")
        else:
            if str(args.test_phase) != "test" or getattr(args, "test_list", None):
                raise RuntimeError(
                    "NYUv2-small ablation evaluation requires the complete test654")
        return
    config = _resolved(getattr(args, "dataset_config", None))
    if config != _FORMAL_NYU_CONFIG:
        raise RuntimeError(
            "formal N requires the locked GeoLayerNet config "
            f"{_FORMAL_NYU_CONFIG}; got {config or '<missing>'}")
    if not os.path.isfile(config):
        raise RuntimeError(f"formal N dataset config is missing: {config}")
    if float(args.depth_unit_scale) != 10.0:
        raise RuntimeError("formal N requires depth_unit_scale=10 metres")
    if str(getattr(args, "preprocess", "none") or "none").lower() != "none":
        raise RuntimeError("formal N requires preprocess=none")

    if mode == "train":
        roots = {
            "train": _resolved(getattr(args, "train_dataroot", None)),
            "val": _resolved(getattr(args, "val_dataroot", None)),
        }
        if roots != {"train": _FORMAL_NYU_ROOT, "val": _FORMAL_NYU_ROOT}:
            raise RuntimeError(
                "formal N train/val must both use the locked GeoLayerNet root "
                f"{_FORMAL_NYU_ROOT}; got {roots}")
        if str(args.train_phase) != "train" or str(args.val_phase) != "val":
            raise RuntimeError("formal N requires train_phase=train and val_phase=val")
        if getattr(args, "train_list", None) or getattr(args, "val_list", None):
            raise RuntimeError(
                "formal N uses the complete directory train/val splits; sample-list "
                "subsetting is forbidden")
    else:
        root = _resolved(getattr(args, "test_dataroot", None))
        if root != _FORMAL_NYU_ROOT:
            raise RuntimeError(
                f"formal N ID evaluation requires {_FORMAL_NYU_ROOT}; got {root}")
        if str(args.test_phase) != "test":
            raise RuntimeError("formal N ID evaluation requires test_phase=test")
        if getattr(args, "test_list", None):
            raise RuntimeError(
                "formal N ID evaluation uses the complete official-654 test split; "
                "--test-list is forbidden")


def _enforce_formal_kitti_source(args, mode, adapter=None):
    """Lock formal K to GeoLayerNet's exact source root and manifests.

    OOD targets have ``dataset.type=ood_npz`` and therefore deliberately skip
    this target-side check; their source-training identity is verified from the
    checkpoint sidecar by :func:`_verify_formal_source_checkpoint`.
    """
    if getattr(args, "_dataset_type", None) != "kitti":
        return
    config = _resolved(getattr(args, "dataset_config", None))
    if config != _FORMAL_KITTI_CONFIG:
        raise RuntimeError(
            "formal K requires the locked GeoLayerNet config "
            f"{_FORMAL_KITTI_CONFIG}; got {config or '<missing>'}")
    if not os.path.isfile(config):
        raise RuntimeError(f"formal K dataset config is missing: {config}")
    if float(args.depth_unit_scale) != 80.0:
        raise RuntimeError("formal K requires depth_unit_scale=80 metres")
    if str(getattr(args, "preprocess", "none") or "none").lower() != "none":
        raise RuntimeError("formal K requires preprocess=none")

    if mode == "train":
        stage = getattr(args, "kitti_stage", None)
        if stage not in ("develop", "final-train"):
            raise RuntimeError(
                "formal K training requires --kitti-stage develop or final-train")
        train_root = _resolved(getattr(args, "train_dataroot", None))
        if train_root != _FORMAL_KITTI_ROOT:
            raise RuntimeError(
                f"formal K train root must be {_FORMAL_KITTI_ROOT}; got {train_root}")
        expected_train = _FORMAL_KITTI_MANIFESTS[
            "train90" if stage == "develop" else "train_full"]
        if _resolved(getattr(args, "train_list", None)) != expected_train:
            raise RuntimeError(
                f"formal K {stage} requires train manifest {expected_train}")
        if stage == "develop":
            val_root = _resolved(getattr(args, "val_dataroot", None))
            if val_root != _FORMAL_KITTI_ROOT:
                raise RuntimeError(
                    f"formal K dev root must be {_FORMAL_KITTI_ROOT}; got {val_root}")
            if _resolved(getattr(args, "val_list", None)) != (
                    _FORMAL_KITTI_MANIFESTS["dev10"]):
                raise RuntimeError(
                    "formal K develop requires the locked dev10 manifest "
                    f"{_FORMAL_KITTI_MANIFESTS['dev10']}")
        else:
            if (bool(getattr(adapter, "REQUIRE_KITTI_FIXED_E_ONLY", False))
                    or not bool(getattr(
                        adapter, "REQUIRE_KITTI_DEVELOP_RECEIPT", True))):
                return
            receipt_path = _resolved(getattr(args, "develop_receipt", None))
            if not os.path.isfile(receipt_path):
                raise RuntimeError(
                    "formal K final-train requires the develop receipt artifact")
            with open(receipt_path) as stream:
                receipt = json.load(stream)
            if (receipt.get("schema") != "geolayernet-kitti-develop-v1"
                    or receipt.get("stage") != "develop"):
                raise RuntimeError("formal K develop receipt schema mismatch")
            if (_resolved(receipt.get("train_manifest")) !=
                    _FORMAL_KITTI_MANIFESTS["train90"]
                    or _resolved(receipt.get("val_manifest")) !=
                    _FORMAL_KITTI_MANIFESTS["dev10"]):
                raise RuntimeError(
                    "formal K develop receipt does not bind the locked train90/dev10 manifests")
            for key in ("train_manifest", "val_manifest"):
                path = _resolved(receipt.get(key))
                if not os.path.isfile(path):
                    raise RuntimeError(
                        f"formal K develop receipt {key} is missing")
                if receipt.get(key + "_sha256") != _sha256_file(path):
                    raise RuntimeError(
                        f"formal K develop receipt {key} SHA256 mismatch")
            if (receipt.get("train_dataroot") != _FORMAL_KITTI_ROOT
                    or receipt.get("val_dataroot") != _FORMAL_KITTI_ROOT):
                raise RuntimeError("formal K develop receipt source root mismatch")
            if receipt.get("dataset_config") != _FORMAL_KITTI_CONFIG:
                raise RuntimeError("formal K develop receipt config path mismatch")
            if receipt.get("dataset_config_sha256") != _sha256_file(
                    _FORMAL_KITTI_CONFIG):
                raise RuntimeError("formal K develop receipt config SHA256 mismatch")
            train_count = int(receipt.get("train_samples", -1))
            val_count = int(receipt.get("val_samples", -1))
            if train_count <= 0 or val_count <= 0:
                raise RuntimeError("formal K develop receipt split counts are invalid")
            _verify_bound_dataset_audit(
                receipt, {"train": train_count, "val": val_count})
        return

    test_root = _resolved(getattr(args, "test_dataroot", None))
    if test_root != _FORMAL_KITTI_ROOT:
        raise RuntimeError(
            f"formal K ID root must be {_FORMAL_KITTI_ROOT}; got {test_root}")
    role = str(getattr(args, "kitti_eval_role", "final") or "final")
    expected_split = _kitti_eval_spec(role)["split"]
    if _resolved(getattr(args, "test_list", None)) != (
            _FORMAL_KITTI_MANIFESTS[expected_split]):
        raise RuntimeError(
            f"formal K {role} evaluation requires the locked {expected_split} "
            f"manifest {_FORMAL_KITTI_MANIFESTS[expected_split]}")
    if str(getattr(args, "test_phase", "")) != expected_split:
        raise RuntimeError(
            f"formal K {role} evaluation requires test_phase={expected_split}")


def _enforce_formal_nyu_loader_counts(args, *, train_loader=None,
                                      val_loader=None, test_loader=None):
    if getattr(args, "_dataset_type", None) != "nyu":
        return
    small = _nyu_small_spec(args)
    expected_counts = small["counts"] if small is not None else _FORMAL_NYU_COUNTS
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    for split, loader in loaders.items():
        if loader is None:
            continue
        actual = len(loader.dataset)
        expected = expected_counts[split]
        if actual != expected:
            raise RuntimeError(
                f"NYU protocol {split} sample count mismatch: expected {expected}, "
                f"got {actual}")


def _enforce_formal_nyu_audit_counts(args, audit, mode):
    if getattr(args, "_dataset_type", None) != "nyu":
        return
    small = _nyu_small_spec(args)
    expected_counts = small["counts"] if small is not None else _FORMAL_NYU_COUNTS
    expected_splits = ("train", "val") if mode == "train" else ("test",)
    counts = audit.get("counts", {})
    for split in expected_splits:
        expected = expected_counts[split]
        actual = counts.get(split)
        if actual != expected:
            raise RuntimeError(
                f"NYU protocol {split} audit count mismatch: expected {expected}, "
                f"got {actual!r}")


def _enforce_formal_kitti_audit_counts(args, audit, mode):
    if getattr(args, "_dataset_type", None) != "kitti":
        return
    if mode == "eval":
        role = str(getattr(args, "kitti_eval_role", "final") or "final")
        expected = _kitti_eval_spec(role)["split"]
        expected_by_audit_split = {
            "test": _FORMAL_KITTI_COUNTS[expected]}
    elif getattr(args, "kitti_stage", None) == "final-train":
        expected_by_audit_split = {
            "train": _FORMAL_KITTI_COUNTS["train_full"]}
    elif getattr(args, "kitti_stage", None) == "develop":
        expected_by_audit_split = {
            "train": _FORMAL_KITTI_COUNTS["train90"],
            "val": _FORMAL_KITTI_COUNTS["dev10"],
        }
    else:
        return
    counts = audit.get("counts", {})
    for split, expected in expected_by_audit_split.items():
        actual = counts.get(split)
        if actual != expected:
            raise RuntimeError(
                f"formal K {split} audit count mismatch: expected {expected}, "
                f"got {actual!r}")


def _enforce_formal_kitti_loader_counts(args, *, train_loader=None,
                                         val_loader=None, test_loader=None):
    if getattr(args, "_dataset_type", None) != "kitti":
        return
    if test_loader is not None:
        role = str(getattr(args, "kitti_eval_role", "final") or "final")
        split = _kitti_eval_spec(role)["split"]
        expected = _FORMAL_KITTI_COUNTS[split]
        actual = len(test_loader.dataset)
        if actual != expected:
            raise RuntimeError(
                f"formal K {split} sample count mismatch: expected {expected}, "
                f"got {actual}")
    stage = getattr(args, "kitti_stage", None)
    if train_loader is not None and stage in ("develop", "final-train"):
        actual = len(train_loader.dataset)
        split = "train90" if stage == "develop" else "train_full"
        expected = _FORMAL_KITTI_COUNTS[split]
        if actual != expected:
            raise RuntimeError(
                f"formal K {split} sample count mismatch: expected {expected}, "
                f"got {actual}")
    if val_loader is not None and stage == "develop":
        actual = len(val_loader.dataset)
        expected = _FORMAL_KITTI_COUNTS["dev10"]
        if actual != expected:
            raise RuntimeError(
                f"formal K dev10 sample count mismatch: expected {expected}, "
                f"got {actual}")


def _enforce_formal_kitti_train_stage(adapter, args):
    """Validate the two-stage KITTI protocol before any split is opened.

    The loader phase deliberately stays ``train`` for both stages so that the
    registered KITTI train crop and colour augmentation remain enabled.  The
    manifest, rather than the phase label, selects train90 versus train_full.
    """
    if not _formal_kitti_stage_enabled(adapter, args):
        return
    stage = getattr(args, "kitti_stage", None)
    if stage not in ("develop", "final-train"):
        raise RuntimeError(
            "formal KITTI training requires --kitti-stage develop or final-train")
    if str(getattr(args, "train_phase", "")) != "train":
        raise RuntimeError(
            "formal KITTI training must keep --train-phase train; the manifest "
            "selects train90/train_full while phase=train preserves train augmentation")
    if getattr(args, "checkpoint", None):
        raise RuntimeError(
            "formal KITTI stages must start from the method's registered initialization; "
            "--checkpoint completion-model initialization/resume is forbidden")
    if getattr(args, "test_dataroot", None) or getattr(args, "test_list", None):
        raise RuntimeError("KITTI training must not receive or audit the final test split")

    if stage == "develop":
        if _manifest_name(args.train_list) != "train90.jsonl":
            raise RuntimeError("KITTI develop requires --train-list train90.jsonl")
        if _manifest_name(args.val_list) != "dev10.jsonl":
            raise RuntimeError("KITTI develop requires --val-list dev10.jsonl")
        if bool(getattr(args, "no_val", False)) or not getattr(args, "val_dataroot", None):
            raise RuntimeError("KITTI develop requires dev10 validation; --no-val is forbidden")
        if str(getattr(args, "val_phase", "")) != "dev10":
            raise RuntimeError("KITTI develop requires --val-phase dev10")
        if getattr(args, "fixed_epochs", None) is not None or getattr(args, "develop_receipt", None):
            raise RuntimeError("KITTI develop does not accept --fixed-epochs/--develop-receipt")
        args._kitti_develop_receipt = None
    else:
        if _manifest_name(args.train_list) != "train_full.jsonl":
            raise RuntimeError("KITTI final-train requires --train-list train_full.jsonl")
        if not bool(getattr(args, "no_val", False)):
            raise RuntimeError("KITTI final-train requires --no-val")
        if getattr(args, "val_list", None):
            raise RuntimeError("KITTI final-train must not receive dev10/val1000 --val-list")
        fixed = getattr(args, "fixed_epochs", None)
        receipt_required = (
            not bool(getattr(adapter, "REQUIRE_KITTI_FIXED_E_ONLY", False))
            and bool(getattr(adapter, "REQUIRE_KITTI_DEVELOP_RECEIPT", True)))
        receipt_path = getattr(args, "develop_receipt", None)
        if fixed is None or int(fixed) <= 0:
            raise RuntimeError("KITTI final-train requires positive --fixed-epochs E")
        if receipt_required and (
                not receipt_path or not os.path.isfile(os.path.expanduser(receipt_path))):
            raise RuntimeError("KITTI final-train requires an existing --develop-receipt")
        if not receipt_required:
            if receipt_path:
                raise RuntimeError(
                    "this adapter forbids --develop-receipt; pass only the explicitly locked --fixed-epochs E")
            args.epochs = int(fixed)
            args._kitti_develop_receipt = None
            args._kitti_develop_receipt_path = None
            args._kitti_develop_receipt_sha256 = None
            return
        receipt_path = os.path.realpath(os.path.expanduser(receipt_path))
        with open(receipt_path) as stream:
            receipt = json.load(stream)
        if receipt.get("stage") != "develop":
            raise RuntimeError("KITTI develop receipt has the wrong stage")
        if receipt.get("method") != getattr(adapter, "METHOD_NAME", None):
            raise RuntimeError("KITTI develop receipt method does not match this adapter")
        if int(receipt.get("seed", -1)) != int(args.seed):
            raise RuntimeError("KITTI final-train seed must match its develop receipt")
        selected = int(receipt.get("selected_epoch", -1))
        if int(fixed) != selected:
            raise RuntimeError(
                f"KITTI --fixed-epochs={int(fixed)} does not match develop selected_epoch={selected}")
        develop_dir = os.path.realpath(os.path.dirname(receipt_path))
        final_dir = os.path.realpath(os.path.expanduser(args.output_dir))
        if develop_dir == final_dir:
            raise RuntimeError("KITTI final-train output directory must differ from develop")
        args.epochs = int(fixed)
        args._kitti_develop_receipt = receipt
        args._kitti_develop_receipt_path = receipt_path
        args._kitti_develop_receipt_sha256 = _sha256_file(receipt_path)


def _enforce_formal_training_splits(args):
    """Keep test data out of training and require dev-based selection."""
    if getattr(args, "test_dataroot", None) or getattr(args, "test_list", None):
        raise RuntimeError(
            "formal training accepts only train and validation splits; test data "
            "must be provided only to a separate evaluation process")
    is_kitti_final = (
        getattr(args, "_dataset_type", None) == "kitti"
        and getattr(args, "kitti_stage", None) == "final-train"
    )
    if is_kitti_final:
        return
    if bool(getattr(args, "no_val", False)):
        raise RuntimeError(
            "formal N/K develop training requires validation; --no-val is allowed "
            "only for KITTI final-train with a verified develop receipt")
    if not getattr(args, "val_dataroot", None) or not getattr(args, "val_phase", None):
        raise RuntimeError(
            "formal training requires an explicit validation split for checkpoint/"
            "epoch selection")


def _formal_kitti_sidecar_path(checkpoint):
    return os.path.join(os.path.dirname(os.path.realpath(checkpoint)),
                        "final_net_G.protocol.json")


def _formal_source_sidecar_path(checkpoint):
    checkpoint = os.path.realpath(os.path.expanduser(str(checkpoint)))
    stem, _extension = os.path.splitext(checkpoint)
    return stem + ".protocol.json"


def _nyu_small_sidecar_path(checkpoint):
    checkpoint = os.path.realpath(os.path.expanduser(str(checkpoint)))
    stem, _extension = os.path.splitext(checkpoint)
    return stem + ".nyu_small_ablation.protocol.json"


def _verify_nyu_small_checkpoint(adapter, args, checkpoint):
    spec = _nyu_small_spec(args)
    checkpoint = _resolved(checkpoint)
    sidecar_path = _nyu_small_sidecar_path(checkpoint)
    if not os.path.isfile(sidecar_path):
        raise RuntimeError(
            "NYUv2-small checkpoint sidecar is missing: " + sidecar_path)
    with open(sidecar_path) as stream:
        sidecar = json.load(stream)
    expected = {
        "schema": "geolayernet-nyu-small-ablation-checkpoint-v1",
        "status": "COMPLETE",
        "method": getattr(adapter, "METHOD_NAME", None),
        "training_domain": "N",
        "seed": int(args.seed),
        "sparse_points": int(spec["points"]),
        "counts": spec["counts"],
        "formal_result": False,
        "result_eligible_for_main_table": False,
    }
    for name, value in expected.items():
        if sidecar.get(name) != value:
            raise RuntimeError(
                "NYUv2-small checkpoint sidecar mismatch "
                f"field={name} actual={sidecar.get(name)!r} expected={value!r}")
    if (spec.get("sparse_percent") is not None and
            sidecar.get("sparse_input_percent") != spec["sparse_percent"]):
        raise RuntimeError(
            "NYUv2-small checkpoint sidecar sparse-percent mismatch")
    if _resolved(sidecar.get("checkpoint")) != checkpoint:
        raise RuntimeError("NYUv2-small checkpoint sidecar path mismatch")
    if sidecar.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise RuntimeError("NYUv2-small checkpoint SHA256 mismatch")
    protocol = spec["path"]
    if (_resolved(sidecar.get("dataset_protocol")) != protocol
            or sidecar.get("dataset_protocol_sha256") != _sha256_file(protocol)):
        raise RuntimeError("NYUv2-small dataset protocol binding mismatch")
    config_source = args.dataset_config
    if (getattr(args, "_dataset_type", None) == "ood_npz"
            and getattr(args, "geolayernet_runtime_config", None)):
        config_source = args.geolayernet_runtime_config
    config = _resolved(config_source)
    if (_resolved(sidecar.get("dataset_config")) != config
            or sidecar.get("dataset_config_sha256") != _sha256_file(config)):
        raise RuntimeError("NYUv2-small dataset config binding mismatch")
    _verify_bound_dataset_audit(sidecar, {"train": 640, "val": 155})
    args._nyu_small_checkpoint_sidecar = sidecar
    args.checkpoint_path = checkpoint
    args.checkpoint_loading = str(getattr(
        adapter, "CHECKPOINT_LOADING", "strict_state_dict"))
    args.checkpoint_registered = True


def _source_training_domain(args):
    domains = {
        str(value).upper() for key, value in vars(args).items()
        if key.endswith("_training_domain")
        and str(value).upper() in {"N", "K"}
    }
    if len(domains) > 1:
        raise RuntimeError(
            f"conflicting source training domains in evaluation args: {domains}")
    return next(iter(domains), None)


def _verify_bound_dataset_audit(sidecar, expected_counts):
    """Verify that a source sidecar is bound to an actual PASS audit.

    The fingerprint copied into the checkpoint sidecar is compared byte-for-
    byte with the hash-bound audit JSON.  Merely placing an arbitrary 64-digit
    string in the sidecar must never be sufficient for a formal result.
    """
    audit_path = sidecar.get("dataset_audit")
    if not audit_path or not os.path.isfile(audit_path):
        raise RuntimeError("formal source-training dataset audit is missing")
    if sidecar.get("dataset_audit_sha256") != _sha256_file(audit_path):
        raise RuntimeError("formal source-training dataset audit SHA256 mismatch")
    with open(audit_path) as stream:
        audit = json.load(stream)
    if audit.get("status") != "PASS":
        raise RuntimeError(
            "formal source-training dataset audit must have status PASS")
    sidecar_fingerprints = sidecar.get("split_fingerprints") or {}
    audit_fingerprints = audit.get("split_fingerprints") or {}
    for split, expected_count in expected_counts.items():
        if audit.get("counts", {}).get(split) != int(expected_count):
            raise RuntimeError(
                f"formal source {split} audit count mismatch")
        item = sidecar_fingerprints.get(split) or {}
        if item != (audit_fingerprints.get(split) or {}):
            raise RuntimeError(
                f"formal source {split} fingerprint does not match its audit")
        if (item.get("schema") != "ordered-sample-content-sha256-v1"
                or item.get("count") != int(expected_count)
                or len(str(item.get("sha256") or "")) != 64):
            raise RuntimeError(
                f"formal source lacks a valid {split} ordered content fingerprint")
    return audit


def _verify_formal_source_checkpoint(adapter, args):
    """Bind ID/OOD evaluation to the exact formal N or K training run."""
    domain = _source_training_domain(args)
    if not bool(getattr(adapter, "REQUIRES_CHECKPOINT", True)):
        return
    if domain is None:
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'checkpoint method')} formal "
            "evaluation has no registered N/K source training domain")
    checkpoint = _resolved(_require_checkpoint(args))
    if _nyu_small_spec(args) is not None:
        _verify_nyu_small_checkpoint(adapter, args, checkpoint)
        return
    if bool(getattr(adapter, "SKIP_SHARED_CHECKPOINT_PROVENANCE", False)):
        k_develop = (
            domain == "K"
            and _kitti_eval_spec(getattr(args, "kitti_eval_role", "final"))[
                "checkpoint"] == "best_net_G.pth")
        expected_name = (
            "best_net_G.pth" if domain == "N" or k_develop
            else "final_net_G.pth")
        if os.path.basename(checkpoint) != expected_name:
            raise RuntimeError(
                f"formal {domain} evaluation accepts only {expected_name} from "
                "the registered source-training stage")
        args.checkpoint_path = checkpoint
        args.checkpoint_loading = str(getattr(
            adapter, "CHECKPOINT_LOADING", "strict_state_dict"))
        args.checkpoint_registered = False
        return
    if domain in ("N", "K") and getattr(adapter, "METHOD_NAME", None) == "SGNet":
        # SGNet owns its checkpoint format and performs strict=True loading
        # in the adapter.  Historical receipts and SHA sidecars are not active
        # evaluation requirements.
        args.checkpoint_path = checkpoint
        args.checkpoint_loading = "strict_state_dict"
        args.checkpoint_registered = False
        return

    k_develop = (
        domain == "K"
        and _kitti_eval_spec(getattr(args, "kitti_eval_role", "final"))[
            "checkpoint"] == "best_net_G.pth")
    expected_name = (
        "best_net_G.pth" if domain == "N" or k_develop
        else "final_net_G.pth")
    if os.path.basename(checkpoint) != expected_name:
        raise RuntimeError(
            f"formal {domain} evaluation accepts only {expected_name} from the "
            "registered source-training stage")
    allow_missing_sidecar = bool(getattr(
        args, "allow_missing_checkpoint_sidecar", False))
    if allow_missing_sidecar:
        if not bool(getattr(
                adapter, "ALLOW_MISSING_CHECKPOINT_SIDECAR_EVAL", False)):
            raise RuntimeError(
                "adapter does not authorize missing checkpoint sidecar evaluation")
        allowed_domains = tuple(getattr(
            adapter, "ALLOW_MISSING_CHECKPOINT_SIDECAR_DOMAINS", ("N",)))
        if domain not in allowed_domains:
            raise RuntimeError(
                "missing checkpoint sidecar evaluation is not registered for "
                f"training domain {domain}")
        args.checkpoint_path = checkpoint
        args.checkpoint_loading = (
            "strict_adapter_checkpoint_with_embedded_protocol")
        args.checkpoint_registered = False
        args.checkpoint_provenance_verified = False
        args.checkpoint_provenance_status = (
            "MISSING_EXTERNAL_SIDECAR_EXPLICIT_UNRANKED")
        args.result_eligible_for_main_table = False
        args.formal_result = False
        args.controlled_ranking = False
        return
    sidecar_path = _formal_source_sidecar_path(checkpoint)
    if not os.path.isfile(sidecar_path):
        raise RuntimeError(
            f"formal {domain} checkpoint protocol sidecar is missing: {sidecar_path}")
    with open(sidecar_path) as stream:
        sidecar = json.load(stream)
    if sidecar.get("method") != getattr(adapter, "METHOD_NAME", None):
        raise RuntimeError(f"formal {domain} checkpoint sidecar method mismatch")
    if _requires_numerical_valid_checkpoint(adapter):
        _validate_numerical_validity_record(
            sidecar, f"formal {domain} checkpoint sidecar")
    if sidecar.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise RuntimeError(
            f"formal {domain} checkpoint SHA256 does not match its sidecar")

    if domain == "K":
        if sidecar.get("schema") != "geolayernet-kitti-final-v1":
            raise RuntimeError("KITTI checkpoint sidecar schema mismatch")
        if sidecar.get("stage") != "final-train":
            raise RuntimeError("KITTI checkpoint sidecar is not final-train")
        if sidecar.get("training_domain") != "K":
            raise RuntimeError("KITTI checkpoint sidecar domain mismatch")
        if int(sidecar.get("seed", -1)) != int(args.seed):
            raise RuntimeError("KITTI checkpoint sidecar seed mismatch")
        manifest = _resolved(sidecar.get("train_manifest"))
        if manifest != _FORMAL_KITTI_MANIFESTS["train_full"]:
            raise RuntimeError(
                "KITTI final checkpoint was not trained from the locked train_full manifest")
        if not os.path.isfile(manifest):
            raise RuntimeError("KITTI train_full manifest referenced by sidecar is missing")
        if sidecar.get("train_manifest_sha256") != _sha256_file(manifest):
            raise RuntimeError("KITTI train_full manifest SHA256 mismatch")
        if int(sidecar.get("train_samples", -1)) != _FORMAL_KITTI_COUNTS["train_full"]:
            raise RuntimeError("KITTI train_full sample count mismatch in sidecar")
        if sidecar.get("train_dataroot") != _FORMAL_KITTI_ROOT:
            raise RuntimeError("KITTI checkpoint sidecar source root mismatch")
        if sidecar.get("dataset_config") != _FORMAL_KITTI_CONFIG:
            raise RuntimeError("KITTI checkpoint sidecar config path mismatch")
        if not os.path.isfile(_FORMAL_KITTI_CONFIG):
            raise RuntimeError("formal K dataset config is missing")
        if sidecar.get("dataset_config_sha256") != _sha256_file(
                _FORMAL_KITTI_CONFIG):
            raise RuntimeError("formal K dataset config SHA256 mismatch")
        receipt_path = sidecar.get("develop_receipt")
        if not receipt_path or not os.path.isfile(receipt_path):
            raise RuntimeError("KITTI final checkpoint develop receipt is missing")
        if sidecar.get("develop_receipt_sha256") != _sha256_file(receipt_path):
            raise RuntimeError("KITTI final checkpoint develop receipt SHA256 mismatch")
        with open(receipt_path) as stream:
            receipt = json.load(stream)
        if (receipt.get("schema") != "geolayernet-kitti-develop-v1"
                or receipt.get("stage") != "develop"
                or receipt.get("training_domain") != "K"
                or receipt.get("method") != getattr(adapter, "METHOD_NAME", None)
                or int(receipt.get("seed", -1)) != int(sidecar.get("seed", -2))
                or int(receipt.get("selected_epoch", -1)) != int(
                    sidecar.get("epochs", -2))):
            raise RuntimeError("KITTI final checkpoint develop receipt binding mismatch")
        if (_resolved(receipt.get("train_manifest")) !=
                _FORMAL_KITTI_MANIFESTS["train90"]
                or _resolved(receipt.get("val_manifest")) !=
                _FORMAL_KITTI_MANIFESTS["dev10"]):
            raise RuntimeError("KITTI develop receipt manifest binding mismatch")
        for key in ("train_manifest", "val_manifest"):
            path = _resolved(receipt.get(key))
            if not os.path.isfile(path):
                raise RuntimeError(f"KITTI develop receipt {key} is missing")
            if receipt.get(key + "_sha256") != _sha256_file(path):
                raise RuntimeError(
                    f"KITTI develop receipt {key} SHA256 mismatch")
        if (receipt.get("train_dataroot") != _FORMAL_KITTI_ROOT
                or receipt.get("val_dataroot") != _FORMAL_KITTI_ROOT
                or receipt.get("dataset_config") != _FORMAL_KITTI_CONFIG
                or receipt.get("dataset_config_sha256") != _sha256_file(
                    _FORMAL_KITTI_CONFIG)):
            raise RuntimeError("KITTI develop receipt source/config binding mismatch")
        train_count = int(receipt.get("train_samples", -1))
        val_count = int(receipt.get("val_samples", -1))
        if train_count <= 0 or val_count <= 0:
            raise RuntimeError("KITTI develop receipt split counts are invalid")
        _verify_bound_dataset_audit(
            receipt, {"train": train_count, "val": val_count})
        selected_develop = receipt.get("selected_checkpoint")
        if not selected_develop or not os.path.isfile(selected_develop):
            raise RuntimeError("KITTI develop selected checkpoint is missing")
        if receipt.get("selected_checkpoint_sha256") != _sha256_file(
                selected_develop):
            raise RuntimeError("KITTI develop selected checkpoint SHA256 mismatch")
        _verify_bound_dataset_audit(
            sidecar, {"train": _FORMAL_KITTI_COUNTS["train_full"]})
        args._kitti_final_sidecar = sidecar
        return

    if sidecar.get("schema") != "geolayernet-nyu48k-final-v1":
        raise RuntimeError("formal N checkpoint sidecar schema mismatch")
    if sidecar.get("training_domain") != "N":
        raise RuntimeError("formal N checkpoint sidecar domain mismatch")
    if sidecar.get("dataset_config") != _FORMAL_NYU_CONFIG:
        raise RuntimeError("formal N checkpoint sidecar config path mismatch")
    if sidecar.get("dataset_config_sha256") != _sha256_file(_FORMAL_NYU_CONFIG):
        raise RuntimeError("formal N dataset config SHA256 mismatch")
    if sidecar.get("train_dataroot") != _FORMAL_NYU_ROOT or sidecar.get(
            "val_dataroot") != _FORMAL_NYU_ROOT:
        raise RuntimeError("formal N checkpoint sidecar source root mismatch")
    if sidecar.get("counts") != {
            "train": _FORMAL_NYU_COUNTS["train"],
            "val": _FORMAL_NYU_COUNTS["val"]}:
        raise RuntimeError("formal N checkpoint sidecar split counts mismatch")
    if int(sidecar.get("seed", -1)) != int(args.seed):
        raise RuntimeError("formal N checkpoint sidecar seed mismatch")
    _verify_bound_dataset_audit(
        sidecar, {"train": _FORMAL_NYU_COUNTS["train"],
                  "val": _FORMAL_NYU_COUNTS["val"]})
    args._nyu_final_sidecar = sidecar


def _verify_formal_kitti_eval_checkpoint(adapter, args):
    if not _formal_kitti_stage_enabled(adapter, args):
        return
    role = str(getattr(args, "kitti_eval_role", "final") or "final")
    spec = _kitti_eval_spec(role)
    expected_split = spec["split"]
    if _manifest_name(getattr(args, "test_list", None)) != f"{expected_split}.jsonl":
        raise RuntimeError(
            f"formal KITTI {role} evaluation requires --test-list "
            f"{expected_split}.jsonl")
    if str(getattr(args, "test_phase", "")) != expected_split:
        raise RuntimeError(
            f"formal KITTI {role} evaluation requires --test-phase "
            f"{expected_split}")
    checkpoint = os.path.realpath(os.path.expanduser(_require_checkpoint(args)))
    expected_checkpoint = spec["checkpoint"]
    if os.path.basename(checkpoint) != expected_checkpoint:
        raise RuntimeError(
            f"KITTI {expected_split} accepts only {expected_checkpoint}")
    stage = _checkpoint_protocol_stage(checkpoint)
    if stage is not None and stage != spec["checkpoint_stage"]:
        raise RuntimeError(
            f"KITTI {role} checkpoint stage metadata mismatch: expected "
            f"{spec['checkpoint_stage']}, got {stage}")
    if role == "develop":
        _verify_formal_source_checkpoint(adapter, args)
        return
    if role == "diagnostic-best":
        _enforce_kitti_diagnostic_best_scope(adapter, args)
        _verify_formal_source_checkpoint(adapter, args)
        return
    if bool(getattr(adapter, "SKIP_SHARED_CHECKPOINT_PROVENANCE", False)):
        return
    if not getattr(args, "_kitti_final_sidecar", None):
        _verify_formal_source_checkpoint(adapter, args)


def _dataset_opts(args):
    cfg = getattr(args, "_dataset_config_info", {}) or {}
    opts = dict(cfg.get("dataset", {}) or {})
    opts["dataset_cfg"] = dict(cfg.get("dataset", {}) or {})
    opts["train_cfg"] = dict(cfg.get("train", {}) or {})
    opts["eval_cfg"] = dict(cfg.get("eval", {}) or {})
    opts["protocol_depth_unit_scale"] = getattr(
        args, "depth_unit_scale", None)
    return opts


def _apply_dataset_config(args, mode):
    """Apply data-only config defaults while preserving every CLI override."""
    cfg = load_dataset_config(getattr(args, "dataset_config", None))
    args._dataset_config_info = {k: v for k, v in cfg.items() if k != "raw"}
    args._dataset_type = cfg.get("dataset_type") or "nyu"
    args._visualization_depth_unit_scale = float(
        cfg.get("depth_unit_scale") or
        getattr(args, "depth_unit_scale", None) or 10.0)
    if mode == "train" and getattr(args, "val_phase", None) is None:
        args.val_phase = cfg.get("val_phase") or "val"
    if getattr(args, "test_phase", None) is None:
        args.test_phase = cfg.get("eval_phase") or "test"
    if getattr(args, "preprocess", None) is None:
        args.preprocess = cfg.get("preprocess") or "none"
    if getattr(args, "depth_unit_scale", None) is None:
        args.depth_unit_scale = float(cfg.get("depth_unit_scale") or 10.0)
    if getattr(args, "min_depth", None) is None:
        args.min_depth = float(cfg.get("min_depth") if cfg.get("min_depth") is not None else 1e-3)
    if getattr(args, "max_depth", None) is None:
        args.max_depth = float(cfg.get("max_depth") if cfg.get("max_depth") is not None else 1.0)
    if mode == "eval":
        if getattr(args, "clip_prediction", None) is None:
            args.clip_prediction = bool(cfg.get("clip_prediction") if cfg.get("clip_prediction") is not None else True)
        if getattr(args, "error_max", None) is None:
            args.error_max = float(
                cfg.get("error_max") if cfg.get("error_max") is not None else 0.05)
            args.error_max_source = "dataset_config_normalized_depth"
            args.error_max_m = (
                args.error_max * args._visualization_depth_unit_scale)
        elif str(getattr(args, "_error_max_cli_unit", "normalized_depth")) == "metres":
            args.error_max_m = float(args.error_max)
            args.error_max = args.error_max_m / float(args.depth_unit_scale)
            args.error_max_source = "explicit_cli_metres"
        else:
            args.error_max = float(args.error_max)
            args.error_max_m = args.error_max * float(args.depth_unit_scale)
            args.error_max_source = "explicit_cli_normalized_depth"
        if getattr(args, "vis_depth_mode", None) is None:
            args.vis_depth_mode = "fixed"
    if getattr(args, "dataset_name", None) == "comparison_dataset" and cfg.get("dataset_name"):
        args.dataset_name = str(cfg["dataset_name"])
    if cfg.get("dataroot"):
        if mode == "eval" and not getattr(args, "test_dataroot", None):
            args.test_dataroot = cfg["dataroot"]
        if mode == "train":
            if not getattr(args, "train_dataroot", None):
                args.train_dataroot = cfg["dataroot"]
            if (not getattr(args, "no_val", False)
                    and not getattr(args, "val_dataroot", None)):
                args.val_dataroot = cfg["dataroot"]
    if mode == "train" and args._dataset_type not in ("nyu", "kitti"):
        raise RuntimeError("OOD datasets are evaluation-only and cannot be used for training")
    return cfg


def _split_specs_for_train(args):
    train_root = _require_path(args, "train_dataroot", "--train-dataroot")
    specs = [{
        "name": "train", "dataroot": train_root,
        "phase": args.train_phase, "sample_list": args.train_list,
        "dataset_type": getattr(args, "_dataset_type", "nyu"),
        "dataset_opts": _dataset_opts(args),
        "dataset_config": getattr(args, "dataset_config", None),
    }]
    if not getattr(args, "no_val", False):
        val_root = _require_path(args, "val_dataroot", "--val-dataroot")
        specs.append({
            "name": "val", "dataroot": val_root,
            "phase": args.val_phase, "sample_list": args.val_list,
            "dataset_type": getattr(args, "_dataset_type", "nyu"),
            "dataset_opts": _dataset_opts(args),
            "dataset_config": getattr(args, "dataset_config", None),
        })
    if getattr(args, "test_dataroot", None):
        specs.append({
            "name": "test", "dataroot": args.test_dataroot,
            "phase": args.test_phase, "sample_list": args.test_list,
            "dataset_type": getattr(args, "_dataset_type", "nyu"),
            "dataset_opts": _dataset_opts(args),
            "dataset_config": getattr(args, "dataset_config", None),
        })
    return specs


def _split_specs_for_eval(args):
    specs = [{
        "name": "train", "dataroot": getattr(args, "train_dataroot", None),
        "phase": getattr(args, "train_phase", "train"), "sample_list": args.train_list,
        "dataset_type": getattr(args, "_dataset_type", "nyu"),
        "dataset_opts": _dataset_opts(args),
        "dataset_config": getattr(args, "dataset_config", None),
    }, {
        "name": "val", "dataroot": getattr(args, "val_dataroot", None),
        "phase": getattr(args, "val_phase", "val"), "sample_list": args.val_list,
        "dataset_type": getattr(args, "_dataset_type", "nyu"),
        "dataset_opts": _dataset_opts(args),
        "dataset_config": getattr(args, "dataset_config", None),
    }]
    test_root = _require_path(args, "test_dataroot", "--test-dataroot")
    specs.append({
        "name": "test", "dataroot": test_root,
        "phase": args.test_phase, "sample_list": args.test_list,
        "dataset_type": getattr(args, "_dataset_type", "nyu"),
        "dataset_opts": _dataset_opts(args),
        "dataset_config": getattr(args, "dataset_config", None),
    })
    return specs


def _write_command_and_config(args, filename="run_config.yaml"):
    result_io.write_command(os.path.join(args.output_dir, "command.txt"),
                            [sys.executable] + sys.argv, overwrite=True)
    result_io.dump_config_yaml(os.path.join(args.output_dir, filename),
                               vars(args).copy(), overwrite=True)


_FORMAL_TRAINING_METADATA_FIELDS = (
    "physical_batch_size", "gradient_accumulation",
    "gradient_accumulation_steps", "effective_batch_size", "amp",
    "epochs", "learning_rate", "formal_protocol_variant",
    "hardware_constraint", "original_psd_batch_size",
    "batch_size_deviation_reason",
    "spnet_compute_mode", "forward_compute_dtype", "loss_compute_dtype",
    "encoder_compute_dtype", "decoder_compute_dtype", "grad_scaler_enabled",
    "loss_formula_modified", "loss_weights_modified", "network_modified",
    "numerical_stability_variant", "official_training_ready",
)


def _formal_training_metadata(payload):
    """Pass through opt-in adapter metadata without changing other methods."""
    return {
        name: payload[name]
        for name in _FORMAL_TRAINING_METADATA_FIELDS
        if name in payload
    }


def _print_formal_training_audit_summary(audit, fairness_payload):
    metadata = _formal_training_metadata(fairness_payload)
    if not metadata:
        return
    counts = audit.get("counts", {})
    print("===== Formal Training Protocol =====")
    print(f"train samples={counts.get('train', 'N/A')}")
    print(f"val samples={counts.get('val', 'N/A')}")
    print(f"physical batch size={metadata.get('physical_batch_size', 'N/A')}")
    print(f"gradient accumulation={metadata.get('gradient_accumulation', 'N/A')}")
    print(f"effective batch size={metadata.get('effective_batch_size', 'N/A')}")
    print(f"epochs={metadata.get('epochs', 'N/A')}")
    print(f"lr={metadata.get('learning_rate', 'N/A')}")
    print(f"AMP={metadata.get('amp', 'N/A')}")
    print(f"formal protocol variant={metadata.get('formal_protocol_variant', 'N/A')}")
    print("====================================")


def _write_v2_protocol_artifacts(adapter, args, mode, audit=None):
    fairness_payload = fairness_io.write(
        args.output_dir, adapter, args, mode)
    if _nyu_small_spec(args) is not None:
        fairness_payload.update({
            "controlled_ranking": False,
            "formal_result": False,
            "result_eligible_for_main_table": False,
            "experiment_scope": args.experiment_scope,
            "reporting_scope": "ablation_only",
            "sparse_points": int(args.nyu_small_sparse_points),
            "sparse_input_percent": getattr(
                args, "nyu_small_sparse_percent", None),
            "small_split_counts": {"train": 640, "val": 155, "test": 654},
            "official_test_used_for_selection": False,
        })
        _write_json(os.path.join(args.output_dir, "fairness.json"),
                    fairness_payload)
    diagnostic_metadata = _kitti_diagnostic_metadata(args)
    if diagnostic_metadata:
        fairness_payload.update(diagnostic_metadata)
        _write_json(os.path.join(args.output_dir, "fairness.json"),
                    fairness_payload)
    if audit is not None:
        protocol_audit = dict(audit)
        protocol_audit["schema"] = "comparison-protocol-v2-audit"
        protocol_audit["method"] = fairness_payload["method"]
        protocol_audit["version"] = fairness_payload["version"]
        protocol_audit["training_domain"] = fairness_payload["training_domain"]
        protocol_audit["seed"] = fairness_payload["seed"]
        for name in (
                "measurable_domain_count", "expected_prediction_count",
                "not_measurable_domains", "not_measurable_reason",
                "rgb_used", "sparse_mask_used", "sensor_type",
                "sensor_model", "noise_model", "camera_model",
                "uncertainty_available", "diffusion_steps",
                "diffusion_model", "ensemble_size",
                "processing_resolution", "guidance_scale",
                "latent_initialization", "seed_policy",
                "sparse_conditioning", "confidence_threshold",
                "metric_alignment",
                "sparse_depth_used", "monocular_prior", "monocular_model",
                "mono_model_input_size", "scale_alignment",
                "shift_alignment", "alignment_source",
                "alignment_per_sample", "alignment_method",
                "alignment_point_count", "alignment_scale",
                "alignment_shift", "alignment_fallback",
                "checkpoint_provenance_verified",
                "checkpoint_provenance_status",
                "result_eligible_for_main_table", "formal_result",
                "experiment_scope", "reporting_scope", "sparse_points",
                "small_split_counts", "official_test_used_for_selection"):
            if name in fairness_payload:
                protocol_audit[name] = fairness_payload[name]
        protocol_audit.update(_formal_training_metadata(fairness_payload))
        protocol_audit.update(diagnostic_metadata)
        for name in (
                "dataset_path_policy", "dataset_path_registry",
                "configured_paths", "resolved_paths", "path_audit_status"):
            if hasattr(args, name):
                protocol_audit[name] = getattr(args, name)
        _write_json(os.path.join(args.output_dir, "protocol_audit.json"),
                    protocol_audit)
    return fairness_payload


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as stream:
        stream.write(json.dumps(
            _json_safe_value(payload), sort_keys=True,
            allow_nan=False) + "\n")


def _load_method_state(adapter, model, args, device):
    if not hasattr(adapter, "load_checkpoint"):
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'method')} has no provenance-aware "
            "load_checkpoint; generic/plain checkpoint loading is forbidden")
    adapter.load_checkpoint(
        model, args.checkpoint, device, args,
        strict=bool(getattr(args, "strict_checkpoint", False)))
    return model


def _run_info_header(adapter, args, **extra):
    dataset_name = getattr(args, "dataset_name", None) or "comparison_dataset"
    h = {
        "method": getattr(adapter, "METHOD_NAME", "method"),
        "comparison_bucket": getattr(adapter, "COMPARISON_BUCKET", "unspecified"),
        "supervision": getattr(adapter, "SUPERVISION", "unspecified"),
        "prior": getattr(adapter, "PRIOR", "none"),
        "protocol_status": getattr(adapter, "PROTOCOL_STATUS", "approximate_original_protocol"),
        "protocol_notes": getattr(adapter, "PROTOCOL_NOTES", "(none)"),
        "dataset_name": dataset_name,
        "train_dataroot": getattr(args, "train_dataroot", None),
        "val_dataroot": getattr(args, "val_dataroot", None),
        "test_dataroot": getattr(args, "test_dataroot", None),
        "train_list": getattr(args, "train_list", None),
        "val_list": getattr(args, "val_list", None),
        "test_list": getattr(args, "test_list", None),
        "depth_unit_scale": args.depth_unit_scale,
        "rgb_normalization": "RGB uint8 -> [0,1] -> normalize mean/std 0.5 => [-1,1]",
        "depth_unit": f"normalized [0,1], metres = value * {args.depth_unit_scale}",
        "depth_min_max": f"[{args.min_depth}, {args.max_depth}] normalized units for metrics; prediction clipping follows --clip-prediction",
        "crop_resize_padding": (
            f"GeoLayerNet loader preprocess={getattr(args, 'preprocess', 'none')}; "
            + (f"adapter registered fixed network/evaluation size "
               f"{getattr(args, 'network_input_height')}x"
               f"{getattr(args, 'network_input_width')}"
               if hasattr(args, "network_input_height") else
               "adapter-specific forward geometry recorded separately")),
        "prediction_npy_unit": getattr(
            args, "prediction_npy_unit", "normalized_depth"),
        "metric_valid_mask": "target_depth > 0 and within [1e-3, 1.0]",
        "missing_mask": "metric_valid_mask AND sparse_depth <= 0",
        "observed_mask": "metric_valid_mask AND sparse_depth > 0",
        "visualization_depth_range": (
            "fixed dataset-wide interval from the selected GeoLayerNet YAML "
            "normalization, or explicit metre-valued --vis-depth-min/max"),
        "visualization_error_range": (
            f"fixed absolute depth error 0..{getattr(args, 'error_max_m', 'n/a')} m"),
        "device": args.device,
        "seed": args.seed,
        "git": result_io.git_hash(),
        "command_line": " ".join([sys.executable] + sys.argv),
    }
    if hasattr(adapter, "fairness_metadata"):
        registered = fairness_io.validate(adapter.fairness_metadata(
            args, getattr(args, "_gln_mode", None)))
        h.update({
            "method_version": registered["version"],
            "training_domain": registered["training_domain"],
            "fairness_class": registered["fairness_class"],
            "controlled_ranking": registered["controlled_ranking"],
            "external_pretraining": registered["external_pretraining"],
            "foundation_model": registered["foundation_model"],
            "OOD_specific_tuning": registered["OOD_specific_tuning"],
            "primary_output": registered["primary_output"],
            "loss_modified": registered["loss_modified"],
            "network_modified": registered["network_modified"],
        })
        h.update(_formal_training_metadata(registered))
    if _nyu_small_spec(args) is not None:
        h.update({
            "controlled_ranking": False,
            "formal_result": False,
            "result_eligible_for_main_table": False,
            "experiment_scope": args.experiment_scope,
            "reporting_scope": "ablation_only",
            "sparse_points": int(args.nyu_small_sparse_points),
            "sparse_input_percent": getattr(
                args, "nyu_small_sparse_percent", None),
            "small_split_counts": {"train": 640, "val": 155, "test": 654},
        })
    h.update(_kitti_diagnostic_metadata(args))
    h.update(extra)
    return h


def _read_id_list(path):
    ids = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                sid = line.split()[0].split(",")[0].replace("\\", "/").lstrip("./")
                suffix = os.path.splitext(sid)[1].lower()
                if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    sid = sid[: -len(suffix)]
                for prefix in ("rgb/", "sparse_depth/", "target_depth/", "valid_mask/"):
                    if sid.startswith(prefix):
                        sid = sid[len(prefix):]
                        break
                ids.append(sid)
    return ids


def _choose_visual_ids(dataset, args, output_dir=None):
    if not getattr(args, "save_visuals", False):
        return set(), []
    samples = dataset_samples(dataset)
    all_ids = [s["sample_id"] for s in samples]
    available = set(all_ids)
    limit = max(int(getattr(args, "save_visual_limit", 0)), 0)
    if getattr(args, "visual_sample_list", None):
        requested = _read_id_list(args.visual_sample_list)
        if not requested:
            raise RuntimeError(
                f"visual sample list is empty: {args.visual_sample_list}")
        if len(requested) != len(set(requested)):
            raise RuntimeError(
                f"visual sample list contains duplicate sample_ids: {args.visual_sample_list}")
        missing = [sid for sid in requested if sid not in available]
        if missing:
            if output_dir:
                _write_json(os.path.join(
                    output_dir, "visualization_missing_sample_ids.json"), {
                        "requested_sample_ids": requested,
                        "missing_sample_ids": missing,
                        "replacement_policy": "forbidden",
                    })
            raise RuntimeError(
                f"{len(missing)} requested visualization sample_ids are absent from test split: "
                + ", ".join(missing[:8]))
        # A fixed paper list is authoritative.  The debug-only numeric limit
        # must never silently remove requested samples.
        chosen = requested
    else:
        seed = int(args.visual_seed if args.visual_seed is not None else args.seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        if limit and limit < len(all_ids):
            order = torch.randperm(len(all_ids), generator=generator).tolist()[:limit]
            chosen = [all_ids[i] for i in order]
        else:
            chosen = all_ids[:limit] if limit else all_ids
    return set(chosen), chosen


def _resize_prediction_to_target(pred_1, target_1):
    if pred_1.shape[-2:] == target_1.shape[-2:]:
        return pred_1
    return F.interpolate(pred_1, size=target_1.shape[-2:], mode="bilinear", align_corners=False)


def _enforce_ood_228x304_geometry(batch, prediction, dataset_type):
    """Fail closed before metrics/output if any OOD tensor leaves fixed geometry."""
    if str(dataset_type or "").lower() != "ood_npz":
        return
    rgb = batch.get("rgb")
    if not torch.is_tensor(rgb) or rgb.ndim != 4:
        raise RuntimeError("OOD protocol requires rgb Bx3x228x304")
    batch_size = int(rgb.shape[0])
    expected = {
        "rgb": (batch_size, 3, 228, 304),
        "sparse_depth": (batch_size, 1, 228, 304),
        "target_depth": (batch_size, 1, 228, 304),
        "valid_mask": (batch_size, 1, 228, 304),
        "target_mask": (batch_size, 1, 228, 304),
        "prediction": (batch_size, 1, 228, 304),
    }
    values = {
        "rgb": rgb,
        "sparse_depth": batch.get("sparse_depth"),
        "target_depth": batch.get("target_depth"),
        "valid_mask": batch.get("valid_mask"),
        "target_mask": batch.get("target_mask"),
        "prediction": prediction,
    }
    bad = {
        name: (None if not torch.is_tensor(values[name])
               else tuple(values[name].shape))
        for name, shape in expected.items()
        if not torch.is_tensor(values[name]) or tuple(values[name].shape) != shape
    }
    supplied_k = batch.get("K", batch.get("intrinsics"))
    if supplied_k is not None and (
            not torch.is_tensor(supplied_k)
            or tuple(supplied_k.shape) != (batch_size, 3, 3)):
        bad["K"] = (
            None if not torch.is_tensor(supplied_k) else tuple(supplied_k.shape))
    if bad:
        raise RuntimeError(
            "OOD protocol forbids prediction/metric resizing outside 228x304; "
            f"mismatched tensors={bad}")


def _ood_sample_geometry_provenance(batch, index):
    out = {}
    for name in ("original_input_size", "intrinsics_source_size"):
        value = batch.get(name)
        if value is None:
            continue
        if torch.is_tensor(value):
            selected = value[index] if value.ndim > 1 else value
            out[name] = [int(item) for item in selected.detach().cpu().tolist()]
        elif isinstance(value, (list, tuple)):
            selected = value[index] if len(value) and isinstance(
                value[index], (list, tuple)) else value
            out[name] = [int(item) for item in selected]
    return out


def _save_prediction_files(output_dir, sample_id, pred_1, depth_unit_scale,
                           storage_unit="normalized_depth", metadata=None):
    import numpy as np
    from PIL import Image

    clean = sample_id.replace("\\", "/").strip("/")
    clean = clean.replace("/", "__")
    npy_dir = os.path.join(output_dir, "predictions_npy")
    png_dir = os.path.join(output_dir, "predictions_png16")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    normalized = np.ascontiguousarray(
        pred_1.detach().float().cpu().numpy()[0, 0], dtype=np.float32)
    if storage_unit == "metres":
        arr = np.ascontiguousarray(
            normalized * float(depth_unit_scale), dtype=np.float32)
    elif storage_unit == "normalized_depth":
        arr = normalized
    else:
        raise RuntimeError(
            f"unsupported prediction NPY storage unit: {storage_unit}")
    npy_path = os.path.join(npy_dir, f"{clean}.npy")
    png_path = os.path.join(png_dir, f"{clean}.png")
    np.save(npy_path, arr, allow_pickle=False)
    png = np.clip(normalized, 0.0, 1.0)
    Image.fromarray((png * 65535.0).round().astype(np.uint16)).save(
        png_path)
    meta = {
        "depth_unit": storage_unit,
        "metres": ("value" if storage_unit == "metres" else
                   f"value * {float(depth_unit_scale)}"),
        "png16_scale": 65535,
        "png16_depth_unit": "normalized_depth",
    }
    if metadata:
        meta.update(metadata)
    _write_json(os.path.join(output_dir, "prediction_protocol.json"), meta)
    return {
        "sample_id": str(sample_id),
        "filename_stem": clean,
        "prediction_npy": str(os.path.realpath(npy_path)),
        "prediction_png16": str(os.path.realpath(png_path)),
        "prediction_sha256": vis.sha256_file(npy_path),
        "dtype": "float32",
        "shape": list(arr.shape),
        "depth_unit": storage_unit,
        "depth_unit_scale": float(depth_unit_scale),
        "metres": ("value" if storage_unit == "metres" else
                   f"value * {float(depth_unit_scale)}"),
        "spatial_resolution": list(arr.shape),
    }


def _eval_protocol_payload(args, adapter, audit):
    dataset_type = getattr(args, "_dataset_type", "nyu")
    if dataset_type == "ood_npz":
        depth_loading = "GeoLayerNet OOD npz metre depth / depth_unit_scale"
    elif dataset_type == "kitti":
        depth_loading = "GeoLayerNet KITTI uint16 PNG / 256 metres / 80 normalization"
    else:
        depth_loading = "GeoLayerNet NYU PNG/TIFF depth to [0,1] with configured/inferred scale"
    payload = {
        "method": getattr(adapter, "METHOD_NAME", "method"),
        "version": (adapter.method_version(args)
                    if hasattr(adapter, "method_version") else None),
        "training_domain": getattr(
            args, "training_domain",
            getattr(args, "psd_training_domain", None)),
        "seed": int(args.seed),
        "dataset_name": args.dataset_name,
        "checkpoint": getattr(args, "checkpoint", None),
        "dataset_config": getattr(args, "dataset_config", None),
        "dataset_type": dataset_type,
        "test_dataroot": args.test_dataroot,
        "test_list": args.test_list,
        "test_phase": args.test_phase,
        "rgb_normalization": (
            "GeoLayerNet OOD NPZ RGB bilinear-resized once to 228x304, then normalized to [-1,1]"
            if dataset_type == "ood_npz" else
            "GeoLayerNet: PIL RGB, bicubic resize if requested, tensor normalized to [-1,1]"),
        "depth_loading": depth_loading,
        "valid_mask": "sparse_depth > 0",
        "metric_mask": f"target_depth > 0 and {args.min_depth} <= target_depth <= {args.max_depth}",
        "missing_mask": "metric_mask and sparse_depth <= 0",
        "observed_mask": "metric_mask and sparse_depth > 0",
        "metric_aggregation": "GeoLayerNet MetricBag pixel-pooled raw sums, then finalize",
        "prediction_resize": (
            "forbidden: shared OOD target and adapter prediction must both be 228x304"
            if dataset_type == "ood_npz" else (
            f"adapter fixed batch and prediction to registered "
            f"{getattr(args, 'network_input_height')}x"
            f"{getattr(args, 'network_input_width')} before metrics; no "
            "prediction-to-original resize"
            if hasattr(args, "network_input_height") else
            "adapter output bilinear-resized to target spatial size for metrics/output files")),
        "prediction_clipping": bool(args.clip_prediction),
        "visualization": {
            "root": "visualization/",
            "depth_mode": args.vis_depth_mode,
            "depth_min_m": args.vis_depth_min,
            "depth_max_m": args.vis_depth_max,
            "error_max_normalized": args.error_max,
            "error_max_m": args.error_max_m,
            "error_max_source": args.error_max_source,
            "colormap": vis.DEPTH_COLORMAP,
            "reference": vis.REFERENCE_FILE,
        },
        "audit_status": audit.get("status") if audit else None,
        "unsupported_model_specific_metrics": "proxy/raw/safe/layer metrics are N/A unless adapter returns those tensors; final metrics are always computed",
    }
    for name in (
            "checkpoint_provenance_verified", "checkpoint_provenance_status",
            "result_eligible_for_main_table", "formal_result",
            "experiment_scope", "reporting_scope", "nyu_small_sparse_points",
            "nyu_small_sparse_percent"):
        if hasattr(args, name):
            payload[name] = getattr(args, name)
    if dataset_type == "ood_npz":
        payload.update({
            "input_resolution": [228, 304],
            "metric_resolution": [228, 304],
            "prediction_resolution": [228, 304],
            "visualization_resolution": [228, 304],
            "intrinsics_transform": (
                "shared OOD loader scales NPZ-RGB K once; adapter second "
                "transform forbidden"),
        })
    for name in (
            "dataset_path_policy", "dataset_path_registry",
            "configured_paths", "resolved_paths", "path_audit_status"):
        if hasattr(args, name):
            payload[name] = getattr(args, name)
    provenance_fields = (
        "checkpoint_path", "checkpoint_loading", "checkpoint_registered",
    )
    for name in provenance_fields:
        if hasattr(args, name):
            payload[name] = getattr(args, name)
    for name in (
            "network_input_height", "network_input_width",
            "rgb_interpolation", "depth_interpolation",
            "mask_interpolation", "per_image_normalization",
            "prediction_spatial_resolution", "prediction_npy_unit",
            "kitti_eval_role", "intrinsics_policy",
            "propagation_type", "propagation_iterations",
            "propagation_stages", "propagation_kernel", "affinity_type",
            "confidence_propagation", "preserve_sparse_input",
            "propagation_padding", "padding",
            "refinement_enabled", "input_multiple", "error_max_m",
            "sparse_depth_used", "monocular_model", "mono_model_input_size",
            "alignment_method", "alignment_point_count", "alignment_scale",
            "alignment_shift", "alignment_fallback",
            "error_max_source"):
        if hasattr(args, name):
            payload[name] = getattr(args, name)
    if hasattr(adapter, "fairness_metadata"):
        registered = fairness_io.validate(adapter.fairness_metadata(args, "eval"))
        for name in fairness_io.NEUTRAL_OPTIONAL_FIELDS:
            payload[name] = registered[name]
    payload.update(_kitti_diagnostic_metadata(args))
    return payload


# --------------------------------------------------------------------------- #
# EVAL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_eval(adapter, args):
    _apply_dataset_config(args, "eval")
    if hasattr(adapter, "configure_args"):
        adapter.configure_args(args, "eval")
    _apply_nyu_small_scope(args)
    _apply_kitti_eval_role_metadata(args)
    _enforce_kitti_diagnostic_best_scope(adapter, args)
    validate_geolayernet_protocol_paths(
        args, "eval", getattr(args, "training_domain", None))
    _enforce_formal_nyu_source(args, "eval")
    _enforce_formal_kitti_source(args, "eval", adapter)
    set_global_seed(args.seed)
    device = torch.device(args.device)
    if hasattr(adapter, "validate_runtime_environment"):
        adapter.validate_runtime_environment(args, device)
    _prep_outdir(args.output_dir, args.overwrite)
    scale = float(args.depth_unit_scale)
    clip_pred = bool(args.clip_prediction)
    if args.save_visuals and not args.save_predictions:
        raise RuntimeError(
            "--save-visuals requires float32 --save-predictions; color PNGs "
            "must remain traceable to the raw metric prediction")
    if args.save_visuals and args.vis_depth_mode != "fixed":
        raise RuntimeError(
            "formal comparison visualization forbids per-image normalization; "
            "use --vis-depth-mode fixed")
    visual_protocol = vis.resolve_visualization_protocol(
        dataset=args.dataset_name,
        dataset_config=getattr(args, "dataset_config", None),
        depth_unit_scale=scale,
        visualization_depth_unit_scale=(
            args._visualization_depth_unit_scale),
        max_depth_normalized=args.max_depth,
        error_max_normalized=args.error_max,
        depth_vmin_m=args.vis_depth_min,
        depth_vmax_m=args.vis_depth_max,
        error_vmax_m=(
            args.error_max_m
            if str(getattr(args, "error_max_source", "")).startswith(
                "explicit_cli_") else None),
    )
    args.error_max_m = float(visual_protocol["error_vmax_m"])
    if not str(getattr(args, "error_max_source", "")).startswith(
            "explicit_cli_"):
        args.error_max_source = str(visual_protocol["fixed_range_source"])
    _write_command_and_config(args)
    prediction_npy_unit = str(getattr(
        adapter, "PREDICTION_NPY_UNIT", "normalized_depth"))
    visual_protocol["raw_prediction_unit"] = prediction_npy_unit
    max_samples = None if args.num_test >= 100000 else args.num_test
    audit = audit_splits(
        _split_specs_for_eval(args), output_dir=args.output_dir,
        preprocess=args.preprocess, depth_scale=args.depth_scale,
        max_samples=max_samples, config_hint=f"{adapter.METHOD_NAME} eval",
        fail_on_leak=True,
        allow_val_test_shared_rgb_gt_with_distinct_sparse=bool(
            getattr(args, "allow_val_test_shared_rgb_gt_with_distinct_sparse", False)))
    _enforce_formal_nyu_audit_counts(args, audit, "eval")
    _enforce_formal_kitti_audit_counts(args, audit, "eval")
    fairness_payload = _write_v2_protocol_artifacts(
        adapter, args, "eval", audit=audit)
    visual_protocol.update({
        "method": fairness_payload["method"],
        "version": fairness_payload["version"],
        "training_domain": fairness_payload["training_domain"],
        "seed": fairness_payload["seed"],
    })
    visual_protocol.update(_kitti_diagnostic_metadata(args))
    if args.audit_only:
        _write_json(os.path.join(args.output_dir, "eval_protocol.json"),
                    _eval_protocol_payload(args, adapter, audit))
        return audit
    requires_checkpoint = bool(getattr(adapter, "REQUIRES_CHECKPOINT", True))
    if requires_checkpoint:
        _require_checkpoint(args)
    _verify_formal_source_checkpoint(adapter, args)
    if hasattr(args, "checkpoint_loading"):
        # The first snapshot precedes checkpoint verification.  Refresh it so
        # the simple active loading mode is recorded without legacy claims.
        _write_command_and_config(args)
    _verify_formal_kitti_eval_checkpoint(adapter, args)

    loader = create_loader(args.test_dataroot, args.test_phase, args.batch_size,
                           num_workers=args.num_workers, shuffle=False,
                           max_samples=max_samples, sample_list=args.test_list,
                           preprocess=args.preprocess, depth_scale=args.depth_scale,
                           config_hint=args.dataset_config or f"{adapter.METHOD_NAME} eval", seed=args.seed,
                           dataset_type=getattr(args, "_dataset_type", "nyu"),
                           dataset_opts=_dataset_opts(args))
    _enforce_formal_nyu_loader_counts(args, test_loader=loader)
    _enforce_formal_kitti_loader_counts(args, test_loader=loader)
    visual_id_set, visual_order = _choose_visual_ids(
        loader.dataset, args, args.output_dir)

    model = adapter.build_model(args, device)
    if getattr(args, "checkpoint", None):
        _load_method_state(adapter, model, args, device)
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    degraded = getattr(model, "geolayernet_degraded_note", None)

    global_bag = MetricBag(scale=scale, extra=True, clip=clip_pred,
                           lo=args.min_depth, hi=args.max_depth)
    per_image_rows = []
    stats = None
    saved_ids = []
    prediction_records = []
    prediction_record_by_id = {}
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] test_dataroot={args.test_dataroot} test_list={args.test_list}")
    for batch in loader:
        batch = _to_device(batch, device)
        if hasattr(adapter, "prepare_batch_for_model"):
            batch = adapter.prepare_batch_for_model(batch, device)
        target = batch["target_depth"]
        metric_target = target
        if "target_mask" in batch:
            metric_target = torch.where(
                batch["target_mask"] > 0.5, target, torch.zeros_like(target))
        valid = batch["valid_mask"]
        sparse = batch["sparse_depth"]
        pred = adapter.forward_completed(model, batch, device)
        _enforce_ood_228x304_geometry(
            batch, pred, getattr(args, "_dataset_type", "nyu"))
        outputs = {"completed_depth": pred}
        for j in range(target.size(0)):
            bag = MetricBag(scale=scale, extra=True, clip=clip_pred,
                            lo=args.min_depth, hi=args.max_depth)
            bag.update({"completed_depth": pred[j:j + 1]}, metric_target[j:j + 1], valid[j:j + 1], sparse[j:j + 1])
            row = bag.finalize()
            row["sample_id"] = batch["name"][j]
            per_image_rows.append(row)
            global_bag.merge(bag)
            pred_j = _resize_prediction_to_target(pred[j:j + 1], target[j:j + 1])
            if clip_pred:
                pred_j = pred_j.clamp(1e-3, 1.0)
            if args.save_predictions:
                prediction_record = _save_prediction_files(
                    args.output_dir, batch["name"][j], pred_j, scale,
                    storage_unit=prediction_npy_unit,
                    metadata=_kitti_diagnostic_metadata(args))
                prediction_record.update({
                    "dataset": args.dataset_name,
                    "dataset_config": getattr(args, "dataset_config", None),
                    "split": args.test_phase,
                    "network_input_height": getattr(
                        args, "network_input_height", None),
                    "network_input_width": getattr(
                        args, "network_input_width", None),
                    "prediction_spatial_resolution": getattr(
                        args, "prediction_spatial_resolution", None),
                    "prediction_path": prediction_record["prediction_npy"],
                    "unit": "m" if prediction_npy_unit == "metres" else prediction_npy_unit,
                    "training_domain": fairness_payload["training_domain"],
                    "method": fairness_payload["method"],
                    "version": fairness_payload["version"],
                    "seed": fairness_payload["seed"],
                    "primary_output": fairness_payload["primary_output"],
                    "checkpoint_provenance_verified": fairness_payload.get(
                        "checkpoint_provenance_verified"),
                    "checkpoint_provenance_status": fairness_payload.get(
                        "checkpoint_provenance_status"),
                    "result_eligible_for_main_table": fairness_payload.get(
                        "result_eligible_for_main_table"),
                    "network_input_size": [
                        getattr(args, "network_input_height", None),
                        getattr(args, "network_input_width", None),
                    ],
                    "evaluation_size": [
                        getattr(args, "evaluation_height", target.shape[-2]),
                        getattr(args, "evaluation_width", target.shape[-1]),
                    ],
                    "intrinsics_policy": fairness_payload.get(
                        "intrinsics_policy",
                        getattr(args, "intrinsics_policy", None)),
                    "propagation_iterations": fairness_payload.get(
                        "propagation_iterations",
                        getattr(args, "propagation_iterations", None)),
                    "propagation_type": fairness_payload.get(
                        "propagation_type",
                        getattr(args, "propagation_type", None)),
                    "propagation_stages": fairness_payload.get(
                        "propagation_stages",
                        getattr(args, "propagation_stages", None)),
                    "propagation_kernel": fairness_payload.get(
                        "propagation_kernel",
                        getattr(args, "propagation_kernel", None)),
                    "affinity_type": fairness_payload.get(
                        "affinity_type", getattr(args, "affinity_type", None)),
                    "confidence_propagation": fairness_payload.get(
                        "confidence_propagation",
                        getattr(args, "confidence_propagation", None)),
                    "preserve_sparse_input": fairness_payload.get(
                        "preserve_sparse_input",
                        getattr(args, "preserve_sparse_input", None)),
                    "padding": fairness_payload.get(
                        "padding", getattr(args, "padding", None)),
                    "refinement_enabled": fairness_payload.get(
                        "refinement_enabled",
                        getattr(args, "refinement_enabled", None)),
                    "mono_model_input_size": fairness_payload.get(
                        "mono_model_input_size",
                        getattr(args, "mono_model_input_size", None)),
                    "monocular_model": fairness_payload.get(
                        "monocular_model",
                        getattr(args, "monocular_model", None)),
                    "alignment_method": fairness_payload.get(
                        "alignment_method",
                        getattr(args, "alignment_method", None)),
                    "alignment_point_count": fairness_payload.get(
                        "alignment_point_count",
                        getattr(args, "alignment_point_count", None)),
                    "alignment_scale": fairness_payload.get(
                        "alignment_scale",
                        getattr(args, "alignment_scale", None)),
                    "alignment_shift": fairness_payload.get(
                        "alignment_shift",
                        getattr(args, "alignment_shift", None)),
                    "alignment_fallback": fairness_payload.get(
                        "alignment_fallback",
                        getattr(args, "alignment_fallback", False)),
                    "sparse_depth_used": fairness_payload.get(
                        "sparse_depth_used",
                        getattr(args, "sparse_depth_used", None)),
                    "sensor_type": fairness_payload.get(
                        "sensor_type", getattr(args, "sensor_type", None)),
                    "sensor_model": fairness_payload.get(
                        "sensor_model", getattr(args, "sensor_model", None)),
                    "noise_model": fairness_payload.get(
                        "noise_model", getattr(args, "noise_model", None)),
                    "camera_model": fairness_payload.get(
                        "camera_model", getattr(args, "camera_model", None)),
                    "uncertainty_available": fairness_payload.get(
                        "uncertainty_available",
                        getattr(args, "uncertainty_available", False)),
                    "diffusion_steps": fairness_payload.get(
                        "diffusion_steps", getattr(args, "diffusion_steps", None)),
                    "diffusion_model": fairness_payload.get(
                        "diffusion_model", getattr(args, "diffusion_model", None)),
                    "ensemble_size": fairness_payload.get(
                        "ensemble_size", getattr(args, "ensemble_size", None)),
                    "processing_resolution": fairness_payload.get(
                        "processing_resolution",
                        getattr(args, "processing_resolution", None)),
                    "guidance_scale": fairness_payload.get(
                        "guidance_scale", getattr(args, "guidance_scale", None)),
                    "latent_initialization": fairness_payload.get(
                        "latent_initialization",
                        getattr(args, "latent_initialization", None)),
                    "seed_policy": fairness_payload.get(
                        "seed_policy", getattr(args, "seed_policy", None)),
                    "sparse_conditioning": fairness_payload.get(
                        "sparse_conditioning",
                        getattr(args, "sparse_conditioning", None)),
                    "confidence_threshold": fairness_payload.get(
                        "confidence_threshold",
                        getattr(args, "confidence_threshold", None)),
                    "metric_alignment": fairness_payload.get(
                        "metric_alignment", getattr(args, "metric_alignment", None)),
                    "dataset_configured_paths": getattr(
                        args, "configured_paths", None),
                    "dataset_resolved_paths": getattr(
                        args, "resolved_paths", None),
                    "dataset_path_registry": getattr(
                        args, "dataset_path_registry", None),
                    "path_audit_status": getattr(
                        args, "path_audit_status", None),
                })
                prediction_record.update(
                    _ood_sample_geometry_provenance(batch, j))
                if hasattr(adapter, "prediction_metadata"):
                    extra = adapter.prediction_metadata(
                        model, batch, j, args)
                    if not isinstance(extra, dict):
                        raise RuntimeError(
                            "adapter.prediction_metadata must return a mapping")
                    prediction_record.update(extra)
                for metadata_name in (
                        "measurable_domain_count", "expected_prediction_count",
                        "not_measurable_domains", "not_measurable_reason"):
                    if metadata_name in fairness_payload:
                        prediction_record[metadata_name] = fairness_payload[metadata_name]
                prediction_record.update(_kitti_diagnostic_metadata(args))
                prediction_records.append(prediction_record)
                prediction_record_by_id[batch["name"][j]] = prediction_record
        if stats is None and _model_stats_enabled(adapter):
            sm = _StatsModule(adapter, model, device).to(device).eval()
            stats = model_stats(
                sm, (batch["rgb"][:1], sparse[:1], valid[:1]),
                measure_speed=bool(getattr(adapter, "PROFILE_SPEED", True)))
        elif stats is None:
            # Mark profiling as deliberately handled so a stateful adapter is
            # not reconsidered on every later batch.
            stats = {}
        if args.save_visuals and visual_id_set:
            cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_for_visuals = _resize_prediction_to_target(pred, target)
            clip_out = pred_for_visuals.clamp(1e-3, 1.0) if clip_pred else pred_for_visuals
            cpu_out = {"completed_depth": clip_out.cpu()}
            for j in range(target.size(0)):
                sid = batch["name"][j]
                if sid not in visual_id_set or sid in saved_ids:
                    continue
                prediction_record = prediction_record_by_id.get(sid)
                if prediction_record is None:
                    raise RuntimeError(
                        f"raw prediction index is unavailable for visual sample {sid}")
                vis.save_sample_visuals(
                    os.path.join(args.output_dir, "visualization"), sid, cpu_batch, cpu_out, j,
                    protocol=visual_protocol,
                    prediction_npy_path=prediction_record["prediction_npy"],
                    prediction_npy_unit=prediction_record["depth_unit"])
                saved_ids.append(sid)

    metrics = global_bag.finalize()
    metrics.update({k: v for k, v in (stats or {}).items()})
    metrics["peak VRAM(GB)"] = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)
        if device.type == "cuda" else None)

    result_io.write_per_image_csv(os.path.join(args.output_dir, "per_image_metrics.csv"), per_image_rows)
    result_io.write_metrics(args.output_dir, metrics)
    args_dict = vars(args).copy()
    result_io.write_args_json(os.path.join(args.output_dir, "args.json"), args_dict, overwrite=True)
    result_io.dump_config_yaml(os.path.join(args.output_dir, "eval_config.yaml"), args_dict, overwrite=True)
    _write_json(os.path.join(args.output_dir, "eval_protocol.json"),
                _eval_protocol_payload(args, adapter, audit))
    if prediction_records:
        vis.write_prediction_index(
            os.path.join(args.output_dir, "prediction_index.jsonl"),
            prediction_records)
    if args.save_visuals:
        missing_saved = [sid for sid in visual_order if sid not in saved_ids]
        if missing_saved:
            _write_json(os.path.join(
                args.output_dir, "visualization_missing_sample_ids.json"), {
                    "requested_sample_ids": visual_order,
                    "missing_sample_ids": missing_saved,
                    "replacement_policy": "forbidden",
                })
            vis.write_visualization_protocol(
                args.output_dir, visual_protocol, visual_order,
                missing_sample_ids=missing_saved,
                selection_source=(args.visual_sample_list or
                                  "visual-seed (debug only)"))
            raise RuntimeError(
                "visualization sample_ids were selected but not reached during eval: "
                + ", ".join(missing_saved[:8]))
        with open(os.path.join(args.output_dir, "visualized_sample_ids.txt"), "w") as fh:
            for sid in visual_order:
                fh.write(sid + "\n")
        vis.write_visualization_protocol(
            args.output_dir, visual_protocol, visual_order,
            missing_sample_ids=[],
            selection_source=(args.visual_sample_list or
                              "visual-seed (debug only)"))
    else:
        vis.write_visualization_protocol(
            args.output_dir, visual_protocol, [], missing_sample_ids=[],
            selection_source="not generated during metric evaluation")
    result_io.write_run_info(
        os.path.join(args.output_dir, "run_info.txt"),
        title=f"{adapter.METHOD_NAME} eval: {os.path.basename(args.output_dir)}",
        header=_run_info_header(adapter, args, eval_name=os.path.basename(args.output_dir),
                                output_dir=args.output_dir, checkpoint=args.checkpoint,
                                phase=args.test_phase, num_test=args.num_test,
                                images_evaluated=metrics.get("images"),
                                clip_prediction=clip_pred,
                                dataset_protocol_audit=audit.get("status"),
                                degraded_note=degraded or "none"),
        metrics=metrics, stats=stats, overwrite=args.overwrite,
    )

    print(f"\n=== {adapter.METHOD_NAME} eval -- {args.output_dir} ===")
    for k in ("images", "RMSE_m", "MAE_m", "REL", "delta1(%)", "FinalMissing_RMSE_m",
              "Observed_RMSE_m", "Params(M)", "GFLOPs", "FPS"):
        v = metrics.get(k)
        print(f"  {k:22s}: {v:.6f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    return metrics


# --------------------------------------------------------------------------- #
# TRAIN
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _quick_val_rmse(adapter, model, loader, device, scale, clip_pred,
                    lo=1e-3, hi=1.0, numerical_failfast=False,
                    validation_context=None):
    bag = MetricBag(scale=scale, extra=False, clip=clip_pred, lo=lo, hi=hi)
    for batch in loader:
        batch = _to_device(batch, device)
        if hasattr(adapter, "prepare_batch_for_model"):
            batch = adapter.prepare_batch_for_model(batch, device)
        if validation_context is not None:
            validation_context.clear()
            validation_context.update({
                "current_stage": "validation",
                "optimizer_step_completed": False,
                "batch": batch,
                "model_inputs": batch,
            })
        if numerical_failfast:
            _check_finite_collection(
                batch, "validation", "validation_batch")
        pred = adapter.forward_completed(model, batch, device)
        if validation_context is not None:
            validation_context["predictions"] = {
                "completed_depth": pred,
            }
        if numerical_failfast:
            # MetricBag may clamp completed_depth to the evaluation range.  The
            # model's unmodified output must therefore be checked first.
            _check_finite_tensor(
                "validation_prediction.completed_depth", pred, "validation")
        target, valid, sparse = batch["target_depth"], batch["valid_mask"], batch["sparse_depth"]
        metric_target = target
        if "target_mask" in batch:
            metric_target = torch.where(
                batch["target_mask"] > 0.5, target, torch.zeros_like(target))
        # MetricBag.update treats a 4-D tensor as batch-of-1, so accumulate per image.
        for j in range(target.size(0)):
            bag.update({"completed_depth": pred[j:j + 1]}, metric_target[j:j + 1], valid[j:j + 1], sparse[j:j + 1])
    return bag.finalize()


def _train_one_step(adapter, model, batch, device, epoch, scale):
    """Run only the method's explicitly registered training objective."""
    del scale
    if not hasattr(adapter, "train_step"):
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'method')} has no registered "
            "train_step; generic runner L1 substitution is forbidden")
    return adapter.train_step(model, batch, device, epoch)


def _train_one_step_with_numerical_guard(
        adapter, model, batch, device, epoch, scale, context):
    """Run the same adapter step while observing its opt-in diagnostic hooks."""
    seen = set()

    def guard(stage, payload):
        seen.add(str(stage))
        if stage == "model_inputs":
            context["current_stage"] = "model_input"
            context["model_inputs"] = payload
            _check_finite_collection(payload, "model_input", "model_inputs")
            return
        if stage == "intermediate":
            context["current_stage"] = "forward"
            context.setdefault("intermediates", {}).update(payload)
            _check_finite_collection(
                payload, "forward", "intermediates")
            return
        if stage == "forward_outputs":
            context["current_stage"] = "forward"
            context["predictions"] = payload.get("predictions") or {}
            context.setdefault("intermediates", {}).update(
                payload.get("intermediates") or {})
            _check_finite_collection(
                context["intermediates"], "forward", "intermediates")
            _check_finite_collection(
                context["predictions"], "forward", "predictions")
            return
        if stage == "loss":
            context["current_stage"] = "loss_component"
            context["loss_components"] = payload.get("loss_components") or {}
            total = context["loss_components"].get("loss")
            _check_finite_loss(total, context["loss_components"])
            return
        if stage == "forward":
            context["current_stage"] = "forward"
            context["loss_components"] = payload.get("loss_components") or {}
            context["predictions"] = payload.get("predictions") or {}
            context.setdefault("intermediates", {}).update(
                payload.get("intermediates") or {})
            context["diagnostic_only"] = payload.get("diagnostic_only") or {}
            # Outputs are created before the model combines its native losses;
            # report an already-bad prediction/intermediate as the first stage
            # instead of attributing its downstream effect to the total loss.
            _check_finite_collection(
                context["intermediates"], "forward", "intermediates")
            _check_finite_collection(
                context["predictions"], "forward", "predictions")
            total = context["loss_components"].get("loss")
            _check_finite_loss(total, context["loss_components"])
            return
        raise RuntimeError(f"unknown adapter numerical guard stage: {stage}")

    model._gln_numerical_guard = guard
    try:
        loss = _train_one_step(adapter, model, batch, device, epoch, scale)
    finally:
        if getattr(model, "_gln_numerical_guard", None) is guard:
            delattr(model, "_gln_numerical_guard")
    complete_combined = {"model_inputs", "forward"}.issubset(seen)
    complete_split = {
        "model_inputs", "forward_outputs", "loss"}.issubset(seen)
    if not (complete_combined or complete_split):
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'method')} enabled numerical "
            f"fail-fast but exposed incomplete adapter diagnostics: {sorted(seen)}")
    _check_finite_loss(loss, context.get("loss_components"))
    return loss


def _save_training_checkpoint(adapter, model, optimizer, epoch, path):
    """Preserve a method's native checkpoint container when it declares one."""
    if _requires_numerical_valid_checkpoint(adapter):
        nonfinite_events = getattr(model, "_gln_nonfinite_events", None)
        if (getattr(model, "_gln_numerical_valid", None) is not True
                or type(nonfinite_events) is not int
                or nonfinite_events != 0):
            raise RuntimeError(
                "refusing to save a numerically invalid formal checkpoint")
        _check_finite_parameters(model, stage="checkpoint")
        _check_finite_optimizer_state(model, optimizer)
    if hasattr(adapter, "save_checkpoint"):
        adapter.save_checkpoint(model, optimizer, epoch, path)
        return
    if hasattr(adapter, "checkpoint_payload"):
        payload = adapter.checkpoint_payload(model, optimizer, epoch)
    else:
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'method')} has no provenance-aware "
            "checkpoint serializer; plain state_dict saving is forbidden")
    if _requires_numerical_valid_checkpoint(adapter):
        _validate_numerical_validity_record(
            payload.get("protocol") or {}, "checkpoint protocol")
    torch.save(payload, path)


def run_train(adapter, args):
    _apply_dataset_config(args, "train")
    if hasattr(adapter, "configure_args"):
        adapter.configure_args(args, "train")
    _apply_nyu_small_scope(args)
    numerical_failfast = bool(getattr(
        adapter, "ENABLE_NONFINITE_FAILFAST", False))
    diagnostic_max_steps = (
        int(getattr(args, "diagnostic_max_steps", 0) or 0)
        if numerical_failfast else 0)
    if diagnostic_max_steps < 0:
        raise ValueError("--diagnostic-max-steps must be >= 0")
    diagnostic_run = diagnostic_max_steps > 0
    if numerical_failfast:
        args.diagnostic_run = diagnostic_run
        args.diagnostic_max_steps = diagnostic_max_steps
        args.formal_result = not diagnostic_run
    validate_geolayernet_protocol_paths(
        args, "train", getattr(args, "training_domain", None))
    _enforce_formal_nyu_source(args, "train")
    _enforce_formal_kitti_source(args, "train", adapter)
    _enforce_formal_kitti_train_stage(adapter, args)
    _enforce_formal_training_splits(args)
    set_global_seed(args.seed)
    device = torch.device(args.device)
    if hasattr(adapter, "validate_runtime_environment"):
        adapter.validate_runtime_environment(args, device)
    _prep_outdir(args.output_dir, args.overwrite)
    if numerical_failfast:
        _enforce_clean_numerical_output_dir(args.output_dir)
    _write_command_and_config(args)
    scale = float(args.depth_unit_scale)

    use_val = (not getattr(args, "no_val", False)) and bool(args.val_phase)
    audit = audit_splits(
        _split_specs_for_train(args), output_dir=args.output_dir,
        preprocess=args.preprocess, depth_scale=args.depth_scale,
        max_samples=None, config_hint=f"{adapter.METHOD_NAME} train",
        fail_on_leak=True,
        allow_val_test_shared_rgb_gt_with_distinct_sparse=bool(
            getattr(args, "allow_val_test_shared_rgb_gt_with_distinct_sparse", False)))
    _enforce_formal_nyu_audit_counts(args, audit, "train")
    _enforce_formal_kitti_audit_counts(args, audit, "train")
    fairness_payload = _write_v2_protocol_artifacts(
        adapter, args, "train", audit=audit)
    if args.audit_only:
        _print_formal_training_audit_summary(audit, fairness_payload)
        return audit

    train_loader = create_loader(args.train_dataroot, args.train_phase, args.batch_size,
                                 num_workers=args.num_workers, shuffle=True,
                                 sample_list=args.train_list, preprocess=args.preprocess,
                                 depth_scale=args.depth_scale,
                                 config_hint=args.dataset_config or f"{adapter.METHOD_NAME} train", seed=args.seed,
                                 dataset_type=getattr(args, "_dataset_type", "nyu"),
                                 dataset_opts=_dataset_opts(args))
    val_loader = None
    if use_val:
        val_loader = create_loader(args.val_dataroot, args.val_phase, args.batch_size,
                                   num_workers=args.num_workers, shuffle=False,
                                   sample_list=args.val_list, preprocess=args.preprocess,
                                   depth_scale=args.depth_scale,
                                   config_hint=args.dataset_config or f"{adapter.METHOD_NAME} val", seed=args.seed,
                                   dataset_type=getattr(args, "_dataset_type", "nyu"),
                                   dataset_opts=_dataset_opts(args))
    else:
        print("[train] validation disabled (--no-val or empty --val-phase): "
              "selecting the final-epoch checkpoint")
    _enforce_formal_nyu_loader_counts(
        args, train_loader=train_loader, val_loader=val_loader)
    _enforce_formal_kitti_loader_counts(
        args, train_loader=train_loader, val_loader=val_loader)
    print(f"[train] train_dataroot={args.train_dataroot} train_list={args.train_list}")
    print(f"[train] val_dataroot={args.val_dataroot} val_list={args.val_list}")

    model = adapter.build_model(args, device).to(device)
    backward_hook_state = None
    backward_hook_handles = []
    if (hasattr(adapter, "enable_readonly_backward_diagnostics")
            and adapter.enable_readonly_backward_diagnostics(args)):
        backward_hook_state, backward_hook_handles = (
            _install_readonly_backward_diagnostics(model))
    if getattr(args, "checkpoint", None):
        _load_method_state(adapter, model, args, device)
    degraded = getattr(model, "geolayernet_degraded_note", None)
    args._train_num_batches = len(train_loader)
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    drop_incomplete_accumulation = bool(getattr(
        adapter, "DROP_INCOMPLETE_ACCUMULATION", False))
    if drop_incomplete_accumulation:
        effective_micro_batches = (
            len(train_loader) // grad_accum_steps) * grad_accum_steps
        if effective_micro_batches == 0:
            raise RuntimeError(
                "training split is smaller than one effective accumulated batch")
        args._optimizer_steps_per_epoch = effective_micro_batches // grad_accum_steps
    else:
        effective_micro_batches = len(train_loader)
        args._optimizer_steps_per_epoch = int(math.ceil(
            len(train_loader) / float(grad_accum_steps)))
    amp_enabled = bool(getattr(args, "amp", False)) and device.type == "cuda"
    if bool(getattr(args, "amp", False)) and not amp_enabled:
        print("[train] AMP requested but CUDA is unavailable; AMP disabled")
    grad_scaler_enabled = bool(getattr(
        args, "grad_scaler_enabled", amp_enabled))
    if grad_scaler_enabled and not amp_enabled:
        raise RuntimeError(
            "GradScaler requires enabled CUDA autocast; refusing fallback")
    scaler = _create_grad_scaler(grad_scaler_enabled)
    args.grad_scaler_enabled = grad_scaler_enabled
    autocast_dtype = (
        torch.bfloat16
        if getattr(args, "forward_compute_dtype", None) == "bfloat16"
        else None)
    if not hasattr(adapter, "make_optimizer"):
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'method')} has no registered "
            "make_optimizer; generic runner Adam substitution is forbidden")
    optimizer = adapter.make_optimizer(model, args)
    opt_note = getattr(adapter, "OPTIMIZER_NOTES", "adapter.make_optimizer")
    scheduler = adapter.make_scheduler(optimizer, args) if hasattr(adapter, "make_scheduler") else None
    loss_note = "adapter.train_step (registered method loss)"
    confhead_tracer = None
    confhead_trace_manager = None
    confhead_trace_source_hashes = None
    if hasattr(adapter, "confhead_bn_trace_spec"):
        confhead_trace_spec = adapter.confhead_bn_trace_spec(model, args)
        if confhead_trace_spec is not None:
            if not diagnostic_run:
                raise RuntimeError(
                    "confhead BatchNorm tracing is restricted to bounded "
                    "diagnostic runs")
            confhead_trace_manager = RollingReplayManager(
                args.output_dir,
                interval=int(getattr(args, "psd_bn_snapshot_interval", 100)),
                keep=int(getattr(args, "psd_bn_snapshot_keep", 5)),
                materialized_batch_keep=(
                    int(getattr(args, "psd_bn_snapshot_interval", 100)) + 20))

            def trace_nonfinite(stage, tensor_name, statistics):
                raise _NonfiniteDetected(stage, tensor_name, statistics)

            confhead_tracer = ConfheadBNTracer(
                model, optimizer, confhead_trace_spec,
                warning_callback=confhead_trace_manager.capture_warning,
                nonfinite_callback=trace_nonfinite)
            confhead_trace_source_hashes = _source_sha256(adapter)
            model._gln_confhead_trace_boundary = confhead_tracer.boundary

    internal_reference_path = os.path.join(
        args.output_dir, ".numerical_finite_reference.pt")
    finite_reference_path = internal_reference_path
    if numerical_failfast:
        try:
            _check_finite_parameters(model, stage="initialization")
            _check_finite_optimizer_state(model, optimizer)
        except _NonfiniteDetected as error:
            context = {
                "current_stage": "initialization",
                "optimizer_step_completed": False,
            }
            event_path = _write_nonfinite_diagnostic(
                adapter, args, model, optimizer, {}, context, error, audit,
                epoch=0, iteration=-1, global_step=0,
                rng_state=_capture_rng_state())
            raise RuntimeError(
                f"numerical fail-fast stopped training; diagnostic={event_path}") from error
        _save_known_finite_reference(
            internal_reference_path, model, optimizer, epoch=0, global_step=0)

    latest_path = os.path.join(args.output_dir, "latest_net_G.pth")
    best_path = os.path.join(args.output_dir, "best_net_G.pth")
    final_path = os.path.join(args.output_dir, "final_net_G.pth")
    val_best_metric = str(getattr(args, "val_best_metric", "RMSE_m") or "RMSE_m")
    val_best_mode = str(getattr(args, "val_best_mode", "min") or "min")
    best_value = float("inf") if val_best_mode == "min" else -float("inf")
    best_epoch = None
    # ``global_step`` is the count of successful optimizer updates.  Failed
    # or GradScaler-skipped attempts never advance it.
    global_step = 0
    completed_iterations = 0
    diagnostic_stop = False
    diagnostic_last_loss = None
    diagnostic_first_loss = None
    attempted_optimizer_steps = 0
    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    scheduler_step_count = 0
    previous_successful_step_losses = {}
    diagnostic_trace_path = os.path.join(
        args.output_dir, "numerical_trace.jsonl")
    amp_consecutive_overflows = 0
    max_consecutive_amp_overflows = int(getattr(
        adapter, "MAX_CONSECUTIVE_AMP_OVERFLOWS", 8))
    if max_consecutive_amp_overflows <= 0:
        raise ValueError("MAX_CONSECUTIVE_AMP_OVERFLOWS must be positive")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        optimizer.zero_grad(set_to_none=True)
        if confhead_trace_manager is not None:
            confhead_trace_manager.begin_epoch(train_loader, epoch)
        train_iterator = iter(train_loader)
        for it, batch in enumerate(train_iterator):
            if it >= effective_micro_batches:
                break
            loader_trace_state = None
            if confhead_trace_manager is not None:
                loader_trace_state = confhead_trace_manager.begin_iteration(
                    train_loader, batch, epoch=epoch, iteration=it,
                    global_step=global_step)
            batch = _to_device(batch, device)
            if hasattr(adapter, "prepare_batch_for_model"):
                batch = adapter.prepare_batch_for_model(batch, device)
            context = {
                "current_stage": "raw_input",
                "optimizer_step_completed": False,
                "amp_consecutive_overflows": amp_consecutive_overflows,
                "amp_successful_optimizer_steps": successful_optimizer_steps,
                "maximum_allowed_consecutive_amp_overflows": (
                    max_consecutive_amp_overflows),
                "previous_successful_step_losses": dict(
                    previous_successful_step_losses),
            }
            if loader_trace_state is not None:
                context["dataloader_replay_state"] = loader_trace_state
            if confhead_tracer is not None:
                confhead_tracer.begin_iteration(
                    context, epoch=epoch, iteration=it,
                    global_step=global_step,
                    batch_metadata=(
                        confhead_trace_manager.current["batch_metadata"]))
            if backward_hook_state is not None:
                backward_hook_state["context"] = context
            if (diagnostic_run and it % grad_accum_steps == 0
                    and device.type == "cuda"
                    and torch.cuda.is_available()):
                torch.cuda.reset_peak_memory_stats(device)
            rng_state = _capture_rng_state() if numerical_failfast else None
            start_buffers = (
                _capture_start_buffers(model) if numerical_failfast else None)
            try:
                if numerical_failfast:
                    _check_finite_collection(batch, "raw_input", "batch")
                if confhead_tracer is not None:
                    confhead_tracer.boundary("A")
                if hasattr(adapter, "before_train_step"):
                    adapter.before_train_step(
                        model, optimizer, args, epoch, it, len(train_loader))
                context["current_stage"] = "forward"
                with _autocast_context(
                        device, amp_enabled, autocast_dtype):
                    if numerical_failfast:
                        loss = _train_one_step_with_numerical_guard(
                            adapter, model, batch, device, epoch, scale, context)
                    else:
                        loss = _train_one_step(
                            adapter, model, batch, device, epoch, scale)
                if numerical_failfast:
                    _check_finite_loss(loss, context.get("loss_components"))
                if confhead_tracer is not None:
                    # PSD emits H and per-component I from its adapter-only
                    # diagnostic wrapper; record the combined native loss here.
                    confhead_tracer.boundary("I", values={"loss": loss})
                group_start = (it // grad_accum_steps) * grad_accum_steps
                group_size = min(
                    grad_accum_steps, effective_micro_batches - group_start)
                scaled_loss = loss / float(group_size)
                context["current_stage"] = "backward"
                if grad_scaler_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if confhead_tracer is not None:
                    confhead_tracer.boundary("J")
                # GradScaler gradients are still loss-scaled here. Inf/NaN
                # is assessed after unscale. BF16 has no scaler, so its live
                # gradients are already the optimizer-boundary gradients.
                update_now = ((it + 1) % grad_accum_steps == 0) or (
                    it + 1 == effective_micro_batches)
                if numerical_failfast and not grad_scaler_enabled:
                    context["unscaled_gradient_report"] = (
                        _gradient_finite_report(model))
                    if (not update_now
                            and not context["unscaled_gradient_report"]["finite"]):
                        attempted_optimizer_steps += 1
                        skipped_optimizer_steps += 1
                        context["optimizer_step_skipped"] = True
                        if diagnostic_run:
                            _append_jsonl(
                                diagnostic_trace_path,
                                _numerical_trace_record(
                                    context, epoch=epoch, iteration=it,
                                    global_step=global_step,
                                    successful_optimizer_steps=(
                                        successful_optimizer_steps),
                                    skipped=True,
                                    scheduler_step_count=scheduler_step_count,
                                    scheduler_advanced=False, device=device))
                        report = context["unscaled_gradient_report"]
                        raise _NonfiniteDetected(
                            "unscaled_gradient",
                            "gradient." + str(report.get(
                                "first_nonfinite_parameter")),
                            report.get("first_nonfinite_statistics") or {})
                if update_now:
                    unscaled = not grad_scaler_enabled
                    if numerical_failfast:
                        context["current_stage"] = "scaled_gradient"
                        context["scaled_gradient_report"] = (
                            _gradient_finite_report(model))
                    if grad_scaler_enabled and (numerical_failfast or hasattr(
                            adapter, "before_optimizer_step")):
                        scaler.unscale_(optimizer)
                        unscaled = True
                    if numerical_failfast and unscaled:
                        context["current_stage"] = "unscaled_gradient"
                        context["unscaled_gradient_report"] = (
                            _gradient_finite_report(model))
                    unscaled_nonfinite = bool(
                        numerical_failfast and unscaled
                        and not context["unscaled_gradient_report"]["finite"])
                    fail_first_unscaled = bool(getattr(
                        adapter,
                        "FAIL_ON_FIRST_NONFINITE_UNSCALED_GRADIENT", False))
                    if unscaled_nonfinite and (
                            not grad_scaler_enabled or fail_first_unscaled):
                        attempted_optimizer_steps += 1
                        context["optimizer_step_skipped"] = True
                        if grad_scaler_enabled:
                            # Let GradScaler perform its documented skipped
                            # step/update so scaler_after and skip semantics
                            # are recorded, but do not clip or mutate gradients.
                            optimizer_step_completed = _checked_optimizer_step(
                                model, optimizer, scaler, context,
                                numerical_failfast=numerical_failfast,
                                amp_enabled=True)
                            if optimizer_step_completed:
                                raise RuntimeError(
                                    "GradScaler unexpectedly completed a step "
                                    "with a non-finite unscaled gradient")
                            skipped_optimizer_steps += 1
                            amp_consecutive_overflows += 1
                        else:
                            skipped_optimizer_steps += 1
                        report = context["unscaled_gradient_report"]
                        if diagnostic_run:
                            _append_jsonl(
                                diagnostic_trace_path,
                                _numerical_trace_record(
                                    context, epoch=epoch, iteration=it,
                                    global_step=global_step,
                                    successful_optimizer_steps=(
                                        successful_optimizer_steps),
                                    skipped=True,
                                    scheduler_step_count=scheduler_step_count,
                                    scheduler_advanced=False,
                                    device=device))
                        raise _NonfiniteDetected(
                            "unscaled_gradient",
                            "gradient." + str(report.get(
                                "first_nonfinite_parameter")),
                            report.get("first_nonfinite_statistics") or {})
                    if hasattr(adapter, "before_optimizer_step"):
                        context["current_stage"] = "gradient_clipping"
                        adapter.before_optimizer_step(
                            model, optimizer, args, epoch, it)
                        if numerical_failfast:
                            _check_finite_gradients(model)
                    if confhead_tracer is not None:
                        confhead_tracer.boundary("K")
                    context["current_stage"] = "optimizer_step"
                    attempted_optimizer_steps += 1
                    optimizer_step_completed = _checked_optimizer_step(
                        model, optimizer, scaler, context,
                        numerical_failfast=numerical_failfast,
                        amp_enabled=grad_scaler_enabled,
                        post_step_diagnostic=(
                            (lambda: confhead_tracer.boundary("L"))
                            if confhead_tracer is not None else None))
                    if optimizer_step_completed:
                        amp_consecutive_overflows = 0
                        successful_optimizer_steps += 1
                        global_step += 1
                        context["amp_consecutive_overflows"] = 0
                        context["amp_successful_optimizer_steps"] = (
                            successful_optimizer_steps)
                        if hasattr(adapter, "after_optimizer_step"):
                            adapter.after_optimizer_step(
                                model, optimizer, scheduler, args, epoch, it)
                            if scheduler is not None:
                                scheduler_step_count += 1
                                context["scheduler_step_completed"] = True
                        elif scheduler is not None and bool(
                                getattr(adapter, "SCHEDULER_STEP_PER_BATCH", False)):
                            scheduler.step()
                            scheduler_step_count += 1
                            context["scheduler_step_completed"] = True
                        previous_successful_step_losses = {
                            "loss_absolute": _component_scalar(
                                context, "absolute", "loss_absolute"),
                            "loss_relative": _component_scalar(
                                context, "relative", "loss_relative"),
                            "loss_gradient": _component_scalar(
                                context, "gradient", "loss_gradient"),
                            "loss_total": _component_scalar(
                                context, "total", "loss"),
                        }
                        if confhead_tracer is not None:
                            context["epoch_end_scheduler_pending"] = bool(
                                scheduler is not None
                                and not bool(getattr(
                                    adapter, "SCHEDULER_STEP_PER_BATCH", False))
                                and it + 1 == effective_micro_batches)
                    else:
                        skipped_optimizer_steps += 1
                        amp_consecutive_overflows += 1
                        context["amp_consecutive_overflows"] = (
                            amp_consecutive_overflows)
                        context["amp_successful_optimizer_steps"] = (
                            successful_optimizer_steps)
                        context["scheduler_step_skipped_due_amp_overflow"] = True
                        if (grad_scaler_enabled
                                and amp_consecutive_overflows
                                >= max_consecutive_amp_overflows):
                            _amp_overflow_fail_closed(
                                context,
                                consecutive_overflows=amp_consecutive_overflows,
                                successful_optimizer_steps=(
                                    successful_optimizer_steps),
                                maximum_allowed_consecutive_amp_overflows=(
                                    max_consecutive_amp_overflows))
                    if diagnostic_run:
                        _append_jsonl(
                            diagnostic_trace_path,
                            _numerical_trace_record(
                                context, epoch=epoch, iteration=it,
                                global_step=global_step,
                                successful_optimizer_steps=(
                                    successful_optimizer_steps),
                                skipped=not optimizer_step_completed,
                                scheduler_step_count=scheduler_step_count,
                                scheduler_advanced=bool(context.get(
                                    "scheduler_step_completed", False)),
                                device=device))
                    optimizer.zero_grad(set_to_none=True)
                    if (confhead_tracer is not None
                            and optimizer_step_completed):
                        # Finalize call-count/root trace fields before any
                        # rolling artifact serializes this iteration.
                        confhead_tracer.finish_iteration()
                        confhead_trace_manager.mark_success()
                        confhead_trace_manager.save_snapshot(
                            model, optimizer, scheduler, train_loader,
                            epoch=epoch, iteration=it,
                            global_step=global_step,
                            tracer=confhead_tracer, context=context,
                            source_sha256=confhead_trace_source_hashes)
                        confhead_trace_manager.append_periodic_trace(
                            context, global_step)
                loss_value = float(loss.item())
                running += loss_value
                n += 1
                completed_iterations += 1
                if diagnostic_first_loss is None:
                    diagnostic_first_loss = loss_value
                diagnostic_last_loss = loss_value
                if it % 50 == 0:
                    print(
                        f"[train] epoch {epoch}/{args.epochs} it {it} "
                        f"loss {loss_value:.6f}")
            except _NonfiniteDetected as error:
                if confhead_tracer is not None:
                    try:
                        # A hook can terminate the iteration before downstream
                        # modules run; record observed counts without treating
                        # those expected missing calls as a new warning.
                        confhead_tracer.finish_iteration(
                            terminated_early=True)
                        trace_failure = confhead_trace_manager.finalize_failure(
                            model, optimizer, scheduler, train_loader,
                            epoch=epoch, iteration=it, global_step=global_step,
                            start_buffers=start_buffers,
                            tracer=confhead_tracer, context=context, error=error,
                            iteration_start_rng=rng_state)
                        context["confhead_bn_failure_artifacts"] = trace_failure
                        context["confhead_optimizer_state_statistics"] = (
                            trace_failure.get("optimizer_state_statistics"))
                    except Exception as trace_error:
                        context["confhead_bn_failure_artifacts"] = {
                            "artifact_error": (
                                f"{type(trace_error).__name__}: {trace_error}")}
                event_path = _write_nonfinite_diagnostic(
                    adapter, args, model, optimizer, batch, context, error,
                    audit, epoch=epoch, iteration=it,
                    global_step=global_step, rng_state=rng_state,
                    start_buffers=start_buffers,
                    finite_reference_path=finite_reference_path)
                if diagnostic_run:
                    blocked = error.failure_stage == "amp_overflow"
                    _write_json(os.path.join(
                        args.output_dir, "diagnostic_run.json"), {
                        "schema": "comparison-diagnostic-run-v1",
                        "method": getattr(adapter, "METHOD_NAME", None),
                        "diagnostic_run": True,
                        "formal_result": False,
                        "requested_steps": diagnostic_max_steps,
                        "diagnostic_max_steps": diagnostic_max_steps,
                        "completed_iterations": completed_iterations,
                        "global_successful_step": global_step,
                        "attempted_optimizer_steps": attempted_optimizer_steps,
                        "successful_optimizer_steps": successful_optimizer_steps,
                        "scheduler_step_count": scheduler_step_count,
                        "skipped_optimizer_steps": skipped_optimizer_steps,
                        "amp_overflow_skipped_steps": (
                            skipped_optimizer_steps
                            if grad_scaler_enabled else 0),
                        "final_amp_scale": (
                            float(scaler.get_scale())
                            if grad_scaler_enabled else None),
                        "first_loss": diagnostic_first_loss,
                        "last_loss": diagnostic_last_loss,
                        "formal_checkpoint_written": False,
                        "formal_checkpoints_written": False,
                        "status": (
                            "BLOCKED_PERSISTENT_AMP_OVERFLOW"
                            if blocked else (
                                "BLOCKED_CONFHEAD_BATCHNORM_STATE_DIVERGENCE"
                                if confhead_tracer is not None
                                else "NONFINITE_FAILFAST")),
                        "failure_stage": error.failure_stage,
                        "failure_tensor": error.failure_tensor,
                        "nonfinite_diagnostic": event_path,
                    })
                if os.path.isfile(internal_reference_path):
                    os.remove(internal_reference_path)
                if confhead_tracer is not None:
                    confhead_tracer.close()
                    if getattr(model, "_gln_confhead_trace_boundary", None) is not None:
                        delattr(model, "_gln_confhead_trace_boundary")
                raise RuntimeError(
                    "numerical fail-fast stopped training at "
                    f"epoch={epoch} iteration={it} global_step={global_step}; "
                    f"diagnostic={event_path}") from error
            if (diagnostic_run
                    and successful_optimizer_steps >= diagnostic_max_steps):
                diagnostic_stop = True
                break
        if diagnostic_stop:
            break
        if scheduler is not None and not bool(
                getattr(adapter, "SCHEDULER_STEP_PER_BATCH", False)):
            scheduler.step()
            scheduler_step_count += 1
        avg = running / max(n, 1)
        if not numerical_failfast and not diagnostic_run:
            _save_training_checkpoint(
                adapter, model, optimizer, epoch, latest_path)

        if val_loader is not None and (epoch % max(args.val_freq, 1) == 0):
            model.eval()
            validation_context = {
                "current_stage": "validation",
                "optimizer_step_completed": False,
            }
            try:
                val_metrics = _quick_val_rmse(
                    adapter, model, val_loader, device, scale,
                    clip_pred=bool(getattr(
                        adapter, "VALIDATION_CLIP_PREDICTION", True)),
                    lo=args.min_depth, hi=args.max_depth,
                    numerical_failfast=numerical_failfast,
                    validation_context=validation_context)
            except _NonfiniteDetected as error:
                failing_batch = validation_context.get("batch") or {}
                event_path = _write_nonfinite_diagnostic(
                    adapter, args, model, optimizer, failing_batch,
                    validation_context, error, audit, epoch=epoch,
                    iteration=-1, global_step=global_step,
                    rng_state=_capture_rng_state(),
                    finite_reference_path=finite_reference_path)
                if os.path.isfile(internal_reference_path):
                    os.remove(internal_reference_path)
                raise RuntimeError(
                    "numerical fail-fast stopped training on non-finite "
                    f"validation input/prediction; diagnostic={event_path}") from error
            metric_value = val_metrics.get(val_best_metric)
            if metric_value is None:
                raise KeyError(
                    f"val_best_metric={val_best_metric} not found; available={sorted(val_metrics.keys())}")
            required_val_metrics = tuple(dict.fromkeys(
                (val_best_metric, "RMSE_m")))
            if numerical_failfast and not _validation_metrics_finite(
                    val_metrics, required_fields=required_val_metrics):
                bad_name, bad_value = _first_invalid_validation_metric(
                    val_metrics, required_fields=required_val_metrics)
                if isinstance(bad_value, (int, float, np.number)):
                    bad_statistics = _tensor_statistics(
                        f"validation_metric.{bad_name}",
                        torch.tensor(float(bad_value), dtype=torch.float64))
                else:
                    bad_statistics = {
                        "missing": bad_name not in val_metrics,
                        "value_type": type(bad_value).__name__,
                        "value_repr": repr(bad_value),
                    }
                error = _NonfiniteDetected(
                    "validation", f"validation_metric.{bad_name}",
                    bad_statistics)
                context = {
                    "current_stage": "validation",
                    "optimizer_step_completed": False,
                    "failure_tensor": f"validation_metric.{bad_name}",
                }
                try:
                    raise error
                except _NonfiniteDetected as caught:
                    event_path = _write_nonfinite_diagnostic(
                        adapter, args, model, optimizer, {}, context,
                        caught, audit, epoch=epoch, iteration=-1,
                        global_step=global_step,
                        rng_state=_capture_rng_state(),
                        finite_reference_path=finite_reference_path)
                if os.path.isfile(internal_reference_path):
                    os.remove(internal_reference_path)
                raise RuntimeError(
                    "numerical fail-fast stopped training on non-finite "
                    f"validation metric; diagnostic={event_path}") from error
            print(f"[train] epoch {epoch} avg_loss {avg:.6f} "
                  f"val_{val_best_metric} {metric_value:.6f} "
                  f"val_RMSE_m {val_metrics.get('RMSE_m', float('nan')):.6f} "
                  f"(best {best_value:.6f})")
            if numerical_failfast:
                improved = (
                    math.isfinite(float(metric_value)) and
                    ((metric_value < best_value)
                     if val_best_mode == "min"
                     else (metric_value > best_value)))
            else:
                improved = (
                    (metric_value < best_value)
                    if val_best_mode == "min"
                    else (metric_value > best_value))
            if numerical_failfast and not diagnostic_run:
                _save_training_checkpoint(
                    adapter, model, optimizer, epoch, latest_path)
            if improved:
                best_value = metric_value
                best_epoch = int(epoch)
                if not diagnostic_run:
                    _save_training_checkpoint(
                        adapter, model, optimizer, epoch, best_path)
                    print(f"[train] new best -> {best_path}")
        else:
            print(f"[train] epoch {epoch} avg_loss {avg:.6f}")
            if numerical_failfast and not diagnostic_run:
                _save_training_checkpoint(
                    adapter, model, optimizer, epoch, latest_path)
        if numerical_failfast:
            if diagnostic_run:
                _save_known_finite_reference(
                    finite_reference_path, model, optimizer,
                    epoch=epoch, global_step=global_step)
            elif os.path.isfile(latest_path):
                if os.path.isfile(internal_reference_path):
                    os.remove(internal_reference_path)
                finite_reference_path = latest_path

    if diagnostic_run:
        strict_checkpoint_roundtrip = False
        diagnostic_checkpoint = None
        if (successful_optimizer_steps >= diagnostic_max_steps
                and bool(getattr(
                    adapter, "DIAGNOSTIC_STRICT_CHECKPOINT_ROUNDTRIP", False))):
            if skipped_optimizer_steps != 0:
                raise RuntimeError(
                    "diagnostic reached its successful-step target with "
                    f"{skipped_optimizer_steps} skipped optimizer steps")
            if (scheduler is not None
                    and scheduler_step_count != successful_optimizer_steps):
                raise RuntimeError(
                    "diagnostic scheduler/optimizer step mismatch: "
                    f"scheduler={scheduler_step_count} "
                    f"optimizer={successful_optimizer_steps}")
            diagnostic_checkpoint = os.path.join(
                args.output_dir, "diagnostic_checkpoint.pth")
            payload = adapter.checkpoint_payload(
                model, optimizer, epoch) if hasattr(
                    adapter, "checkpoint_payload") else None
            if payload is None:
                raise RuntimeError(
                    "diagnostic strict round-trip requires checkpoint_payload")
            torch.save(payload, diagnostic_checkpoint)
            adapter.load_checkpoint(
                model, diagnostic_checkpoint, device, args, strict=True)
            strict_checkpoint_roundtrip = True
        status = {
            "schema": "comparison-diagnostic-run-v1",
            "method": getattr(adapter, "METHOD_NAME", None),
            "diagnostic_run": True,
            "formal_result": False,
            "requested_steps": diagnostic_max_steps,
            "diagnostic_max_steps": diagnostic_max_steps,
            "completed_iterations": completed_iterations,
            "global_successful_step": global_step,
            "successful_steps": successful_optimizer_steps,
            "attempted_optimizer_steps": attempted_optimizer_steps,
            "successful_optimizer_steps": successful_optimizer_steps,
            "scheduler_step_count": scheduler_step_count,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "amp_overflow_skipped_steps": (
                skipped_optimizer_steps if grad_scaler_enabled else 0),
            "final_amp_scale": (
                float(scaler.get_scale())
                if grad_scaler_enabled else None),
            "first_loss": diagnostic_first_loss,
            "last_loss": diagnostic_last_loss,
            "formal_checkpoint_written": False,
            "formal_checkpoints_written": False,
            "diagnostic_checkpoint": diagnostic_checkpoint,
            "strict_checkpoint_roundtrip": strict_checkpoint_roundtrip,
            "status": (
                "FINITE_STEPS_COMPLETE"
                if successful_optimizer_steps >= diagnostic_max_steps
                else "FINITE_TRAINING_PLAN_EXHAUSTED"),
        }
        _write_json(os.path.join(
            args.output_dir, "diagnostic_run.json"), status)
        result_io.write_run_info(
            os.path.join(args.output_dir, "train_run_info.txt"),
            title=f"{adapter.METHOD_NAME} diagnostic train: {os.path.basename(args.output_dir)}",
            header=_run_info_header(
                adapter, args, output_dir=args.output_dir,
                diagnostic_run=True, formal_result=False,
                diagnostic_max_steps=diagnostic_max_steps,
                successful_steps=successful_optimizer_steps,
                successful_optimizer_steps=successful_optimizer_steps,
                skipped_optimizer_steps=skipped_optimizer_steps,
                checkpoint_selection="disabled for diagnostic run",
                selected_checkpoint="none", optimizer=opt_note,
                loss=loss_note, amp=amp_enabled,
                grad_accum_steps=grad_accum_steps,
                dataset_protocol_audit=audit.get("status")),
            overwrite=args.overwrite)
        result_io.write_args_json(
            os.path.join(args.output_dir, "train_args.json"),
            vars(args).copy(), overwrite=True)
        if os.path.isfile(internal_reference_path):
            os.remove(internal_reference_path)
        if confhead_tracer is not None:
            confhead_tracer.close()
            if getattr(model, "_gln_confhead_trace_boundary", None) is not None:
                delattr(model, "_gln_confhead_trace_boundary")
        _remove_hook_handles(backward_hook_handles)
        print(
            f"\n=== {adapter.METHOD_NAME} diagnostic done: "
            f"completed_iterations={completed_iterations}, "
            f"successful_optimizer_steps={successful_optimizer_steps}, "
            f"skipped_optimizer_steps={skipped_optimizer_steps}, "
            f"no formal checkpoint written ===")
        return status

    # final-epoch checkpoint always saved; it is the selected one when val is off.
    _save_training_checkpoint(adapter, model, optimizer, args.epochs, final_path)
    if (_formal_kitti_stage_enabled(adapter, args)
            and getattr(args, "kitti_stage", None) == "develop"
            and best_epoch is None):
        raise RuntimeError(
            "KITTI develop produced no finite selected validation checkpoint")
    if getattr(args, "_dataset_type", None) == "nyu" and best_epoch is None:
        raise RuntimeError(
            "formal N training produced no finite selected validation checkpoint")
    if val_loader is not None and not os.path.exists(best_path):
        _save_training_checkpoint(adapter, model, optimizer, args.epochs, best_path)
    selected = best_path if val_loader is not None else final_path
    ckpt_rule = (f"best_net_G.pth ({val_best_mode} val {val_best_metric})" if val_loader is not None
                 else "final_net_G.pth (last epoch; no validation)")

    if _formal_kitti_stage_enabled(adapter, args):
        manifest = os.path.realpath(os.path.expanduser(args.train_list))
        config_path = (os.path.realpath(os.path.expanduser(args.dataset_config))
                       if getattr(args, "dataset_config", None) else None)
        audit_path = _resolved(os.path.join(
            args.output_dir, "dataset_protocol_audit.json"))
        if not os.path.isfile(audit_path):
            raise RuntimeError("formal K training dataset audit file is missing")
        if args.kitti_stage == "develop":
            val_manifest = os.path.realpath(os.path.expanduser(args.val_list))
            if (bool(getattr(adapter, "REQUIRE_KITTI_FIXED_E_ONLY", False))
                    or not bool(getattr(
                        adapter, "REQUIRE_KITTI_DEVELOP_RECEIPT", True))):
                _write_json(os.path.join(args.output_dir, "selected_epoch.json"), {
                    "stage": "develop",
                    "training_domain": "K",
                    "method": getattr(adapter, "METHOD_NAME", None),
                    "seed": int(args.seed),
                    "selected_epoch": int(best_epoch),
                    "selection_metric": val_best_metric,
                    "selection_mode": val_best_mode,
                    "selection_value": float(best_value),
                    "selected_checkpoint": os.path.realpath(best_path),
                    "note": "informational epoch selection only; final training accepts the explicitly fixed epoch E",
                })
                receipt = None
            else:
                receipt = {
                    "schema": "geolayernet-kitti-develop-v1",
                    "stage": "develop",
                    "training_domain": "K",
                    "method": getattr(adapter, "METHOD_NAME", None),
                    "seed": int(args.seed),
                    "selected_epoch": int(best_epoch),
                    "selection_metric": val_best_metric,
                    "selection_mode": val_best_mode,
                    "selection_value": float(best_value),
                    "train_manifest": manifest,
                    "train_manifest_sha256": _sha256_file(manifest),
                    "train_samples": len(train_loader.dataset),
                    "train_dataroot": _resolved(args.train_dataroot),
                    "val_manifest": val_manifest,
                    "val_manifest_sha256": _sha256_file(val_manifest),
                    "val_samples": len(val_loader.dataset),
                    "val_dataroot": _resolved(args.val_dataroot),
                    "dataset_config": config_path,
                    "dataset_config_sha256": (_sha256_file(config_path) if config_path else None),
                    "split_fingerprints": audit.get("split_fingerprints", {}),
                    "dataset_audit": audit_path,
                    "dataset_audit_sha256": _sha256_file(audit_path),
                    "selected_checkpoint": os.path.realpath(best_path),
                    "selected_checkpoint_sha256": _sha256_file(best_path),
                }
            if receipt is not None:
                receipt.update(_numerical_validity_fields(adapter))
                _write_json(os.path.join(args.output_dir, "kitti_develop_receipt.json"), receipt)
        else:
            expected = ((getattr(args, "_dataset_config_info", {}) or {}).get("train", {})
                        or {}).get("expected_train_full_samples")
            if expected is not None and len(train_loader.dataset) != int(expected):
                raise RuntimeError(
                    f"KITTI train_full sample count mismatch: expected {int(expected)}, "
                    f"got {len(train_loader.dataset)}")
            sidecar = {
                "schema": "geolayernet-kitti-final-v1",
                "stage": "final-train",
                "training_domain": "K",
                "method": getattr(adapter, "METHOD_NAME", None),
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "train_manifest": manifest,
                "train_manifest_sha256": _sha256_file(manifest),
                "train_samples": len(train_loader.dataset),
                "train_dataroot": _resolved(args.train_dataroot),
                "dataset_config": config_path,
                "dataset_config_sha256": (_sha256_file(config_path) if config_path else None),
                "split_fingerprints": audit.get("split_fingerprints", {}),
                "dataset_audit": audit_path,
                "dataset_audit_sha256": _sha256_file(audit_path),
                "develop_receipt": getattr(args, "_kitti_develop_receipt_path", None),
                "develop_receipt_sha256": getattr(
                    args, "_kitti_develop_receipt_sha256", None),
                "checkpoint": os.path.realpath(final_path),
                "checkpoint_sha256": _sha256_file(final_path),
            }
            sidecar.update(_numerical_validity_fields(adapter))
            if not bool(getattr(adapter, "SKIP_SHARED_CHECKPOINT_PROVENANCE", False)):
                _write_json(_formal_kitti_sidecar_path(final_path), sidecar)

    if (getattr(args, "_dataset_type", None) == "nyu"
            and _nyu_small_spec(args) is None
            and not bool(getattr(adapter, "SKIP_SHARED_CHECKPOINT_PROVENANCE", False))):
        audit_path = _resolved(os.path.join(
            args.output_dir, "dataset_protocol_audit.json"))
        if not os.path.isfile(audit_path):
            raise RuntimeError("formal N training dataset audit file is missing")
        sidecar = {
            "schema": "geolayernet-nyu48k-final-v1",
            "training_domain": "N",
            "method": getattr(adapter, "METHOD_NAME", None),
            "seed": int(args.seed),
            "selected_epoch": int(best_epoch),
            "selection_metric": val_best_metric,
            "selection_mode": val_best_mode,
            "selection_value": float(best_value),
            "dataset_config": _resolved(args.dataset_config),
            "dataset_config_sha256": _sha256_file(args.dataset_config),
            "train_dataroot": _resolved(args.train_dataroot),
            "val_dataroot": _resolved(args.val_dataroot),
            "counts": {
                "train": len(train_loader.dataset),
                "val": len(val_loader.dataset),
            },
            "split_fingerprints": audit.get("split_fingerprints", {}),
            "dataset_audit": audit_path,
            "dataset_audit_sha256": _sha256_file(audit_path),
            "checkpoint": _resolved(selected),
            "checkpoint_sha256": _sha256_file(selected),
        }
        sidecar.update(_numerical_validity_fields(adapter))
        sidecar.update(_formal_training_metadata(fairness_payload))
        _write_json(_formal_source_sidecar_path(selected), sidecar)

    if os.path.isfile(internal_reference_path):
        os.remove(internal_reference_path)

    result_io.write_run_info(
        os.path.join(args.output_dir, "train_run_info.txt"),
        title=f"{adapter.METHOD_NAME} train: {os.path.basename(args.output_dir)}",
        header=_run_info_header(adapter, args, output_dir=args.output_dir, epochs=args.epochs,
                                lr=args.lr, batch_size=args.batch_size,
                                validation_used=val_loader is not None,
                                checkpoint_selection=ckpt_rule, selected_checkpoint=selected,
                                val_best_metric=val_best_metric,
                                val_best_mode=val_best_mode,
                                best_val_metric_value=best_value if val_loader is not None else "n/a",
                                best_epoch=best_epoch if best_epoch is not None else "n/a",
                                init_checkpoint=getattr(args, "checkpoint", None),
                                optimizer=opt_note, loss=loss_note,
                                amp=amp_enabled,
                                grad_accum_steps=grad_accum_steps,
                                drop_incomplete_accumulation=drop_incomplete_accumulation,
                                optimizer_steps_per_epoch=args._optimizer_steps_per_epoch,
                                kitti_stage=getattr(args, "kitti_stage", None),
                                dataset_protocol_audit=audit.get("status"),
                                degraded_note=degraded or "none"),
        overwrite=args.overwrite,
    )
    result_io.write_args_json(os.path.join(args.output_dir, "train_args.json"), vars(args).copy(), overwrite=True)
    print(f"\n=== {adapter.METHOD_NAME} train done: selected={selected} latest={latest_path} final={final_path} ===")

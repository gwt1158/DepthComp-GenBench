"""Comparison Protocol V2 bridge for the unchanged official CompletionFormer.

The released network, PVT/JCAT backbone, six-step NLSPN refinement, native
``1.0*L1+1.0*L2`` objective, AdamW optimizer and MultiStepLR schedule under
``src/`` are not modified.  This module owns only protocol data conversion,
metadata and deterministic strict state-dict I/O.
"""

from collections import OrderedDict
from contextlib import contextmanager
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


METHOD_NAME = "CompletionFormer"
COMPARISON_BUCKET = "supervised_same_data"
SUPERVISION = "N-only or K-only supervised training on registered source data"
PRIOR = "ordinary ImageNet ResNet-34 and PVT initialization"
PROTOCOL_STATUS = "blocked_extension"
PROTOCOL_VARIANT = "CompletionFormer-N/K and CompletionFormer-N/K-Unified"
REQUIRES_CHECKPOINT = True
REQUIRE_KITTI_TWO_STAGE = True
REQUIRE_KITTI_DEVELOP_RECEIPT = False
SKIP_SHARED_CHECKPOINT_PROVENANCE = True
TRAIN_AMP = False
TRAIN_BATCH_SIZE = 12
TRAIN_NUM_WORKERS = 1
EVAL_BATCH_SIZE = 1
EVAL_NUM_WORKERS = 1
PROFILE_SPEED = False
PREDICTION_NPY_UNIT = "metres"
ERROR_MAX_CLI_UNIT = "metres"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = WORKSPACE_ROOT / "CompletionFormer-main"
_SRC = METHOD_ROOT / "src"
_PRETRAINED = {
    "resnet34": _SRC / "pretrained" / "resnet34.pth",
    "pvt": _SRC / "pretrained" / "pvt.pth",
}
_REQUIRED_DEPENDENCIES = ("torchvision", "timm", "mmcv", "mmseg")
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)
_SOURCE_MAX_DEPTH_M = {"N": 10.0, "K": 80.0}
_NETWORK_INPUT_HW = (228, 304)
_INPUT_MULTIPLE = (1, 1)
_PADDING = (0, 0, 0, 0)
_PROPAGATION_ITERATIONS = 6
_PROPAGATION_KERNEL = 3
_AFFINITY_TYPE = "TGASS"
_AFFINITY_GAMMA = 0.5
_CONFIDENCE_PROPAGATION = True
_PRESERVE_SPARSE_INPUT = False
_REFINEMENT_ENABLED = True
_LOSS_SPEC = "1.0*L1+1.0*L2"
_N_EPOCHS = 72
_K_NATIVE_EPOCHS = 250
_K_DEVELOP_EPOCH_CAP = 50
_N_MILESTONES = [36, 48, 56, 64]
_K_MILESTONES = [150, 180, 210, 240]
_LR = 0.001
_BETAS = (0.9, 0.999)
_EPSILON = 1e-8
_WEIGHT_DECAY = 0.01
_GAMMA = 0.5

PROTOCOL_NOTES = (
    "Unchanged official CompletionFormer with PVT/JCAT features and six native "
    "NLSPN propagation steps. The primary output is dict['pred']; pred_init is "
    "the coarse residual depth and pred_inter contains propagation states. "
    "preserve_input is false, so sparse observations are not hard-overwritten. "
    "The native L1 and L2 terms clamp to the source maximum and mask gt<=1e-4; "
    "zero-coded KITTI unlabelled pixels are excluded, hence the K method is "
    "CompletionFormer-K rather than K-Naive. N and K consume only registered "
    "fixed sparse depth at 10 m and original LiDAR at 80 m. RGB and dep are the "
    "only forward inputs; intrinsics are updated by the shared dataset geometry "
    "but not consumed by the model. Runtime inputs are deterministically resized "
    "to 228x304; the model has no additional divisibility requirement, so padding "
    "is zero. Unified OOD uses one source-domain configuration, no TTA, no target "
    "tuning and no test-time refinement switch. Checkpoints are ordinary strict "
    "model state dictionaries with no receipt, SHA gate, legacy flag or key remap."
)
OPTIMIZER_NOTES = (
    "native AdamW(lr=1e-3, betas=(0.9,0.999), eps=1e-8, "
    "weight_decay=0.01), first-epoch linear warm-up and MultiStepLR(gamma=0.5); "
    "N milestones 36/48/56/64, K milestones 150/180/210/240"
)


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _ensure_src_path():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def asset_status():
    missing = [str(path) for path in _PRETRAINED.values() if not path.is_file()]
    return {
        "status": "BLOCKED_ASSETS" if missing else "READY",
        "required": [str(path) for path in _PRETRAINED.values()],
        "missing": missing,
    }


def extension_status():
    """Inspect the mandatory released DCNv2 module without compiling it."""
    spec = importlib.util.find_spec("DCN")
    if spec is None:
        return {
            "status": "BLOCKED_EXTENSION", "module": "DCN", "path": None,
            "error": "official CompletionFormer DCNv2 extension is not installed",
        }
    try:
        module = importlib.import_module("DCN")
    except (ImportError, OSError, RuntimeError) as exc:
        return {
            "status": "BLOCKED_ENVIRONMENT", "module": "DCN",
            "path": getattr(spec, "origin", None), "error": str(exc),
        }
    if not torch.cuda.is_available():
        return {
            "status": "BLOCKED_ENVIRONMENT", "module": "DCN",
            "path": getattr(module, "__file__", getattr(spec, "origin", None)),
            "error": "DCN is present but the active PyTorch runtime has no CUDA",
        }
    return {
        "status": "READY", "module": "DCN",
        "path": getattr(module, "__file__", getattr(spec, "origin", None)),
        "error": None,
    }


def dependency_status():
    missing = [name for name in _REQUIRED_DEPENDENCIES
               if importlib.util.find_spec(name) is None]
    return {
        "status": "BLOCKED_ENVIRONMENT" if missing else "READY",
        "required": list(_REQUIRED_DEPENDENCIES),
        "missing": missing,
    }


def runtime_status():
    assets = asset_status()
    extension = extension_status()
    dependencies = dependency_status()
    if assets["status"] != "READY":
        status = assets["status"]
    elif extension["status"] != "READY":
        status = extension["status"]
    else:
        status = dependencies["status"]
    return {"status": status, "assets": assets, "extension": extension,
            "dependencies": dependencies}


def _require_runtime():
    status = runtime_status()
    if status["status"] == "BLOCKED_ASSETS":
        raise RuntimeError(
            "CompletionFormer is BLOCKED_ASSETS; missing released initializer: "
            + ", ".join(status["assets"]["missing"]))
    if status["status"] == "BLOCKED_EXTENSION":
        raise RuntimeError(
            "CompletionFormer is BLOCKED_EXTENSION; compile the unchanged official "
            "DCNv2 extension in src/model/deformconv on the target CUDA/PyTorch "
            "runtime. Refinement cannot be disabled or replaced.")
    if status["status"] == "BLOCKED_ENVIRONMENT":
        details = status["extension"].get("error") or (
            "missing dependencies: " + ", ".join(status["dependencies"]["missing"]))
        raise RuntimeError(
            "CompletionFormer is BLOCKED_ENVIRONMENT; the released runtime is "
            f"not usable: {details}")
    return status


def add_method_args(parser):
    parser.add_argument(
        "--completionformer-training-domain", default="auto",
        choices=("auto", "N", "K"),
        help="source training domain; OOD evaluation requires explicit N or K")
    parser.add_argument(
        "--completionformer-ood-policy", default="unified", choices=("unified",),
        help="one fixed source-domain configuration for every OOD-15 target")
    parser.set_defaults(epochs=_N_EPOCHS, lr=_LR, val_best_metric="RMSE_m")


def _resolve_training_domain(args, mode):
    requested = str(getattr(
        args, "completionformer_training_domain", "auto"))
    dataset_type = str(getattr(args, "_dataset_type", "nyu"))
    registered = "N" if dataset_type == "nyu" else (
        "K" if dataset_type == "kitti" else None)
    if requested == "auto":
        if registered is None:
            raise ValueError(
                "CompletionFormer OOD evaluation requires "
                "--completionformer-training-domain N or K")
        requested = registered
    if registered is not None and requested != registered:
        raise ValueError(
            f"CompletionFormer domain mismatch: dataset type {dataset_type} "
            f"requires {registered}, got {requested}")
    if mode == "train" and registered is None:
        raise RuntimeError(
            "CompletionFormer can train only on registered N or K source data")
    return requested


def method_version(args):
    domain = str(getattr(args, "completionformer_training_domain", "N"))
    suffix = "-Unified" if str(getattr(args, "_dataset_type", "nyu")) == "ood_npz" else ""
    return f"CompletionFormer-{domain}{suffix}"


def configure_args(args, mode):
    domain = _resolve_training_domain(args, mode)
    expected_scale = _SOURCE_MAX_DEPTH_M[domain]
    configured_scale = getattr(args, "depth_unit_scale", None)
    dataset_type = str(getattr(args, "_dataset_type", "nyu"))
    if (dataset_type in ("nyu", "kitti") and configured_scale is not None
            and float(configured_scale) != expected_scale):
        raise ValueError(
            f"CompletionFormer-{domain} requires depth_unit_scale={expected_scale:g}, "
            f"got {configured_scale}")
    if configured_scale is None:
        args.depth_unit_scale = expected_scale
    elif float(configured_scale) <= 0:
        raise ValueError("CompletionFormer depth_unit_scale must be positive")
    args.completionformer_training_domain = domain
    args.training_domain = domain
    args.completionformer_runtime_mode = str(mode)
    args.completionformer_native_max_depth = expected_scale
    args.completionformer_ood_policy = "unified"
    args.method_name = method_version(args)
    args.method_version = args.method_name
    args.prediction_npy_unit = PREDICTION_NPY_UNIT
    args.network_input_height, args.network_input_width = _NETWORK_INPUT_HW
    args.evaluation_height, args.evaluation_width = _NETWORK_INPUT_HW
    args.prediction_spatial_resolution = list(_NETWORK_INPUT_HW)
    args.rgb_interpolation = "bilinear"
    args.depth_interpolation = "nearest"
    args.mask_interpolation = "nearest"
    args.per_image_normalization = False
    args.strict_checkpoint = True
    args.primary_output = "pred"
    args.intrinsics_policy = "updated_for_geometry_but_not_used_by_forward"
    args.propagation_iterations = _PROPAGATION_ITERATIONS
    args.propagation_padding = list(_PADDING)
    args.padding = list(_PADDING)
    args.input_multiple = list(_INPUT_MULTIPLE)
    args.refinement_enabled = _REFINEMENT_ENABLED
    args.runtime_status = runtime_status()["status"]
    if mode == "train":
        if getattr(args, "checkpoint", None):
            raise RuntimeError(
                "CompletionFormer formal N/K training starts from the released "
                "backbone initializers; task checkpoint initialization is forbidden")
        stage = getattr(args, "kitti_stage", None)
        if domain == "N":
            if int(args.epochs) != _N_EPOCHS:
                raise ValueError("CompletionFormer-N locks the native 72-epoch schedule")
        elif stage == "develop":
            if int(args.epochs) not in (_N_EPOCHS, _K_DEVELOP_EPOCH_CAP):
                raise ValueError(
                    "CompletionFormer-K develop locks the protocol 50-epoch cap")
            args.epochs = _K_DEVELOP_EPOCH_CAP
        elif stage == "final-train":
            fixed = getattr(args, "fixed_epochs", None)
            if fixed is not None:
                fixed = int(fixed)
                if fixed < 1 or fixed > _K_DEVELOP_EPOCH_CAP:
                    raise ValueError(
                        "CompletionFormer-K final fixed epoch must be selected "
                        "within the Stage-A range 1..50")
                args.epochs = fixed
        if float(args.lr) != _LR:
            raise ValueError("CompletionFormer locks the native learning rate to 0.001")
        args.batch_size = TRAIN_BATCH_SIZE
        args.grad_accum_steps = 1
        args.amp = False
    else:
        args.batch_size = EVAL_BATCH_SIZE
    return args


def fairness_metadata(args, mode):
    del mode
    domain = str(getattr(args, "completionformer_training_domain", "N"))
    status = runtime_status()
    return {
        "method": METHOD_NAME,
        "version": method_version(args),
        "training_domain": domain,
        "seed": int(getattr(args, "seed", 42)),
        "fairness_class": "CONTROLLED_NATIVE",
        "controlled_ranking": True,
        "unfair_reason": None,
        "native_loss": _LOSS_SPEC,
        "valid_pixel_masking": True,
        "zero_coded_unlabelled_pixels_in_loss": False,
        "external_pretraining": True,
        "pretrained_backbone": "ImageNet ResNet-34 plus PVT",
        "pretrained_weight_path": [
            "src/pretrained/resnet34.pth", "src/pretrained/pvt.pth"],
        "foundation_model": None,
        "official_mixed_training": False,
        "custom_extension": True,
        "custom_cuda_extension": True,
        "runtime_status": status["status"],
        "intrinsics_required": False,
        "intrinsics_policy": "updated_for_geometry_but_not_used_by_forward",
        "refinement_enabled": _REFINEMENT_ENABLED,
        "preserve_sparse_input": _PRESERVE_SPARSE_INPUT,
        "input_multiple": list(_INPUT_MULTIPLE),
        "padding": list(_PADDING),
        "propagation_iterations": _PROPAGATION_ITERATIONS,
        "affinity_type": _AFFINITY_TYPE,
        "confidence_propagation": _CONFIDENCE_PROPAGATION,
        "TTA": False,
        "OOD_specific_tuning": False,
        "target_overlap": False,
        "loss_modified": False,
        "network_modified": False,
        "optimizer_modified": False,
        "scheduler_modified": False,
        "training_epoch_budget_modified": domain == "K",
        "official_k_epoch_budget": _K_NATIVE_EPOCHS,
        "k_develop_epoch_cap": _K_DEVELOP_EPOCH_CAP,
        "formal_protocol_variant": (
            "K_develop_50_epoch_cap" if domain == "K"
            else "N_native_72_epochs"),
        "primary_output": "pred (final depth after six native propagation steps)",
        "notes": PROTOCOL_NOTES,
    }


def _official_args(args):
    return SimpleNamespace(
        model="CompletionFormer",
        prop_time=_PROPAGATION_ITERATIONS,
        prop_kernel=_PROPAGATION_KERNEL,
        preserve_input=_PRESERVE_SPARSE_INPUT,
        affinity=_AFFINITY_TYPE,
        affinity_gamma=_AFFINITY_GAMMA,
        conf_prop=_CONFIDENCE_PROPAGATION,
        legacy=False,
        lr=_LR,
        max_depth=float(args.completionformer_native_max_depth),
        loss=_LOSS_SPEC,
    )


class CompletionFormerModelV2(nn.Module):
    """Register native loss modules around the untouched released network."""

    def __init__(self, net, native_loss, args):
        super().__init__()
        self.net = net
        self.native_loss = native_loss
        self.native_loss_modules = native_loss.loss_module
        self.shared_depth_scale = float(args.depth_unit_scale)
        self.training_domain = str(args.completionformer_training_domain)
        self.geolayernet_degraded_note = "none; native refinement is mandatory"

    def forward(self, sample):
        return self.net(sample)


def build_model(args, device):
    del device
    _require_runtime()
    _ensure_src_path()
    native_args = _official_args(args)
    try:
        with _working_directory(_SRC):
            from model.completionformer import CompletionFormer
            from loss.l1l2loss import L1L2Loss
            net = CompletionFormer(native_args)
            native_loss = L1L2Loss(native_args)
    except ModuleNotFoundError as exc:
        if exc.name == "DCN":
            raise RuntimeError(
                "CompletionFormer is BLOCKED_EXTENSION; official DCNv2 is "
                "required and refinement has not been disabled.") from exc
        raise RuntimeError(
            f"CompletionFormer is BLOCKED_ENVIRONMENT; missing dependency {exc.name}") from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "CompletionFormer is BLOCKED_ENVIRONMENT; the released model or "
            f"extension cannot be constructed: {exc}") from exc
    return CompletionFormerModelV2(net, native_loss, args)


def _resize_tensor(value, size, mode):
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    if mode == "bilinear":
        return F.interpolate(value, size=size, mode=mode, align_corners=False)
    return F.interpolate(value, size=size, mode=mode)


def _resize_intrinsics(intrinsics, source_hw, target_hw):
    out = intrinsics.clone().to(dtype=torch.float32)
    sy = float(target_hw[0]) / float(source_hw[0])
    sx = float(target_hw[1]) / float(source_hw[1])
    out[..., 0, 0] *= sx
    out[..., 0, 2] *= sx
    out[..., 1, 1] *= sy
    out[..., 1, 2] *= sy
    return out


def prepare_batch_for_model(batch, device):
    del device
    if "rgb" not in batch or "sparse_depth" not in batch:
        raise KeyError("CompletionFormer batch requires rgb and sparse_depth")
    source_hw = tuple(batch["rgb"].shape[-2:])
    target_hw = _NETWORK_INPUT_HW
    out = dict(batch)
    out["rgb"] = _resize_tensor(batch["rgb"], target_hw, "bilinear")
    out["sparse_depth"] = _resize_tensor(
        batch["sparse_depth"], target_hw, "nearest")
    if "target_depth" in batch:
        out["target_depth"] = _resize_tensor(
            batch["target_depth"], target_hw, "nearest")
    out["valid_mask"] = (out["sparse_depth"] > 0).to(
        out["sparse_depth"].dtype)
    if "target_depth" in out:
        target = out["target_depth"]
        target_valid = torch.isfinite(target) & (target > 0) & (target <= 1.0)
        if "target_mask" in batch:
            source_mask = _resize_tensor(
                batch["target_mask"].float(), target_hw, "nearest") > 0.5
            target_valid = target_valid & source_mask
        out["target_mask"] = target_valid.to(target.dtype)
    if "intrinsics" in batch:
        out["intrinsics"] = _resize_intrinsics(
            batch["intrinsics"], source_hw, target_hw)
    out["network_input_size"] = target_hw
    out["evaluation_size"] = target_hw
    out["padding"] = _PADDING
    return out


def _native_inputs(model, batch, device, include_target=False):
    rgb = batch["rgb"].to(device=device, dtype=torch.float32)
    sparse = batch["sparse_depth"].to(device=device, dtype=torch.float32)
    if rgb.ndim != 4 or rgb.size(1) != 3:
        raise ValueError("CompletionFormer expects rgb[B,3,H,W]")
    if sparse.ndim != 4 or sparse.size(1) != 1:
        raise ValueError("CompletionFormer expects sparse_depth[B,1,H,W]")
    if rgb.shape[0] != sparse.shape[0] or rgb.shape[-2:] != sparse.shape[-2:]:
        raise ValueError("CompletionFormer requires aligned RGB and sparse depth")
    if tuple(rgb.shape[-2:]) != _NETWORK_INPUT_HW:
        raise ValueError(
            "CompletionFormer adapter requires prepare_batch_for_model -> 228x304")
    if not torch.isfinite(rgb).all() or not torch.isfinite(sparse).all():
        raise ValueError("CompletionFormer inputs must be finite")
    if torch.any(sparse < 0):
        raise ValueError("CompletionFormer sparse depth must be non-negative")
    rgb01 = (rgb * 0.5 + 0.5).clamp(0.0, 1.0)
    mean = rgb01.new_tensor(_RGB_MEAN).view(1, 3, 1, 1)
    std = rgb01.new_tensor(_RGB_STD).view(1, 3, 1, 1)
    sample = {
        "rgb": (rgb01 - mean) / std,
        "dep": sparse * model.shared_depth_scale,
    }
    if include_target:
        target = batch["target_depth"].to(device=device, dtype=torch.float32)
        if target.shape != sparse.shape:
            raise ValueError("CompletionFormer target_depth must align with sparse depth")
        if not torch.isfinite(target).all() or torch.any(target < 0):
            raise ValueError(
                "CompletionFormer target depth must be finite and non-negative")
        sample["gt"] = target * model.shared_depth_scale
    return sample


def _prediction(output):
    if not isinstance(output, dict) or "pred" not in output:
        raise RuntimeError(
            "official CompletionFormer forward must return dict['pred']")
    pred_m = output["pred"]
    if pred_m.ndim == 3:
        pred_m = pred_m.unsqueeze(1)
    if pred_m.ndim != 4 or pred_m.size(1) != 1:
        raise ValueError(
            f"CompletionFormer prediction must be [B,1,H,W], got {tuple(pred_m.shape)}")
    if tuple(pred_m.shape[-2:]) != _NETWORK_INPUT_HW:
        raise RuntimeError(
            "CompletionFormer refined output must remain 228x304, got "
            f"{tuple(pred_m.shape[-2:])}")
    if not torch.isfinite(pred_m).all():
        raise RuntimeError("CompletionFormer produced non-finite metric depth")
    return pred_m


def forward_completed(model, batch, device):
    output = model(_native_inputs(model, batch, device, include_target=False))
    return _prediction(output) / model.shared_depth_scale


def train_step(model, batch, device, epoch):
    del epoch
    sample = _native_inputs(model, batch, device, include_target=True)
    output = model(sample)
    loss_sum, _loss_values = model.native_loss(sample, output)
    loss = loss_sum / float(sample["rgb"].size(0))
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise RuntimeError(
            "CompletionFormer native L1+L2 returned an invalid loss")
    return loss


def make_optimizer(model, args):
    return torch.optim.AdamW(
        model.net.parameters(), lr=float(args.lr), betas=_BETAS,
        eps=_EPSILON, weight_decay=_WEIGHT_DECAY)


def _milestones(args):
    domain = str(args.completionformer_training_domain)
    return list(_N_MILESTONES if domain == "N" else _K_MILESTONES)


def make_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=_milestones(args), gamma=_GAMMA)


def before_train_step(model, optimizer, args, epoch, iteration, loader_length):
    del model, args
    if int(epoch) != 1:
        return
    for group in optimizer.param_groups:
        initial = float(group.get("initial_lr", _LR))
        group["lr"] = initial * float(iteration + 1) / float(loader_length + 1)


def checkpoint_payload(model, optimizer, epoch):
    del optimizer, epoch
    return model.net.state_dict()


def _is_tensor_state_dict(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items())


def _extract_state_dict(payload):
    if _is_tensor_state_dict(payload):
        weights = payload
    elif isinstance(payload, dict):
        recognized = [key for key in ("state_dict", "model", "net")
                      if key in payload and _is_tensor_state_dict(payload[key])]
        if len(recognized) != 1:
            raise RuntimeError(
                "CompletionFormer checkpoint must contain exactly one recognized "
                "tensor state_dict key (state_dict/model/net)")
        weights = payload[recognized[0]]
    else:
        raise RuntimeError(
            "CompletionFormer checkpoint is not a state_dict mapping")
    prefixed = [key.startswith("module.") for key in weights]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError(
            "CompletionFormer checkpoint has a mixed module. prefix")
    if all(prefixed):
        return OrderedDict(
            (key[len("module."):], value) for key, value in weights.items())
    return OrderedDict(weights.items())


def load_checkpoint(model, checkpoint, device, args, strict=False):
    del args, strict
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"CompletionFormer checkpoint does not exist: {path}")
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    weights = _extract_state_dict(payload)
    result = model.net.load_state_dict(weights, strict=True)
    print(
        f"[checkpoint] CompletionFormer strict load: {path} "
        f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    return result

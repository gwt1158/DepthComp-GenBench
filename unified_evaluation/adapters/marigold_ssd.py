"""Comparison Protocol V2 bridge for the released Marigold-SSD model.

Only the registered data and checkpoint interfaces live here.  The official
UnetDecMid, conditional VAE decoder, one-step diffusion forward, masked L1,
AdamW groups, IterExponential schedule, VaePipeline and sparse-only least
squares alignment are imported from ``ssd`` without changing their equations.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


METHOD_NAME = "Marigold-SSD"
COMPARISON_BUCKET = "external_foundation_same_data_retraining"
SUPERVISION = (
    "N-only or K-only registered dense GT through official masked relative L1")
PRIOR = "GonzaloMG/marigold-e2e-ft-depth external diffusion foundation"
PROTOCOL_STATUS = "FROZEN"
REQUIRES_CHECKPOINT = True
REQUIRE_KITTI_TWO_STAGE = True
REQUIRE_KITTI_FIXED_E_ONLY = True
SKIP_SHARED_CHECKPOINT_PROVENANCE = True
TRAIN_BATCH_SIZE = 1
TRAIN_NUM_WORKERS = 8
TRAIN_AMP = False
EVAL_BATCH_SIZE = 1
EVAL_NUM_WORKERS = 4
DROP_INCOMPLETE_ACCUMULATION = True
SCHEDULER_STEP_PER_BATCH = True
SKIP_MODEL_STATS = True
PROFILE_SPEED = False
PREDICTION_NPY_UNIT = "metres"
ERROR_MAX_CLI_UNIT = "metres"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = WORKSPACE_ROOT / "marigold-ssd-main"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
_DEFAULT_FOUNDATION = (
    METHOD_ROOT / "weights" / "GonzaloMG_marigold-e2e-ft-depth")
_EVALUATION_HW = (228, 304)
_PADDED_HW = (232, 304)
_INPUT_MULTIPLE = 8
_EPOCHS = 15
_UNET_LR = 3e-6
_VAE_LR = 3e-5
_WEIGHT_DECAY = 1e-2
_GRAD_ACCUMULATION = 32
_WARMUP_STEPS = 100
_SCHEDULE_STEPS = 20_000
_DIFFUSION_STEPS = 1
_ENSEMBLE_SIZE = 1
_NOISE = "zeros"
_TIMESTEP_SPACING = "trailing"
_NATIVE_LOSS = (
    "official ssd.loss.L1Loss(prediction[GT_METRIC_MASK], "
    "GT_RELATIVE[GT_METRIC_MASK])")

PROTOCOL_NOTES = (
    "Marigold-SSD is an RGB-guided sparse metric depth-completion method. "
    "V2 changes only the dataset/checkpoint bridge: N uses the fixed on-disk "
    "1 percent sparse map and K uses original KITTI LiDAR; no GT-derived "
    "sparse resampling is used. Training retains official q02-q98 relative "
    "target preprocessing and masked L1, so finite positive valid GT only is "
    "supervised and KITTI zero-coded unlabelled pixels are excluded. Inference "
    "retains official one-step DDIM with trailing timesteps, one-member "
    "ensemble and zero latent. Metric scale and shift are fitted per image by "
    "official align_depth_least_square using sparse input only; target GT is "
    "never passed to inference or alignment. The mandatory Marigold E2E "
    "foundation makes this EXTERNAL_PRIOR and excludes it from controlled "
    "ranking. The released 20k-update synthetic-mixture launcher is recorded; "
    "the formal same-data N/K protocol uses the released 15-epoch CLI budget "
    "and KITTI selects a fixed epoch E. All 15 current OOD domains are "
    "measurable without intrinsics (8888 predictions); ScanNet is not "
    "evaluated and Macro-16 is unavailable."
)
OPTIMIZER_NOTES = (
    "official AdamW: UNet lr=3e-6; VAE decoder and post_quant_conv lr=3e-5; "
    "betas=(0.9,0.999), eps=1e-8, weight_decay=0.01; official "
    "IterExponential(total=20000,warmup=100,final_ratio=0.01) per update")


def asset_status(path=None):
    root = Path(path or _DEFAULT_FOUNDATION).expanduser().resolve()
    required_files = [
        root / "scheduler" / "scheduler_config.json",
        root / "tokenizer" / "tokenizer_config.json",
        root / "tokenizer" / "vocab.json",
        root / "tokenizer" / "merges.txt",
        root / "text_encoder" / "config.json",
        root / "vae" / "config.json",
        root / "unet" / "config.json",
    ]
    weight_groups = [
        (root / "text_encoder" / "model.safetensors",
         root / "text_encoder" / "pytorch_model.bin"),
        (root / "vae" / "diffusion_pytorch_model.safetensors",
         root / "vae" / "diffusion_pytorch_model.bin"),
        (root / "unet" / "diffusion_pytorch_model.safetensors",
         root / "unet" / "diffusion_pytorch_model.bin"),
    ]
    missing = [str(item) for item in required_files if not item.is_file()]
    missing.extend(
        "one_of(" + ", ".join(str(item) for item in group) + ")"
        for group in weight_groups if not any(item.is_file() for item in group))
    required = [str(item) for item in required_files]
    required.extend(
        "one_of(" + ", ".join(str(item) for item in group) + ")"
        for group in weight_groups)
    return {
        "status": "READY" if not missing else "BLOCKED_ASSETS",
        "root": str(root),
        "required": required,
        "missing": missing,
        "foundation_model": "GonzaloMG/marigold-e2e-ft-depth",
        "official_task_checkpoint_required": False,
        "official_task_checkpoint_used": False,
        "network_access_allowed": False,
    }


def extension_status():
    return {
        "status": "NOT_REQUIRED", "module": None,
        "mandatory": False, "fallback": "forbidden"}


def runtime_status(path=None):
    assets = asset_status(path)
    return {
        "status": (
            "READY_FOR_SERVER_SMOKE"
            if assets["status"] == "READY" else "BLOCKED_ASSETS"),
        "assets": assets,
        "extension": extension_status(),
    }


def add_method_args(parser):
    parser.add_argument(
        "--marigold-ssd-training-domain", default="auto",
        choices=("auto", "N", "K"),
        help="source training domain; OOD evaluation requires explicit N or K")
    parser.add_argument(
        "--marigold-ssd-foundation", default=str(_DEFAULT_FOUNDATION),
        help=("complete local GonzaloMG/marigold-e2e-ft-depth snapshot; "
              "network download and official task checkpoints are forbidden"))
    parser.add_argument(
        "--marigold-ssd-ood-policy", default="unified", choices=("unified",),
        help="one fixed source checkpoint and inference policy for OOD-15")
    parser.set_defaults(
        epochs=_EPOCHS, lr=_UNET_LR, batch_size=TRAIN_BATCH_SIZE,
        grad_accum_steps=_GRAD_ACCUMULATION, amp=False,
        val_best_metric="RMSE_m", val_best_mode="min", val_freq=1,
        clip_prediction=False)


def _resolve_training_domain(args, mode):
    requested = str(getattr(args, "marigold_ssd_training_domain", "auto"))
    dataset_type = str(getattr(args, "_dataset_type", "nyu"))
    registered = (
        "N" if dataset_type == "nyu" else
        "K" if dataset_type == "kitti" else None)
    if requested == "auto":
        if registered is None:
            raise ValueError(
                "Marigold-SSD OOD evaluation requires explicit "
                "--marigold-ssd-training-domain N or K")
        requested = registered
    if registered is not None and requested != registered:
        raise ValueError(
            f"Marigold-SSD domain mismatch: {dataset_type} requires "
            f"{registered}, got {requested}")
    if mode == "train" and registered is None:
        raise RuntimeError("Marigold-SSD can train only on registered N or K data")
    return requested


def method_version(args):
    domain = str(getattr(args, "marigold_ssd_training_domain", "N"))
    suffix = (
        "-Unified" if str(getattr(args, "_dataset_type", "nyu")) == "ood_npz"
        else "")
    return f"Marigold-SSD-{domain}{suffix}"


def _require_equal(name, actual, expected):
    if actual != expected:
        raise ValueError(
            f"Marigold-SSD formal protocol fixes {name}={expected!r}; "
            f"got {actual!r}")


def configure_args(args, mode):
    domain = _resolve_training_domain(args, mode)
    dataset_type = str(getattr(args, "_dataset_type", "nyu"))
    if dataset_type == "nyu":
        _require_equal("depth_unit_scale", float(args.depth_unit_scale), 10.0)
    elif dataset_type == "kitti":
        _require_equal("depth_unit_scale", float(args.depth_unit_scale), 80.0)
    elif float(args.depth_unit_scale) <= 0:
        raise ValueError("OOD dataset depth_unit_scale must be positive")
    args.marigold_ssd_training_domain = domain
    args.training_domain = domain
    args.method_name = method_version(args)
    args.method_version = args.method_name
    args.marigold_ssd_ood_policy = "unified"
    args.prediction_npy_unit = PREDICTION_NPY_UNIT
    args.network_input_height, args.network_input_width = _PADDED_HW
    args.evaluation_height, args.evaluation_width = _EVALUATION_HW
    args.prediction_spatial_resolution = list(_EVALUATION_HW)
    args.rgb_interpolation = "bilinear"
    args.depth_interpolation = "nearest"
    args.mask_interpolation = "nearest"
    args.per_image_normalization = False
    args.strict_checkpoint = True
    args.input_multiple = _INPUT_MULTIPLE
    args.padding = [0, 0, 4, 0]
    args.primary_output = (
        "official VaePipeline depth_np after sparse-only least-squares alignment")
    args.intrinsics_policy = "not_consumed"
    args.sparse_depth_used = True
    args.sensor_type = "registered_on_disk_metric_sparse_depth"
    args.sensor_model = "generic_sparse_depth_condition"
    args.noise_model = None
    args.camera_model = "camera_model_agnostic_2d_forward"
    args.uncertainty_available = False
    args.diffusion_steps = _DIFFUSION_STEPS
    args.diffusion_model = "Marigold-SSD one-step latent diffusion"
    args.ensemble_size = _ENSEMBLE_SIZE
    args.processing_resolution = list(_PADDED_HW)
    args.guidance_scale = None
    args.latent_initialization = _NOISE
    args.seed_policy = "seed42_recorded_zero_latent_deterministic_forward"
    args.sparse_conditioning = "per_sample_sparse_minmax_to_minus1_plus1"
    args.confidence_threshold = None
    args.metric_alignment = "sparse_depth_least_square_scale_shift"
    args.monocular_model = None
    args.mono_model_input_size = None
    args.alignment_method = "official_sparse_least_square_scale_shift"
    args.alignment_point_count = None
    args.alignment_scale = None
    args.alignment_shift = None
    args.alignment_fallback = False
    args.runtime_status = runtime_status(args.marigold_ssd_foundation)["status"]
    args._gln_mode = str(mode)
    if mode == "train":
        if getattr(args, "checkpoint", None):
            raise RuntimeError(
                "formal N/K training starts from the local Marigold foundation; "
                "task-checkpoint initialization is forbidden")
        stage = getattr(args, "kitti_stage", None)
        if not (domain == "K" and stage == "final-train"):
            _require_equal("epochs", int(args.epochs), _EPOCHS)
        if not math.isclose(float(args.lr), _UNET_LR, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Marigold-SSD locks UNet lr=3e-6")
        _require_equal("physical batch size", int(args.batch_size), 1)
        _require_equal(
            "gradient accumulation", int(args.grad_accum_steps),
            _GRAD_ACCUMULATION)
        _require_equal("AMP", bool(args.amp), False)
        _require_equal("validation metric", str(args.val_best_metric), "RMSE_m")
        _require_equal("validation mode", str(args.val_best_mode), "min")
        _require_equal("validation frequency", int(args.val_freq), 1)
    else:
        args.batch_size = EVAL_BATCH_SIZE
    return args


def fairness_metadata(args, mode):
    del mode
    domain = str(getattr(args, "marigold_ssd_training_domain", "N"))
    dataset_type = str(getattr(args, "_dataset_type", "nyu"))
    sparse_distribution = {
        "nyu": "fixed_on_disk_1_percent",
        "kitti": "original_KITTI_LiDAR",
        "ood_npz": "registered_per_domain_on_disk_sparse_depth",
    }.get(dataset_type, "registered_on_disk_sparse_depth")
    return {
        "method": METHOD_NAME,
        "version": method_version(args),
        "training_domain": domain,
        "seed": int(getattr(args, "seed", 42)),
        "fairness_class": "EXTERNAL_PRIOR",
        "controlled_ranking": False,
        "unfair_reason": (
            "mandatory GonzaloMG/marigold-e2e-ft-depth diffusion foundation "
            "pretraining is not shared by controlled native methods"),
        "rgb_used": True,
        "sparse_depth_used": True,
        "sparse_mask_used": True,
        "sensor_type": "dataset_native_on_disk_sparse_depth",
        "sensor_model": "generic_sparse_depth_condition",
        "sensor_distribution": sparse_distribution,
        "synthetic_sparse_training": False,
        "official_synthetic_sparse_training": (
            "Hypersim/VKITTI dense GT RandomPointsFromDepth; not used by V2"),
        "noise_model": None,
        "camera_model": "camera_model_agnostic_2d_forward",
        "external_pretraining": True,
        "foundation_model": "GonzaloMG/marigold-e2e-ft-depth",
        "foundation_checkpoint": str(Path(getattr(
            args, "marigold_ssd_foundation", _DEFAULT_FOUNDATION))),
        "foundation_training_data": "Marigold E2E foundation; upstream provenance",
        "pretrained_backbone": "Marigold latent diffusion foundation",
        "target_overlap": True,
        "target_overlap_details": (
            "foundation/official SSD training families include synthetic "
            "Hypersim and Virtual KITTI; report Hypersim and VKITTI2 OOD "
            "results as external-prior overlap-risk, not controlled ranking"),
        "native_loss": _NATIVE_LOSS,
        "loss_valid_masks": {
            "L1Loss": (
                "finite target_m>0 within domain max, registered target_mask, "
                "then inclusive q02-q98 GT_METRIC_MASK")},
        "valid_pixel_masking": True,
        "zero_coded_unlabelled_pixels_in_loss": False,
        "official_mixed_training": False,
        "TTA": False,
        "OOD_specific_tuning": False,
        "loss_modified": False,
        "network_modified": False,
        "optimizer_modified": False,
        "scheduler_modified": False,
        "primary_output": args.primary_output,
        "intrinsics_required": False,
        "intrinsics_policy": "not_consumed",
        "uncertainty_output": False,
        "uncertainty_available": False,
        "confidence_output": False,
        "diffusion_steps": _DIFFUSION_STEPS,
        "diffusion_model": "Marigold-SSD one-step latent diffusion",
        "ensemble_size": _ENSEMBLE_SIZE,
        "processing_resolution": list(_PADDED_HW),
        "guidance_scale": None,
        "latent_initialization": _NOISE,
        "seed_policy": "seed42_recorded_zero_latent_deterministic_forward",
        "sparse_conditioning": "per_sample_sparse_minmax_to_minus1_plus1",
        "confidence_threshold": None,
        "metric_alignment": "sparse_depth_least_square_scale_shift",
        "scale_alignment": True,
        "shift_alignment": True,
        "alignment_source": "registered_sparse_depth_only",
        "alignment_per_sample": True,
        "alignment_method": "official_sparse_least_square_scale_shift",
        "alignment_point_count": None,
        "alignment_scale": None,
        "alignment_shift": None,
        "alignment_fallback": False,
        "monocular_prior": False,
        "monocular_model": None,
        "mono_model_input_size": None,
        "test_time_optimization": False,
        "network_input_size": list(_PADDED_HW),
        "evaluation_size": list(_EVALUATION_HW),
        "padding": [0, 0, 4, 0],
        "measurable_domain_count": 15,
        "expected_prediction_count": 8888,
        "not_measurable_domains": [],
        "not_measurable_reason": None,
        "ScanNet": "not_evaluated",
        "Macro16": "unavailable",
        "runtime_status": runtime_status(getattr(
            args, "marigold_ssd_foundation", _DEFAULT_FOUNDATION))["status"],
        "notes": PROTOCOL_NOTES,
    }


class MarigoldSSDV2(nn.Module):
    def __init__(self, unet, vae, text_encoder, tokenizer, train_scheduler,
                 eval_scheduler, empty_encoding, args):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.register_buffer("empty_encoding", empty_encoding)
        self.register_buffer(
            "alpha_prod", train_scheduler.alphas_cumprod.detach().float())
        self.register_buffer("beta_prod", 1.0 - self.alpha_prod)
        object.__setattr__(self, "_text_encoder", text_encoder)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_train_scheduler", train_scheduler)
        object.__setattr__(self, "_eval_scheduler", eval_scheduler)
        self.runtime_depth_scale = float(args.depth_unit_scale)
        self.training_domain = str(args.marigold_ssd_training_domain)
        self.last_alignment = []
        self.geolayernet_degraded_note = (
            "external foundation; official one-step diffusion and sparse-only "
            "alignment; registered data bridge only")


def build_model(args, device):
    foundation = Path(args.marigold_ssd_foundation).expanduser().resolve()
    status = asset_status(foundation)
    if status["status"] != "READY":
        raise FileNotFoundError(
            "BLOCKED_ASSETS: incomplete local "
            f"GonzaloMG/marigold-e2e-ft-depth snapshot: {foundation}; "
            "missing=" + ", ".join(status["missing"]))
    try:
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        from ssd.model.depth_autoencoder_mid import DepthAutoencoderMid
        from ssd.model.unetdecmid import UnetDecMid
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "BLOCKED_ENVIRONMENT: Marigold-SSD V2 requires its official "
            f"Python environment: {exc!r}") from exc
    try:
        load = {"local_files_only": True}
        train_scheduler = DDPMScheduler.from_pretrained(
            foundation, subfolder="scheduler", **load)
        eval_scheduler = DDIMScheduler.from_pretrained(
            foundation, subfolder="scheduler",
            timestep_spacing=_TIMESTEP_SPACING, **load)
        tokenizer = CLIPTokenizer.from_pretrained(
            foundation, subfolder="tokenizer", **load)
        text_encoder = CLIPTextModel.from_pretrained(
            foundation, subfolder="text_encoder", **load)
        vae = DepthAutoencoderMid.from_pretrained(
            foundation, subfolder="vae", low_cpu_mem_usage=False,
            device_map=None, **load)
        unet = UNet2DConditionModel.from_pretrained(
            foundation, subfolder="unet", **load)
        initializer = UnetDecMid(unet, vae)
        initializer.initialize_fusion("zeros")
        initializer.initialize_condition_encoder()
    except Exception as exc:
        raise RuntimeError(
            "BLOCKED_ASSETS: failed to load the complete local Marigold "
            f"foundation without network access: {exc!r}") from exc
    text_encoder.requires_grad_(False)
    vae.encoder.requires_grad_(False)
    vae.quant_conv.requires_grad_(False)
    vae.post_quant_conv.requires_grad_(True)
    vae.decoder.requires_grad_(True)
    text_encoder = text_encoder.to(device=device, dtype=torch.float32)
    empty = tokenizer(
        [""], padding="max_length", truncation=True,
        return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        empty_encoding = text_encoder(empty, return_dict=False)[0].float()
    return MarigoldSSDV2(
        unet, vae, text_encoder, tokenizer, train_scheduler, eval_scheduler,
        empty_encoding, args)


def _resize(tensor, size, mode):
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    if mode == "bilinear":
        return F.interpolate(tensor.float(), size=size, mode=mode,
                             align_corners=False)
    return F.interpolate(tensor.float(), size=size, mode=mode)


def prepare_batch_for_model(batch, device):
    del device
    for key in ("rgb", "sparse_depth"):
        if key not in batch:
            raise KeyError(f"Marigold-SSD batch requires {key}")
    out = dict(batch)
    out["rgb"] = _resize(batch["rgb"], _EVALUATION_HW, "bilinear").clamp(-1, 1)
    sparse = _resize(batch["sparse_depth"], _EVALUATION_HW, "nearest")
    # The shared loader already expresses every registered sensor measurement
    # in the active runtime unit.  OOD measurements may legitimately exceed
    # the source-domain training range, so rebuilding the sensor mask must not
    # discard them merely because their normalized value is greater than one.
    sparse_valid = torch.isfinite(sparse) & (sparse > 0)
    out["sparse_depth"] = torch.where(
        sparse_valid, sparse, torch.zeros_like(sparse))
    out["valid_mask"] = sparse_valid.to(sparse.dtype)
    if "target_depth" in batch:
        target = _resize(batch["target_depth"], _EVALUATION_HW, "nearest")
        target_valid = torch.isfinite(target) & (target > 0) & (target <= 1.0)
        if "target_mask" in batch:
            target_valid &= _resize(
                batch["target_mask"].float(), _EVALUATION_HW,
                "nearest") > 0.5
        out["target_depth"] = torch.where(
            target_valid, target, torch.zeros_like(target))
        out["target_mask"] = target_valid.to(target.dtype)
    out["network_input_size"] = _PADDED_HW
    out["evaluation_size"] = _EVALUATION_HW
    return out


def _pad_model_input(tensor, value):
    height, width = tensor.shape[-2:]
    pad_h = (-height) % _INPUT_MULTIPLE
    pad_w = (-width) % _INPUT_MULTIPLE
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=value)


def _relative_training_target(target, registered_mask=None):
    relative = torch.full_like(target, -1.0)
    masks = torch.zeros_like(target, dtype=torch.bool)
    bounds = []
    for index in range(target.shape[0]):
        valid = torch.isfinite(target[index]) & (target[index] > 0)
        if registered_mask is not None:
            valid &= registered_mask[index] > 0.5
        values = target[index][valid]
        if values.numel() < 2:
            raise RuntimeError(
                "BLOCKED_SAMPLE: fewer than two finite positive GT pixels")
        low, high = torch.quantile(values, torch.tensor(
            [0.02, 0.98], device=values.device, dtype=values.dtype))
        if not bool(torch.isfinite(low + high)) or float(high - low) <= 1e-8:
            raise RuntimeError("BLOCKED_SAMPLE: q02-q98 target range is degenerate")
        mask = valid & (target[index] >= low) & (target[index] <= high)
        kept = target[index][mask]
        kept_low, kept_high = kept.min(), kept.max()
        if float(kept_high - kept_low) <= 1e-8:
            raise RuntimeError("BLOCKED_SAMPLE: masked target range is degenerate")
        relative[index][mask] = (
            -1.0 + 2.0 * (target[index][mask] - kept_low)
            / (kept_high - kept_low))
        masks[index] = mask
        bounds.append((kept_low, kept_high))
    return relative, masks, bounds


def _condition_from_bounds(sparse, bounds):
    condition = torch.full_like(sparse, -1.0)
    mask = torch.isfinite(sparse) & (sparse > 0)
    for index, (low, high) in enumerate(bounds):
        values = sparse[index][mask[index]]
        if values.numel() == 0:
            raise RuntimeError("BLOCKED_SAMPLE: registered sparse input is empty")
        condition[index][mask[index]] = (
            -1.0 + 2.0 * (values - low) / (high - low)).clamp(-1.0, 1.0)
    return condition


def _sparse_relative(sparse_m):
    condition = torch.full_like(sparse_m, -1.0)
    mask = torch.isfinite(sparse_m) & (sparse_m > 0)
    for index in range(sparse_m.shape[0]):
        values = sparse_m[index][mask[index]]
        if values.numel() < 2:
            raise RuntimeError(
                "BLOCKED_SAMPLE: Marigold-SSD alignment needs at least two "
                "registered sparse points")
        low, high = values.min(), values.max()
        if not bool(torch.isfinite(low + high)) or float(high - low) <= 1e-8:
            raise RuntimeError(
                "BLOCKED_SAMPLE: sparse-only normalization is degenerate")
        condition[index][mask[index]] = (
            -1.0 + 2.0 * (values - low) / (high - low))
    return condition, mask


def train_step(model, batch, device, epoch):
    del epoch
    from ssd.loss import L1Loss
    from ssd.model.unetdecmid import UnetDecMid

    rgb = batch["rgb"].to(device=device, dtype=torch.float32)
    sparse = batch["sparse_depth"].to(device=device, dtype=torch.float32)
    target = batch["target_depth"].to(device=device, dtype=torch.float32)
    registered = batch.get("target_mask")
    if registered is not None:
        registered = registered.to(device=device)
    target_relative, target_mask, bounds = _relative_training_target(
        target, registered)
    condition = _condition_from_bounds(sparse, bounds)
    rgb = _pad_model_input(rgb, -1.0)
    condition = _pad_model_input(condition, -1.0)
    core = UnetDecMid(model.unet, model.vae)
    estimate = core(
        rgb, condition, model._train_scheduler, model.empty_encoding,
        model.alpha_prod, model.beta_prod)
    estimate = estimate.mean(dim=1, keepdim=True).clamp(-1.0, 1.0)
    estimate = estimate[..., :_EVALUATION_HW[0], :_EVALUATION_HW[1]]
    loss = L1Loss()(estimate, target_relative, target_mask)
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise RuntimeError("official Marigold-SSD L1Loss returned invalid loss")
    return loss.reshape(())


def _make_eval_pipeline(model):
    from ssd.model.pipe import VaePipeline
    return VaePipeline(
        unet=model.unet, vae=model.vae, scheduler=model._eval_scheduler,
        text_encoder=model._text_encoder, tokenizer=model._tokenizer)


def forward_completed(model, batch, device):
    from ssd.alignment import align_depth_least_square

    rgb = batch["rgb"].to(device=device, dtype=torch.float32)
    sparse_m = (
        batch["sparse_depth"].to(device=device, dtype=torch.float32)
        * float(model.runtime_depth_scale))
    condition, sparse_mask = _sparse_relative(sparse_m)
    pipeline = _make_eval_pipeline(model).to(device)
    outputs = []
    alignments = []
    for index in range(rgb.shape[0]):
        rgb_i = _pad_model_input(rgb[index], -1.0)
        condition_i = _pad_model_input(condition[index], -1.0)
        result = pipeline(
            rgb_i, condition_i, batch_size=1, color_map=None,
            show_progress_bar=False, noise=_NOISE)
        relative = np.asarray(result.depth_np, dtype=np.float32).squeeze()
        relative = relative[:_EVALUATION_HW[0], :_EVALUATION_HW[1]]
        sparse_np = sparse_m[index, 0].detach().cpu().numpy().astype(np.float32)
        mask_np = sparse_mask[index, 0].detach().cpu().numpy().astype(bool)
        aligned, scale, shift = align_depth_least_square(
            gt_arr=sparse_np, pred_arr=relative,
            valid_mask_arr=mask_np, return_scale_shift=True,
            max_resolution=None)
        if not np.isfinite(aligned).all():
            raise RuntimeError("Marigold-SSD sparse alignment produced non-finite depth")
        aligned = np.clip(aligned, a_min=1e-6, a_max=None).astype(np.float32)
        outputs.append(torch.from_numpy(aligned)[None, None].to(device))
        alignments.append({
            "alignment_method": "official_sparse_least_square_scale_shift",
            "alignment_source": "registered_sparse_depth_only",
            "alignment_point_count": int(mask_np.sum()),
            "alignment_scale": float(np.asarray(scale).reshape(-1)[0]),
            "alignment_shift": float(np.asarray(shift).reshape(-1)[0]),
            "alignment_fallback": False,
            "diffusion_steps": _DIFFUSION_STEPS,
            "ensemble_size": _ENSEMBLE_SIZE,
            "latent_initialization": _NOISE,
            "timestep_spacing": _TIMESTEP_SPACING,
            "guidance_scale": None,
            "uncertainty_available": False,
        })
    model.last_alignment = alignments
    return torch.cat(outputs, dim=0) / float(model.runtime_depth_scale)


def prediction_metadata(model, batch, index, args):
    del batch, args
    if index >= len(model.last_alignment):
        raise RuntimeError("Marigold-SSD alignment metadata is unavailable")
    return dict(model.last_alignment[index])


def make_optimizer(model, args):
    return torch.optim.AdamW([
        {"params": model.unet.parameters(), "lr": float(args.lr),
         "name": "unet"},
        {"params": model.vae.decoder.parameters(), "lr": _VAE_LR,
         "name": "vae_decoder"},
        {"params": model.vae.post_quant_conv.parameters(), "lr": _VAE_LR,
         "name": "vae_post_quant_conv"},
    ], lr=float(args.lr), betas=(0.9, 0.999), eps=1e-8,
       weight_decay=_WEIGHT_DECAY)


def make_scheduler(optimizer, args):
    del args
    from ssd.lr_scheduler import IterExponential
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=IterExponential(
            total_iter_length=_SCHEDULE_STEPS, final_ratio=0.01,
            warmup_steps=_WARMUP_STEPS))


def before_optimizer_step(model, optimizer, args, epoch, iteration):
    del optimizer, args, epoch, iteration
    torch.nn.utils.clip_grad_norm_(model.vae.decoder.parameters(), 1.0)


def checkpoint_payload(model, optimizer, epoch):
    del optimizer, epoch
    return model.state_dict()


def _is_tensor_state_dict(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items())


def _extract_state_dict(payload):
    if _is_tensor_state_dict(payload):
        weights = payload
    elif isinstance(payload, dict):
        recognized = [
            key for key in ("state_dict", "model", "net")
            if key in payload and _is_tensor_state_dict(payload[key])]
        if len(recognized) != 1:
            raise RuntimeError(
                "Marigold-SSD V2 checkpoint must contain exactly one recognized "
                "tensor state dict (state_dict/model/net)")
        weights = payload[recognized[0]]
    else:
        raise RuntimeError("Marigold-SSD V2 checkpoint is not a state dict")
    prefixes = [key.startswith("module.") for key in weights]
    if any(prefixes) and not all(prefixes):
        raise RuntimeError("checkpoint has a mixed module. prefix")
    if all(prefixes):
        return OrderedDict(
            (key[len("module."):], value) for key, value in weights.items())
    return OrderedDict(weights.items())


def load_checkpoint(model, checkpoint, device, args, strict=False):
    del strict
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Marigold-SSD V2 checkpoint does not exist: {path}")
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    result = model.load_state_dict(_extract_state_dict(payload), strict=True)
    args.checkpoint_path = str(path)
    args.checkpoint_loading = "strict_state_dict"
    args.checkpoint_registered = False
    print(
        f"[checkpoint] Marigold-SSD strict load: {path} "
        f"missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}")
    return result

"""SGNet adapter for the shared evaluation runner."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = WORKSPACE_ROOT / "SGNet-main"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD_NAME = "SGNet"
ERROR_MAX_CLI_UNIT = "metres"
PREDICTION_NPY_UNIT = "metres"
SKIP_SHARED_CHECKPOINT_PROVENANCE = True
EVAL_BATCH_SIZE = 1
NETWORK_SIZE = (228, 304)


def add_method_args(parser):
    parser.add_argument("--training-domain", required=True, choices=("N", "K"))
    parser.add_argument("--sgnet-scale", type=int, default=8, choices=(8,))
    parser.add_argument("--sgnet-num-feats", type=int, default=40)


def configure_args(args, mode):
    if mode != "eval":
        raise RuntimeError("the SGNet adapter is evaluation-only")
    if args.sgnet_num_feats != 40:
        raise RuntimeError("the published SGNet configuration uses 40 features")
    args.sgnet_training_domain = args.training_domain
    args.method_name = f"SGNet-{args.training_domain}"
    args.method_version = args.method_name
    args.prediction_npy_unit = PREDICTION_NPY_UNIT
    args.primary_output = "completed_depth"
    args.network_input_height, args.network_input_width = NETWORK_SIZE
    args.evaluation_height, args.evaluation_width = NETWORK_SIZE
    args.prediction_spatial_resolution = list(NETWORK_SIZE)
    args.strict_checkpoint = True
    return args


def fairness_metadata(args, mode=None):
    del mode
    return {
        "method": METHOD_NAME,
        "version": f"SGNet-{args.training_domain}",
        "training_domain": args.training_domain,
        "seed": int(args.seed),
        "fairness_class": "CONTROLLED_NATIVE",
        "controlled_ranking": True,
        "unfair_reason": None,
        "native_loss": "published SGNet training objective",
        "valid_pixel_masking": False,
        "zero_coded_unlabelled_pixels_in_loss": args.training_domain == "K",
        "external_pretraining": False,
        "foundation_model": None,
        "official_mixed_training": False,
        "TTA": False,
        "OOD_specific_tuning": False,
        "target_overlap": False,
        "loss_modified": False,
        "network_modified": False,
        "optimizer_modified": False,
        "scheduler_modified": False,
        "primary_output": "completed_depth",
        "notes": "central evaluation adapter with mask-aware sparse pooling",
    }


def build_model(args, device):
    del device
    from models.SGNet import SGNet
    model = SGNet(num_feats=args.sgnet_num_feats, kernel_size=3, scale=8)
    model.source_depth_scale = 10.0 if args.training_domain == "N" else 80.0
    model.runtime_depth_scale = float(args.depth_unit_scale)
    return model


def _resize_batch(batch):
    if batch["rgb"].shape[-2:] != NETWORK_SIZE:
        batch["rgb"] = F.interpolate(
            batch["rgb"], size=NETWORK_SIZE, mode="bilinear", align_corners=False
        )
    for key in ("sparse_depth", "target_depth", "target_mask"):
        value = batch.get(key)
        if value is not None and value.shape[-2:] != NETWORK_SIZE:
            batch[key] = F.interpolate(
                value.float(), size=NETWORK_SIZE, mode="nearest"
            )
    batch["valid_mask"] = (batch["sparse_depth"] > 0).to(
        batch["sparse_depth"].dtype
    )


def forward_completed(model, batch, device):
    _resize_batch(batch)
    rgb = ((batch["rgb"].to(device) + 1.0) * 0.5).clamp(0.0, 1.0)
    sparse = batch["sparse_depth"].to(device)
    sparse = sparse * model.runtime_depth_scale / model.source_depth_scale
    mask = (sparse > 0).to(sparse.dtype)
    sparse_sum = F.avg_pool2d(sparse, 8) * 64
    sample_count = F.avg_pool2d(mask, 8) * 64
    low_resolution = torch.where(
        sample_count > 0,
        sparse_sum / sample_count.clamp_min(1.0),
        torch.zeros_like(sparse_sum),
    )
    prediction, _ = model((rgb, low_resolution))
    return prediction * model.source_depth_scale / model.runtime_depth_scale


def load_checkpoint(model, checkpoint, device, args, strict=True):
    del args
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state:
        raise RuntimeError("SGNet checkpoint does not contain model weights")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(cleaned, strict=bool(strict))
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return result

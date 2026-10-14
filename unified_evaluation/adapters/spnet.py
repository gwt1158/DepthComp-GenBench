"""SPNet-Large adapter for the shared evaluation runner."""

from collections import OrderedDict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = WORKSPACE_ROOT / "SPNet-main"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD_NAME = "SPNet-Large"
ERROR_MAX_CLI_UNIT = "metres"
PREDICTION_NPY_UNIT = "metres"
SKIP_SHARED_CHECKPOINT_PROVENANCE = True
EVAL_BATCH_SIZE = 1
EVALUATION_SIZE = (228, 304)


def add_method_args(parser):
    parser.add_argument("--training-domain", required=True, choices=("N", "K"))


def configure_args(args, mode):
    if mode != "eval":
        raise RuntimeError("the SPNet adapter is evaluation-only")
    expected_scale = 10.0 if args.training_domain == "N" else 80.0
    if float(args.depth_unit_scale) != expected_scale:
        raise ValueError(
            f"SPNet-{args.training_domain} requires "
            f"depth_unit_scale={expected_scale:g}"
        )
    args.spnet_training_domain = args.training_domain
    args.method_name = f"SPNet-Large-{args.training_domain}"
    args.method_version = args.method_name
    args.prediction_npy_unit = PREDICTION_NPY_UNIT
    args.primary_output = "V2Net depth tensor"
    args.network_input_height, args.network_input_width = (256, 320)
    args.evaluation_height, args.evaluation_width = EVALUATION_SIZE
    args.prediction_spatial_resolution = list(EVALUATION_SIZE)
    args.strict_checkpoint = True
    args.batch_size = 1
    return args


def fairness_metadata(args, mode=None):
    del mode
    return {
        "method": METHOD_NAME,
        "version": f"SPNet-Large-{args.training_domain}",
        "training_domain": args.training_domain,
        "seed": int(args.seed),
        "fairness_class": "CONTROLLED_NATIVE",
        "controlled_ranking": True,
        "unfair_reason": None,
        "native_loss": "published SPNet training objective",
        "valid_pixel_masking": True,
        "zero_coded_unlabelled_pixels_in_loss": False,
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
        "primary_output": "V2Net depth tensor",
        "notes": "central evaluation adapter for the published Large model",
    }


def build_model(args, device):
    del args, device
    from src.networks import V2Net
    return V2Net(
        [192, 384, 768, 1536],
        [3, 3, 27, 3],
        0.2,
        "CNX",
    )


def prepare_batch_for_model(batch, device=None):
    del device
    for key, mode in (
        ("rgb", "bilinear"),
        ("sparse_depth", "nearest"),
        ("target_depth", "nearest"),
        ("target_mask", "nearest"),
    ):
        value = batch.get(key)
        if value is None or value.shape[-2:] == EVALUATION_SIZE:
            continue
        options = {"align_corners": False} if mode == "bilinear" else {}
        batch[key] = F.interpolate(
            value.float(), size=EVALUATION_SIZE, mode=mode, **options
        )
    batch["valid_mask"] = (batch["sparse_depth"] > 0).to(
        batch["sparse_depth"].dtype
    )
    return batch


def forward_completed(model, batch, device):
    prepare_batch_for_model(batch)
    rgb = ((batch["rgb"].to(device) + 1.0) * 0.5).clamp(0.0, 1.0)
    sparse = batch["sparse_depth"].to(device=device, dtype=torch.float32)
    valid = (sparse > 0).to(sparse.dtype)
    height, width = sparse.shape[-2:]
    pad = (0, (-width) % 32, 0, (-height) % 32)
    rgb = F.pad(rgb, pad, mode="replicate")
    sparse = F.pad(sparse, pad)
    valid = F.pad(valid, pad)
    prediction = model(rgb, sparse, valid)
    return prediction[..., :height, :width]


def _extract_state_dict(payload):
    def is_state_dict(value):
        return (
            isinstance(value, dict)
            and bool(value)
            and all(isinstance(key, str) and torch.is_tensor(item)
                    for key, item in value.items())
        )

    if is_state_dict(payload):
        state = payload
    elif isinstance(payload, dict):
        matches = [
            payload[key] for key in ("network", "state_dict", "model", "net")
            if key in payload and is_state_dict(payload[key])
        ]
        if len(matches) != 1:
            raise RuntimeError("SPNet checkpoint must contain one model state dict")
        state = matches[0]
    else:
        raise RuntimeError("SPNet checkpoint is not a state dict")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("SPNet checkpoint has inconsistent module prefixes")
    if all(prefixed):
        return OrderedDict((key.removeprefix("module."), value)
                           for key, value in state.items())
    return OrderedDict(state.items())


def load_checkpoint(model, checkpoint, device, args, strict=True):
    del args
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    result = model.load_state_dict(_extract_state_dict(payload), strict=bool(strict))
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return result

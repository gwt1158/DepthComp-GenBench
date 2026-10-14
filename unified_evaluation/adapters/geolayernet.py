"""GeoLayerNet adapter for the shared evaluation runner."""

import sys
from pathlib import Path

import torch
import torch.nn as nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = WORKSPACE_ROOT / "GeoLayerNet"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD_NAME = "GeoLayerNet"
ERROR_MAX_CLI_UNIT = "metres"
PREDICTION_NPY_UNIT = "metres"
SKIP_SHARED_CHECKPOINT_PROVENANCE = True
EVAL_BATCH_SIZE = 1


def add_method_args(parser):
    parser.add_argument(
        "--training-domain", required=True, choices=("N", "K"),
        help="source domain used to train the checkpoint",
    )
    parser.add_argument(
        "--runtime-config", required=True,
        help="workspace-relative GeoLayerNet model configuration",
    )


def configure_args(args, mode):
    if mode != "eval":
        raise RuntimeError("the GeoLayerNet adapter is evaluation-only")
    args.method_name = f"GeoLayerNet-{args.training_domain}"
    args.method_version = args.method_name
    args.prediction_npy_unit = PREDICTION_NPY_UNIT
    args.primary_output = "completed_depth"
    args.network_input_height = 228
    args.network_input_width = 304
    args.evaluation_height = 228
    args.evaluation_width = 304
    args.prediction_spatial_resolution = [228, 304]
    args.strict_checkpoint = True
    return args


def fairness_metadata(args, mode=None):
    del mode
    return {
        "method": METHOD_NAME,
        "version": f"GeoLayerNet-{args.training_domain}",
        "training_domain": args.training_domain,
        "seed": int(args.seed),
        "fairness_class": "CONTROLLED_NATIVE",
        "controlled_ranking": True,
        "unfair_reason": None,
        "native_loss": "unchanged training checkpoint",
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
        "primary_output": "completed_depth",
        "notes": "central evaluation adapter",
    }


class EvalModel(nn.Module):
    def __init__(self, network, source_scale, runtime_scale):
        super().__init__()
        self.network = network
        self.source_scale = float(source_scale)
        self.runtime_scale = float(runtime_scale)


def build_model(args, device):
    del device
    from geolayernet.config import load_config, model_config_from_yaml
    from geolayernet.models import GeoLayerNet

    config_path = Path(args.runtime_config)
    if not config_path.is_absolute():
        config_path = WORKSPACE_ROOT / config_path
    config = load_config(config_path)
    network = GeoLayerNet(model_config_from_yaml(config))
    source_scale = 10.0 if args.training_domain == "N" else 80.0
    return EvalModel(network, source_scale, float(args.depth_unit_scale))


def forward_completed(model, batch, device):
    del device
    sparse = batch["sparse_depth"] * model.runtime_scale / model.source_scale
    output = model.network(
        batch["rgb"], sparse, batch["valid_mask"],
        output_mode=model.network.output_mode,
    )["completed_depth"]
    return output * model.source_scale / model.runtime_scale


def load_checkpoint(model, checkpoint, device, args, strict=True):
    del args
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    cleaned = {
        key.removeprefix("module."): value for key, value in state.items()
    }
    result = model.network.load_state_dict(cleaned, strict=bool(strict))
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return result

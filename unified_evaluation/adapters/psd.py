"""PSD-D adapter for the shared evaluation runner."""

from collections import OrderedDict
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = WORKSPACE_ROOT / "PSD-main"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD_NAME = "PSD"
ERROR_MAX_CLI_UNIT = "metres"
PREDICTION_NPY_UNIT = "metres"
SKIP_SHARED_CHECKPOINT_PROVENANCE = True
EVAL_BATCH_SIZE = 1
EVALUATION_SIZE = (228, 304)
MDE_SIZE = (392, 392)


def add_method_args(parser):
    parser.add_argument("--training-domain", required=True, choices=("N", "K"))
    parser.add_argument(
        "--psd-config", default=str(METHOD_ROOT / "configs" / "cfg.yml")
    )


def configure_args(args, mode):
    if mode != "eval":
        raise RuntimeError("the compact PSD adapter is evaluation-only")
    expected_scale = 10.0 if args.training_domain == "N" else 80.0
    if float(args.depth_unit_scale) != expected_scale:
        raise ValueError(
            f"PSD-{args.training_domain}-D requires depth_unit_scale={expected_scale:g}"
        )
    args.method_name = f"PSD-{args.training_domain}-D"
    args.method_version = args.method_name
    args.prediction_npy_unit = PREDICTION_NPY_UNIT
    args.primary_output = "final refined depth"
    args.network_input_height, args.network_input_width = EVALUATION_SIZE
    args.evaluation_height, args.evaluation_width = EVALUATION_SIZE
    args.prediction_spatial_resolution = list(EVALUATION_SIZE)
    args.strict_checkpoint = True
    args.batch_size = 1
    return args


def fairness_metadata(args, mode=None):
    del mode
    return {
        "method": METHOD_NAME,
        "version": f"PSD-{args.training_domain}-D",
        "training_domain": args.training_domain,
        "seed": int(args.seed),
        "fairness_class": "CONTROLLED_NATIVE",
        "controlled_ranking": True,
        "unfair_reason": None,
        "native_loss": "published PSD objective",
        "valid_pixel_masking": True,
        "zero_coded_unlabelled_pixels_in_loss": False,
        "external_pretraining": True,
        "foundation_model": "Depth Anything V1 ViT-L",
        "official_mixed_training": False,
        "TTA": False,
        "OOD_specific_tuning": False,
        "target_overlap": False,
        "loss_modified": False,
        "network_modified": False,
        "optimizer_modified": False,
        "scheduler_modified": False,
        "primary_output": "final refined depth",
        "notes": "published PSD-D network with a fixed source-domain configuration",
    }


class EvalModel(nn.Module):
    def __init__(self, network, scale, data_name):
        super().__init__()
        self.network = network
        self.scale = float(scale)
        self.data_name = data_name


def _resolve_weight_paths(cfg):
    for node in (
        cfg.MDEBranch.depany,
        cfg.RegressBranch.resnet,
        cfg.CSLBranch.resnet,
    ):
        path = Path(str(node.pretrained))
        node.pretrained = str(path if path.is_absolute() else METHOD_ROOT / path)


def build_model(args, device):
    del device
    from omegaconf import OmegaConf
    from network.psdnet import DCNet

    cfg = OmegaConf.load(args.psd_config)
    OmegaConf.set_struct(cfg, False)
    _resolve_weight_paths(cfg)
    network = DCNet(cfg)
    data_name = "NYUv2" if args.training_domain == "N" else "KITTI"
    return EvalModel(network, args.depth_unit_scale, data_name)


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
        batch[key] = F.interpolate(value.float(), EVALUATION_SIZE, mode=mode, **options)
    return batch


def _camera_matrix(batch, device, count):
    matrix = batch.get("intrinsics")
    if matrix is None:
        matrix = torch.tensor(
            [[518.8579, 0.0, 325.5824], [0.0, 519.4696, 253.7362], [0.0, 0.0, 1.0]],
            device=device,
        ).unsqueeze(0).repeat(count, 1, 1)
    else:
        matrix = matrix.to(device=device, dtype=torch.float32)
    return matrix


def forward_completed(model, batch, device):
    prepare_batch_for_model(batch)
    rgb = ((batch["rgb"].to(device) + 1.0) * 0.5).clamp(0.0, 1.0)
    sparse = batch["sparse_depth"].to(device=device, dtype=torch.float32) * model.scale
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    rgb_mde = F.interpolate(rgb, MDE_SIZE, mode="bicubic", align_corners=False)
    rgb_mde = (rgb_mde - mean) / std
    rgb_native = (rgb - mean) / std

    from datasets.ip_basic import fill_in_fast
    filled = []
    for item in sparse.detach().cpu().numpy():
        depth = item[0]
        maximum = float(depth.max()) if depth.max() > 0 else model.scale
        filled.append(torch.from_numpy(fill_in_fast(depth.copy(), max_depth=maximum)))
    ip = torch.stack(filled, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
    inputs = {
        "rgb": rgb_native,
        "rgb_mde": rgb_mde,
        "sparse": sparse,
        "ip": ip,
        "K": _camera_matrix(batch, device, rgb.shape[0]),
    }
    output = model.network(inputs=inputs, epoch=10**6, is_test=True, data=model.data_name)
    return output["depth"][-1] / model.scale


def _extract_state_dict(payload):
    if isinstance(payload, dict) and payload and all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()
    ):
        state = payload
    elif isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
    else:
        raise RuntimeError("PSD checkpoint must be a state dict or contain model")
    if state and all(key.startswith("module.") for key in state):
        state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    return state


def load_checkpoint(model, checkpoint, device, args, strict=True):
    del args
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    return model.network.load_state_dict(_extract_state_dict(payload), strict=bool(strict))

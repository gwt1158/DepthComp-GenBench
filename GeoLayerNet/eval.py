"""Evaluate GeoLayerNet using only user-specified paths and result names."""

import argparse
import copy
import csv
from contextlib import nullcontext
import json
import math
import os
import sys
import time


# ==================== User settings ====================
# Paths may be absolute or relative to the project directory.
USER_DATASET_NAME = ""
USER_CONFIG_PATH = ""
USER_TEST_DATAROOT = ""
USER_CHECKPOINT_PATH = ""
USER_RESULT_DIR = ""
USER_EVAL_NAME = ""
USER_PHASE = "test"
USER_OUTPUT_MODE = "safe"  # Choices: safe / raw / final / proxy_only
USER_GPU = None  # Example: 1. None preserves CUDA_VISIBLE_DEVICES or defaults to 0.
# ================================================================


DEFAULT_GPU = 0


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action):
        help_text = action.help or ""
        if (
            action.default is not None
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in help_text
        ):
            help_text += " (default: %(default)s)"
        return help_text


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the final GeoLayerNet method.",
        epilog=(
            "Edit the USER_* block at the top of eval.py, then run directly.\n"
            "Command-line arguments remain available as temporary overrides."
        ),
        formatter_class=_HelpFormatter,
    )

    entry = parser.add_argument_group("evaluation selection")
    entry.add_argument(
        "--dataset-name", default=USER_DATASET_NAME,
        help="user-selected dataset label recorded in result metadata",
    )
    entry.add_argument(
        "--config", "--config-path", dest="config",
        default=USER_CONFIG_PATH, help="evaluation config file path",
    )
    entry.add_argument(
        "--dataroot", "--data-path", dest="dataroot",
        default=USER_TEST_DATAROOT,
        help="evaluation dataset root or phase directory",
    )
    entry.add_argument(
        "--checkpoint", "--model-path", dest="checkpoint",
        default=USER_CHECKPOINT_PATH, help="model checkpoint file to evaluate",
    )
    entry.add_argument(
        "--output-dir", "--result-dir", dest="output_dir",
        default=USER_RESULT_DIR, help="directory for evaluation results",
    )
    entry.add_argument(
        "--eval-name", default=USER_EVAL_NAME,
        help="user-selected evaluation/result name",
    )

    hardware = parser.add_argument_group("hardware and data loading")
    hardware.add_argument(
        "--gpu", "--gpu-id", dest="gpu", type=_non_negative_int,
        default=USER_GPU,
        help=("physical GPU id; overrides CUDA_VISIBLE_DEVICES before torch import; "
              f"otherwise preserves the environment or uses {DEFAULT_GPU}"),
    )
    hardware.add_argument("--device", help="torch device; auto-selects cuda/cpu when omitted")
    hardware.add_argument("--batch-size", type=_positive_int, help="eval batch size; defaults to config")
    hardware.add_argument("--num-workers", type=_non_negative_int, help="data-loader workers; defaults to config")

    protocol = parser.add_argument_group("evaluation protocol")
    protocol.add_argument("--phase", default=USER_PHASE, help="dataset split")
    protocol.add_argument("--num-test", type=_positive_int, help="maximum samples; defaults to config")
    protocol.add_argument(
        "--output-mode", choices=("safe", "raw", "final", "proxy_only"),
        default=USER_OUTPUT_MODE, help="model output used for metrics",
    )
    protocol.add_argument("--seed", type=int, default=42, help="random seed")
    protocol.add_argument(
        "--clip-prediction", action=argparse.BooleanOptionalAction, default=None,
        help="clip predictions before metrics; defaults to config",
    )
    protocol.add_argument(
        "--extra-metrics", action=argparse.BooleanOptionalAction, default=True,
        help="compute boundary/normal/gradient/distance metrics",
    )
    protocol.add_argument(
        "--model-stats", action=argparse.BooleanOptionalAction, default=False,
        help="run one independent Params/GFLOPs/FPS measurement at 228x304",
    )
    protocol.add_argument(
        "--per-image-metrics", action=argparse.BooleanOptionalAction, default=True,
        help="write per-image metrics and preserve the per-image aggregation path",
    )
    protocol.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False,
        help="enable optional CUDA autocast; default evaluation remains FP32",
    )
    protocol.add_argument(
        "--progress-every", type=_positive_int, default=50,
        help="print progress every N processed images",
    )
    protocol.add_argument(
        "--profile-first-batches", type=_non_negative_int, default=3,
        help="print timing and CUDA memory for the first N batches; 0 disables",
    )
    protocol.add_argument(
        "--print-output-tensors", action=argparse.BooleanOptionalAction, default=False,
        help="print first-batch output tensor shapes/dtypes/sizes without retaining them",
    )
    protocol.add_argument(
        "--eval-height", type=_non_negative_int, default=None,
        help="evaluation height; 0 disables resizing and None uses YAML/default",
    )
    protocol.add_argument(
        "--eval-width", type=_non_negative_int, default=None,
        help="evaluation width; 0 disables resizing and None uses YAML/default",
    )

    visuals = parser.add_argument_group("visualization")
    visuals.add_argument("--save-visuals", action="store_true", help="save sample visualizations")
    visuals.add_argument("--save-count", type=_non_negative_int, help="maximum visualized samples; defaults to config")
    visuals.add_argument("--error-max", type=float, help="error-map saturation; defaults to config")
    visuals.add_argument(
        "--vis-depth-mode", choices=("auto", "shared", "both"),
        help="depth visualization scaling; defaults to config",
    )

    control = parser.add_argument_group("execution control")
    control.add_argument("--overwrite", action="store_true", help="overwrite existing run records")
    control.add_argument(
        "--dry-run", action="store_true",
        help="resolve paths/config values, then exit before model/data creation",
    )
    return parser


def resolve_entry_args(args):
    required = {
        "USER_DATASET_NAME": args.dataset_name,
        "USER_CONFIG_PATH": args.config,
        "USER_TEST_DATAROOT": args.dataroot,
        "USER_CHECKPOINT_PATH": args.checkpoint,
        "USER_RESULT_DIR": args.output_dir,
        "USER_EVAL_NAME": args.eval_name,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise SystemExit(
            "Set these values in the user settings block at the top of eval.py: "
            + ", ".join(missing)
        )
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        args.gpu = DEFAULT_GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_GPU)
    return args


def print_entry(args):
    print(f"[entry] dataset_name={args.dataset_name}")
    print(f"[entry] config={args.config}")
    print(f"[entry] dataroot={args.dataroot}")
    print(f"[entry] checkpoint={args.checkpoint}")
    print(f"[entry] output_dir={args.output_dir}")
    print(f"[entry] eval_name={args.eval_name}")
    print(f"[entry] gpu={args.gpu}")
    print(f"[entry] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


def _get(cfg, section, key, default=None):
    sec = getattr(cfg, section, None)
    return getattr(sec, key, default) if sec is not None else default


def _dry_run_config_fields(path):
    """Read basic dataset/eval fields without importing torch or PyYAML."""
    wanted = {
        "dataset": {"dataroot"},
        "eval": {
            "phase", "output_mode", "num_test", "batch_size", "num_threads",
            "clip_prediction", "min_depth", "max_depth",
            "save_layer_maps", "error_max", "vis_depth_mode", "eval_height", "eval_width",
        },
    }
    found = {}
    section = None
    with open(path) as fh:
        for raw_line in fh:
            content = raw_line.split("#", 1)[0].rstrip()
            if not content:
                continue
            if not content.startswith((" ", "\t")) and content.endswith(":"):
                section = content[:-1].strip()
                continue
            if section not in wanted or ":" not in content:
                continue
            key, value = content.strip().split(":", 1)
            if key in wanted[section]:
                found[key] = value.strip().strip("'\"")
    return found


def write_per_image_csv(path, rows):
    """One row per test image. NaN -> empty cell so pandas/csv never chokes."""
    from geolayernet.utils.extra_metrics import PER_IMAGE_FIELDS

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_IMAGE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k in PER_IMAGE_FIELDS:
                v = row.get(k, "")
                if isinstance(v, float) and math.isnan(v):
                    v = ""
                clean[k] = v
            w.writerow(clean)


def _cuda_sync(device, torch):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_dict(batch):
    return {
        key: list(batch[key].shape)
        for key in ("rgb", "sparse_depth", "valid_mask", "target_depth")
        if key in batch and torch_is_tensor(batch[key])
    }


def torch_is_tensor(value):
    # Kept as a tiny helper so shape/diagnostic code remains import-light in
    # --dry-run mode; the real torch module is only imported inside main().
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "numel")


def _print_output_tensor_diagnostics(outputs):
    total_bytes = 0
    print("[diagnostic] first-batch output tensors")
    for key, value in outputs.items():
        if not torch_is_tensor(value):
            continue
        bytes_used = int(value.numel()) * int(value.element_size())
        total_bytes += bytes_used
        print(
            f"  key={key} shape={list(value.shape)} dtype={value.dtype} "
            f"mb={bytes_used / (1024.0 ** 2):.6f}"
        )
    print(f"[diagnostic] output tensor total_mb={total_bytes / (1024.0 ** 2):.6f}")


def _resize_stats_sample(sample_cpu, device, torch):
    """Prepare one independent 228x304 stats input without touching eval data."""
    import torch.nn.functional as F

    rgb = sample_cpu["rgb"].to(device)
    sparse = sample_cpu["sparse_depth"].to(device)
    valid = sample_cpu["valid_mask"].to(device)
    if tuple(rgb.shape[-2:]) != (228, 304):
        rgb = F.interpolate(rgb, size=(228, 304), mode="bilinear", align_corners=False)
        sparse = F.interpolate(sparse, size=(228, 304), mode="nearest")
        valid = F.interpolate(valid, size=(228, 304), mode="nearest")
    return rgb, sparse, valid


def _optional_eval_dim(value, name):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or 0/None, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer or 0/None, got {value!r}")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer or 0/None, got {value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a positive integer or 0/None, got {value}")
    return None if value == 0 else value


def _resolve_eval_size(cli_height, cli_width, yaml_height=None, yaml_width=None):
    if (cli_height is None) != (cli_width is None):
        raise ValueError(
            "--eval-height and --eval-width must be provided together; provide both values "
            "or omit both"
        )
    height = _optional_eval_dim(
        cli_height if cli_height is not None else yaml_height, "eval_height")
    width = _optional_eval_dim(
        cli_width if cli_width is not None else yaml_width, "eval_width")
    if (height is None) != (width is None):
        raise ValueError(
            "eval_height and eval_width must be set together; provide both positive values "
            "or leave both unset/zero"
        )
    return height, width


def _resize_eval_tensors(rgb, sparse, valid, target, eval_hw):
    """Resize all evaluation tensors before forward with one shared geometry."""
    import torch.nn.functional as F

    if tuple(rgb.shape[-2:]) == tuple(eval_hw):
        valid = (valid > 0.5).to(dtype=valid.dtype)
        sparse = sparse * valid
        return rgb, sparse, valid, target, False
    rgb = F.interpolate(rgb, size=eval_hw, mode="bilinear", align_corners=False)
    sparse = F.interpolate(sparse, size=eval_hw, mode="nearest")
    valid = F.interpolate(valid, size=eval_hw, mode="nearest")
    valid = (valid > 0.5).to(dtype=valid.dtype)
    target = F.interpolate(target, size=eval_hw, mode="nearest")
    sparse = sparse * valid
    return rgb, sparse, valid, target, True


def _pad_eval_tensors(rgb, sparse, valid, target, multiple=64):
    """Pad only when needed for a multiple-of-64 model input, then crop later."""
    import torch.nn.functional as F

    h, w = rgb.shape[-2:]
    pad_h = (multiple - int(h) % multiple) % multiple
    pad_w = (multiple - int(w) % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return rgb, sparse, valid, target, (0, 0)
    padding = (0, pad_w, 0, pad_h)
    rgb = F.pad(rgb, padding, mode="replicate")
    sparse = F.pad(sparse, padding, mode="constant", value=0.0)
    valid = F.pad(valid, padding, mode="constant", value=0.0)
    target = F.pad(target, padding, mode="constant", value=0.0)
    return rgb, sparse, valid, target, (pad_h, pad_w)


def _crop_model_outputs(outputs, eval_hw, padded_hw):
    if tuple(eval_hw) == tuple(padded_hw):
        return outputs
    h, w = eval_hw
    for key, value in list(outputs.items()):
        if torch_is_tensor(value) and value.dim() >= 3 and tuple(value.shape[-2:]) == tuple(padded_hw):
            outputs[key] = value[..., :h, :w]
    return outputs


def main(argv=None):
    parser = build_parser()
    args = resolve_entry_args(parser.parse_args(argv))
    print_entry(args)

    fields = _dry_run_config_fields(args.config)
    output_mode = args.output_mode or fields.get("output_mode", "safe")
    phase = args.phase or fields.get("phase", "test") or "test"
    num_test = args.num_test or int(fields.get("num_test", 100000))
    bs = args.batch_size or int(fields.get("batch_size", 2))
    nthreads = (
        args.num_workers
        if args.num_workers is not None
        else int(fields.get("num_threads", 4))
    )
    save_cap = (
        args.save_count
        if args.save_count is not None
        else int(fields.get("save_layer_maps", 50))
    )
    err_max = (
        args.error_max
        if args.error_max is not None
        else float(fields.get("error_max", 0.05))
    )
    vis_mode = args.vis_depth_mode or fields.get("vis_depth_mode", "both")
    dry_eval_height, dry_eval_width = _resolve_eval_size(
        args.eval_height, args.eval_width,
        fields.get("eval_height"), fields.get("eval_width"),
    )

    if args.dry_run:
        print(f"[dry-run] config_dataroot_ignored={fields.get('dataroot')}")
        print(f"[dry-run] dataroot={args.dataroot}")
        print(f"[dry-run] phase={phase}")
        print(f"[dry-run] output_mode={output_mode}")
        print(f"[dry-run] num_test={num_test}")
        print(f"[dry-run] batch_size={bs}")
        print(f"[dry-run] num_workers={nthreads}")
        print(f"[dry-run] save_count={save_cap}")
        print(f"[dry-run] extra_metrics={args.extra_metrics}")
        print(f"[dry-run] per_image_metrics={args.per_image_metrics}")
        print(f"[dry-run] model_stats={args.model_stats}")
        print(f"[dry-run] amp={args.amp}")
        print(f"[dry-run] progress_every={args.progress_every}")
        print(f"[dry-run] profile_first_batches={args.profile_first_batches}")
        print(f"[dry-run] print_output_tensors={args.print_output_tensors}")
        print(f"[dry-run] eval_height={dry_eval_height}")
        print(f"[dry-run] eval_width={dry_eval_width}")
        print(f"[dry-run] eval_name={args.eval_name}")
        return

    import torch
    from geolayernet.config import load_config, model_config_from_yaml, config_to_dict
    from geolayernet.models import GeoLayerNet
    from geolayernet.data import create_loader
    from geolayernet.utils import set_global_seed, load_checkpoint, model_stats
    from geolayernet.utils.extra_metrics import MetricBag
    from geolayernet.utils import result_io, vis

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    set_global_seed(args.seed)
    cfg = load_config(args.config)
    gc = model_config_from_yaml(cfg)
    gc.output_mode = output_mode
    eval_height, eval_width = _resolve_eval_size(
        args.eval_height, args.eval_width,
        _get(cfg, "eval", "eval_height", None),
        _get(cfg, "eval", "eval_width", None),
    )

    dataroot = args.dataroot
    preprocess = _get(cfg, "dataset", "preprocess", "none")
    scale = float(_get(cfg, "dataset", "depth_unit_scale", 10))

    model = GeoLayerNet(gc).to(device).eval()
    load_checkpoint(model, args.checkpoint, map_location=device)

    # eval-only sparsity (low-density experiments): when eval.dynamic_sparse is
    # true the loader deterministically samples eval.sparse_points from the GT as
    # the sparse INPUT (the model still only sees sparse points + mask). Default
    # off -> uses the on-disk sparse depth unchanged.
    eval_dyn = bool(_get(cfg, "eval", "dynamic_sparse", False))
    eval_sp = int(_get(cfg, "eval", "sparse_points", 500))
    eval_sp_pat = _get(cfg, "eval", "sparse_pattern", "random")
    if eval_dyn:
        print(f"[eval-sparsity] dynamic_sparse=ON points={eval_sp} pattern={eval_sp_pat} (deterministic per sample)")
    # Dataset routing: NYU by default, plus KITTI and generic OOD NPZ inputs.
    dataset_type = _get(cfg, "dataset", "type", "nyu")
    dataset_opts = dict(cfg.raw.get("dataset", {}) if getattr(cfg, "raw", None) else {})
    if getattr(cfg, "raw", None):
        dataset_opts["dataset_cfg"] = cfg.raw.get("dataset", {})
        dataset_opts["eval_cfg"] = cfg.raw.get("eval", {})
    cuda_eval = device.type == "cuda"
    amp_enabled = bool(args.amp and cuda_eval)
    if args.amp and not cuda_eval:
        print(f"[eval] amp requested but device={device}; AMP disabled")
    pin_memory = cuda_eval
    persistent_workers = cuda_eval and nthreads > 0
    prefetch_factor = 2 if nthreads > 0 else None
    loader = create_loader(
        dataroot, phase, bs, num_workers=nthreads, preprocess=preprocess,
        shuffle=False, max_samples=None if num_test >= 100000 else num_test,
        config_hint=args.config, dynamic_sparse=eval_dyn,
        sparse_points=eval_sp, sparse_pattern=eval_sp_pat,
        dataset_type=dataset_type, dataset_opts=dataset_opts,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    loader_batch_size = getattr(loader, "batch_size", None)
    if loader_batch_size != bs:
        raise RuntimeError(
            f"resolved batch_size={bs} does not match loader.batch_size={loader_batch_size}")
    dataset_size = len(loader.dataset)

    clip_pred = (
        args.clip_prediction
        if args.clip_prediction is not None
        else bool(_get(cfg, "eval", "clip_prediction", True))
    )
    eval_min_depth = float(_get(cfg, "eval", "min_depth", 1e-3))
    eval_max_depth = float(_get(cfg, "eval", "max_depth", 1.0))
    print(f"[eval] metric_depth_range=[{eval_min_depth}, {eval_max_depth}] scale={scale} clip_prediction={clip_pred}")
    print(
        f"[eval] resolved_batch_size={bs} loader.batch_size={loader_batch_size} "
        f"num_workers={nthreads} dataset_size={dataset_size} output_mode={output_mode} "
        f"extra_metrics={args.extra_metrics} per_image_metrics={args.per_image_metrics} "
        f"model_stats={args.model_stats} amp={amp_enabled} pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers} prefetch_factor={prefetch_factor}"
    )
    data_iter = iter(loader)
    data_wait_start = time.perf_counter()
    try:
        first_batch = next(data_iter)
    except StopIteration as exc:
        raise RuntimeError("evaluation loader is empty") from exc
    first_shape = _shape_dict(first_batch)
    original_h, original_w = map(int, first_batch["rgb"].shape[-2:])
    first_input_hws = {
        key: tuple(first_batch[key].shape[-2:])
        for key in ("rgb", "sparse_depth", "valid_mask", "target_depth")
    }
    if len(set(first_input_hws.values())) != 1:
        raise RuntimeError(
            f"evaluation input spatial shapes must match before resize, got {first_input_hws}")
    requested_eval_hw = (eval_height, eval_width) if eval_height is not None else None
    first_eval_h, first_eval_w = requested_eval_hw or (original_h, original_w)
    resize_enabled = (first_eval_h, first_eval_w) != (original_h, original_w)
    original_pixels = original_h * original_w
    eval_pixels = int(first_eval_h) * int(first_eval_w)
    pixel_reduction_ratio = original_pixels / float(eval_pixels)
    eval_shape = {
        key: (list(value[:-2]) + [int(first_eval_h), int(first_eval_w)])
        for key, value in first_shape.items()
    }
    eval_rgb_shape = list(first_shape["rgb"][:-2]) + [int(first_eval_h), int(first_eval_w)]
    eval_pixel_ratio_vs_228x304 = eval_pixels / float(228 * 304)
    print(
        f"[eval] original_shape={first_shape} eval_shape={eval_shape} "
        f"first_batch_eval_rgb_shape={eval_rgb_shape} H={first_eval_h} W={first_eval_w} "
        f"resize_enabled={resize_enabled} resize_scale_h={first_eval_h / original_h:.6f} "
        f"resize_scale_w={first_eval_w / original_w:.6f} original_pixels={original_pixels} "
        f"eval_pixels={eval_pixels} pixel_reduction_ratio={pixel_reduction_ratio:.6f} "
        f"pixel_ratio_vs_228x304={eval_pixel_ratio_vs_228x304:.6f}"
    )
    if cuda_eval:
        print(
            f"[eval] cuda_memory_initial_allocated_mb={torch.cuda.memory_allocated(device) / (1024.0 ** 2):.3f} "
            f"cuda_memory_initial_reserved_mb={torch.cuda.memory_reserved(device) / (1024.0 ** 2):.3f}"
        )
    stats_sample_cpu = None
    if args.model_stats:
        stats_sample_cpu = {
            key: first_batch[key][:1].detach().cpu().clone()
            for key in ("rgb", "sparse_depth", "valid_mask")
        }
    non_blocking = bool(getattr(loader, "pin_memory", False))
    global_bag = MetricBag(scale=scale, extra=args.extra_metrics, clip=clip_pred,
                           lo=eval_min_depth, hi=eval_max_depth)
    per_image_rows = [] if args.per_image_metrics else None
    stats = None
    saved = 0
    resize_enabled_any = resize_enabled
    padding_height = 0
    padding_width = 0
    total_data_time = 0.0
    total_forward_time = 0.0
    total_metric_time = 0.0
    processed = 0
    batch_count = 0
    run_start = time.perf_counter()
    batch = first_batch
    next_progress = args.progress_every
    if cuda_eval:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        while batch is not None:
            batch_count += 1
            data_time = time.perf_counter() - data_wait_start
            prepare_start = time.perf_counter()
            batch_rgb = batch["rgb"]
            batch_sparse = batch["sparse_depth"]
            batch_valid = batch["valid_mask"]
            batch_target = batch["target_depth"]
            batch_original_hw = tuple(batch_rgb.shape[-2:])
            batch_eval_hw = requested_eval_hw or batch_original_hw
            if requested_eval_hw is not None:
                batch_rgb, batch_sparse, batch_valid, batch_target, batch_resized = _resize_eval_tensors(
                    batch_rgb, batch_sparse, batch_valid, batch_target, batch_eval_hw
                )
            else:
                batch_resized = False
            resize_enabled_any = resize_enabled_any or batch_resized
            if requested_eval_hw is not None:
                batch_rgb, batch_sparse, batch_valid, batch_target, (batch_pad_h, batch_pad_w) = _pad_eval_tensors(
                    batch_rgb, batch_sparse, batch_valid, batch_target, multiple=64
                )
            else:
                batch_pad_h, batch_pad_w = 0, 0
            padding_height = max(padding_height, int(batch_pad_h))
            padding_width = max(padding_width, int(batch_pad_w))
            rgb = batch_rgb.to(device, non_blocking=non_blocking)
            sparse = batch_sparse.to(device, non_blocking=non_blocking)
            valid = batch_valid.to(device, non_blocking=non_blocking)
            target = batch_target.to(device, non_blocking=non_blocking)
            _cuda_sync(device, torch)
            model_input_hw = tuple(rgb.shape[-2:])
            data_time += time.perf_counter() - prepare_start

            forward_start = time.perf_counter()
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled else nullcontext()
            )
            with autocast_context:
                outputs = model(rgb, sparse, valid)
            _cuda_sync(device, torch)
            forward_time = time.perf_counter() - forward_start
            outputs = _crop_model_outputs(outputs, batch_eval_hw, model_input_hw)
            if (batch_pad_h, batch_pad_w) != (0, 0):
                eval_h_now, eval_w_now = batch_eval_hw
                rgb = rgb[..., :eval_h_now, :eval_w_now]
                sparse = sparse[..., :eval_h_now, :eval_w_now]
                valid = valid[..., :eval_h_now, :eval_w_now]
                target = target[..., :eval_h_now, :eval_w_now]

            if args.print_output_tensors and batch_count == 1:
                _print_output_tensor_diagnostics(outputs)

            metric_start = time.perf_counter()
            # MetricBag.update currently folds a leading batch of one, but does
            # not implement a mathematically equivalent batch reduction. Keep
            # batch>1 on the old per-image merge path for exactness.
            if not args.per_image_metrics and rgb.size(0) == 1:
                global_bag.update(outputs, target, valid, sparse)
            else:
                for j in range(rgb.size(0)):
                    bag = MetricBag(scale=scale, extra=args.extra_metrics, clip=clip_pred,
                                    lo=eval_min_depth, hi=eval_max_depth)
                    out_j = {
                        k: (v[j:j + 1] if torch.is_tensor(v) and v.dim() == 4 else v)
                        for k, v in outputs.items()
                    }
                    bag.update(out_j, target[j:j + 1], valid[j:j + 1], sparse[j:j + 1])
                    if args.per_image_metrics:
                        row = bag.finalize()
                        row["sample_id"] = batch["name"][j]
                        per_image_rows.append(row)
                    global_bag.merge(bag)
                    del bag, out_j
            _cuda_sync(device, torch)
            metric_time = time.perf_counter() - metric_start

            if args.save_visuals and saved < save_cap:
                cpu_batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
                eval_h_now, eval_w_now = batch_eval_hw
                cpu_batch["rgb"] = batch_rgb[..., :eval_h_now, :eval_w_now].detach().cpu()
                cpu_batch["sparse_depth"] = batch_sparse[..., :eval_h_now, :eval_w_now].detach().cpu()
                cpu_batch["valid_mask"] = batch_valid[..., :eval_h_now, :eval_w_now].detach().cpu()
                cpu_batch["target_depth"] = batch_target[..., :eval_h_now, :eval_w_now].detach().cpu()
                cpu_out = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()}
                for j in range(rgb.size(0)):
                    if saved >= save_cap:
                        break
                    vis.save_sample_visuals(args.output_dir, batch["name"][j], cpu_batch, cpu_out, j,
                                            err_max=err_max, vis_depth_mode=vis_mode)
                    vis.save_panel_materials(args.output_dir, batch["name"][j], cpu_batch, cpu_out, j,
                                             output_mode=output_mode, scale=scale)
                    saved += 1
                del cpu_batch, cpu_out

            batch_size_now = int(rgb.size(0))
            processed += batch_size_now
            total_data_time += data_time
            total_forward_time += forward_time
            total_metric_time += metric_time
            if cuda_eval and batch_count <= args.profile_first_batches:
                print(
                    f"[profile] batch={batch_count} processed={processed} "
                    f"allocated_mb={torch.cuda.memory_allocated(device) / (1024.0 ** 2):.3f} "
                    f"reserved_mb={torch.cuda.memory_reserved(device) / (1024.0 ** 2):.3f} "
                    f"max_allocated_mb={torch.cuda.max_memory_allocated(device) / (1024.0 ** 2):.3f} "
                    f"max_reserved_mb={torch.cuda.max_memory_reserved(device) / (1024.0 ** 2):.3f}"
                )
            if processed >= next_progress or processed == dataset_size:
                elapsed = max(time.perf_counter() - run_start, 1e-9)
                print(
                    f"[progress] processed={processed}/{dataset_size} "
                    f"avg_data_ms={total_data_time / batch_count * 1000.0:.3f} "
                    f"avg_forward_ms={total_forward_time / batch_count * 1000.0:.3f} "
                    f"avg_metric_ms={total_metric_time / batch_count * 1000.0:.3f} "
                    f"samples_per_s={processed / elapsed:.3f}"
                )
                while next_progress <= processed:
                    next_progress += args.progress_every

            del outputs, rgb, sparse, valid, target, batch_rgb, batch_sparse, batch_valid, batch_target
            batch = None
            data_wait_start = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                batch = None

    del data_iter, first_batch
    eval_elapsed = max(time.perf_counter() - run_start, 1e-9)
    if cuda_eval:
        _cuda_sync(device, torch)
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
    average_forward_ms = total_forward_time / max(batch_count, 1) * 1000.0
    average_total_ms = (total_data_time + total_forward_time + total_metric_time) / max(batch_count, 1) * 1000.0
    throughput_images_per_s = processed / eval_elapsed if processed else float("nan")

    metrics = global_bag.finalize()
    if args.model_stats:
        if stats_sample_cpu is None:
            raise RuntimeError("--model-stats requires at least one evaluation sample")
        _cuda_sync(device, torch)
        # This is the only cache clear in eval.py, and it is outside the timed
        # evaluation interval. Stats always use a fixed 228x304 input.
        if cuda_eval:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        stats_rgb, stats_sparse, stats_valid = _resize_stats_sample(stats_sample_cpu, device, torch)
        _cuda_sync(device, torch)
        stats_start = time.perf_counter()
        with torch.inference_mode():
            stats = model_stats(model, (stats_rgb, stats_sparse, stats_valid), measure_speed=True)
        _cuda_sync(device, torch)
        print(f"[model-stats] fixed_input=1x3x228x304 elapsed_s={time.perf_counter() - stats_start:.3f}")
        del stats_rgb, stats_sparse, stats_valid, stats_sample_cpu
    metrics.update({k: v for k, v in (stats or {}).items()})

    saved_raw_config = copy.deepcopy(cfg.raw or {})
    saved_eval_config = saved_raw_config.setdefault("eval", {})
    saved_eval_config.update({
        "original_height": original_h,
        "original_width": original_w,
        "eval_height": int(eval_height or 0),
        "eval_width": int(eval_width or 0),
        "resolved_eval_height": int(first_eval_h),
        "resolved_eval_width": int(first_eval_w),
        "resize_enabled": bool(resize_enabled_any),
        "rgb_resize_mode": "bilinear, align_corners=False",
        "depth_resize_mode": "nearest",
        "mask_resize_mode": "nearest_then_binary",
        "resize_scale_h": float(first_eval_h / original_h),
        "resize_scale_w": float(first_eval_w / original_w),
        "pixel_reduction_ratio": float(pixel_reduction_ratio),
        "model_padding_multiple": 64,
        "model_padding_height": int(padding_height),
        "model_padding_width": int(padding_width),
    })

    # --- per-image metrics CSV (one row per test image) ---
    if args.per_image_metrics:
        write_per_image_csv(os.path.join(args.output_dir, "per_image_metrics.csv"), per_image_rows)

    # --- per-run records (scoped to this output-dir; never overwrite silently) ---
    result_io.write_metrics(args.output_dir, metrics)
    result_io.dump_config_yaml(os.path.join(args.output_dir, "eval_config.yaml"), saved_raw_config, overwrite=True)
    result_io.write_run_info(
        os.path.join(args.output_dir, "run_info.txt"),
        title=f"GeoLayerNet eval: {args.eval_name} ({output_mode})",
        header={
            "eval_name": args.eval_name, "output_dir": args.output_dir,
            "checkpoint": args.checkpoint, "output_mode": output_mode,
            "config": args.config, "dataset": args.dataset_name, "dataroot": dataroot,
            "phase": phase, "num_test": num_test,
            "images_evaluated": metrics.get("images"), "gpu": args.gpu,
            "device": args.device, "seed": args.seed,
            "resolved_batch_size": bs, "loader_batch_size": loader_batch_size,
            "num_workers": nthreads, "dataset_size": dataset_size,
            "original_height": original_h, "original_width": original_w,
            "eval_height": int(first_eval_h), "eval_width": int(first_eval_w),
            "resize_enabled": resize_enabled_any,
            "rgb_resize_mode": "bilinear, align_corners=False",
            "depth_resize_mode": "nearest",
            "mask_resize_mode": "nearest_then_binary",
            "first_batch_shape": json.dumps(first_shape, sort_keys=True),
            "eval_batch_shape": json.dumps(eval_shape, sort_keys=True),
            "resize_scale_h": f"{first_eval_h / original_h:.9f}",
            "resize_scale_w": f"{first_eval_w / original_w:.9f}",
            "original_pixels": original_pixels,
            "eval_pixels": eval_pixels,
            "pixel_reduction_ratio": f"{pixel_reduction_ratio:.9f}",
            "pixel_ratio_vs_228x304": f"{eval_pixel_ratio_vs_228x304:.9f}",
            "model_padding_height": padding_height,
            "model_padding_width": padding_width,
            "extra_metrics": args.extra_metrics,
            "per_image_metrics": args.per_image_metrics,
            "model_stats_enabled": args.model_stats,
            "amp": amp_enabled,
            "peak_allocated_mb": f"{peak_allocated_mb:.6f}",
            "peak_reserved_mb": f"{peak_reserved_mb:.6f}",
            "average_forward_ms": f"{average_forward_ms:.6f}",
            "average_total_ms": f"{average_total_ms:.6f}",
            "throughput_images_per_s": f"{throughput_images_per_s:.6f}",
            "git": result_io.git_hash(),
            "command_line": " ".join([sys.executable] + sys.argv),
        },
        metrics=metrics, stats=stats,
        config={"model": config_to_dict(gc), "raw_config": saved_raw_config},
        overwrite=args.overwrite,
    )

    print(f"\n=== eval ({output_mode}) — {args.output_dir} ===")
    for k in ("images", "RMSE_m", "MAE_m", "REL", "delta1(%)", "FinalMissing_RMSE_m",
              "ProxyMissing_RMSE_m", "Params(M)", "GFLOPs", "FPS"):
        v = metrics.get(k)
        print(f"  {k:22s}: {v:.6f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    print(f"  {'average_forward_ms':22s}: {average_forward_ms:.6f}")
    print(f"  {'average_total_ms':22s}: {average_total_ms:.6f}")
    print(f"  {'throughput_images_per_s':22s}: {throughput_images_per_s:.6f}")
    print(f"  {'peak_allocated_mb':22s}: {peak_allocated_mb:.6f}")
    print(f"  {'peak_reserved_mb':22s}: {peak_reserved_mb:.6f}")


if __name__ == "__main__":
    main()

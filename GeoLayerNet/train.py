"""Train GeoLayerNet using only user-specified paths and run names."""

import argparse
import inspect
import os
import sys
from pathlib import Path


# ==================== User settings ====================
# Paths may be absolute or relative to the project directory.
USER_DATASET_NAME = "nyu48k_1pct"
USER_CONFIG_PATH = "configs/geolayernet_final_nyu48k_1pct_fullval.yaml"
USER_TRAIN_DATAROOT = "data/nyu48k_1pct"
USER_VAL_DATAROOT = "data/nyu48k_1pct"
USER_TRAIN_PHASE = "train"
USER_VAL_PHASE = "val"
USER_MODEL_NAME = "geolayernet_nyu48k_1pct"
USER_RUN_OUTPUT_DIR = "output/geolayernet_nyu48k_1pct"
USER_CHECKPOINT_DIR = "checkpoints/geolayernet_nyu48k_1pct"
USER_BEST_CHECKPOINT_NAME = "best_net_G.pth"
USER_LATEST_CHECKPOINT_NAME = "latest_net_G.pth"
USER_EPOCH_CHECKPOINT_PATTERN = "{epoch}_net_G.pth"
USER_GPU = None
# ================================================================


DEFAULT_GPU = 0


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show grouped defaults while preserving multiline examples."""

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


def _non_negative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _beta(value):
    value = float(value)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("must satisfy 0 <= value < 1")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the final GeoLayerNet method.",
        epilog=(
            "Edit the USER_* block at the top of train.py, then run directly.\n"
            "Command-line arguments remain available as temporary overrides."
        ),
        formatter_class=_HelpFormatter,
    )

    entry = parser.add_argument_group("run selection")
    entry.add_argument(
        "--dataset-name", default=USER_DATASET_NAME,
        help="user-selected dataset label recorded in run metadata",
    )
    entry.add_argument(
        "--config", "--config-path", dest="config",
        default=USER_CONFIG_PATH, help="training config file path",
    )
    entry.add_argument(
        "--train-dataroot", "--train-data-path", dest="train_dataroot",
        default=USER_TRAIN_DATAROOT,
        help="training dataset root or the training phase directory itself",
    )
    entry.add_argument(
        "--val-dataroot", default=USER_VAL_DATAROOT,
        help="validation dataset root or the validation phase directory itself",
    )
    entry.add_argument(
        "--train-phase", default=USER_TRAIN_PHASE,
        help="training phase/subdirectory name",
    )
    entry.add_argument(
        "--name", "--model-name", dest="name",
        default=USER_MODEL_NAME, help="training run/model name",
    )
    entry.add_argument(
        "--output-dir", "--run-output-dir", dest="output_dir",
        default=USER_RUN_OUTPUT_DIR,
        help="directory for logs, resolved config, and run records",
    )

    hardware = parser.add_argument_group("hardware and data loading")
    hardware.add_argument(
        "--gpu", "--gpu-id", dest="gpu", type=_non_negative_int,
        default=USER_GPU,
        help=("physical GPU id; overrides CUDA_VISIBLE_DEVICES before torch import; "
              f"otherwise preserves the environment or uses {DEFAULT_GPU}"),
    )
    hardware.add_argument(
        "--device", help="torch device; auto-selects cuda/cpu when omitted"
    )
    hardware.add_argument(
        "--batch-size", type=_positive_int,
        help="training batch size; defaults to config",
    )
    hardware.add_argument(
        "--num-workers", type=_non_negative_int,
        help="data-loader workers; defaults to config",
    )

    optim = parser.add_argument_group("optimization overrides")
    optim.add_argument(
        "--seed", type=int, help="random seed; defaults to config"
    )
    optim.add_argument(
        "--lr", type=_positive_float, help="learning rate; defaults to config"
    )
    optim.add_argument(
        "--beta1", type=_beta, help="AdamW beta1; defaults to config"
    )
    optim.add_argument(
        "--weight-decay", type=_non_negative_float,
        help="AdamW weight decay; defaults to config",
    )
    optim.add_argument(
        "--epochs", type=_positive_int,
        help="primary scheduler-cycle epochs; defaults to config",
    )
    optim.add_argument(
        "--epochs-decay", type=_non_negative_int,
        help="additional epochs; defaults to config",
    )
    optim.add_argument(
        "--cosine-t-max", type=_positive_int,
        help="CosineAnnealingLR T_max; defaults to config, then primary epochs",
    )
    optim.add_argument(
        "--warmup-epochs", type=_non_negative_int,
        help="residual warmup epochs; defaults to config",
    )
    optim.add_argument(
        "--grad-clip", type=_non_negative_float,
        help="gradient clipping norm; defaults to config",
    )

    validation = parser.add_argument_group("validation and checkpoints")
    validation.add_argument(
        "--val-phase", default=USER_VAL_PHASE, help="validation split"
    )
    validation.add_argument(
        "--val-num-test", type=_positive_int,
        help="validation sample cap; defaults to config",
    )
    validation.add_argument(
        "--val-best-metric",
        help="metric used to select best checkpoint; defaults to config",
    )
    validation.add_argument(
        "--val-batch-size", type=_positive_int, default=1,
        help="validation batch size",
    )
    validation.add_argument(
        "--save-epoch-freq", type=_non_negative_int, default=5,
        help="save numbered checkpoints every N epochs; 0 disables them",
    )
    validation.add_argument(
        "--checkpoint-dir", "--save-model-dir", dest="checkpoint_dir",
        default=USER_CHECKPOINT_DIR, help="directory for saved model checkpoints",
    )
    validation.add_argument(
        "--best-checkpoint-name", "--save-model-name",
        dest="best_checkpoint_name", default=USER_BEST_CHECKPOINT_NAME,
        help="best checkpoint filename, or an absolute path",
    )
    validation.add_argument(
        "--latest-checkpoint-name", default=USER_LATEST_CHECKPOINT_NAME,
        help="latest checkpoint filename, or an absolute path",
    )
    validation.add_argument(
        "--epoch-checkpoint-pattern", default=USER_EPOCH_CHECKPOINT_PATTERN,
        help="numbered checkpoint filename pattern; use {epoch}",
    )
    validation.add_argument(
        "--early-stopping-patience", type=_non_negative_int,
        help="stop after N validation epochs without improvement; 0 disables",
    )
    validation.add_argument(
        "--early-stopping-min-delta", type=_non_negative_float,
        help="minimum validation improvement counted by early stopping",
    )

    control = parser.add_argument_group("execution control")
    control.add_argument(
        "--overwrite", action="store_true", help="overwrite existing run records"
    )
    control.add_argument(
        "--no-train-diagnostics", action="store_true",
        help="skip the first-batch loss-breakdown diagnostic forward",
    )
    control.add_argument(
        "--dry-run", action="store_true",
        help="resolve and read the entry config, then exit before model/data creation",
    )
    return parser


def resolve_entry_args(args):
    """Validate the top-of-file settings/CLI overrides and select the GPU."""
    required = {
        "USER_DATASET_NAME": args.dataset_name,
        "USER_CONFIG_PATH": args.config,
        "USER_TRAIN_DATAROOT": args.train_dataroot,
        "USER_VAL_DATAROOT": args.val_dataroot,
        "USER_MODEL_NAME": args.name,
        "USER_RUN_OUTPUT_DIR": args.output_dir,
        "USER_CHECKPOINT_DIR": args.checkpoint_dir,
        "USER_BEST_CHECKPOINT_NAME": args.best_checkpoint_name,
        "USER_LATEST_CHECKPOINT_NAME": args.latest_checkpoint_name,
        "USER_EPOCH_CHECKPOINT_PATTERN": args.epoch_checkpoint_pattern,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise SystemExit(
            "Set these values in the user settings block at the top of train.py: "
            + ", ".join(missing)
        )
    if "{epoch}" not in args.epoch_checkpoint_pattern:
        raise SystemExit("USER_EPOCH_CHECKPOINT_PATTERN must contain {epoch}")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        args.gpu = DEFAULT_GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_GPU)
    return args


def _checkpoint_path(checkpoint_dir, name_or_path, epoch=None):
    """Resolve a checkpoint filename or absolute path without touching disk."""
    value = name_or_path.format(epoch=epoch) if epoch is not None else name_or_path
    return value if os.path.isabs(value) else os.path.join(checkpoint_dir, value)


def print_entry(args):
    print(f"[entry] dataset_name={args.dataset_name}")
    print(f"[entry] config={args.config}")
    print(f"[entry] train_dataroot={args.train_dataroot}")
    print(f"[entry] val_dataroot={args.val_dataroot}")
    print(f"[entry] name={args.name}")
    print(f"[entry] output_dir={args.output_dir}")
    print(f"[entry] checkpoint_dir={args.checkpoint_dir}")
    print(f"[entry] best_checkpoint={_checkpoint_path(args.checkpoint_dir, args.best_checkpoint_name)}")
    print(f"[entry] latest_checkpoint={_checkpoint_path(args.checkpoint_dir, args.latest_checkpoint_name)}")
    print(f"[entry] gpu={args.gpu}")
    print(f"[entry] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


def _get(cfg, section, key, default=None):
    sec = getattr(cfg, section, None)
    return getattr(sec, key, default) if sec is not None else default


def _arg_or_config(cli_value, cfg, section, key, default, cast):
    """Use an explicit argparse value, otherwise read and cast the config value."""
    value = cli_value if cli_value is not None else _get(cfg, section, key, default)
    return cast(value)


def _dry_run_config_fields(path):
    """Read entry/training fields needed by --dry-run without ML/YAML imports."""
    wanted = {
        "dataset": {"dataroot", "train_dataroot", "val_dataroot"},
        "train": {
            "seed", "batch_size", "num_threads", "lr", "beta1",
            "weight_decay", "n_epochs", "n_epochs_decay",
            "residual_warmup_epochs", "grad_clip", "val_phase",
            "val_num_test", "val_best_metric", "cosine_t_max",
            "early_stopping_patience", "early_stopping_min_delta", "train_phase",
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


def build_criterion(all_lambdas):
    """Construct GeoLayerNetLoss with ONLY the kwargs it actually supports.

    Record-only fields may appear in the merged lambdas but are not constructor
    args of the simplified loss.
    Such kwargs are dropped ONLY if their value is inert (0/0.0/False/None);
    a non-zero unsupported term raises (never silently drop a real loss term).
    """
    from geolayernet.models import GeoLayerNetLoss

    supported = set(inspect.signature(GeoLayerNetLoss.__init__).parameters)
    kwargs, ignored = {}, []
    for k, v in all_lambdas.items():
        if k in supported:
            kwargs[k] = v
        elif v in (0, 0.0, False, None):
            ignored.append(k)
        else:
            raise ValueError(
                f"loss kwarg {k}={v!r} is not a GeoLayerNetLoss constructor arg and is "
                f"non-zero — refusing to silently drop a real loss term. Map it or set 0.")
    if "lambda_uncertainty" in all_lambdas and "lambda_uncertainty" not in kwargs:
        raise RuntimeError("lambda_uncertainty is set but GeoLayerNetLoss does not accept it")
    if ignored:
        print(f"[loss] ignored {len(ignored)} inert/record-only kwargs (value 0/False/None) "
              f"not in GeoLayerNetLoss: {sorted(ignored)}")
    return GeoLayerNetLoss(**kwargs)


def write_resolved_train_config(path, gc, lambdas, lr, beta1, weight_decay,
                                n_epochs, n_decay, cosine_t_max, warmup_epochs, grad_clip,
                                scale, train_dataroot, val_dataroot, dataset_tag,
                                dataset_mode, preprocess, train_phase, val_phase,
                                val_num, val_batch_size, val_best_metric,
                                early_stopping_patience, early_stopping_min_delta):
    """Emit the resolved final training configuration."""
    rows = {
        # dataset
        "dataroot": train_dataroot, "train_dataroot": train_dataroot,
        "val_dataroot": val_dataroot, "train_phase": train_phase,
        "dataset_mode": dataset_mode, "dataset_tag": dataset_tag,
        "preprocess": preprocess, "input_nc": 4, "output_nc": 1, "depth_unit_scale": scale,
        # optimizer / schedule (training)
        "gp_optimizer": "adamw", "lr": lr, "beta1": beta1, "gp_weight_decay": weight_decay,
        "gp_grad_clip": grad_clip, "lr_policy": "cosine", "n_epochs": n_epochs,
        "n_epochs_decay": n_decay, "cosine_T_max": cosine_t_max,
        "gp_residual_warmup_epochs": warmup_epochs,
        # model (resolved from GeoLayerConfig)
        "gp_preset": "final", "gp_layer_encoder": "acm_dual", "gp_base_channels": gc.base_channels,
        "gp_num_layers": gc.num_layers, "gp_num_classes": 0, "gp_layer_source": "sparse_depth",
        "gp_partition": "quantile", "gp_output_mode": "safe", "gp_relation_steps": gc.relation_steps,
        "gp_residual_scale": gc.residual_scale, "gp_fusion_type": gc.fusion_type,
        "gp_fusion_mask_smooth": 5, "gp_gate_bias_init": gc.gate_bias_init,
        "gp_global_pool": gc.global_pool, "gp_attention_heads": gc.attention_heads,
        "gp_layer_drop_mode": "feature", "gp_layer_drop_prob": gc.layer_drop_prob,
        "gp_sparse_layer_support_mode": "gaussian_bilateral",
        "gp_sparse_layer_expand_radius": gc.sparse_layer_expand_radius,
        "gp_sparse_layer_expand_kernel": "gaussian", "gp_sparse_layer_sigma_xy": gc.sparse_layer_sigma_xy,
        "gp_sparse_layer_rgb_sigma": gc.sparse_layer_rgb_sigma,
        "gp_sparse_layer_rgb_guided": gc.sparse_layer_rgb_guided,
        "gp_sparse_layer_unknown_channel": True, "gp_sparse_layer_max_coverage": 0.35,
        "gp_sparse_layer_min_points": 20,
        # reverse-boolean switches (resolved to their effective positive meaning)
        "gp_no_global_layer_assignment": True, "gp_no_preserve_observed_depth": False,
        "gp_no_rgb_guidance": (not gc.sparse_layer_rgb_guided), "gp_no_uncertainty": False,
        "gp_continuous_modulation": False, "gp_predict_rgb": False, "gp_geometry_refine": False,
        "gp_proxy_safe_refine": True, "preserve_observed_depth": True,
        # validation
        "val_freq": 1, "val_num_test": val_num, "val_batch_size": val_batch_size,
        "val_phase": val_phase,
        "val_best_metric": val_best_metric, "val_min_depth": 0.001, "val_max_depth": 1.0,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    for k, v in lambdas.items():
        rows[k] = v
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("----------------- Resolved GeoLayerNet final train config ---------------\n")
        for k in sorted(rows):
            fh.write(f"{k:>40}: {rows[k]}\n")
        fh.write("----------------- End -------------------\n")


def validate(model, loader, scale, device):
    import torch
    from geolayernet.utils.metrics import DepthEvaluator

    with torch.no_grad():
        model.eval()
        saved = model.output_mode
        model.output_mode = "safe"
        ev = DepthEvaluator(depth_unit_scale=scale)
        for batch in loader:
            rgb = batch["rgb"].to(device)
            sparse = batch["sparse_depth"].to(device)
            valid = batch["valid_mask"].to(device)
            target = batch["target_depth"].to(device)
            ev.update(model(rgb, sparse, valid), target, valid)
        model.output_mode = saved
        return ev.result()


def main(argv=None):
    parser = build_parser()
    args = resolve_entry_args(parser.parse_args(argv))
    print_entry(args)

    if args.dry_run:
        fields = _dry_run_config_fields(args.config)

        def resolved(cli_value, config_key, fallback):
            return cli_value if cli_value is not None else fields.get(config_key, fallback)

        print(f"[dry-run] config_dataroot_ignored={fields.get('dataroot')}")
        print(f"[dry-run] train_dataroot={args.train_dataroot}")
        print(f"[dry-run] val_dataroot={args.val_dataroot}")
        print(f"[dry-run] train_phase={resolved(args.train_phase, 'train_phase', 'train')}")
        print(f"[dry-run] device={args.device or 'auto'}")
        print(f"[dry-run] seed={resolved(args.seed, 'seed', 42)}")
        print(f"[dry-run] batch_size={resolved(args.batch_size, 'batch_size', 4)}")
        print(f"[dry-run] num_workers={resolved(args.num_workers, 'num_threads', 4)}")
        print(f"[dry-run] lr={resolved(args.lr, 'lr', 1e-4)}")
        print(f"[dry-run] beta1={resolved(args.beta1, 'beta1', 0.5)}")
        print(f"[dry-run] weight_decay={resolved(args.weight_decay, 'weight_decay', 1e-4)}")
        print(f"[dry-run] epochs={resolved(args.epochs, 'n_epochs', 30)}")
        print(f"[dry-run] epochs_decay={resolved(args.epochs_decay, 'n_epochs_decay', 30)}")
        print(f"[dry-run] cosine_t_max={resolved(args.cosine_t_max, 'cosine_t_max', resolved(args.epochs, 'n_epochs', 30))}")
        print(f"[dry-run] warmup_epochs={resolved(args.warmup_epochs, 'residual_warmup_epochs', 5)}")
        print(f"[dry-run] grad_clip={resolved(args.grad_clip, 'grad_clip', 1.0)}")
        print(f"[dry-run] val_phase={resolved(args.val_phase, 'val_phase', 'val')}")
        print(f"[dry-run] val_num_test={resolved(args.val_num_test, 'val_num_test', 100000)}")
        print(f"[dry-run] val_batch_size={args.val_batch_size}")
        print(f"[dry-run] val_best_metric={resolved(args.val_best_metric, 'val_best_metric', 'SafeMissing_RMSE_m')}")
        print(f"[dry-run] early_stopping_patience={resolved(args.early_stopping_patience, 'early_stopping_patience', 0)}")
        print(f"[dry-run] early_stopping_min_delta={resolved(args.early_stopping_min_delta, 'early_stopping_min_delta', 0.0)}")
        print(f"[dry-run] save_epoch_freq={args.save_epoch_freq}")
        print(f"[dry-run] output_dir={args.output_dir}")
        print(f"[dry-run] checkpoint_dir={args.checkpoint_dir}")
        print(f"[dry-run] best_checkpoint={_checkpoint_path(args.checkpoint_dir, args.best_checkpoint_name)}")
        print(f"[dry-run] latest_checkpoint={_checkpoint_path(args.checkpoint_dir, args.latest_checkpoint_name)}")
        return

    import torch
    from geolayernet.config import load_config, model_config_from_yaml, config_to_dict
    from geolayernet.models import GeoLayerNet, loss_lambdas_from_yaml
    from geolayernet.data import create_loader
    from geolayernet.utils import set_global_seed, save_checkpoint, model_stats
    from geolayernet.utils import result_io
    from tools.tip_revision_common import capture_provenance

    cfg = load_config(args.config)
    seed = _arg_or_config(args.seed, cfg, "train", "seed", 42, int)
    set_global_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    capture_provenance(
        Path(__file__).resolve().parent, args.output_dir,
        [sys.executable] + sys.argv,
        {"seed": seed, "run_name": args.name, "training_entrypoint": "train.py"})
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    gc = model_config_from_yaml(cfg)
    gc.output_mode = "safe"
    print(f"[model] coarse={gc.use_coarse_densifier} "
          f"proxy_use_coarse={gc.proxy_use_coarse} selective={gc.selective_residual}")
    model = GeoLayerNet(gc).to(device)
    lambdas = loss_lambdas_from_yaml(cfg)
    criterion = build_criterion(lambdas).to(device)  # filtered to supported kwargs
    print(f"[sharpness-loss] grad_loss_weight={getattr(criterion,'grad_loss_weight',0.0)} "
          f"boundary_l1_weight={getattr(criterion,'boundary_l1_weight',0.0)} "
          f"lap_loss_weight={getattr(criterion,'lap_loss_weight',0.0)} "
          f"boundary_top_percent={getattr(criterion,'boundary_top_percent',10.0)}")

    lr = _arg_or_config(args.lr, cfg, "train", "lr", 1e-4, float)
    beta1 = _arg_or_config(args.beta1, cfg, "train", "beta1", 0.5, float)
    weight_decay = _arg_or_config(
        args.weight_decay, cfg, "train", "weight_decay", 1e-4, float)
    n_epochs = _arg_or_config(args.epochs, cfg, "train", "n_epochs", 30, int)
    n_decay = _arg_or_config(
        args.epochs_decay, cfg, "train", "n_epochs_decay", 30, int)
    total = n_epochs + n_decay
    cosine_t_max = _arg_or_config(
        args.cosine_t_max, cfg, "train", "cosine_t_max", n_epochs, int)
    warmup_epochs = _arg_or_config(
        args.warmup_epochs, cfg, "train", "residual_warmup_epochs", 5, int)
    grad_clip = _arg_or_config(args.grad_clip, cfg, "train", "grad_clip", 1.0, float)
    bs = _arg_or_config(args.batch_size, cfg, "train", "batch_size", 4, int)
    nthreads = _arg_or_config(
        args.num_workers, cfg, "train", "num_threads", 4, int)
    train_phase = args.train_phase or str(_get(cfg, "train", "train_phase", "train"))
    val_phase = args.val_phase or str(_get(cfg, "train", "val_phase", "val"))
    val_num = _arg_or_config(
        args.val_num_test, cfg, "train", "val_num_test", 100000, int)
    val_batch_size = args.val_batch_size
    val_best_metric = args.val_best_metric or str(
        _get(cfg, "train", "val_best_metric", "SafeMissing_RMSE_m"))
    early_stopping_patience = _arg_or_config(
        args.early_stopping_patience, cfg, "train", "early_stopping_patience", 0, int)
    early_stopping_min_delta = _arg_or_config(
        args.early_stopping_min_delta, cfg, "train", "early_stopping_min_delta", 0.0, float)
    scale = float(_get(cfg, "dataset", "depth_unit_scale", 10))
    train_dataroot = args.train_dataroot
    val_dataroot = args.val_dataroot
    dataset_mode = _get(cfg, "dataset", "dataset_mode", "rgbd_completion")
    preprocess = _get(cfg, "dataset", "preprocess", "none")
    teacher_dir = _get(cfg, "dataset", "teacher_depth_dir", None)
    print(
        f"[data] pre-generated sparse input; train={train_dataroot}/{train_phase} "
        f"val={val_dataroot}/{val_phase} teacher_dir={teacher_dir}"
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(beta1, 0.999),
        weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cosine_t_max, eta_min=0)

    # dataset routing: nyu (default) or kitti (forwards the dataset config block).
    dataset_type = _get(cfg, "dataset", "type", "nyu")
    dataset_opts = (cfg.raw.get("dataset", {}) if getattr(cfg, "raw", None) else {})
    train_loader = create_loader(train_dataroot, train_phase, bs, num_workers=nthreads, preprocess=preprocess, shuffle=True,
                                 teacher_depth_dir=teacher_dir,
                                 config_hint=args.config, dataset_type=dataset_type, dataset_opts=dataset_opts)
    val_loader = create_loader(
        val_dataroot, val_phase, val_batch_size, num_workers=nthreads,
        preprocess=preprocess, shuffle=False, max_samples=val_num,
        config_hint=args.config, dataset_type=dataset_type,
        dataset_opts=dataset_opts,
    )

    write_resolved_train_config(
        os.path.join(args.output_dir, "resolved_train_config.txt"),
        gc, lambdas, lr, beta1, weight_decay, n_epochs, n_decay, cosine_t_max,
        warmup_epochs, grad_clip, scale, train_dataroot, val_dataroot, args.dataset_name,
        dataset_mode, preprocess, train_phase, val_phase, val_num, val_batch_size,
        val_best_metric, early_stopping_patience, early_stopping_min_delta,
    )

    # --- startup diagnostics ---
    import os.path as _op
    print("\n================= ACTUAL RESOLVED TRAINING VALUES =================")
    print(f"actual train batch_size        = {bs}")
    print(f"actual train output_mode       = {model.output_mode}   (model trains with safe)")
    print(f"actual eval output_mode        = safe")
    print(f"actual lambda_uncertainty      = {lambdas.get('lambda_uncertainty')}")
    print(f"actual preserve_observed_depth = True")
    print(f"actual rgb_guidance / sparse_layer_rgb_guided = {gc.sparse_layer_rgb_guided}")
    print(f"actual uncertainty (head ON)   = True")
    print(f"actual layer_masks output source = fusion_masks (support-normalized)")
    print("==================================================================")
    print("\n================= TRAIN DIAGNOSTICS =================")
    print(f"run_name={args.name}  output_dir={args.output_dir}  config={args.config}")
    print(f"train_dataroot(realpath)={_op.realpath(train_dataroot)}")
    print(f"val_dataroot(realpath)={_op.realpath(val_dataroot)}")
    print(f"#train={len(train_loader.dataset)}  #val={len(val_loader.dataset)}")
    b0 = next(iter(train_loader))
    print("first batch keys:", sorted(b0.keys()))
    for k in ("rgb", "sparse_depth", "target_depth", "valid_mask"):
        t = b0[k]
        print(f"  {k:13s} shape={tuple(t.shape)} min={t.min():.4f} max={t.max():.4f} mean={t.float().mean():.4f}")
    sparse_ratio = (b0["sparse_depth"] > 0).float().mean()
    print(
        f"  sparse nonzero ratio={sparse_ratio:.5f}  "
        f"valid_mask mean={b0['valid_mask'].mean():.5f}"
    )
    print(
        f"optimizer=AdamW lr={lr} betas=({beta1},0.999) "
        f"wd={weight_decay} scheduler=CosineAnnealingLR T_max={cosine_t_max}"
    )
    # Closed-form preview of the LR used at the start of each epoch. Avoid
    # stepping a scheduler without an optimizer step (which triggers a warning).
    import math
    preview = {
        ep: 0.5 * lr * (1.0 + math.cos(math.pi * (ep - 1) / cosine_t_max))
        for ep in range(1, min(total, cosine_t_max + 1) + 1)
    }
    milestones = sorted({1, 5, n_epochs, n_epochs + 1, cosine_t_max, total})
    preview_values = {
        epoch: round(preview[epoch], 8)
        for epoch in milestones
        if epoch in preview
    }
    print("lr preview:", preview_values)
    print(f"early stopping: patience={early_stopping_patience} "
          f"min_delta={early_stopping_min_delta} (0 patience disables)")
    from geolayernet.utils.metrics import count_params
    print(f"Params(M)={count_params(model)[0]/1e6:.6f}")
    # first-batch loss breakdown (diagnostic only). Everything below is detached to
    # Python floats immediately and ALL GPU tensors are freed before training, so
    # the diagnostic never holds the graph or activations into the training loop.
    do_diagnostics = (not args.no_train_diagnostics) and bool(_get(cfg, "train", "print_diagnostics", True))
    if do_diagnostics:
        model.train()
        model.set_residual_warmup(min(1.0, 1 / max(warmup_epochs, 1)))
        with torch.no_grad():
            d_rgb = b0["rgb"].to(device)
            d_sparse = b0["sparse_depth"].to(device)
            d_valid = b0["valid_mask"].to(device)
            d_target = b0["target_depth"].to(device)
            d_tmask = b0["target_mask"].to(device)
            d_teacher = b0["teacher_depth"].to(device) if b0.get("teacher_depth") is not None else None
            d_tmaskk = b0["teacher_mask"].to(device) if b0.get("teacher_mask") is not None else None
            _out = model(d_rgb, d_sparse, d_valid)
            _ls = criterion(_out, d_rgb, d_sparse, d_target, target_rgb=None,
                            valid_mask=d_valid, target_mask=d_tmask,
                            teacher_depth=d_teacher, teacher_mask=d_tmaskk)
            # detach to host floats right away — do NOT keep GPU tensors in a dict
            ls_floats = {}
            for k, v in _ls.items():
                try: ls_floats[k] = float(v.detach().cpu())
                except Exception: pass
        print("first-batch loss breakdown:")
        for k in sorted(ls_floats):
            print(f"  {k:22s}={ls_floats[k]:.6f}")
        # release the diagnostic forward's GPU memory before training starts
        del _out, _ls, d_rgb, d_sparse, d_valid, d_target, d_tmask, d_teacher, d_tmaskk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("first-batch loss breakdown: skipped (--no-train-diagnostics / print_diagnostics=false)")
    del b0
    print("====================================================\n")

    start_time = result_io.now_str()
    best = float("inf")
    best_metrics = None
    best_epoch = -1
    epochs_without_improvement = 0
    final_metrics = None
    final_epoch = 0
    best_checkpoint_path = _checkpoint_path(
        args.checkpoint_dir, args.best_checkpoint_name)
    latest_checkpoint_path = _checkpoint_path(
        args.checkpoint_dir, args.latest_checkpoint_name)
    os.makedirs(os.path.dirname(best_checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(latest_checkpoint_path) or ".", exist_ok=True)
    for epoch in range(1, total + 1):
        model.train()
        model.set_residual_warmup(min(1.0, epoch / max(warmup_epochs, 1)))
        running = 0.0
        epoch_lr = optim.param_groups[0]["lr"]
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            sparse = batch["sparse_depth"].to(device)
            valid = batch["valid_mask"].to(device)
            target = batch["target_depth"].to(device)
            tmask = batch["target_mask"].to(device)
            teacher_d = batch.get("teacher_depth")
            teacher_m = batch.get("teacher_mask")
            teacher_d = teacher_d.to(device) if teacher_d is not None else None
            teacher_m = teacher_m.to(device) if teacher_m is not None else None
            out = model(rgb, sparse, valid)
            loss = criterion(out, rgb, sparse, target, target_rgb=None, valid_mask=valid, target_mask=tmask,
                             teacher_depth=teacher_d, teacher_mask=teacher_m)["total"]
            optim.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            running += float(loss.item())
        # One scheduler step per completed epoch, after all optimizer steps.
        sched.step()

        res = validate(model, val_loader, scale, device)
        final_metrics, final_epoch = res, epoch
        vmetric = res.get(val_best_metric)
        if vmetric is None:
            raise KeyError(
                f"val_best_metric={val_best_metric} not found; available={sorted(res.keys())}")
        save_checkpoint(model, latest_checkpoint_path)
        if args.save_epoch_freq > 0 and epoch % args.save_epoch_freq == 0:
            epoch_checkpoint_path = _checkpoint_path(
                args.checkpoint_dir, args.epoch_checkpoint_pattern, epoch=epoch)
            os.makedirs(os.path.dirname(epoch_checkpoint_path) or ".", exist_ok=True)
            save_checkpoint(model, epoch_checkpoint_path)
        improved = vmetric < (best - early_stopping_min_delta)
        if improved:
            best, best_metrics, best_epoch = vmetric, res, epoch
            epochs_without_improvement = 0
            save_checkpoint(model, best_checkpoint_path)
            with open(os.path.join(args.output_dir, "best_epoch.txt"), "w") as fh:
                fh.write(f"epoch {best_epoch}  {val_best_metric} {vmetric:.6f}\n")
            print(f"    -> new best saved (epoch {best_epoch}, {val_best_metric}={vmetric:.6f})")
        else:
            epochs_without_improvement += 1
        print(
            f"[epoch {epoch:3d}/{total}] lr={epoch_lr:.2e} "
            f"loss={running/max(len(train_loader),1):.4f} "
            f"current_{val_best_metric}={vmetric:.6f} current_RMSE_m={res['RMSE_m']:.6f} "
            f"best_{val_best_metric}={best:.6f} best_RMSE_m={best_metrics['RMSE_m']:.6f} "
            f"best_epoch={best_epoch} no_improve={epochs_without_improvement}"
        )
        if (early_stopping_patience > 0 and
                epochs_without_improvement >= early_stopping_patience):
            print(f"[early-stop] epoch={epoch} patience={early_stopping_patience} "
                  f"best_epoch={best_epoch} best_{val_best_metric}={best:.6f}")
            break
    end_time = result_io.now_str()

    # --- per-run records (scoped to output-dir; never overwrite silently) ---
    sample = next(iter(val_loader))
    stats = model_stats(model.eval(), (sample["rgb"][:1].to(device), sample["sparse_depth"][:1].to(device),
                                       sample["valid_mask"][:1].to(device)))
    merged = {"model": config_to_dict(gc), "raw_config": cfg.raw}
    result_io.dump_config_yaml(os.path.join(args.output_dir, "train_config.yaml"), cfg.raw, overwrite=True)
    result_io.write_command(os.path.join(args.output_dir, "train_command.txt"),
                            [sys.executable] + sys.argv, overwrite=True)
    result_io.write_run_info(
        os.path.join(args.output_dir, "train_run_info.txt"),
        title=f"GeoLayerNet training: {args.name}",
        header={
            "run_name": args.name, "output_dir": args.output_dir, "config": args.config,
            "seed": seed, "dataset": args.dataset_name,
            "train_dataroot": train_dataroot, "train_phase": train_phase,
            "val_dataroot": val_dataroot,
            "checkpoint_dir": args.checkpoint_dir,
            "best_checkpoint_path": best_checkpoint_path,
            "latest_checkpoint_path": latest_checkpoint_path,
            "gpu": args.gpu, "device": device, "num_workers": nthreads,
            "optimizer": f"AdamW(lr={lr}, betas=({beta1},0.999), wd={weight_decay})",
            "scheduler": f"CosineAnnealingLR(T_max={cosine_t_max}, eta_min=0) over {total} epochs",
            "grad_clip": grad_clip, "batch_size": bs, "epochs": total,
            "val_phase": val_phase, "val_num_test": val_num,
            "val_batch_size": val_batch_size, "save_epoch_freq": args.save_epoch_freq,
            "best_epoch": best_epoch, "best_val_metric": val_best_metric,
            f"best_val_{val_best_metric}": f"{best:.6f}",
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "final_epoch": final_epoch,
            "train_output_mode": "safe", "eval_output_mode": "safe",
            "start_time": start_time, "end_time": end_time, "git": result_io.git_hash(),
            "command_line": " ".join([sys.executable] + sys.argv),
        },
        metrics=best_metrics, stats=stats, config=merged,
        extra=f"best checkpoint selected by {val_best_metric} on full val; evaluate with --output-mode safe.",
        overwrite=args.overwrite,
    )
    print(f"done. best epoch={best_epoch} {val_best_metric}={best:.6f}  -> {args.output_dir}")
    print(
        "[training-summary] "
        f"best_epoch={best_epoch}, "
        f"best_SafeMissing_RMSE_m={best_metrics['SafeMissing_RMSE_m']:.6f}, "
        f"best_RMSE_m={best_metrics['RMSE_m']:.6f}, "
        f"final_epoch_RMSE_m={final_metrics['RMSE_m']:.6f}, "
        f"checkpoint_path={best_checkpoint_path}"
    )


if __name__ == "__main__":
    main()

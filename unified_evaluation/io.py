"""Per-run result writers (mirrors GeoLayerNet/geolayernet/utils/io.py).

metrics.json/csv, run_info.txt, eval_config.yaml, args.json, per_image_metrics.csv.
All writers are scoped to a run's output-dir; existing run_info is backed up with a
timestamp unless ``overwrite=True``.
"""

import csv
import datetime
import json
import math
import os
import subprocess

from .metrics import PER_IMAGE_FIELDS


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def git_hash(cwd=None):
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def backup_if_exists(path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        bak = f"{path}.{_stamp()}.bak"
        os.replace(path, bak)
        return bak
    return None


def write_metrics(out_dir, metrics):
    def clean_json(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    def clean_csv(v):
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        return v

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({k: clean_json(v) for k, v in metrics.items()}, fh, indent=2)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(metrics.keys()))
        w.writerow([clean_csv(metrics[k]) for k in metrics])


def write_per_image_csv(path, rows):
    """One row per test image. NaN -> empty cell so pandas/csv never chokes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_IMAGE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k in PER_IMAGE_FIELDS:
                v = row.get(k, "")
                if isinstance(v, float) and math.isnan(v):
                    v = "N/A"
                clean[k] = v
            w.writerow(clean)


def dump_config_yaml(path, raw_config, overwrite=True):
    """Write the run config to a YAML file (falls back to JSON if no pyyaml)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    backup_if_exists(path, overwrite=overwrite)
    try:
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(raw_config, fh, sort_keys=False, default_flow_style=False)
    except Exception:
        with open(path, "w") as fh:
            json.dump(raw_config, fh, indent=2, default=str)


def write_args_json(path, args_dict, overwrite=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    backup_if_exists(path, overwrite=overwrite)
    with open(path, "w") as fh:
        json.dump(args_dict, fh, indent=2, default=str)


def write_command(path, argv, overwrite=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    backup_if_exists(path, overwrite=overwrite)
    with open(path, "w") as fh:
        fh.write(" ".join(argv) + "\n")


def write_run_info(path, *, title, header=None, metrics=None, stats=None,
                   config=None, extra=None, overwrite=False):
    """Human-readable run record. ``header`` is an ordered dict of run fields."""
    backup = backup_if_exists(path, overwrite=overwrite)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [f"# {title}", ""]
    if backup:
        lines += [f"(previous run_info backed up to {os.path.basename(backup)})", ""]
    if header:
        lines.append("## run")
        for k, v in header.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    if stats:
        lines.append("## model stats")
        for k in ("Params(M)", "Trainable(M)", "GFLOPs", "FPS"):
            if k in stats:
                v = stats[k]
                lines.append(f"{k}: {'N/A' if v is None else (f'{v:.6f}' if isinstance(v, float) else v)}")
        lines.append("")
    if metrics:
        lines.append("## metrics")
        for k, v in metrics.items():
            lines.append(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        lines.append("")
    if config is not None:
        lines.append("## final merged config")
        lines.append(json.dumps(config, indent=2, default=str))
        lines.append("")
    if extra:
        lines.append("## notes")
        lines.append(str(extra))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

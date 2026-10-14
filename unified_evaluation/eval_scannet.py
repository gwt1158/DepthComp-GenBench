"""Strict single-domain ScanNet-1038 evaluation wrapper.

Every method is evaluated through its existing ``eval_geolayernet_v2.py``;
this module only fixes the registered data/config, checks completeness, and
writes an aggregation-free ScanNet status record.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .geolayernet_protocol import load_dataset_config
from .ood_registry import load_ood16


EXPECTED = 1038


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _jsonl_count(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed comparison checkpoint on ScanNet-1038")
    parser.add_argument("--eval-script", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ood-config-dir", required=True)
    parser.add_argument("--ood-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--runtime-depth-unit-scale", type=float, required=True,
        choices=(10.0, 80.0),
        help=("source checkpoint normalization: 10 for N, 80 for K; fixed "
              "ScanNet visualization remains 0-10 m independently"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--save-visual-limit", type=int, default=EXPECTED)
    parser.add_argument("--visual-sample-list", default=None)
    parser.add_argument("--eval-extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    registry = load_ood16(args.ood_config_dir, args.ood_root)
    item = next(row for row in registry["datasets"] if row["dataset_name"] == "ScanNet")
    if item["status"] != "COMPLETE" or item["sample_count"] != EXPECTED:
        raise RuntimeError(
            f"ScanNet registry is not complete: status={item['status']} "
            f"count={item['sample_count']} reasons={item['reasons']}")
    config = load_dataset_config(item["config"])
    if config["dataset_type"] != "ood_npz":
        raise RuntimeError("ScanNet formal config must select ood_npz")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    eval_script = Path(args.eval_script).expanduser().resolve()
    if not checkpoint.is_file() or not eval_script.is_file():
        raise RuntimeError("checkpoint or eval script is missing")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"ScanNet output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(eval_script), "--dataset-config", item["config"],
        "--test-dataroot", item["dataroot"], "--test-phase", "test",
        "--dataset-name", "ScanNet", "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--device", args.device,
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--seed", str(args.seed), "--num-test", str(EXPECTED),
        "--depth-unit-scale", str(args.runtime_depth_unit_scale),
        "--protocol-ood-config-dir", str(Path(args.ood_config_dir).resolve()),
        "--protocol-ood-root", str(Path(args.ood_root).resolve()),
        "--no-clip-prediction", "--vis-depth-mode", "fixed",
        "--vis-depth-min", "0", "--vis-depth-max", "10", "--error-max", "1.0",
    ]
    if args.strict_checkpoint:
        command.append("--strict-checkpoint")
    if args.save_predictions:
        command.append("--save-predictions")
    if args.save_visuals:
        command.extend(["--save-visuals", "--save-visual-limit", str(args.save_visual_limit)])
        if args.visual_sample_list:
            command.extend(["--visual-sample-list", args.visual_sample_list])
    command.extend(args.eval_extra_arg)
    (output / "batch_command.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[ScanNet1038] " + " ".join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(eval_script.parent), check=False)
    if completed.returncode:
        return completed.returncode
    metrics = _read_json(output / "metrics.json") or {}
    if int(metrics.get("images", -1)) != EXPECTED:
        raise RuntimeError(f"ScanNet metrics images must be {EXPECTED}, got {metrics.get('images')}")
    prediction_count = None
    if args.save_predictions:
        index = output / "prediction_index.jsonl"
        prediction_count = _jsonl_count(index) if index.is_file() else 0
        if prediction_count != EXPECTED:
            raise RuntimeError(
                f"ScanNet prediction index must have {EXPECTED} entries; got {prediction_count}")
    # The one canonical formal panel per sample is written by the shared
    # renderer under visualization/concat.  The visualization root itself
    # intentionally contains modality subdirectories rather than PNG files.
    visual_count = len(list((output / "visualization" / "concat").glob("*.png")))
    if args.save_visuals and visual_count != min(args.save_visual_limit, EXPECTED):
        raise RuntimeError(
            f"ScanNet visualization count mismatch: got {visual_count}, "
            f"expected {min(args.save_visual_limit, EXPECTED)}")
    fairness = _read_json(output / "fairness.json") or {}
    provenance_verified = bool(fairness.get(
        "checkpoint_provenance_verified", True))
    completion_status = (
        "SCANNET1038_COMPLETE" if provenance_verified else
        "SCANNET1038_COMPLETE_UNVERIFIED_CHECKPOINT_PROVENANCE")
    status = {
        "schema": "comparison-scannet1038-eval-v1",
        "status": completion_status, "dataset": "ScanNet",
        "expected_samples": EXPECTED, "metrics_images": int(metrics["images"]),
        "prediction_count": prediction_count, "visualization_count": visual_count,
        "depth_unit": "m", "input_resolution": [228, 304],
        "metric_resolution": [228, 304], "prediction_resolution": [228, 304],
        "visualization_resolution": [228, 304], "checkpoint": str(checkpoint),
        "runtime_depth_unit_scale_m": args.runtime_depth_unit_scale,
        "checkpoint_provenance_verified": provenance_verified,
        "checkpoint_provenance_status": fairness.get(
            "checkpoint_provenance_status"),
        "controlled_ranking": bool(fairness.get("controlled_ranking", False)),
    }
    (output / "scannet1038_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PASS] {completion_status} predictions={prediction_count} visuals={visual_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

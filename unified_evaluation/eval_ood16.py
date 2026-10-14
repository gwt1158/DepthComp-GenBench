"""Run one fixed checkpoint over GeoLayerNet's formal OOD-16 registry.

The script only loops over existing data/config selections and invokes the
comparison model's normal eval entrypoint. It never trains, changes a model
option between domains, chooses a checkpoint from OOD metrics, or aggregates an
incomplete OOD-16 run.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from .dataset_paths import validate_ood_wrapper_paths
from .ood_registry import OOD16_TOTAL, load_ood16, write_registry


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _write_macro(path, registry, results):
    dataset_entries = registry.get("datasets") or [
        {
            "dataset_name": row.get("dataset_name"),
            "status": "COMPLETE" if row.get("status") == "PASS" else row.get("status"),
            "expected_sample_count": int((row.get("metrics") or {}).get("images", 0) or 0),
            "reasons": [],
        }
        for row in results
    ]
    not_measurable = [
        item for item in dataset_entries
        if item.get("status") == "NOT_MEASURABLE"]
    measurable_entries = [
        item for item in dataset_entries
        if item.get("status") != "NOT_MEASURABLE"]
    full_complete = not not_measurable and registry["status"] == "COMPLETE" and all(
        row.get("status") == "PASS" for row in results)
    measurable_complete = bool(not_measurable) and all(
        item.get("status") == "COMPLETE" for item in measurable_entries) and all(
            row.get("status") in ("PASS", "NOT_MEASURABLE") for row in results) and sum(
                row.get("status") == "PASS" for row in results) == len(measurable_entries)
    passed = [row for row in results if row.get("status") == "PASS"]
    if full_complete:
        status = "COMPLETE"
    elif measurable_complete:
        status = f"OOD-{len(measurable_entries)}-MEASURABLE_COMPLETE"
    else:
        status = "INCOMPLETE"
    payload = {
        "status": status,
        "dataset_order": registry["dataset_order"],
        "groups": registry["groups"],
        "checkpoint": results[0].get("checkpoint") if results else None,
        "aggregation": (
            "unweighted mean of all 16 per-domain finalized metric fields"
            if full_complete else
            "unweighted mean of measurable per-domain finalized metric fields"
            if measurable_complete else None),
        "metrics": {},
        "expected_prediction_count": OOD16_TOTAL,
        "expected_measurable_prediction_count": sum(
            int(item["expected_sample_count"]) for item in measurable_entries),
        "evaluated_prediction_count": sum(
            int((row.get("metrics") or {}).get("images", 0) or 0)
            for row in results),
        "measurable_domain_count": len(measurable_entries),
        "not_measurable_domains": [item["dataset_name"] for item in not_measurable],
        "not_measurable_reasons": {
            item["dataset_name"]: list(item.get("reasons", []))
            for item in not_measurable},
        "macro_16": "available" if full_complete else "unavailable",
        "macro_measurable": (
            f"Macro-{len(measurable_entries)}" if measurable_complete else
            "Macro-16" if full_complete else "unavailable"),
    }
    protocols = [row.get("protocol") for row in results
                 if isinstance(row.get("protocol"), dict)]
    provenance_fields = (
        "checkpoint_path", "checkpoint_loading", "checkpoint_registered",
    )
    if protocols:
        for name in provenance_fields:
            values = [protocol.get(name) for protocol in protocols]
            if values and all(value == values[0] for value in values):
                payload[name] = values[0]
    if full_complete or measurable_complete:
        metric_names = ("RMSE_m", "MAE_m", "REL", "delta1(%)", "delta2(%)", "delta3(%)",
                        "FinalMissing_RMSE_m", "Observed_RMSE_m")
        for name in metric_names:
            values = [r["metrics"].get(name) for r in passed
                      if isinstance(r.get("metrics", {}).get(name), (int, float))]
            if len(values) == len(passed):
                payload["metrics"][name] = sum(values) / len(values)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["status"] + list(payload["metrics"].keys()))
        writer.writerow([payload["status"]] + list(payload["metrics"].values()))
    return payload


def main(argv=None, checkpoint_required=True):
    parser = argparse.ArgumentParser(description="Evaluate one comparison checkpoint on GeoLayerNet OOD-16")
    parser.add_argument("--eval-script", required=True,
                        help="current comparison repository eval entrypoint")
    parser.add_argument(
        "--checkpoint", required=checkpoint_required, default=None,
        help=("fixed learned-method checkpoint; omitted only when the calling "
              "adapter explicitly declares a checkpoint-free method"))
    parser.add_argument("--ood-config-dir", required=True,
                        help="single OOD template YAML or a directory of per-domain YAML files")
    parser.add_argument("--ood-root", required=True,
                        help="exact registered GeoLayerNet OOD root")
    parser.add_argument("--output-root", required=True,
                        help="one run root; each domain is written below it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test", type=int, default=100000)
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--save-visual-limit", type=int, default=8)
    parser.add_argument("--visual-sample-list", default=None)
    parser.add_argument("--visual-seed", type=int, default=None,
                        help="debug-only deterministic selection when no fixed list is supplied")
    parser.add_argument("--eval-extra-arg", action="append", default=[],
                        help="repeatable argument forwarded verbatim to the method eval entrypoint")
    parser.add_argument("--require-npz-key", action="append", default=[],
                        help="repeatable strict per-sample NPZ metadata requirement, e.g. intrinsics")
    parser.add_argument(
        "--require-npz-key-for", action="append", default=[], metavar="DATASET:KEY",
        help=("repeatable dataset-specific NPZ requirement, e.g. "
              "VOID1500:intrinsics; unlike --require-npz-key it does not block "
              "domains whose official protocol supplies fixed camera geometry"))
    parser.add_argument(
        "--unsupported-dataset", action="append", default=[], metavar="DATASET:REASON",
        help=("repeatable method capability block. The domain is recorded and skipped, "
              "and Macro-16 remains unavailable; it is never silently dropped"))
    parser.add_argument(
        "--not-measurable-dataset", action="append", default=[], metavar="DATASET:REASON",
        help=("repeatable declared measurement limitation. The domain is recorded "
              "as NOT_MEASURABLE, skipped, and Macro-16 remains unavailable"))
    parser.add_argument(
        "--require-ood16", action="store_true",
        help=("fail unless all 16 domains are measurable and pass; by default a "
              "declared, otherwise complete measurable subset exits successfully"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run:
        validate_ood_wrapper_paths(args.ood_config_dir, args.ood_root)

    checkpoint = (str(Path(args.checkpoint).expanduser().resolve())
                  if args.checkpoint else None)
    eval_script = str(Path(args.eval_script).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    required_by_dataset = {}
    for value in args.require_npz_key_for:
        if ":" not in value:
            parser.error("--require-npz-key-for must use DATASET:KEY")
        dataset, key = (part.strip() for part in value.split(":", 1))
        if not dataset or not key:
            parser.error("--require-npz-key-for must use non-empty DATASET:KEY")
        required_by_dataset.setdefault(dataset, []).append(key)
    unsupported = {}
    for value in args.unsupported_dataset:
        if ":" not in value:
            parser.error("--unsupported-dataset must use DATASET:REASON")
        dataset, reason = (part.strip() for part in value.split(":", 1))
        if not dataset or not reason:
            parser.error("--unsupported-dataset must use non-empty DATASET:REASON")
        unsupported[dataset] = reason
    not_measurable = {}
    for value in args.not_measurable_dataset:
        if ":" not in value:
            parser.error("--not-measurable-dataset must use DATASET:REASON")
        dataset, reason = (part.strip() for part in value.split(":", 1))
        if not dataset or not reason:
            parser.error("--not-measurable-dataset requires non-empty DATASET:REASON")
        not_measurable[dataset] = reason
    registry = load_ood16(
        args.ood_config_dir, args.ood_root,
        required_npz_keys=args.require_npz_key,
        required_npz_keys_by_dataset=required_by_dataset,
        unsupported_datasets=unsupported,
        not_measurable_datasets=not_measurable)
    output_root.mkdir(parents=True, exist_ok=True)
    write_registry(output_root / "ood16_registry.json", registry)
    results = []

    for item in registry["datasets"]:
        display = item["dataset_name"]
        leaf = output_root / display
        row = {"dataset_name": display, "config": item["config"],
               "dataroot": item["dataroot"], "checkpoint": checkpoint,
               "status": item["status"], "metrics": {}}
        if item["status"] != "COMPLETE":
            if item["status"] == "BLOCKED_METADATA":
                row["status"] = "SKIPPED_BLOCKED_METADATA"
            elif item["status"] == "BLOCKED_METHOD":
                row["status"] = "SKIPPED_UNSUPPORTED_METHOD"
            elif item["status"] == "NOT_MEASURABLE":
                row["status"] = "NOT_MEASURABLE"
            else:
                row["status"] = "SKIPPED_INCOMPLETE"
            row["reasons"] = item["reasons"]
            results.append(row)
            print(f"[ood16] skip {display}: {'; '.join(item['reasons'])}")
            continue

        command = [
            sys.executable, eval_script,
            "--dataset-config", item["config"],
            "--test-dataroot", item["dataroot"],
            "--dataset-name", display,
            "--output-dir", str(leaf),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--num-test", str(args.num_test),
            "--protocol-ood-config-dir", str(
                Path(args.ood_config_dir).expanduser().resolve()),
            "--protocol-ood-root", str(
                Path(args.ood_root).expanduser().resolve()),
        ]
        if checkpoint is not None:
            command += ["--checkpoint", checkpoint]
        command.extend(args.eval_extra_arg)
        if args.strict_checkpoint:
            command.append("--strict-checkpoint")
        if args.save_visuals:
            command += ["--save-visuals", "--save-visual-limit", str(args.save_visual_limit)]
            if args.visual_sample_list:
                command += ["--visual-sample-list", args.visual_sample_list]
            if args.visual_seed is not None:
                command += ["--visual-seed", str(args.visual_seed)]
        (leaf / "batch_command.txt").parent.mkdir(parents=True, exist_ok=True)
        (leaf / "batch_command.txt").write_text(" ".join(command) + "\n")
        print("[ood16] " + " ".join(command))
        if args.dry_run:
            row["status"] = "DRY_RUN"
            results.append(row)
            continue
        completed = subprocess.run(command, cwd=str(Path.cwd()))
        protocol = _json(leaf / "eval_protocol.json") or {}
        if protocol:
            row["protocol"] = protocol
        if completed.returncode != 0:
            row["status"] = "FAIL"
            row["returncode"] = completed.returncode
        else:
            metrics = _json(leaf / "metrics.json") or {}
            actual = int(metrics.get("images", -1))
            expected = int(item["expected_sample_count"])
            row["status"] = "PASS" if actual == expected else "FAIL"
            if actual != expected:
                row["returncode"] = 4
                row["reason"] = (
                    f"metrics image count mismatch: expected {expected}, got {actual}")
            row["metrics"] = metrics
        results.append(row)
        (output_root / "ood16_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    (output_root / "ood16_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    macro = _write_macro(output_root / "ood16_macro.json", registry, results)
    if macro["status"].endswith("MEASURABLE_COMPLETE"):
        print(f"[ood16] status={macro['status']}; {macro['macro_measurable']} available; "
              "Macro-16 unavailable")
    elif registry["status"] != "COMPLETE":
        print("[ood16] status=INCOMPLETE; Macro-16 was not calculated")
    elif not all(r.get("status") == "PASS" for r in results):
        print("[ood16] status=INCOMPLETE; at least one domain evaluation failed, Macro-16 was not calculated")
    else:
        print(f"[ood16] status=COMPLETE; evaluated {len(results)} formal domains")
    if any(row.get("status") == "FAIL" for row in results):
        return 2
    if any(row.get("status") in ("SKIPPED_BLOCKED_METADATA", "SKIPPED_UNSUPPORTED_METHOD")
           for row in results):
        return 3
    if args.require_ood16 and macro["status"] != "COMPLETE":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate one fixed checkpoint on the registered OOD-15 partial matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .ood_registry import (OOD15_PARTIAL_TOTAL, load_ood15_partial,
                           write_registry)
from .dataset_paths import validate_ood_wrapper_paths


METRIC_NAMES = (
    "RMSE_m", "MAE_m", "REL", "Log10", "iRMSE", "iMAE",
    "delta1(%)", "delta2(%)", "delta3(%)", "FinalMissing_RMSE_m",
    "Observed_RMSE_m",
)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _prediction_count(path):
    index = Path(path) / "prediction_index.jsonl"
    if not index.is_file():
        return 0
    return sum(1 for line in index.read_text(encoding="utf-8").splitlines()
               if line.strip())


def _metric_macro(rows):
    metrics = {}
    for name in METRIC_NAMES:
        values = [row.get("metrics", {}).get(name) for row in rows]
        if values and all(isinstance(value, (int, float)) for value in values):
            metrics[name] = sum(values) / len(values)
    return metrics


def _write_macro(output_root, rows, measurable_protocol=None):
    if measurable_protocol is not None:
        excluded = set(measurable_protocol["not_measurable_domains"])
        measurable_rows = [row for row in rows
                           if row["dataset_name"] not in excluded]
        count = sum(int(row.get("prediction_count", 0))
                    for row in measurable_rows)
        complete = (
            len(measurable_rows) == measurable_protocol["domain_count"]
            and all(row.get("status") == "PASS" for row in measurable_rows)
            and count == measurable_protocol["prediction_count"]
        )
        metrics = _metric_macro(measurable_rows) if complete else {}
        payload = {
            "protocol_name": measurable_protocol["protocol_name"],
            "status": (measurable_protocol["complete_status"]
                       if complete else "INCOMPLETE"),
            "measurable_domain_count": measurable_protocol["domain_count"],
            "prediction_count": count,
            "expected_prediction_count": measurable_protocol["prediction_count"],
            "not_measurable_domains": list(
                measurable_protocol["not_measurable_domains"]),
            "not_measurable_reason": measurable_protocol["reason"],
            "domain_status": {
                name: "NOT_MEASURABLE"
                for name in measurable_protocol["not_measurable_domains"]
            },
            "Macro-13 measurable": metrics,
            "Macro-15 partial": "unavailable",
            "Macro-16": "unavailable",
            "aggregation": (
                "unweighted mean of 13 measurable per-domain finalized metrics"),
            "metrics": metrics,
        }
        stem = measurable_protocol["output_stem"]
        (output_root / f"{stem}_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        with (output_root / f"{stem}_macro.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "status", "measurable_domain_count", "prediction_count",
                *metrics])
            writer.writerow([
                payload["status"], payload["measurable_domain_count"], count,
                *metrics.values()])
        return complete

    passed = bool(rows) and all(row.get("status") == "PASS" for row in rows)
    count = sum(int(row.get("prediction_count", 0)) for row in rows)
    complete = passed and len(rows) == 15 and count == OOD15_PARTIAL_TOTAL
    metrics = _metric_macro(rows) if complete else {}
    payload = {
        "protocol_name": "Macro-15 partial",
        "status": "OOD-15_PARTIAL_COMPLETE" if complete else "INCOMPLETE",
        "domains": len(rows),
        "prediction_count": count,
        "expected_prediction_count": OOD15_PARTIAL_TOTAL,
        "ScanNet": "not evaluated",
        "Macro-16": "unavailable",
        "aggregation": "unweighted mean of 15 per-domain finalized metrics",
        "metrics": metrics,
    }
    (output_root / "ood15_partial_macro.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (output_root / "ood15_partial_macro.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["status", "domains", "prediction_count"] + list(metrics))
        writer.writerow([payload["status"], 15, count] + list(metrics.values()))
    return complete


def _parser(description=None, include_eval_script=True):
    parser = argparse.ArgumentParser(
        description=(description or
                     "Evaluate one fixed comparison checkpoint on OOD-15 partial (ScanNet absent)"))
    if include_eval_script:
        parser.add_argument("--eval-script", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ood-config-dir", required=True,
                        help="single OOD template YAML or a directory of per-domain YAML files")
    parser.add_argument("--ood-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test", type=int, default=100000)
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--save-visual-limit", type=int, default=100000)
    parser.add_argument("--eval-extra-arg", action="append", default=[])
    parser.add_argument("--require-npz-key-for", action="append", default=[])
    parser.add_argument(
        "--not-measurable-dataset", action="append", default=[],
        metavar="DATASET:REASON",
        help=("record a method-geometry limitation explicitly; the domain is "
              "not evaluated, strict OOD-15 remains unavailable, and a configured "
              "measurable subset may still complete"))
    parser.add_argument(
        "--require-ood15", action="store_true",
        help=("require all 15 domains even when a method declares domains "
              "NOT_MEASURABLE; otherwise a configured measurable subset may complete"))
    parser.add_argument("--measurable-protocol-name", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--measurable-complete-status", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--measurable-output-stem", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--expected-measurable-domain-count", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--expected-measurable-prediction-count", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--not-measurable-reason-code", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        validate_ood_wrapper_paths(args.ood_config_dir, args.ood_root)

    required = {}
    for item in args.require_npz_key_for:
        if ":" not in item:
            parser.error("--require-npz-key-for must use DATASET:KEY")
        dataset, key = item.split(":", 1)
        required.setdefault(dataset, []).append(key)
    not_measurable = {}
    for item in args.not_measurable_dataset:
        if ":" not in item:
            parser.error("--not-measurable-dataset must use DATASET:REASON")
        dataset, reason = item.split(":", 1)
        if not dataset or not reason:
            parser.error("--not-measurable-dataset requires non-empty DATASET and REASON")
        not_measurable[dataset] = reason
    measurable_values = (
        args.measurable_protocol_name, args.measurable_complete_status,
        args.measurable_output_stem, args.expected_measurable_domain_count,
        args.expected_measurable_prediction_count,
        args.not_measurable_reason_code,
    )
    if any(value is not None for value in measurable_values) and not all(
            value is not None for value in measurable_values):
        parser.error("measurable protocol configuration must be complete")
    measurable_protocol = None
    if all(value is not None for value in measurable_values):
        measurable_protocol = {
            "protocol_name": args.measurable_protocol_name,
            "complete_status": args.measurable_complete_status,
            "output_stem": args.measurable_output_stem,
            "domain_count": int(args.expected_measurable_domain_count),
            "prediction_count": int(args.expected_measurable_prediction_count),
            "not_measurable_domains": list(not_measurable),
            "reason": args.not_measurable_reason_code,
        }
        if len(not_measurable) + measurable_protocol["domain_count"] != 15:
            parser.error("measurable and NOT_MEASURABLE domain counts must total 15")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_ood15_partial(
        args.ood_config_dir, args.ood_root,
        required_npz_keys_by_dataset=required,
        not_measurable_datasets=not_measurable)
    if measurable_protocol is not None:
        measurable_entries = [
            item for item in registry["datasets"]
            if item["dataset_name"] not in set(
                measurable_protocol["not_measurable_domains"])
        ]
        registry["protocol_name"] = measurable_protocol["protocol_name"]
        registry["measurable_domain_count"] = measurable_protocol["domain_count"]
        registry["expected_prediction_count"] = measurable_protocol["prediction_count"]
        registry["measurable_prediction_count"] = sum(
            int(item["sample_count"]) for item in measurable_entries)
        registry["not_measurable_domains"] = list(
            measurable_protocol["not_measurable_domains"])
        registry["not_measurable_reason"] = measurable_protocol["reason"]
        registry["Macro-13 measurable"] = "available after complete evaluation"
        registry["Macro-15 partial"] = "unavailable"
        registry["Macro-16"] = "unavailable"
        measurable_ready = (
            len(measurable_entries) == measurable_protocol["domain_count"]
            and all(item["status"] == "COMPLETE" for item in measurable_entries)
            and registry["measurable_prediction_count"]
            == measurable_protocol["prediction_count"]
        )
        registry["status"] = (
            "OOD-13_MEASURABLE_READY" if measurable_ready else "INCOMPLETE")
        file_stem = measurable_protocol["output_stem"]
    else:
        measurable_ready = False
        file_stem = "ood15_partial"
    write_registry(output_root / f"{file_stem}_registry.json", registry)
    rows = []
    for item in registry["datasets"]:
        leaf = output_root / item["dataset_name"]
        row = {
            "dataset_name": item["dataset_name"],
            "expected_sample_count": item["expected_sample_count"],
            "status": item["status"],
            "reasons": item.get("reasons", []),
        }
        if item["status"] != "COMPLETE":
            row["status"] = (
                "SKIPPED_NOT_MEASURABLE"
                if item["status"] == "NOT_MEASURABLE"
                else "SKIPPED_INCOMPLETE")
            rows.append(row)
            continue
        command = [
            sys.executable, str(Path(args.eval_script).expanduser().resolve()),
            "--dataset-config", item["config"], "--test-dataroot", item["dataroot"],
            "--dataset-name", item["dataset_name"], "--output-dir", str(leaf),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers), "--seed", str(args.seed),
            "--num-test", str(args.num_test), "--checkpoint",
            str(Path(args.checkpoint).expanduser().resolve()),
            "--protocol-ood-config-dir", str(
                Path(args.ood_config_dir).expanduser().resolve()),
            "--protocol-ood-root", str(
                Path(args.ood_root).expanduser().resolve()),
        ]
        command.extend(args.eval_extra_arg)
        if args.strict_checkpoint:
            command.append("--strict-checkpoint")
        if args.save_visuals:
            command.extend(["--save-visuals", "--save-visual-limit",
                            str(args.save_visual_limit)])
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "batch_command.txt").write_text(" ".join(command) + "\n")
        label = "ood13-measurable" if measurable_protocol is not None else "ood15-partial"
        print(f"[{label}] " + " ".join(command))
        if args.dry_run:
            row["status"] = "DRY_RUN"
        else:
            completed = subprocess.run(command, cwd=str(Path.cwd()))
            row["returncode"] = completed.returncode
            row["metrics"] = _read_json(leaf / "metrics.json") or {}
            row["prediction_count"] = _prediction_count(leaf)
            row["status"] = "PASS" if (
                completed.returncode == 0
                and row["prediction_count"] == item["expected_sample_count"]
            ) else "FAIL"
        rows.append(row)
        (output_root / f"{file_stem}_results.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    (output_root / f"{file_stem}_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    if args.dry_run:
        if measurable_protocol is not None and not args.require_ood15:
            return 0 if measurable_ready else 3
        return 0 if registry["status"] == "OOD-15_PARTIAL_COMPLETE" else 3
    complete = _write_macro(output_root, rows, measurable_protocol)
    if measurable_protocol is not None and args.require_ood15:
        return 2
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

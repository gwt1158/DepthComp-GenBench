#!/usr/bin/env python3
"""Prepare GeoLayerNet KITTI manifest protocol.

This script scans KITTI official raw/depth-completion folders and writes only
JSONL manifests + metadata. It never copies, re-encodes, or symlinks PNG files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_TRAIN_FULL = 85898
EXPECTED_VAL1000 = 1000


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lines(lines) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _drive_date(drive: str) -> str:
    return "_".join(drive.split("_")[:3])


def _sample_id(drive: str, camera: str, frame: str) -> str:
    return f"{drive}/{camera}/{Path(frame).stem}"


def _parse_val_selection_name(stem: str, folder_name: str = ""):
    drive_match = re.search(r"(20\d\d_\d\d_\d\d_drive_\d+_sync)", stem)
    cam_match = re.search(r"(image_0[23])", stem)
    frame_match = re.search(r"(\d{10})", stem)
    drive = drive_match.group(1) if drive_match else "val_selection_cropped"
    camera = cam_match.group(1) if cam_match else (folder_name if folder_name in ("image_02", "image_03") else "image")
    frame = f"{frame_match.group(1)}.png" if frame_match else f"{stem}.png"
    date = _drive_date(drive) if drive != "val_selection_cropped" else ""
    return date, drive, camera, frame


def scan_train(src: Path):
    records = []
    missing_sparse = 0
    missing_rgb = 0
    gt_paths = sorted((src / "train").glob("*/proj_depth/groundtruth/*/*.png"))
    for gt in gt_paths:
        drive = gt.parents[3].name
        camera = gt.parent.name
        frame = gt.name
        date = _drive_date(drive)
        sparse = src / "train" / drive / "proj_depth" / "velodyne_raw" / camera / frame
        rgb = src / date / drive / camera / "data" / frame
        if not sparse.is_file():
            missing_sparse += 1
            continue
        if not rgb.is_file():
            missing_rgb += 1
            continue
        records.append({
            "sample_id": _sample_id(drive, camera, frame),
            "date": date,
            "drive": drive,
            "camera": camera,
            "frame": frame,
            "rgb": _rel(rgb, src),
            "sparse_depth": _rel(sparse, src),
            "target_depth": _rel(gt, src),
        })
    return records, {"gt": len(gt_paths), "missing_sparse": missing_sparse, "missing_rgb": missing_rgb}


def _iter_val_rgb_paths(base: Path):
    for folder in ("image", "rgb", "image_02", "image_03"):
        d = base / folder
        if d.is_dir():
            yield from sorted(d.glob("*.png"))


def _val_key(path: Path):
    date, drive, camera, frame = _parse_val_selection_name(path.stem, path.parent.name)
    return (drive, camera, Path(frame).stem), (date, drive, camera, frame)


def _unique_val_index(paths):
    index = {}
    duplicates = 0
    for p in sorted(paths):
        key, _ = _val_key(p)
        if key in index:
            duplicates += 1
            index[key] = None
        else:
            index[key] = p
    return {k: v for k, v in index.items() if v is not None}, duplicates


def scan_val1000(src: Path):
    base = src / "depth_selection" / "val_selection_cropped"
    gt_dir = base / "groundtruth_depth"
    sparse_dir = base / "velodyne_raw"
    records = []
    missing_sparse = 0
    missing_rgb = 0
    sparse_index, sparse_dupes = _unique_val_index(sparse_dir.glob("*.png"))
    rgb_index, rgb_dupes = _unique_val_index(_iter_val_rgb_paths(base))
    gt_paths = sorted(gt_dir.glob("*.png"))
    for gt in gt_paths:
        key, parsed = _val_key(gt)
        sparse = sparse_index.get(key)
        rgb = rgb_index.get(key)
        if sparse is None:
            missing_sparse += 1
            continue
        if rgb is None:
            missing_rgb += 1
            continue
        date, drive, camera, frame = parsed
        records.append({
            "sample_id": _sample_id(drive, camera, frame),
            "date": date,
            "drive": drive,
            "camera": camera,
            "frame": frame,
            "rgb": _rel(rgb, src),
            "sparse_depth": _rel(sparse, src),
            "target_depth": _rel(gt, src),
        })
    return records, {
        "gt": len(gt_paths),
        "sparse": len(sparse_index),
        "rgb": len(rgb_index),
        "missing_sparse": missing_sparse,
        "missing_rgb": missing_rgb,
        "duplicate_sparse_keys": sparse_dupes,
        "duplicate_rgb_keys": rgb_dupes,
    }


def split_by_drive(records, seed: int, dev_ratio: float):
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["date"], rec["drive"])].append(rec)
    items = sorted(groups.items(), key=lambda kv: kv[0])
    rng = random.Random(seed)
    rng.shuffle(items)
    counts = [len(v) for _, v in items]
    total = sum(counts)
    target = int(round(total * dev_ratio))

    reachable = {0: None}
    parents = {}
    for idx, count in enumerate(counts):
        for s in list(reachable.keys()):
            ns = s + count
            if ns not in reachable:
                reachable[ns] = (s, idx)
                parents[ns] = (s, idx)
    best_sum = min(reachable.keys(), key=lambda s: (abs(s - target), abs(s - int(total * dev_ratio)), s > target, s))
    dev_indices = set()
    cur = best_sum
    while cur:
        prev, idx = parents[cur]
        dev_indices.add(idx)
        cur = prev

    train, dev = [], []
    for idx, (_, recs) in enumerate(items):
        (dev if idx in dev_indices else train).extend(recs)
    train = sorted(train, key=lambda r: r["sample_id"])
    dev = sorted(dev, key=lambda r: r["sample_id"])
    return train, dev, {"target_dev": target, "actual_dev": len(dev), "drive_groups": len(items)}


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _stats(records):
    return {
        "samples": len(records),
        "dates": len({r["date"] for r in records if r.get("date")}),
        "drives": len({r["drive"] for r in records}),
        "cameras": dict(Counter(r["camera"] for r in records)),
    }


def _ids(records):
    return {r["sample_id"] for r in records}


def _drives(records):
    return {(r["date"], r["drive"]) for r in records}


def _assert_unique(records, name):
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        dup = [k for k, v in Counter(ids).items() if v > 1][:10]
        raise RuntimeError(f"{name}: duplicate sample_id: {dup}")


def write_audit_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["check", "value", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare KITTI GeoLayerNet protocol manifests.")
    parser.add_argument("--src", default="data/kitti_official_raw")
    parser.add_argument("--dst", default="data/kitti_geolayernet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--expected-train-count", type=int, default=EXPECTED_TRAIN_FULL)
    parser.add_argument("--expected-val1000-count", type=int, default=EXPECTED_VAL1000)
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    protocol = dst / "protocol"
    if not src.is_dir():
        raise FileNotFoundError(f"KITTI official source root not found: {src}")

    train_full, train_scan = scan_train(src)
    val1000, val_scan = scan_val1000(src)
    _assert_unique(train_full, "train_full")
    _assert_unique(val1000, "val1000")
    if len(train_full) != args.expected_train_count:
        raise RuntimeError(f"train_full expected {args.expected_train_count}, got {len(train_full)}; scan={train_scan}")
    if len(val1000) != args.expected_val1000_count:
        raise RuntimeError(f"val1000 expected {args.expected_val1000_count}, got {len(val1000)}; scan={val_scan}")

    train90, dev10, split_info = split_by_drive(train_full, args.seed, args.dev_ratio)
    if len(train90) + len(dev10) != len(train_full):
        raise RuntimeError("train90 + dev10 does not equal train_full")
    if _ids(train90) & _ids(dev10):
        raise RuntimeError("train/dev sample_id overlap is non-zero")
    drive_overlap = _drives(train90) & _drives(dev10)
    if drive_overlap:
        raise RuntimeError(f"train/dev date+drive overlap is non-zero: {sorted(drive_overlap)[:5]}")
    val_train_overlap = _ids(val1000) & _ids(train_full)
    if val_train_overlap:
        raise RuntimeError(f"val1000/train_full sample_id overlap is non-zero: {sorted(val_train_overlap)[:5]}")

    files = {
        "train90": protocol / "train90.jsonl",
        "dev10": protocol / "dev10.jsonl",
        "train_full": protocol / "train_full.jsonl",
        "val1000": protocol / "val1000.jsonl",
    }
    for name, records in (("train90", train90), ("dev10", dev10), ("train_full", train_full), ("val1000", val1000)):
        _write_jsonl(files[name], records)

    hashes = {name: _sha256_file(path) for name, path in files.items()}
    summary = {
        "protocol_status": "COMPLETE",
        "source_root": str(src),
        "source_root_realpath": str(src.resolve()),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "split_rule": "grouped by complete KITTI date+drive; deterministic subset closest to target dev sample count",
        "train_scan": train_scan,
        "val1000_scan": val_scan,
        "split_info": split_info,
        "counts": {
            "train_full": len(train_full),
            "train90": len(train90),
            "dev10": len(dev10),
            "val1000": len(val1000),
            "train_dev_sample_overlap": len(_ids(train90) & _ids(dev10)),
            "train_dev_drive_overlap": len(drive_overlap),
            "val1000_train_full_sample_overlap": len(val_train_overlap),
        },
        "stats": {
            "train_full": _stats(train_full),
            "train90": _stats(train90),
            "dev10": _stats(dev10),
            "val1000": _stats(val1000),
        },
        "manifest_sha256": hashes,
    }
    (protocol / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    audit_rows = [
        {"check": "official_train_total", "value": len(train_full), "status": "OK"},
        {"check": "train90", "value": len(train90), "status": "OK"},
        {"check": "dev10", "value": len(dev10), "status": "OK"},
        {"check": "train_dev_drive_overlap", "value": len(drive_overlap), "status": "OK"},
        {"check": "val1000", "value": len(val1000), "status": "OK"},
        {"check": "train_dev_val1000_sample_overlap", "value": len((_ids(train90) | _ids(dev10)) & _ids(val1000)), "status": "OK"},
        {"check": "protocol_status", "value": "COMPLETE", "status": "OK"},
    ]
    write_audit_csv(protocol / "audit.csv", audit_rows)

    print(f"official train total={len(train_full)}")
    print(f"train90={len(train90)}")
    print(f"dev10={len(dev10)}")
    print(f"train/dev date+drive overlap={len(drive_overlap)}")
    print(f"official val1000={len(val1000)}")
    print(f"train/dev/val1000 sample_id intersection={len((_ids(train90) | _ids(dev10)) & _ids(val1000))}")
    print(f"protocol status=COMPLETE -> {protocol}")


if __name__ == "__main__":
    main()

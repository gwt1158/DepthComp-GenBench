"""Aggregate per-run metrics.csv files into one summary table.

Scans ``<results-dir>/*/metrics.csv`` (each written by the comparison eval runner
in the same GeoLayerNet schema), unions all columns, and writes a single
``summary_metrics.csv`` with one row per run (the run/method name is the eval
directory name). NaN/empty cells are preserved.

Usage:
  python unified_evaluation/collect_metrics.py --results-dir results_compare --out results_compare/summary_metrics.csv
"""

import argparse
import csv
import glob
import os


def _read_metrics_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    keys, vals = rows[0], rows[1]
    return dict(zip(keys, vals))


def main():
    ap = argparse.ArgumentParser(description="collect comparison metrics into one CSV")
    ap.add_argument("--results-dir", required=True, help="dir containing <run>/metrics.csv subfolders")
    ap.add_argument("--out", required=True, help="output summary CSV path")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*", "metrics.csv")))
    if not paths:
        print(f"[collect] no metrics.csv found under {args.results_dir}/*/")
    records = []
    all_cols = []
    for p in paths:
        m = _read_metrics_csv(p)
        if m is None:
            print(f"[collect] skipping malformed {p}")
            continue
        run_name = os.path.basename(os.path.dirname(p))
        rec = {"run": run_name}
        rec.update(m)
        records.append(rec)
        for k in rec:
            if k not in all_cols:
                all_cols.append(k)

    # ensure "run" first
    if "run" in all_cols:
        all_cols = ["run"] + [c for c in all_cols if c != "run"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    print(f"[collect] wrote {len(records)} runs x {len(all_cols)} cols -> {args.out}")


if __name__ == "__main__":
    main()

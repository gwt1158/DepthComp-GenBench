"""Summarize several eval runs into one paper main-table (CSV + Markdown).

Usage:
  python tools/summarize_results.py \
    --runs r11_proxy=results/r11_retrain_seed42_proxyonly_eval \
           r11_safe=results/r11_retrain_seed42_safe_eval \
           enh_proxy=results/geolayernet_enhanced_full_proxyonly_eval \
           enh_safe=results/geolayernet_enhanced_full_safe_eval \
    --out results/summary_main_table.csv

Reads each run's metrics.json (preferred) or metrics.csv. Missing fields (e.g.
older runs without the new metrics) are left blank — never crashes.
"""

import argparse
import csv
import json
import os

# (column in table, key(s) in metrics with fallback order)
COLUMNS = [
    ("Method", None), ("Output", None),
    ("RMSE_m", ["RMSE_m"]), ("RMSE_mm", ["RMSE_mm"]),
    ("MAE_m", ["MAE_m"]), ("MAE_mm", ["MAE_mm"]),
    ("AbsRel", ["AbsRel", "REL"]), ("SqRel", ["SqRel"]),
    ("Log10", ["Log10"]), ("LogRMSE", ["LogRMSE"]), ("SILog", ["SILog"]),
    ("delta1", ["delta1", "delta1(%)"]), ("delta2", ["delta2", "delta2(%)"]),
    ("delta3", ["delta3", "delta3(%)"]),
    ("iRMSE", ["iRMSE"]), ("iMAE", ["iMAE"]),
    ("Final_Missing_RMSE_m", ["Final_Missing_RMSE_m", "FinalMissing_RMSE_m"]),
    ("Boundary_RMSE_m", ["Boundary_RMSE_m"]),
    ("Gradient_MAE_m", ["Gradient_MAE_m"]),
    ("Normal_MAE_deg", ["Normal_MAE_deg"]),
    ("Params(M)", ["Params(M)"]), ("FPS", ["FPS"]),
]

# Relative-improvement columns (computed, appended after the metric columns).
# Main baseline is r11_safe (final-output vs final-output); r11_proxy is also
# provided for the proxy comparison. Error metrics use (base - method)/base*100;
# delta1 uses method - base in percentage points (pp).
IMPROVE_COLS = [
    "Improve_vs_r11_proxy_RMSE(%)",
    "Improve_vs_r11_safe_RMSE(%)",
    "Improve_vs_r11_safe_MAE(%)",
    "Improve_vs_r11_safe_REL(%)",
    "Improve_vs_r11_safe_delta1(pp)",
]
PROXY_BASE, SAFE_BASE = "r11_proxy", "r11_safe"


def _num(metrics, keys):
    """First present numeric value among keys, else None."""
    for k in keys:
        v = metrics.get(k)
        if v in (None, ""):
            continue
        try:
            f = float(v)
            if f == f:  # not NaN
                return f
        except (TypeError, ValueError):
            pass
    return None


def _rel_improve(base, method):
    if base in (None, "") or method in (None, "") or base == 0:
        return ""
    return (base - method) / base * 100.0


def _abs_improve(base, method):
    if base in (None, "") or method in (None, ""):
        return ""
    return method - base


def _coerce_floats(d):
    """Best-effort convert string numeric values to float (leave the rest as-is)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


def load_metrics(run_dir, name=""):
    """Return a flat dict of metrics for one run, robust to a broken metrics.json.

    Priority: (1) metrics.json if present, non-empty and valid JSON;
    (2) else metrics.csv first data row (numeric fields coerced to float) — if the
    json is broken but the csv is usable, warn and continue, do NOT crash;
    (3) if neither is usable, print a clear error (run name, dir, missing files)
    and return {} so the rest of the table still builds."""
    jp = os.path.join(run_dir, "metrics.json")
    cp = os.path.join(run_dir, "metrics.csv")
    tag = name or run_dir

    if os.path.isfile(jp) and os.path.getsize(jp) > 0:
        try:
            with open(jp) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, ValueError, OSError) as ex:
            print(f"[warn] {tag}: metrics.json is invalid ({ex}); falling back to metrics.csv")

    if os.path.isfile(cp) and os.path.getsize(cp) > 0:
        try:
            rows = list(csv.reader(open(cp)))
            if len(rows) >= 2 and rows[0]:
                return _coerce_floats(dict(zip(rows[0], rows[1])))
            print(f"[warn] {tag}: metrics.csv has no data row ({cp})")
        except OSError as ex:
            print(f"[warn] {tag}: metrics.csv unreadable ({ex})")

    def _state(p):
        if not os.path.isfile(p):
            return "absent"
        return "empty" if os.path.getsize(p) == 0 else "present-but-unreadable"
    print(f"[error] run '{name}' at '{run_dir}': no usable metrics "
          f"(metrics.json: {_state(jp)}; metrics.csv: {_state(cp)})")
    return {}


def infer_output(name, run_dir):
    """Best-effort output-mode label (proxy_only / safe / raw)."""
    low = name.lower() + " " + run_dir.lower()
    info = os.path.join(run_dir, "run_info.txt")
    if os.path.isfile(info):
        for line in open(info):
            if line.lower().startswith("output_mode:"):
                return line.split(":", 1)[1].strip()
    for tag in ("proxy_only", "proxyonly", "proxy", "safe", "raw", "final"):
        if tag in low:
            return "proxy_only" if "proxy" in tag else tag
    return ""


def _fmt(v):
    if v in (None, ""):
        return ""
    try:
        f = float(v)
        if f != f:  # NaN
            return ""
        return f"{f:.4f}" if abs(f) < 1000 else f"{f:.2f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path entries")
    ap.add_argument("--out", required=True, help="output CSV path (.md written alongside)")
    args = ap.parse_args()

    table = []
    raw_by_name = {}
    for entry in args.runs:
        if "=" not in entry:
            print(f"[warn] skipping malformed --runs entry (need name=path): {entry}")
            continue
        name, path = entry.split("=", 1)
        m = load_metrics(path, name)   # robust: bad metrics.json -> metrics.csv -> clear error
        raw_by_name[name] = m
        row = {"Method": name, "Output": infer_output(name, path), "_metrics": m}
        for col, keys in COLUMNS:
            if keys is None:
                continue
            val = ""
            for k in keys:
                if k in m and m[k] not in (None, ""):
                    val = m[k]
                    break
            row[col] = val
        table.append(row)

    # ---- relative improvement vs the r11 baselines (blank if baseline absent) ----
    bp = raw_by_name.get(PROXY_BASE)
    bs = raw_by_name.get(SAFE_BASE)
    bp_rmse = _num(bp, ["RMSE_m"]) if bp else None
    bs_rmse = _num(bs, ["RMSE_m"]) if bs else None
    bs_mae = _num(bs, ["MAE_m"]) if bs else None
    bs_rel = _num(bs, ["AbsRel", "REL"]) if bs else None
    bs_d1 = _num(bs, ["delta1", "delta1(%)"]) if bs else None
    for row in table:
        m = row["_metrics"]
        row["Improve_vs_r11_proxy_RMSE(%)"] = _rel_improve(bp_rmse, _num(m, ["RMSE_m"]))
        row["Improve_vs_r11_safe_RMSE(%)"] = _rel_improve(bs_rmse, _num(m, ["RMSE_m"]))
        row["Improve_vs_r11_safe_MAE(%)"] = _rel_improve(bs_mae, _num(m, ["MAE_m"]))
        row["Improve_vs_r11_safe_REL(%)"] = _rel_improve(bs_rel, _num(m, ["AbsRel", "REL"]))
        row["Improve_vs_r11_safe_delta1(pp)"] = _abs_improve(bs_d1, _num(m, ["delta1", "delta1(%)"]))

    headers = [c for c, _ in COLUMNS] + IMPROVE_COLS
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for row in table:
            w.writerow({h: (row.get(h, "") if h in ("Method", "Output") else _fmt(row.get(h, ""))) for h in headers})

    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in table:
            cells = [str(row.get(h, "")) if h in ("Method", "Output") else _fmt(row.get(h, "")) for h in headers]
            fh.write("| " + " | ".join(cells) + " |\n")

    print(f"[summary] wrote {args.out} and {md_path} ({len(table)} runs)")


if __name__ == "__main__":
    main()

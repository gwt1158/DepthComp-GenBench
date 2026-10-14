"""Build NYUv2-48K (HF / FastDepth) sparse depth-completion datasets for GeoLayerNet.

Source: the HF / FastDepth NYUv2-48K webdataset tars (NOT the labeled 795/654 mat —
that small set is handled by tools/build_nyu_sparse_sets.py, which is left intact):
  train : train-000000.tar .. train-000011.tar   (12 tars)
  val   : val-000000.tar   .. val-000001.tar      (2 tars)
each tar holds .h5 files with datasets 'rgb' (-> H,W,3) and 'depth' (H,W meters).

Phases (HF has only train + validation):
  train -> 12 train tars ;  test -> 2 val tars ;  val -> the SAME 2 val tars
  (val/test share the same source -> audit records val_test_same_source=true).
splits.mat is NOT used here (that is only for the labeled-mat small set).

ONE protocol (PSD / TIP-2025 image preprocessing, identical to the mat tool):
  read RGB + dense depth -> resize to 320x240 -> center-crop to 304x228 (x=8:312, y=6:234)
  -> saved arrays are H x W = 228 x 304 -> sample sparse FROM THE CROPPED target.

Output (directly readable by Nyu500Dataset), one root per protocol:
  data/nyu48k_1pct   sparse = 1% of valid target pixels (~693)
  data/nyu48k_500/200/100/50/10   fixed point counts
  <root>/<phase>/{rgb, target_depth, sparse_depth, valid_mask}/<i:08d>.png
  rgb uint8; target_depth/sparse_depth uint16 (depth_m/10*65535); valid_mask uint8 0/255.

To save disk, --link-static hardlinks (else symlinks, else copies) rgb + target_depth
from data/nyu48k_1pct into the fixed-point roots; sparse_depth + valid_mask are always
generated per root. Deterministic sampling per crc32(split, sample_id, protocol, seed).

Example smoke (one line):
  python tools/build_nyu48k_sparse_sets.py --protocols 1pct,500,100,10 --limit 20 --overwrite --verify
"""

import argparse
import csv
import io
import os
import shutil
import sys
import tarfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_TAR_ROOT = "data/nyu_depth_v2_hf"
TRAIN_TARS = [f"train-{i:06d}.tar" for i in range(12)]
VAL_TARS = [f"val-{i:06d}.tar" for i in range(2)]
PHASE_TARS = {"train": TRAIN_TARS, "test": VAL_TARS, "val": VAL_TARS}

MAX_DEPTH = 10.0                      # meters -> 65535 (same GeoLayerNet depth scale as nyu500_mat)
RESIZE_W, RESIZE_H = 320, 240         # PSD-style resize (W x H)
CROP_X0, CROP_X1 = 8, 312             # center crop -> width 304
CROP_Y0, CROP_Y1 = 6, 234            # center crop -> height 228
OUT_H, OUT_W = 228, 304              # final saved H x W
PREPROCESS_TAG = "PSD_NYU_resize320x240_center_crop304x228"
SPLIT_SOURCE_TAG = "HF_tar_train12_val2"
ONEPCT_FRAC = 0.01
_SUBDIRS = ("rgb", "target_depth", "sparse_depth", "valid_mask")
_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in _EXTS}


def to_u16(depth_m):
    return np.round(np.clip(depth_m / MAX_DEPTH, 0.0, 1.0) * 65535.0).astype(np.uint16)


def psd_rgb(rgb_hwc):
    im = Image.fromarray(rgb_hwc.astype(np.uint8)).resize((RESIZE_W, RESIZE_H), Image.LANCZOS)
    return np.asarray(im)[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1, :]


def psd_depth(depth_hw):
    im = Image.fromarray(depth_hw.astype(np.float32), mode="F").resize((RESIZE_W, RESIZE_H), Image.NEAREST)
    return np.asarray(im)[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]


def read_h5(data_bytes):
    """Return (rgb HxWx3 uint8, depth HxW float meters) from one .h5 byte blob."""
    import h5py
    with h5py.File(io.BytesIO(data_bytes), "r") as f:
        rgb = np.array(f["rgb"])
        depth = np.array(f["depth"]).astype(np.float32)
    if rgb.ndim == 3 and rgb.shape[0] == 3:      # (3,H,W) -> (H,W,3)
        rgb = np.transpose(rgb, (1, 2, 0))
    elif rgb.ndim == 3 and rgb.shape[2] != 3:
        ch = next((ax for ax, s in enumerate(rgb.shape) if s == 3), None)
        if ch is not None:
            rgb = np.moveaxis(rgb, ch, -1)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    return rgb.astype(np.uint8), depth


def iter_phase(tar_root, phase, limit):
    """Yield (idx, h5_name, h5_bytes) for a phase, in a fixed deterministic order."""
    idx = 0
    for tarname in PHASE_TARS[phase]:
        path = Path(tar_root) / tarname
        if not path.is_file():
            raise RuntimeError(f"missing tar: {path}")
        with tarfile.open(path) as tf:
            members = sorted((m for m in tf.getmembers() if m.name.endswith(".h5")), key=lambda m: m.name)
            for m in members:
                if limit and idx >= limit:
                    return
                yield idx, os.path.basename(m.name), tf.extractfile(m).read()
                idx += 1


def n_points_for(protocol, target_u16):
    if protocol == "1pct":
        valid = int((target_u16 > 0).sum())
        return max(1, int(round(ONEPCT_FRAC * valid)))
    return int(protocol)


def sample_sparse(target_u16, n_points, sample_key, seed):
    valid = target_u16 > 0
    ys, xs = np.nonzero(valid)
    sparse = np.zeros_like(target_u16)
    mask = np.zeros(target_u16.shape, np.uint8)
    if ys.size == 0:
        return sparse, mask, 0
    rng = np.random.RandomState(zlib.crc32(f"{sample_key}_{n_points}_{seed}".encode()) & 0xFFFFFFFF)
    sel = np.arange(ys.size) if ys.size <= n_points else rng.choice(ys.size, n_points, replace=False)
    sy, sx = ys[sel], xs[sel]
    mask[sy, sx] = 255
    sparse[sy, sx] = target_u16[sy, sx]
    return sparse, mask, int(sel.size)


def place_static(src, dst, link_static):
    """hardlink -> symlink -> copy src to dst. Returns the method used."""
    if os.path.lexists(dst):
        os.remove(dst)
    if link_static:
        try:
            os.link(src, dst); return "hardlink"
        except OSError:
            try:
                os.symlink(os.path.abspath(src), dst); return "symlink"
            except OSError:
                pass
    shutil.copy(src, dst); return "copy"


def build(tar_root, out_root, protocols, seed, limit, link_static, warns):
    roots = {p: Path(out_root) / f"nyu48k_{p}" for p in protocols}
    src_root = roots.get("1pct")                # canonical link source (may be None)
    for phase in ("train", "test", "val"):
        for p, root in roots.items():
            for sub in _SUBDIRS:
                (root / phase / sub).mkdir(parents=True, exist_ok=True)
        for idx, h5name, blob in iter_phase(tar_root, phase, limit):
            rgb, depth = read_h5(blob)
            rgb_c = psd_rgb(rgb)
            tgt_u16 = to_u16(psd_depth(depth))
            name = f"{idx:08d}.png"
            # source root (1pct) writes the real rgb/target first
            if src_root is not None and "1pct" in roots:
                Image.fromarray(rgb_c).save(roots["1pct"] / phase / "rgb" / name)
                Image.fromarray(tgt_u16).save(roots["1pct"] / phase / "target_depth" / name)
            for p, root in roots.items():
                rgb_dst = root / phase / "rgb" / name
                tgt_dst = root / phase / "target_depth" / name
                if p != "1pct":
                    s_rgb = (src_root / phase / "rgb" / name) if src_root is not None else None
                    s_tgt = (src_root / phase / "target_depth" / name) if src_root is not None else None
                    if link_static and s_rgb is not None and s_rgb.exists():
                        place_static(s_rgb, rgb_dst, link_static)
                        place_static(s_tgt, tgt_dst, link_static)
                    else:
                        Image.fromarray(rgb_c).save(rgb_dst)
                        Image.fromarray(tgt_u16).save(tgt_dst)
                n = n_points_for(p, tgt_u16)
                sparse, mask, kept = sample_sparse(tgt_u16, n, f"{phase}_{h5name}_{p}", seed)
                Image.fromarray(sparse).save(root / phase / "sparse_depth" / name)
                Image.fromarray(mask).save(root / phase / "valid_mask" / name)
                if kept < n:
                    warns.append(f"nyu48k_{p}/{phase}/{name}: {kept} valid pts (< {n})")
        print(f"[build] phase={phase} done for protocols={list(roots)}")


def _arr(path):
    return np.array(Image.open(path))


def audit(out_root, protocols, audit_csv):
    rows, errors = [], []
    fields = ["dataset_root", "protocol", "phase", "num_rgb", "num_target_depth", "num_sparse_depth",
              "num_valid_mask", "shape_rgb", "shape_depth", "sparse_nonzero_mean", "sparse_nonzero_min",
              "sparse_nonzero_max", "valid_mask_unique", "target_depth_min", "target_depth_max",
              "sparse_depth_nonzero_min", "sparse_depth_nonzero_max", "split_source", "preprocess",
              "val_test_same_source", "tar_count", "h5_count", "warnings"]
    for p in protocols:
        ddir = Path(out_root) / f"nyu48k_{p}"
        if not ddir.is_dir():
            continue
        for phase in ("train", "test", "val"):
            rgb = _list(ddir / phase / "rgb"); gt = _list(ddir / phase / "target_depth")
            spd = _list(ddir / phase / "sparse_depth"); mk = _list(ddir / phase / "valid_mask")
            if not rgb:
                continue
            if not (sorted(rgb) == sorted(gt) == sorted(spd) == sorted(mk)):
                errors.append(f"nyu48k_{p}/{phase}: rgb/target/sparse/mask stems mismatch")
            shp_rgb = _arr(next(iter(rgb.values()))).shape
            shp_dep = _arr(next(iter(gt.values()))).shape
            sp_nz, mk_nz, vmu, tmin, tmax, sdmin, sdmax = [], [], set(), 1 << 30, 0, 1 << 30, 0
            for stem, mp in mk.items():
                m = _arr(mp); mk_nz.append(int((m > 0).sum())); vmu |= set(np.unique(m).tolist())
            for stem, sp in spd.items():
                s = _arr(sp); sp_nz.append(int((s > 0).sum()))
                snz = s[s > 0]
                if snz.size:
                    sdmin = min(sdmin, int(snz.min())); sdmax = max(sdmax, int(snz.max()))
            for stem, gp in list(gt.items())[:50]:
                g = _arr(gp); tmin = min(tmin, int(g.min())); tmax = max(tmax, int(g.max()))
            sp_nz = sp_nz or [0]; mk_nz = mk_nz or [0]
            n_warn = 0  # filled below if needed
            rows.append({
                "dataset_root": str(ddir), "protocol": p, "phase": phase, "num_rgb": len(rgb),
                "num_target_depth": len(gt), "num_sparse_depth": len(spd), "num_valid_mask": len(mk),
                "shape_rgb": "x".join(map(str, shp_rgb)), "shape_depth": "x".join(map(str, shp_dep)),
                "sparse_nonzero_mean": round(float(np.mean(sp_nz)), 2), "sparse_nonzero_min": int(np.min(sp_nz)),
                "sparse_nonzero_max": int(np.max(sp_nz)), "valid_mask_unique": str(sorted(vmu)),
                "target_depth_min": tmin, "target_depth_max": tmax,
                "sparse_depth_nonzero_min": sdmin if sdmin < (1 << 30) else 0, "sparse_depth_nonzero_max": sdmax,
                "split_source": SPLIT_SOURCE_TAG, "preprocess": PREPROCESS_TAG, "val_test_same_source": True,
                "tar_count": len(PHASE_TARS[phase]), "h5_count": len(rgb), "warnings": n_warn,
            })
            # ---- hard sanity checks ----
            if tuple(shp_rgb) != (OUT_H, OUT_W, 3):
                errors.append(f"nyu48k_{p}/{phase}: rgb shape {shp_rgb} != ({OUT_H},{OUT_W},3)")
            if tuple(shp_dep) != (OUT_H, OUT_W):
                errors.append(f"nyu48k_{p}/{phase}: depth shape {shp_dep} != ({OUT_H},{OUT_W})")
            if not set(vmu).issubset({0, 255}):
                errors.append(f"nyu48k_{p}/{phase}: valid_mask values {sorted(vmu)} not in {{0,255}}")
            if not (len(rgb) == len(gt) == len(spd) == len(mk)):
                errors.append(f"nyu48k_{p}/{phase}: dir counts differ")
            if sorted(sp_nz) != sorted(mk_nz):
                errors.append(f"nyu48k_{p}/{phase}: sparse_nonzero != valid_mask_nonzero")
            mean_nz = float(np.mean(sp_nz))
            if p != "1pct":
                pts = int(p)
                if np.max(sp_nz) > pts or mean_nz < pts - 0.5:
                    errors.append(f"nyu48k_{p}/{phase}: sparse nonzero mean={mean_nz:.2f} max={np.max(sp_nz)} != {pts}")
            else:
                if mean_nz <= 0:
                    errors.append(f"nyu48k_1pct/{phase}: sparse nonzero mean is 0")

    os.makedirs(os.path.dirname(audit_csv) or ".", exist_ok=True)
    with open(audit_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"[audit] -> {audit_csv}")
    for r in rows:
        print(f"  {r['dataset_root']:20s} {r['phase']:5s} n={r['num_rgb']:<6d} "
              f"shape_rgb={r['shape_rgb']:11s} sparse_nz(mean/min/max)="
              f"{r['sparse_nonzero_mean']}/{r['sparse_nonzero_min']}/{r['sparse_nonzero_max']} "
              f"vmask_unique={r['valid_mask_unique']} val_test_same_source={r['val_test_same_source']}")
    if errors:
        print(f"\n[AUDIT FAILED] {len(errors)} problem(s):")
        for e in errors:
            print("  - " + e)
        sys.exit(1)
    print("[audit] all checks passed.")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar-root", default=DEFAULT_TAR_ROOT, help="dir with train-0000NN.tar / val-0000NN.tar")
    ap.add_argument("--out-root", default="data", help="output root (-> <out-root>/nyu48k_<protocol>)")
    ap.add_argument("--protocols", default="1pct,500,200,100,50,10", help="comma list: 1pct and/or point counts")
    ap.add_argument("--seed", type=int, default=42, help="per-sample sampling seed")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N samples per split (0=all)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if an output root exists")
    ap.add_argument("--link-static", dest="link_static", action="store_true", default=True,
                    help="hardlink/symlink rgb+target_depth from nyu48k_1pct to the fixed-point roots (default on)")
    ap.add_argument("--no-link-static", dest="link_static", action="store_false")
    ap.add_argument("--num-workers", type=int, default=0, help="reserved (single-process for determinism)")
    ap.add_argument("--verify", action="store_true", help="audit -> output/nyu48k_sparse_dataset_audit.csv")
    ap.add_argument("--audit-out", default="output/nyu48k_sparse_dataset_audit.csv")
    args = ap.parse_args()

    protocols = [p.strip() for p in str(args.protocols).replace(" ", "").split(",") if p.strip()]
    # 1pct first so it is the link source
    protocols = (["1pct"] if "1pct" in protocols else []) + [p for p in protocols if p != "1pct"]
    print(f"[source] tar-root = {args.tar_root}  ({SPLIT_SOURCE_TAG})")
    print(f"[preprocess] {PREPROCESS_TAG}  (final saved arrays {OUT_H}x{OUT_W})")
    print(f"[protocols] {protocols}  seed={args.seed}  limit={args.limit or 'all'}  link_static={args.link_static}")

    to_build = []
    for p in protocols:
        root = Path(args.out_root) / f"nyu48k_{p}"
        if root.exists() and not args.overwrite:
            print(f"[skip] {root} exists (use --overwrite to rebuild) — not mixed with new data")
            continue
        if root.exists() and args.overwrite:
            shutil.rmtree(root)
        to_build.append(p)
    if not to_build:
        print("[done] nothing to build (all targets exist; pass --overwrite to rebuild).")
        if args.verify:
            audit(args.out_root, protocols, args.audit_out)
        return

    warns = []
    build(args.tar_root, args.out_root, to_build, args.seed, args.limit, args.link_static, warns)
    if warns:
        print(f"[warn] {len(warns)} sample(s) had fewer valid pixels than requested; first few: {warns[:5]}")
    if args.verify:
        audit(args.out_root, protocols, args.audit_out)


if __name__ == "__main__":
    main()

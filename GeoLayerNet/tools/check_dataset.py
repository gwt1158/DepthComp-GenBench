"""Verify a prepared nyu500 dataset layout / counts (no torch).

  python tools/check_dataset.py --dataroot data/nyu500
"""
import argparse
import os

MODALITIES = ["rgb", "sparse_depth", "target_depth", "valid_mask"]
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def count_dir(path):
    if not os.path.isdir(path):
        return None
    return sum(1 for f in os.listdir(path) if f.lower().endswith(EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    args = ap.parse_args()
    print(f"dataset root: {args.dataroot}\n")
    ok = True
    for split in ("train", "val", "test"):
        sroot = os.path.join(args.dataroot, split)
        if not os.path.isdir(sroot):
            print(f"[{split}] MISSING"); ok = False; continue
        counts = {m: count_dir(os.path.join(sroot, m)) for m in MODALITIES}
        core = [c for c in counts.values() if c is not None]
        consistent = len(set(core)) <= 1 and all(counts[m] is not None for m in MODALITIES)
        print(f"[{split}] {'OK' if consistent else 'INCONSISTENT/MISSING'}  "
              + "  ".join(f"{m}={counts[m]}" for m in MODALITIES))
        if not consistent:
            ok = False
        if split == "test" and counts["rgb"] not in (None, 654):
            print(f"    note: test has {counts['rgb']} images (canonical NYU test = 654).")
    print("\nRESULT:", "PASS" if ok else "CHECK FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

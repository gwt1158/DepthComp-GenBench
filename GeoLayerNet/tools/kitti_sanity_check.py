"""KITTI dataloader sanity check (NO model, NO training).

Loads a few batches through the KITTI routing (create_loader -> KittiDataset) using
a KITTI config, and prints: batch keys, tensor shapes, depth ranges back in METRES
(normalized x max_depth), and the sparse / target valid ratios. Use this to confirm
the data path / format / crop / units before any training.

  python tools/kitti_sanity_check.py --config configs/geolayernet_kitti_psd.yaml --num 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from geolayernet.config import load_config  # noqa: E402
from geolayernet.data import create_loader   # noqa: E402


def _get(cfg, section, key, default=None):
    sec = getattr(cfg, section, None)
    return getattr(sec, key, default) if sec is not None else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", default=None, help="train/val/test (default: eval.phase)")
    ap.add_argument("--num", type=int, default=4, help="number of samples to inspect")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataroot = _get(cfg, "dataset", "dataroot", "data/kitti_depth_completion")
    dataset_type = _get(cfg, "dataset", "type", "nyu")
    max_depth = float(_get(cfg, "dataset", "max_depth", 80.0))
    scale = float(_get(cfg, "dataset", "depth_unit_scale", 80))
    phase = args.phase or _get(cfg, "eval", "phase", "val") or "val"
    dataset_opts = cfg.raw.get("dataset", {}) if getattr(cfg, "raw", None) else {}

    print(f"[sanity] config={args.config} type={dataset_type} dataroot={dataroot} "
          f"phase={phase} max_depth={max_depth} depth_unit_scale={scale}")
    if str(dataset_type).lower() != "kitti":
        print("[warn] dataset.type is not 'kitti' — this checker is for KITTI configs.")

    loader = create_loader(dataroot, phase, batch_size=1, num_workers=0, shuffle=False,
                           config_hint=args.config, dataset_type=dataset_type, dataset_opts=dataset_opts)

    required = {"rgb", "sparse_depth", "target_depth", "valid_mask", "target_mask", "name"}
    seen = 0
    for batch in loader:
        missing = required - set(batch.keys())
        assert not missing, f"missing batch keys: {missing}"
        rgb, sp, gt = batch["rgb"], batch["sparse_depth"], batch["target_depth"]
        vm, tm = batch["valid_mask"], batch["target_mask"]
        n_px = gt[0, 0].numel()
        sp_m = sp * scale
        gt_m = gt * scale
        print(f"\n[{seen}] name={batch['name'][0]}")
        print(f"  rgb          {tuple(rgb.shape)}  range[{rgb.min():.2f},{rgb.max():.2f}]")
        print(f"  sparse_depth {tuple(sp.shape)}  metres[{float(sp_m[sp>0].min()) if (sp>0).any() else 0:.2f},{float(sp_m.max()):.2f}]")
        print(f"  target_depth {tuple(gt.shape)}  metres[{float(gt_m[gt>0].min()) if (gt>0).any() else 0:.2f},{float(gt_m.max()):.2f}]")
        print(f"  valid_mask   ratio={float(vm.mean()):.4f}  (sparse>0)")
        print(f"  target_mask  ratio={float(tm.mean()):.4f}  (0<gt<= {max_depth:.0f} m)")
        print(f"  H x W = {gt.shape[-2]} x {gt.shape[-1]}   pixels/img={n_px}")
        seen += 1
        if seen >= args.num:
            break
    print(f"\n[sanity] inspected {seen} sample(s) — keys OK, units in metres via x{scale:.0f}. No model was run.")


if __name__ == "__main__":
    main()

"""nyu500 RGB-D completion dataset (config-driven, no legacy option framework).

Reads <dataroot>/<phase>/{rgb, sparse_depth, target_depth, valid_mask}, matched
by filename stem. Transforms reproduce the original pipeline:
  - preprocess "none": native resolution (no resize/crop).
  - rgb -> [-1,1]; depth PNG -> [0,1] (scale inferred from bit depth: 65535/255).
  - valid_mask from file, else sparse_depth>0.
Returns rgb, sparse_depth, target_depth, valid_mask, target_mask, name.
"""

import zlib
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in _EXTS}


class Nyu500Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, phase="test", preprocess="none",
                 load_size=286, crop_size=256, no_flip=True, depth_scale=0.0,
                 teacher_depth_dir=None, dynamic_sparse=False, sparse_points=500,
                 sparse_pattern="random", depth_noise_std=0.0, missing_block_prob=0.0,
                 config_hint=None):
        self.config_hint = config_hint
        root = Path(dataroot) / phase
        if not root.is_dir():
            root = Path(dataroot)
        self.phase = phase
        self.preprocess = preprocess
        self.load_size = load_size
        self.crop_size = crop_size
        self.no_flip = no_flip
        self.depth_scale = depth_scale
        # Phase 6 — dynamic sparse sampling (train only; OFF == file sparse path).
        self.dynamic_sparse = bool(dynamic_sparse)
        self.sparse_points = int(sparse_points)
        self.sparse_pattern = str(sparse_pattern)
        self.depth_noise_std = float(depth_noise_std)
        self.missing_block_prob = float(missing_block_prob)

        rgb = _list(root / "rgb")
        sparse = _list(root / "sparse_depth")
        target = _list(root / "target_depth")
        vmask = _list(root / "valid_mask")
        if not rgb:
            # Lightweight diagnostic before raising (does not change behaviour):
            # surface the resolved dataset path / config / cwd so a wrong dataroot
            # is obvious, then re-raise the original error (never swallow it).
            print("[data][ERROR] no RGB images found — check the dataset path.")
            print(f"  configured dataroot : {dataroot}")
            print(f"  resolved data dir   : {root.resolve()}")
            print(f"  expected rgb dir    : {(root / 'rgb').resolve()}")
            print(f"  config              : {self.config_hint or '(not provided)'}")
            print(f"  current working dir : {Path.cwd()}")
            raise RuntimeError(f"no rgb images under {root/'rgb'}")
        if not sparse and not self.dynamic_sparse:
            raise RuntimeError(f"no sparse_depth under {root/'sparse_depth'}")
        # Phase 5 — optional foundation/teacher depth (inert if dir absent).
        teacher = {}
        if teacher_depth_dir:
            teacher = _list(root / teacher_depth_dir)
            if not teacher:
                print(f"[data] teacher dir '{teacher_depth_dir}' empty/absent under {root} -> teacher inert")
        self.samples = []
        for k, rp in rgb.items():
            if k not in sparse and not self.dynamic_sparse:
                continue
            self.samples.append({
                "name": k, "rgb": rp, "sparse": sparse.get(k),
                "target": target.get(k, sparse.get(k)), "valid": vmask.get(k),
                "teacher": teacher.get(k),
            })
        if not self.samples:
            raise RuntimeError(f"no matched samples under {root}")

    def __len__(self):
        return len(self.samples)

    def _geom(self, img, flip, interp):
        if "resize" in self.preprocess:
            img = TF.resize(img, [self.load_size, self.load_size], interpolation=interp)
        if "crop" in self.preprocess:
            img = TF.center_crop(img, [self.crop_size, self.crop_size])
        if flip:
            img = TF.hflip(img)
        return img

    def _rgb(self, path, flip):
        img = self._geom(Image.open(path).convert("RGB"), flip, TF.InterpolationMode.BICUBIC)
        return TF.normalize(TF.to_tensor(img), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _depth(self, path, flip):
        img = self._geom(Image.open(path), flip, TF.InterpolationMode.NEAREST)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        scale = self.depth_scale
        if scale <= 0:
            scale = 65535.0 if arr.max(initial=0) > 255.0 else 255.0
        arr = np.clip(arr / max(scale, 1.0), 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)

    def _mask(self, path, flip):
        img = self._geom(Image.open(path), flip, TF.InterpolationMode.NEAREST)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)

    def _sample_sparse(self, target):
        """Generate sparse depth and its valid mask from the dense target.
        This is input construction, not a forward-time leak —
        the model still only ever sees the generated sparse_depth + valid_mask.
        ``target`` is a [1,H,W] tensor in [0,1]; returns (sparse[1,H,W], valid[1,H,W])."""
        t = target[0].numpy()
        h, w = t.shape
        valid_t = t > 0
        ys, xs = np.nonzero(valid_t)
        n_avail = ys.size
        sparse = np.zeros((h, w), np.float32)
        valid = np.zeros((h, w), np.float32)
        if n_avail == 0:
            return torch.from_numpy(sparse).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)

        pattern = self.sparse_pattern
        if pattern == "mixed":
            pattern = np.random.choice(["random", "grid", "line", "orb"])
        n = min(self.sparse_points, n_avail)

        if pattern == "grid":
            step = max(int(round((h * w / max(self.sparse_points, 1)) ** 0.5)), 1)
            gy, gx = np.mgrid[0:h:step, 0:w:step]
            gy = gy.ravel(); gx = gx.ravel()
            keep = valid_t[gy, gx]
            sel_y, sel_x = gy[keep], gx[keep]
        elif pattern == "line":
            n_lines = max(int(round(self.sparse_points ** 0.5)), 1)
            rows = np.random.choice(h, size=min(n_lines, h), replace=False)
            line_mask = np.zeros((h, w), bool)
            line_mask[rows, :] = True
            sel = line_mask & valid_t
            sel_y, sel_x = np.nonzero(sel)
            if sel_y.size > self.sparse_points:
                idx = np.random.choice(sel_y.size, self.sparse_points, replace=False)
                sel_y, sel_x = sel_y[idx], sel_x[idx]
        elif pattern == "orb":
            # gradient-peak placeholder for ORB/SIFT-like keypoints: prefer pixels
            # with high local depth gradient (corners / edges).
            gy_, gx_ = np.gradient(t)
            grad = np.sqrt(gy_ ** 2 + gx_ ** 2)
            grad[~valid_t] = -1.0
            order = np.argsort(grad.ravel())[::-1][:n]
            sel_y, sel_x = np.unravel_index(order, (h, w))
        else:  # random (uniform)
            idx = np.random.choice(n_avail, n, replace=False)
            sel_y, sel_x = ys[idx], xs[idx]

        vals = t[sel_y, sel_x].astype(np.float32)
        if self.depth_noise_std > 0:
            vals = vals + np.random.randn(*vals.shape).astype(np.float32) * self.depth_noise_std
            vals = np.clip(vals, 0.0, 1.0)
        sparse[sel_y, sel_x] = vals
        valid[sel_y, sel_x] = 1.0

        if self.missing_block_prob > 0 and np.random.rand() < self.missing_block_prob:
            bh, bw = h // 4, w // 4
            y0 = np.random.randint(0, max(h - bh, 1)); x0 = np.random.randint(0, max(w - bw, 1))
            sparse[y0:y0 + bh, x0:x0 + bw] = 0.0
            valid[y0:y0 + bh, x0:x0 + bw] = 0.0
        return torch.from_numpy(sparse).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)

    def __getitem__(self, i):
        s = self.samples[i]
        flip = (self.phase == "train") and (not self.no_flip) and (np.random.rand() > 0.5)
        rgb = self._rgb(s["rgb"], flip)
        target = self._depth(s["target"], flip)

        # Phase 6: dynamically resample sparse_depth from target.
        #   * train  -> stochastic augmentation (unchanged behaviour).
        #   * eval    -> fixed-density sparsity for sparsity experiments, made
        #     DETERMINISTIC per (sample, density) so r11 and GeoLayerNet see the
        #     exact same sparse input at each point count (fair comparison).
        if self.dynamic_sparse and self.phase == "train":
            sparse, valid = self._sample_sparse(target)
        elif self.dynamic_sparse:
            seed = zlib.crc32(f"{s['name']}_{self.sparse_points}".encode()) & 0x7FFFFFFF
            state = np.random.get_state()
            np.random.seed(seed)
            sparse, valid = self._sample_sparse(target)
            np.random.set_state(state)
        else:
            sparse = self._depth(s["sparse"], flip) if s["sparse"] else torch.zeros_like(target)
            valid = self._mask(s["valid"], flip) if s["valid"] else (sparse > 0).float()

        out = {"rgb": rgb, "sparse_depth": sparse, "target_depth": target,
               "valid_mask": valid, "target_mask": (target > 0).float(), "name": s["name"]}

        # Phase 5: teacher depth (zeros + mask 0 when absent, so collate works
        # and the distillation loss is inert).
        if s.get("teacher"):
            teacher = self._depth(s["teacher"], flip)
            out["teacher_depth"] = teacher
            out["teacher_mask"] = (teacher > 0).float()
        else:
            out["teacher_depth"] = torch.zeros_like(target)
            out["teacher_mask"] = torch.zeros_like(target)
        return out


def create_loader(dataroot, phase, batch_size, num_workers=4, preprocess="none",
                  shuffle=False, depth_scale=0.0, max_samples=None,
                  teacher_depth_dir=None, dynamic_sparse=False, sparse_points=500,
                  sparse_pattern="random", depth_noise_std=0.0, missing_block_prob=0.0,
                  config_hint=None, dataset_type="nyu", dataset_opts=None,
                  pin_memory=False, persistent_workers=None, prefetch_factor=None):
    # dataset_type defaults to NYU; KITTI uses its dedicated reader.
    dtype = str(dataset_type).lower()
    if dtype == "kitti":
        from .kitti_dataset import KittiDataset
        opts = {k: v for k, v in (dataset_opts or {}).items()
                if k not in ("dataroot", "type", "dataset_mode", "dataset_tag",
                             "preprocess", "input_nc", "depth_unit_scale")}
        ds = KittiDataset(dataroot, phase=phase, config_hint=config_hint, **opts)
    elif dtype == "ood_npz":
        from .ood_npz_dataset import OodNpzDataset
        opts = {k: v for k, v in (dataset_opts or {}).items()
                if k not in ("dataroot", "type", "dataset_mode", "dataset_tag",
                             "preprocess", "input_nc")}
        ds = OodNpzDataset(dataroot, phase=phase, config_hint=config_hint, **opts)
    else:
        ds = Nyu500Dataset(dataroot, phase=phase, preprocess=preprocess,
                           no_flip=(phase != "train"), depth_scale=depth_scale,
                           teacher_depth_dir=teacher_depth_dir, dynamic_sparse=dynamic_sparse,
                           sparse_points=sparse_points, sparse_pattern=sparse_pattern,
                           depth_noise_std=depth_noise_std, missing_block_prob=missing_block_prob,
                           config_hint=config_hint)
    if max_samples is not None and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    print(f"[data] {phase}: {len(ds)} samples")
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": shuffle,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            False if persistent_workers is None else bool(persistent_workers)
        )
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return torch.utils.data.DataLoader(ds, **loader_kwargs)

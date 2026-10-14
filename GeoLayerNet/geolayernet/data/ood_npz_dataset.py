"""Generic NPZ reader for out-of-distribution depth-completion datasets."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _rgb_chw(value):
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"rgb must have three dimensions, got {array.shape}")
    if array.shape[0] != 3 and array.shape[-1] == 3:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != 3:
        raise ValueError(f"rgb must have three channels, got {array.shape}")
    array = array.astype(np.float32)
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def _depth_1hw(value, key):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.moveaxis(array, -1, 0)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"{key} must have shape HxW or 1xHxW, got {array.shape}")
    return np.ascontiguousarray(array)


class OodNpzDataset(torch.utils.data.Dataset):
    """Load RGB, sparse depth, and target depth from generic NPZ samples.

    Depth arrays are expected in metres. Samples are resized to the fixed
    GeoLayerNet input size and normalized by ``depth_unit_scale``.
    """

    def __init__(self, dataroot, phase="test", depth_unit_scale=10.0,
                 resize_to=(228, 304), config_hint=None, **_ignored):
        self.root = Path(dataroot)
        self.phase = str(phase)
        self.scale = float(depth_unit_scale)
        self.resize_to = tuple(int(value) for value in resize_to)
        self.config_hint = config_hint
        if self.scale <= 0:
            raise ValueError("depth_unit_scale must be positive")
        self.samples = self._index()
        if not self.samples:
            raise RuntimeError(
                f"no NPZ samples found under {self.root}; config={config_hint}")

    def _index(self):
        manifest = self.root / f"{self.phase}.txt"
        if manifest.is_file():
            samples = []
            for raw in manifest.read_text().splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                path = self.root / line
                if path.suffix.lower() != ".npz":
                    path = path.with_suffix(".npz")
                if not path.is_file():
                    raise FileNotFoundError(f"manifest sample not found: {path}")
                samples.append(path)
            return samples
        phase_root = self.root / self.phase
        search_root = phase_root if phase_root.is_dir() else self.root
        return sorted(search_root.glob("*.npz"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        with np.load(path, allow_pickle=False) as archive:
            required = ("rgb", "sparse_depth", "target_depth")
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise KeyError(f"{path} is missing keys: {missing}")
            rgb = torch.from_numpy(_rgb_chw(archive["rgb"]))
            sparse = torch.from_numpy(_depth_1hw(archive["sparse_depth"], "sparse_depth"))
            target = torch.from_numpy(_depth_1hw(archive["target_depth"], "target_depth"))

        rgb = F.interpolate(
            rgb[None], size=self.resize_to, mode="bilinear", align_corners=False)[0]
        sparse = F.interpolate(sparse[None], size=self.resize_to, mode="nearest")[0]
        target = F.interpolate(target[None], size=self.resize_to, mode="nearest")[0]
        sparse = sparse / self.scale
        target = target / self.scale
        return {
            "rgb": (rgb - 0.5) / 0.5,
            "sparse_depth": sparse,
            "target_depth": target,
            "valid_mask": (sparse > 0).float(),
            "target_mask": (target > 0).float(),
            "name": path.stem,
        }

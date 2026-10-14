import numpy
from PIL import Image
from os.path import join, normpath, sep, splitext
from glob import glob
from torch.utils.data import Dataset
from typing import Optional, Callable
from ssd.datasets.constants import (
    SAMPLE_ID_, 
    RGB_, 
    DEPTH_METRIC_, 
    GT_METRIC_,
    INTRINSICS_
)


def ddad_rgb_path_to_sample_id(path: str) -> str:
    path_components = normpath(path).split(sep)
    image = splitext(path_components[-1])[0]
    return image


class DDAD(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Optional[Callable], limit_size: Optional[int] = None):
        assert subset in ["val"]
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        
        self._rgb_paths = sorted(glob(join(self._path, subset, "rgb", "*.png")))
        self._gt_paths = sorted(glob(join(self._path, subset, "gt", "*.png")))
        self._depth_paths = sorted(glob(join(self._path, subset, "hints", "*.png")))
        self._intrinsics_paths = sorted(glob(join(self._path, subset, "intrinsics", "*.txt")))
        self._ids = [ddad_rgb_path_to_sample_id(p) for p in self._rgb_paths]

        assert len(self._rgb_paths) == len(self._gt_paths) == len(self._depth_paths) == len(self._intrinsics_paths), "some dataset files are missing"

        if limit_size:
            self._rgb_paths = self._rgb_paths[:limit_size]
            self._gt_paths = self._gt_paths[:limit_size]
            self._depth_paths = self._depth_paths[:limit_size]
            self._intrinsics_paths = self._intrinsics_paths[:limit_size]
            self._ids = self._ids[:limit_size]

        self._sample_id_to_idx = { sample_id: idx for idx, sample_id in enumerate(self._ids) }

    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx):
        sample = {
            SAMPLE_ID_: self._ids[idx],
            RGB_: numpy.array(Image.open(self._rgb_paths[idx])), 
            DEPTH_METRIC_: numpy.array(Image.open(self._depth_paths[idx])) / 256.0,
            GT_METRIC_: numpy.array(Image.open(self._gt_paths[idx])) / 256.0,
            INTRINSICS_: numpy.loadtxt(self._intrinsics_paths[idx], dtype=numpy.float32, delimiter=" ")
        }
    
        if self._preprocessing is not None:
            sample = self._preprocessing(sample)

        return sample
    
    def by_id(self, sample_id):
        return self.__getitem__(self._sample_id_to_idx[sample_id])

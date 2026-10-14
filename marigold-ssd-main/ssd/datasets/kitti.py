import numpy
from glob import glob
from PIL import Image
from typing import Callable
from os.path import join, normpath, sep, splitext
from torch.utils.data import Dataset
from ssd.datasets.constants import (
    SAMPLE_ID_,
    RGB_,
    DEPTH_METRIC_, 
    GT_METRIC_,
    INTRINSICS_
)


def val_selection_rgb_path_to_id(path: str) -> str:
    path_components = normpath(path).split(sep)
    image_name = splitext(path_components[-1])[0]
    return image_name


class KittiDepth(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Callable, limit_size: int = None):
        assert subset in ["val"]
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        self._limit_size = limit_size

        self._rgb_paths = sorted(glob(join(self._path, "data_depth_selection", "val_selection_cropped", "image", "*.png")))
        self._gt_paths = sorted(glob(join(self._path, "data_depth_selection", "val_selection_cropped", "groundtruth_depth", "*.png")))
        self._depth_paths = sorted(glob(join(self._path, "data_depth_selection", "val_selection_cropped", "velodyne_raw", "*.png")))
        self._intrinsics_paths = sorted(glob(join(self._path, "data_depth_selection", "val_selection_cropped", "intrinsics", "*.txt")))
        self._ids = [val_selection_rgb_path_to_id(p) for p in self._rgb_paths]

        assert len(self._rgb_paths) == len(self._gt_paths) == len(self._depth_paths) == len(self._intrinsics_paths), "some dataset files are missing"

        if self._limit_size is not None:
            self._rgb_paths = self._rgb_paths[:self._limit_size]
            self._gt_paths = self._gt_paths[:self._limit_size]
            self._depth_paths = self._depth_paths[:self._limit_size]
            self._intrinsics_paths = self._intrinsics_paths[:self._limit_size]
            self._ids = self._ids[:self._limit_size]

        self._sample_id_to_idx = { sample_id: idx for idx, sample_id in enumerate(self._ids) }

    def __getitem__(self, index):
        item = {
            SAMPLE_ID_: self._ids[index],
            RGB_: numpy.array(Image.open(self._rgb_paths[index])), 
            GT_METRIC_: numpy.array(Image.open(self._gt_paths[index])) / 256.0, 
            DEPTH_METRIC_: numpy.array(Image.open(self._depth_paths[index])) / 256.0,
            INTRINSICS_: numpy.loadtxt(self._intrinsics_paths[index], dtype=numpy.float32).reshape((3, 3))
        }

        if self._preprocessing is not None:
            item = self._preprocessing(item)

        return item

    def __len__(self):
        return len(self._ids)
    
    def by_id(self, sample_id):
        return self.__getitem__(self._sample_id_to_idx[sample_id])
import numpy
from glob import glob
from typing import Callable
from PIL import Image
from os.path import (
    join,
    normpath, 
    sep, 
    splitext
)
from ssd.datasets.constants import (
    SAMPLE_ID_,
    RGB_,
    GT_METRIC_,
    INTRINSICS_
)

MARIGOLD_DC_TEST_745 = "ssd/configs/scannet-marigolddc-test"


def load_marigold_dc_sample_names(file_path: str):
    sample_names = []

    with open(file_path, "r") as f:
        for line in f.readlines():
            s = line.strip()
            if len(s) == 0:
                continue

            sample_names.append("_".join(s.split("/")))

    return sample_names


class ScanNet():
    def __init__(self, path: str, subset: str, preprocessing: Callable, limit_size: int = None):
        assert subset in ["test", "marigolddctest"]
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        self._limit_size = limit_size

        if self._subset == "test":
            self._rgb_paths = sorted(glob(join(self._path, self._subset, "images", "*.png")))
            self._gt_paths = sorted(glob(join(self._path, self._subset, "groundtruth", "*.png")))
            self._intrinsic_paths = sorted(glob(join(self._path, self._subset, "intrinsics", "*.txt")))
            self._ids = [self._rgb_path_to_id(p) for p in self._rgb_paths]

        elif self._subset == "marigolddctest":
            sample_names = load_marigold_dc_sample_names(MARIGOLD_DC_TEST_745)
            self._rgb_paths = sorted([join(self._path, "test", "images", f"{sn}.png") for sn in sample_names])
            self._gt_paths = sorted([join(self._path, "test", "groundtruth", f"{sn}.png") for sn in sample_names])
            self._intrinsic_paths = sorted([join(self._path, "test", "intrinsics", f"{sn}.txt") for sn in sample_names])
            self._ids = [self._rgb_path_to_id(p) for p in self._rgb_paths]

        assert len(self._rgb_paths) == len(self._gt_paths) == len(self._intrinsic_paths)

        if limit_size is not None:
            self._ids = self._ids[:limit_size]
            self._rgb_paths = self._rgb_paths[:limit_size]
            self._gt_paths = self._gt_paths[:limit_size]
            self._intrinsic_paths = self._intrinsic_paths[:limit_size]

        self._sample_id_to_idx = { sample_id: idx for idx, sample_id in enumerate(self._ids) }


    @staticmethod
    def _rgb_path_to_id(path: str) -> str:
        path_components = normpath(path).split(sep)
        scene, scene2, frame = splitext(path_components[-1])[0].split("_")
        return '_'.join((scene, scene2, frame))

    def __getitem__(self, index):
        item = {
            SAMPLE_ID_: str(self._ids[index]),
            RGB_: numpy.array(Image.open(self._rgb_paths[index]).convert('RGB')),
            GT_METRIC_: numpy.array(Image.open(self._gt_paths[index])) / 1000.0,
            INTRINSICS_: numpy.loadtxt(self._intrinsic_paths[index], dtype=numpy.float32, delimiter=" ")
        }

        if self._preprocessing is not None:
            item = self._preprocessing(item)

        return item

    def __len__(self):
        return len(self._rgb_paths)
    
    def by_id(self, sample_id):
        return self.__getitem__(self._sample_id_to_idx[sample_id])

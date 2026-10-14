import numpy
from PIL import Image
from os.path import join, normpath, sep, splitext
from glob import glob
from torch.utils.data import Dataset
from scipy.io import loadmat
from typing import Optional, Callable
from ssd.datasets.constants import (
    SAMPLE_ID_, 
    RGB_, 
    GT_METRIC_,
    INVALID_MASK_,
    TRANSPARENT_MASK_,
    INTRINSICS_,
    EDGES_MASK_
)


def ibims1_mat_path_to_sample_id(path: str) -> str:
    path_components = normpath(path).split(sep)
    image = splitext(path_components[-1])[0]
    return image


class Ibims1(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Optional[Callable], limit_size: Optional[int] = None):
        assert subset in ["all"]
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing

        self._mat_paths = sorted(glob(join(join(self._path, "ibims1_core_mat", "*.mat"))))
        self._ids = [ibims1_mat_path_to_sample_id(p) for p in self._mat_paths]

        assert len(self._mat_paths) > 0, "iBims-1 dataset: no files found"

        if limit_size:
            self._mat_paths = self._mat_paths[:limit_size]
            self._ids = self._ids[:limit_size]

        self._sample_id_to_idx = { sample_id: idx for idx, sample_id in enumerate(self._ids) }

    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx):
        data = loadmat(self._mat_paths[idx])['data']
        sample = {
            SAMPLE_ID_: self._ids[idx],
            RGB_: data['rgb'][0][0], 
            GT_METRIC_: data['depth'][0][0].astype(numpy.float32),
            INVALID_MASK_: data['mask_invalid'][0][0].astype(numpy.bool_),
            TRANSPARENT_MASK_: data['mask_transp'][0][0].astype(numpy.bool_),
            EDGES_MASK_: data['edges'][0][0].astype(numpy.bool_),
            INTRINSICS_: data['calib'][0][0].astype(numpy.float32)
        }

        if self._preprocessing is not None:
            sample = self._preprocessing(sample)

        return sample
    
    def by_id(self, sample_id):
        return self.__getitem__(self._sample_id_to_idx[sample_id])
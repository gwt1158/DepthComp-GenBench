import numpy
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, List
from os.path import (
    join, 
    normpath, 
    sep, 
    splitext
)
from ssd.datasets.constants import (
    SAMPLE_ID_,
    RGB_,
    DEPTH_METRIC_,
    DEPTH_METRIC_MASK_,
    GT_METRIC_,
    INTRINSICS_
)


def load_txt_paths(txt_path: str, prepend_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [join(prepend_path, line.strip()) for line in f.readlines()]


def load_intrinsics(txt_path: str) -> numpy.ndarray:
    matrix = numpy.zeros(shape=(3, 3), dtype=numpy.float32)

    with open(txt_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            matrix[i] = [float(p) for p in line.split()]
        
    return matrix


def image_path_to_id(path: str) -> str:
    path_components = normpath(path).split(sep)
    scene = path_components[-3]
    image = splitext(path_components[-1])[0]
    return '_'.join((scene, image))


class Void(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Callable, limit_size: int = None):
        assert subset in ["train_150", "test_150", "train_500", "test_500", "train_1500", "test_1500"]

        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        self._limit_size = limit_size

        sub_path = join(self._path, f"void_{subset.split('_')[1]}")

        self._ground_truth_paths = load_txt_paths(join(sub_path, f"{subset.split('_')[0]}_ground_truth.txt"), self._path)
        self._image_paths = load_txt_paths(join(sub_path, f"{subset.split('_')[0]}_image.txt"), self._path)
        self._intrinsic_paths = load_txt_paths(join(sub_path, f"{subset.split('_')[0]}_intrinsics.txt"), self._path)
        self._sparse_depth_paths = load_txt_paths(join(sub_path, f"{subset.split('_')[0]}_sparse_depth.txt"), self._path)
        self._validity_map_paths = load_txt_paths(join(sub_path, f"{subset.split('_')[0]}_validity_map.txt"), self._path)
        self._ids = [image_path_to_id(p) for p in self._image_paths]

        if limit_size is not None:
            self._ground_truth_paths = self._ground_truth_paths[:limit_size]
            self._image_paths = self._image_paths[:limit_size]
            self._intrinsic_paths = self._intrinsic_paths[:limit_size]
            self._sparse_depth_paths = self._sparse_depth_paths[:limit_size]
            self._validity_map_paths = self._validity_map_paths[:limit_size]
            self._ids = self._ids[:limit_size]

        self._sample_id_to_idx = { sample_id: idx for idx, sample_id in enumerate(self._ids) }

    def __getitem__(self, index):
        item = {
            RGB_: numpy.array(Image.open(self._image_paths[index])),
            DEPTH_METRIC_: numpy.array(Image.open(self._sparse_depth_paths[index]), dtype=numpy.float32) / 256.0,
            DEPTH_METRIC_MASK_: numpy.array(Image.open(self._validity_map_paths[index]), dtype=numpy.bool_),
            GT_METRIC_: numpy.array(Image.open(self._ground_truth_paths[index]), dtype=numpy.float32) / 256.0,
            INTRINSICS_: load_intrinsics(self._intrinsic_paths[index]),
            SAMPLE_ID_: str(self._ids[index])
        }

        if self._preprocessing is not None:
            item = self._preprocessing(item)

        return item

    def __len__(self):
        return len(self._image_paths)
    
    def by_id(self, sample_id):
        return self.__getitem__(self._sample_id_to_idx[sample_id])
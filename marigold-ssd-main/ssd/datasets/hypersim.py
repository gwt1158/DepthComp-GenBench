import os
import numpy
import pandas

from torch.utils.data import Dataset

from PIL import Image
from typing import Optional, Callable
from os.path import sep, splitext, normpath, join
from ssd.datasets.constants import (
    SAMPLE_ID_, 
    RGB_, 
    # NORMALS_, 
    GT_METRIC_
)


def hypersim_rgb_path_to_sample_id(path: str) -> str:
    path_components = normpath(path).split(sep)
    scene = path_components[-2]
    camera = f"cam{splitext(path_components[-1])[0].split('_')[-2]}"
    frame = splitext(path_components[-1])[0].split('_')[-1]
    return '_'.join((scene, camera, frame))


class HypersimDiffusionE2E(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Optional[Callable], limit_size: Optional[int] = None):
        assert subset in ["train"]
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        self._pairs = self._find_pairs()

        if limit_size is not None:
            self._pairs = self._pairs[:limit_size]


    def _find_pairs(self):
        csv_path = join(self._path, f"{self._subset}/filename_meta_{self._subset}.csv")
        df = pandas.read_csv(csv_path)
        pairs = []
        for _, row in df.iterrows():
            if row['included_in_public_release'] and (row['split_partition_name'] == "train"):
                rgb_path = os.path.join(self._path, "train", row['rgb_path'])
                depth_path = os.path.join(self._path, "train", row['depth_path'])
                head, _ = os.path.split(os.path.join(self._path, "train"))
                normal_dir = os.path.join(os.path.join(head, 'normals'), row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_preview',f'frame.{str(row["frame_id"]).zfill(4) }.normal_cam.png')
                if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(normal_dir):
                    pair_info = {'rgb_path': rgb_path, 'depth_path': depth_path, 'normal_path': normal_dir}    
                    pairs.append(pair_info)
        return pairs

    def __len__(self):
        return len(self._pairs)
        
    def __getitem__(self, idx):
        pairs = self._pairs[idx]

        sample = {
            SAMPLE_ID_: hypersim_rgb_path_to_sample_id(pairs['rgb_path']),
            RGB_: numpy.array(Image.open(pairs['rgb_path']).convert('RGB')), 
            # NORMALS_: numpy.array(Image.open(pairs['normal_path']).convert('RGB')), 
            GT_METRIC_: numpy.array(Image.open(pairs['depth_path'])) / 1000.0,
        }
    
        if self._preprocessing is not None:
            sample = self._preprocessing(sample)

        return sample

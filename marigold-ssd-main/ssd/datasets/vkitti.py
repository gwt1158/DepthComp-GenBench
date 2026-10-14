import numpy
from PIL import Image
from os import listdir
from os.path import join, exists, normpath, sep, splitext
from torch.utils.data import Dataset
from typing import Optional, Callable
from ssd.datasets.constants import (
    SAMPLE_ID_, 
    RGB_, 
    GT_METRIC_
)


def vkitti_rgb_path_to_sample_id(path: str) -> str:
    path_components = normpath(path).split(sep)
    scene = path_components[-6]
    weather = path_components[-5]
    camera = path_components[-2].split('_')[-1]
    frame = splitext(path_components[-1])[0].split('_')[-1]
    return '_'.join((scene, weather, camera, frame))


class VkittiDiffusionE2E(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Optional[Callable], limit_size: Optional[int] = None):
        assert subset in ["train"]
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        self._pairs = self._find_pairs()

        if limit_size is not None:
            self._pairs = self._pairs[:limit_size]

    def _find_pairs(self):
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
        cameras = ["Camera_0", "Camera_1"]
        vkitti2_rgb_path = join(self._path, "vkitti_2.0.3_rgb")
        vkitti2_depth_path =  join(self._path, "vkitti_2.0.3_depth")
        pairs = []
        for scene in scenes:
            for weather in weather_conditions:
                for camera in cameras:
                    rgb_dir = join(vkitti2_rgb_path, scene, weather, "frames", "rgb" ,camera)
                    depth_dir = join(vkitti2_depth_path, scene, weather, "frames","depth" , camera)
                    if exists(rgb_dir) and exists(depth_dir):
                        rgb_files = [f for f in listdir(rgb_dir) if f.endswith(".jpg")]
                        rgb_files  = [file[3:] for file in rgb_files]
                        for file in rgb_files:
                            rgb_file = "rgb" + file
                            depth_file = "depth" + file.replace('.jpg', '.png')
                            rgb_path = join(rgb_dir, rgb_file)
                            depth_path = join(depth_dir, depth_file)
                            pairs.append((rgb_path, depth_path))
        return pairs
    
    def __len__(self):
        return len(self._pairs)
    
    def __getitem__(self, idx):
        rgb_path, depth_path = self._pairs[idx]

        sample_id = vkitti_rgb_path_to_sample_id(rgb_path)
        rgb = numpy.array(Image.open(rgb_path).convert('RGB'))
        gt = numpy.array(Image.open(depth_path)) / 100.0 # cm to meters

        sample = {
            SAMPLE_ID_: sample_id,
            RGB_: rgb, 
            GT_METRIC_: gt
        }
    
        if self._preprocessing is not None:
            sample = self._preprocessing(sample)

        return sample
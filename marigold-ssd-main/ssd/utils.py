import cv2
import h5py
import json
import torch
import numpy
import random
import matplotlib
import importlib
from typing import Tuple, Optional, Dict


def seed_all(seed: int = 0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def getattr_from_string(function_name: str):
    components = function_name.split(".")
    module = importlib.import_module(".".join(components[:-1]))
    func = getattr(module, components[-1])
    return func


def float_depth_to_8bit_rgb(depth: numpy.ndarray, map: str = "Spectral") -> numpy.ndarray:
    cm = matplotlib.colormaps[map]
    colored = cm(depth, bytes=False)[:,:,:3]
    colored = colored * 255
    return colored.astype(numpy.uint8)


def visualization_depth0to1(
        depth: numpy.ndarray, 
        valid_mask: numpy.ndarray = None, 
        map: str = "Spectral",
        mark_mask: Optional[numpy.ndarray] = None,
        mark_color: Optional[Tuple[int]] = None) -> numpy.ndarray:
    colored_rgb = float_depth_to_8bit_rgb(depth, map)
    
    if valid_mask is not None:
        colored_rgb[numpy.logical_not(valid_mask)] = (0, 0, 0)

    if mark_mask is not None and mark_color is not None:
        colored_rgb[mark_mask] = mark_color
    
    return colored_rgb


def save_visualization_depth0to1(
        depth: numpy.ndarray, 
        path: str, 
        valid_mask: numpy.ndarray = None, 
        map: str = "Spectral",
        mark_mask: Optional[numpy.ndarray] = None,
        mark_color: Optional[Tuple[int]] = None) -> None:
    
    colored_rgb = visualization_depth0to1(depth, valid_mask, map, mark_mask, mark_color)
    cv2.imwrite(path, cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))


def visualization_depthfloat(
        depth: numpy.ndarray | torch.Tensor, 
        valid_mask: numpy.ndarray | torch.Tensor,
        map: str = "Spectral",
        mark_mask: Optional[numpy.ndarray] = None,
        mark_color: Optional[Tuple[int]] = None) -> numpy.ndarray:

    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().detach().numpy()

    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().detach().numpy()

    if valid_mask is not None:
        rescaled = numpy.zeros(shape=depth.shape, dtype=depth.dtype)
        rescaled[valid_mask] = (depth[valid_mask] - depth[valid_mask].min()) / (depth[valid_mask].max() - depth[valid_mask].min())
    else:
        rescaled = (depth - depth.min()) / (depth.max() - depth.min())

    rescaled = numpy.squeeze(rescaled)

    if valid_mask is not None:
        valid_mask = numpy.squeeze(valid_mask)

    return visualization_depth0to1(rescaled, valid_mask, map, mark_mask, mark_color)


def save_visualization_depthfloat(
        depth: numpy.ndarray | torch.Tensor, 
        path: str, 
        valid_mask: numpy.ndarray | torch.Tensor = None,
        map: str = "Spectral",
        mark_mask: Optional[numpy.ndarray] = None,
        mark_color: Optional[Tuple[int]] = None) -> None:
    colored_rgb = visualization_depthfloat(depth, valid_mask, map, mark_mask, mark_color)
    cv2.imwrite(path, cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))


def visualize_absolute_difference(
        prediction: torch.Tensor, 
        target: torch.Tensor, 
        path: str, 
        valid_mask: Optional[torch.Tensor], 
        colormap: str = "GnBu", 
        mark_mask: Optional[numpy.ndarray] = None, 
        mark_color: Optional[Tuple[int]] = None):
    abs_diff = torch.abs(prediction - target)
    save_visualization_depthfloat(abs_diff, path, valid_mask, colormap, mark_mask, mark_color)


def write_h5(data: Dict[str, numpy.ndarray], path: str) -> None:
    with h5py.File(path, 'w') as h5_file:
        for k, v in data.items():
            h5_file.create_dataset(k, data=v, compression="gzip")


def load_h5(path: str) -> Dict[str, numpy.ndarray]:
    result = dict()

    with h5py.File(path, 'r') as h5_file:
        for key in h5_file.keys():
            result[key] = numpy.array(h5_file.get(key))

    return result


def encode_image(vae, image):
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent


def intrinsic_matrix(fx, fy, cx, cy) -> numpy.ndarray:
    matrix = numpy.zeros(shape=(3, 3), dtype=numpy.float32)
    matrix[0, 0] = fx
    matrix[1, 1] = fy
    matrix[0, 2] = cx
    matrix[1, 2] = cy
    matrix[2, 2] = 1
    return matrix


def load_json_to_dict(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

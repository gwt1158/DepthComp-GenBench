import numpy
import torch
from PIL import Image
from argparse import Namespace
from typing import Dict
from copy import deepcopy
from os.path import join
from torchvision.transforms import Resize
from ssd.datasets.constants import (
    SAMPLE_ID_,
    RGB_,
    DEPTH_METRIC_,
    GT_METRIC_
)
from ssd.preprocessing.constants import (
    RGB, 
    DEPTH_METRIC,
    DEPTH_METRIC_MASK,
    DEPTH_RELATIVE,
    DEPTH_RELATIVE_MASK,
    GT_METRIC,
    GT_METRIC_MASK
)
from ssd.preprocessing.utils.functions import reproject_points
from ssd.utils import load_h5


def ddad_default(args: Namespace, sample: Dict):
    if args.preprocessing_paths is not None:
        depth_path = args.preprocessing_paths[0]
    else:
        depth_path = None

    max_depth = args.max_depth
    min_depth = args.min_depth

    # RGB
    rgb_0to255 = sample[RGB_]
    
    rgb_tensor = 2 * (rgb_0to255 / 255) - 1
    rgb_tensor = numpy.transpose(rgb_tensor, (2, 0, 1)).astype(numpy.float32)
    sample[RGB] = torch.from_numpy(rgb_tensor)

    # groundtruth
    gt_metric = sample[GT_METRIC_]
    gt_mask = gt_metric > 0

    if min_depth is not None:
        gt_mask = numpy.logical_and(gt_mask, gt_metric >= min_depth)
    if max_depth is not None:
        gt_mask = numpy.logical_and(gt_mask, gt_metric <= max_depth)

    sample[GT_METRIC] = torch.from_numpy(numpy.expand_dims(gt_metric, axis=0))
    sample[GT_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(gt_mask, axis=0))

    # Depth Metric
    if depth_path is None:
        depth_metric = sample[DEPTH_METRIC_]
    else:
        depth_metric = load_h5(join(depth_path, f"{sample[SAMPLE_ID_]}.h5"))[DEPTH_METRIC]

    depth_mask = depth_metric > 0.0

    if min_depth is not None:
        depth_mask = numpy.logical_and(depth_mask, depth_metric >= min_depth)
    if max_depth is not None:
        depth_mask = numpy.logical_and(depth_mask, depth_metric <= max_depth)

    sample[DEPTH_METRIC] = torch.from_numpy(numpy.expand_dims(depth_metric, axis=0))
    sample[DEPTH_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(depth_mask, axis=0))

    # Depth Relative
    depth_relative = deepcopy(depth_metric)
    depth_relative_mask = deepcopy(depth_mask)
    depth_relative[numpy.logical_not(depth_relative_mask)] = -1.0

    depth_relative[depth_relative_mask] = (depth_relative[depth_relative_mask] - depth_relative[depth_relative_mask].min()) / (depth_relative[depth_relative_mask].max() - depth_relative[depth_relative_mask].min())
    depth_relative[depth_relative_mask] = depth_relative[depth_relative_mask] * 2.0 - 1.0

    sample[DEPTH_RELATIVE] = torch.from_numpy(numpy.expand_dims(depth_relative, axis=0))
    sample[DEPTH_RELATIVE_MASK] = torch.from_numpy(numpy.expand_dims(depth_relative_mask, axis=0))

    return sample


def ddad_default_small2(args: Namespace, sample: Dict):
    if args.preprocessing_paths is not None:
        depth_path = args.preprocessing_paths[0]
    else:
        depth_path = None

    max_depth = args.max_depth
    min_depth = args.min_depth

    # RGB
    rgb_0to255 = sample[RGB_]
    rgb_0to255 = numpy.array(Resize((480, 768))(Image.fromarray(rgb_0to255)))
    rgb_tensor = 2 * (rgb_0to255 / 255) - 1
    rgb_tensor = numpy.transpose(rgb_tensor, (2, 0, 1)).astype(numpy.float32)
    sample[RGB] = torch.from_numpy(rgb_tensor)
    
    # groundtruth
    gt_metric = sample[GT_METRIC_]
    gt_mask = gt_metric > 0

    if min_depth is not None:
        gt_mask = numpy.logical_and(gt_mask, gt_metric >= min_depth)
    if max_depth is not None:
        gt_mask = numpy.logical_and(gt_mask, gt_metric <= max_depth)

    sample[GT_METRIC] = torch.from_numpy(numpy.expand_dims(gt_metric, axis=0))
    sample[GT_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(gt_mask, axis=0))

    # Depth Metric
    new_height, new_width = rgb_0to255.shape[:2]

    if depth_path is None:
        depth_metric_large = sample[DEPTH_METRIC_]
    else:
        depth_metric_large = load_h5(join(depth_path, f"{sample[SAMPLE_ID_]}.h5"))[DEPTH_METRIC]

    depth_metric = numpy.zeros(shape=(new_height, new_width), dtype=numpy.float32)
    reproject_points(depth_metric_large, depth_metric)

    depth_mask = depth_metric > 0.0

    if min_depth is not None:
        depth_mask = numpy.logical_and(depth_mask, depth_metric >= min_depth)
    if max_depth is not None:
        depth_mask = numpy.logical_and(depth_mask, depth_metric <= max_depth)

    sample[DEPTH_METRIC] = torch.from_numpy(numpy.expand_dims(depth_metric, axis=0))
    sample[DEPTH_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(depth_mask, axis=0))

    # Depth Relative
    depth_relative = deepcopy(depth_metric)
    depth_relative_mask = deepcopy(depth_mask)
    depth_relative[numpy.logical_not(depth_relative_mask)] = -1.0

    depth_relative[depth_relative_mask] = (depth_relative[depth_relative_mask] - depth_relative[depth_relative_mask].min()) / (depth_relative[depth_relative_mask].max() - depth_relative[depth_relative_mask].min())
    depth_relative[depth_relative_mask] = depth_relative[depth_relative_mask] * 2.0 - 1.0

    sample[DEPTH_RELATIVE] = torch.from_numpy(numpy.expand_dims(depth_relative, axis=0))
    sample[DEPTH_RELATIVE_MASK] = torch.from_numpy(numpy.expand_dims(depth_relative_mask, axis=0))

    return sample

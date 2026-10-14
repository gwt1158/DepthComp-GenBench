import torch
import numpy

from argparse import Namespace
from typing import Dict
from copy import deepcopy
from ssd.datasets.constants import (
    RGB_, 
    DEPTH_METRIC_,
    DEPTH_METRIC_MASK_,
    GT_METRIC_,
)
from ssd.preprocessing.constants import *


def void_default(args: Namespace, sample: Dict) -> Dict:
    max_depth = args.max_depth
    min_depth = args.min_depth

    # RGB
    rgb_0to255 = sample[RGB_]
    
    rgb_tensor = 2 * (rgb_0to255 / 255) - 1
    rgb_tensor = numpy.transpose(rgb_tensor, (2, 0, 1)).astype(numpy.float32)
    sample[RGB] = torch.from_numpy(rgb_tensor)

    # Ground Truth
    gt_metric = sample[GT_METRIC_]
    gt_mask = gt_metric > 0

    if min_depth is not None:
        gt_mask = numpy.logical_and(gt_mask, gt_metric >= min_depth)
    if max_depth is not None:
        gt_mask = numpy.logical_and(gt_mask, gt_metric <= max_depth)

    sample[GT_METRIC] = torch.from_numpy(numpy.expand_dims(gt_metric, axis=0))
    sample[GT_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(gt_mask, axis=0))

    # Depth Metric
    depth_metric = sample[DEPTH_METRIC_]
    depth_mask = sample[DEPTH_METRIC_MASK_]

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

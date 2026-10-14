import numpy
import torch
from argparse import Namespace
from typing import Dict
from random import random, uniform
from PIL import Image
from torchvision.transforms import Resize
from ssd.datasets.constants import (
    RGB_,
    GT_METRIC_
)
from ssd.preprocessing.utils.callables import (
    RandomPointsFromDepth
)
from ssd.preprocessing.constants import (
    RGB,
    DEPTH_RELATIVE,
    DEPTH_RELATIVE_MASK,
    DEPTH_METRIC,
    DEPTH_METRIC_MASK,
    GT_RELATIVE,
    GT_METRIC,
    GT_METRIC_MASK,
    INVALID_DEPTH
)

def hypersim_vae_scaling(args: Namespace, sample: Dict) -> Dict:
    assert args.naive_lidar is not None, "naive_lidar must be specified"

    height, width = 480, 640
    near_plane = 1e-5
    # this is value from original code, this hypersim should not reach this value in theory
    far_plane = 65.0
    horizontal_flip = random() > 0.5

    # RGB
    rgb_0to255 = sample[RGB_]
    rgb_0to255 = numpy.array(Resize((height, width))(Image.fromarray(rgb_0to255)))

    if horizontal_flip:
        rgb_0to255 = numpy.copy(numpy.flip(rgb_0to255, axis=1))

    rgb_tensor = (rgb_0to255 / 255.0) * 2 - 1
    rgb_tensor = numpy.transpose(rgb_tensor, (2, 0, 1)).astype(numpy.float32)
    rgb_tensor =  torch.from_numpy(rgb_tensor)
    sample[RGB] = rgb_tensor

    # GT
    gt_metric = sample[GT_METRIC_]
    gt_metric = numpy.array(Resize((height, width), interpolation=Image.NEAREST)(Image.fromarray(gt_metric)))

    if horizontal_flip:
        gt_metric = numpy.copy(numpy.flip(gt_metric, axis=1))

    gt_mask = numpy.logical_and(gt_metric > near_plane, gt_metric < far_plane)

    if gt_mask.any():
        min_depth, max_depth = numpy.quantile(gt_metric[gt_mask], (0.02, 0.98))  
        
        if min_depth == max_depth:
            gt_mask = numpy.zeros_like(gt_mask, dtype=numpy.bool_)
            gt_metric = numpy.zeros_like(gt_metric)
            gt_relative = numpy.zeros_like(gt_metric)
            
        else:            
            gt_mask = numpy.logical_and(gt_mask, min_depth <= gt_metric)
            gt_mask = numpy.logical_and(gt_mask, gt_metric <= max_depth)
            gt_relative = numpy.full_like(gt_metric, dtype=numpy.float32, fill_value=-1.0)
            gt_relative[gt_mask] = -1 + 2.0 * ((gt_metric[gt_mask] - gt_metric[gt_mask].min()) / (gt_metric[gt_mask].max() - gt_metric[gt_mask].min()))
    else:
        gt_relative = numpy.zeros_like(gt_metric)
        gt_metric = numpy.zeros_like(gt_metric)

    sample[GT_METRIC] = torch.from_numpy(numpy.expand_dims(gt_metric, axis=0))
    sample[GT_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(gt_mask, axis=0))
    sample[GT_RELATIVE] = torch.from_numpy(numpy.expand_dims(gt_relative, axis=0))

    # Depth Metric
    if args.naive_lidar_max is None:
        preserve_portion = args.naive_lidar
    else:
        preserve_portion = uniform(args.naive_lidar, args.naive_lidar_max)

    depth_metric, depth_mask = RandomPointsFromDepth(preserve_portion, INVALID_DEPTH, valid_mask=gt_mask, return_mask=True)(gt_metric)
    sample[DEPTH_METRIC] = torch.from_numpy(numpy.expand_dims(depth_metric, axis=0))
    sample[DEPTH_METRIC_MASK] = torch.from_numpy(numpy.expand_dims(depth_mask, axis=0))

    # Depth Relative
    depth_relative = numpy.full_like(depth_metric, fill_value=-1.0)
    depth_relative[depth_mask] = gt_relative[depth_mask]

    sample[DEPTH_RELATIVE] = torch.from_numpy(numpy.expand_dims(depth_relative, axis=0))
    sample[DEPTH_RELATIVE_MASK] = torch.from_numpy(numpy.expand_dims(depth_mask, axis=0))

    return sample 

# The code was adopted from:
# https://github.com/prs-eth/Marigold/blob/ac915287aaa4e5fd5fe578322735c7c411a1f263/script/depth/eval.py
# license notice:

# Last modified: 2024-03-11
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import logging
import os

import cv2
import numpy
import torch

from os import makedirs
from os.path import join, exists
from functools import partial
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ssd.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from ssd import metrics
from ssd.metrics import MetricTracker
from ssd.utils import getattr_from_string, visualize_absolute_difference
from ssd.args import save_arguments_to_json, ARGS_JSON, parse_eval_args
from ssd.datasets.constants import SAMPLE_ID_
from ssd.preprocessing.constants import (
    DEPTH_METRIC,
    DEPTH_METRIC_MASK,
    GT_METRIC, 
    GT_METRIC_MASK
)

CSV_RESULTS = "per_sample_metrics.csv"
JSON_RESULTS = "average.json"
VIS_ERROR_DIR = "error_visualizations"

EVAL_METRICS = [
    "abs_relative_difference",
    "squared_relative_difference",
    "mean_absolute_error",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]

if "__main__" == __name__:
    args = parse_eval_args() 
    makedirs(args.output, exist_ok=True)
    save_arguments_to_json(args, join(args.output, ARGS_JSON))

    input_dir = args.input
    output_dir = args.output
    alignment = args.alignment
    alignment_max_res = args.alignment_max_res
    no_cuda = args.no_cuda

    vis_abs_err_dir = join(args.output, VIS_ERROR_DIR)
    if args.visualize_error:
        makedirs(vis_abs_err_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    preprocessing = partial(getattr_from_string(args.eval_preprocessing), args)
    dataset = getattr_from_string(args.eval_dataset)(args.eval_dataset_path, args.eval_dataset_subset, preprocessing, args.limit_dataset_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metrics, _met) for _met in EVAL_METRICS]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metric file head --------------------
    per_sample_filename = os.path.join(output_dir, CSV_RESULTS)
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc=f"Evaluating {type(dataset).__name__}"):
        # Load predictions        
        sample_id = data[SAMPLE_ID_][0]
        pred_path = join(input_dir, f"{sample_id}.npy")

        if not exists(pred_path):
            print(f"sample {sample_id} was not found and won't be evaluated")
            continue

        depth_pred = numpy.load(pred_path)

        # Align prediction
        if args.alignment is not None and args.align_with is not None:
            if args.align_with == "groundtruth":
                align_target_ts = data[GT_METRIC].squeeze()
                align_mask_ts = data[GT_METRIC_MASK].squeeze()
            else:
                align_target_ts = data[DEPTH_METRIC].squeeze()
                align_mask_ts = data[DEPTH_METRIC_MASK].squeeze()
            
            align_target = align_target_ts.numpy()
            align_mask = align_mask_ts.numpy()

            align_target_ts = align_target_ts.to(device)
            align_mask_ts = align_mask_ts.to(device)

            # Align least square or least square disparity
            if "least_square" == alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=align_target,
                    pred_arr=depth_pred,
                    valid_mask_arr=align_mask,
                    return_scale_shift=True,
                    max_resolution=alignment_max_res,
                )
            
            elif "least_square_disparity" == alignment:
                # convert GT depth -> GT disparity
                gt_disparity, gt_non_neg_mask = depth2disparity(
                    depth=align_target, return_mask=True
                )
                # LS alignment in disparity space
                pred_non_neg_mask = depth_pred > 0
                valid_nonnegative_mask = align_mask & gt_non_neg_mask & pred_non_neg_mask

                disparity_pred, scale, shift = align_depth_least_square(
                    gt_arr=gt_disparity,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_nonnegative_mask,
                    return_scale_shift=True,
                    max_resolution=alignment_max_res,
                )
                # convert to depth
                disparity_pred = numpy.clip(
                    disparity_pred, a_min=1e-3, a_max=None
                )  # avoid 0 disparity
                depth_pred = disparity2depth(disparity_pred)

        # clip to min and max depth
        if args.min_depth is not None or args.max_depth is not None:
            depth_pred = numpy.clip(depth_pred, a_min=args.min_depth, a_max=args.max_depth)

        # clip to d > 0 for evaluation
        depth_pred = numpy.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate (using CUDA if available)
        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)

        # GT data
        gt_ts = data[GT_METRIC].squeeze().to(device)
        gt_mask_ts = data[GT_METRIC_MASK].squeeze().to(device)

        # postprocessing
        if args.postprocessing:
            postprocessing_func = getattr_from_string(args.postprocessing)
            depth_pred_ts, gt_ts, gt_mask_ts = postprocessing_func(depth_pred_ts, gt_ts, gt_mask_ts)

        # metrics
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_ts, gt_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(sample_id + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

        if args.visualize_error:
            abs_diff_path = join(vis_abs_err_dir, f"{sample_id}_absolute_difference.png")

            if DEPTH_METRIC_MASK in data.keys() and args.postprocessing is None:
                mark_mask = data[DEPTH_METRIC_MASK].squeeze().numpy().astype(numpy.uint8)
                mark_mask = cv2.dilate(mark_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)), iterations=1).astype(numpy.bool_)
            else:
                mark_mask = None
                
            visualize_absolute_difference(depth_pred_ts, gt_ts, valid_mask=gt_mask_ts, path=abs_diff_path, mark_mask=mark_mask, mark_color=(255, 0, 0))

    metric_tracker.save_average_to_json(join(output_dir, JSON_RESULTS))
    metric_tracker.save_count_to_json(join(output_dir, "count.json"))
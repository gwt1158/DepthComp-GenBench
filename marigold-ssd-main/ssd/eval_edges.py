# this code takes inspiration from:
# https://github.com/prs-eth/Marigold/blob/ac915287aaa4e5fd5fe578322735c7c411a1f263/script/depth/eval.py
# original license notice:

# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
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

import logging
import os
import numpy
import torch
import json
import math
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
from ssd.utils import getattr_from_string
from ssd.args import (
    save_arguments_to_json, 
    parse_eval_args
)
from ssd.evaluation.evaluate_ibims_error_metrics import compute_depth_boundary_error
from ssd.datasets.constants import (
    SAMPLE_ID_, 
    EDGES_MASK_
)
from ssd.preprocessing.constants import (
    DEPTH_METRIC,
    DEPTH_METRIC_MASK,
    GT_METRIC, 
    GT_METRIC_MASK
)

EDGE_EVAL_ARGS = "args-eval-edges.json"
EDGE_CSV_RESULTS = "per-sample-metrics-edges.csv"
EDGE_JSON_RESULTS = "average-edges.json"
EDGE_JSON_COUNTS = "count-edges.json"


class EdgeMetricTracker:
    def __init__(self, metrics):
        self._metrics = {}
        self._counts = {}

        for k in metrics:
            self._metrics[k] = 0
            self._counts[k] = 0

    def update(self, metric, value):
        if math.isnan(value):
            return

        self._metrics[metric] = self._metrics[metric] + value
        self._counts[metric] = self._counts[metric] + 1


    def result(self):
        result_dict = {}

        for k in self._metrics.keys():
            result_dict[k] = self._metrics[k] / self._counts[k]

        return result_dict
    
    def save_average_to_json(self, path: str):
        with open(path, "w") as file:
            json.dump({k: float(v) for k, v in self.result().items()}, file, indent=2, sort_keys=True)

    def save_count_to_json(self, path: str):
        with open(path, "w") as file:
            json.dump({k: float(v) for k, v in self._counts.items()}, file, indent=2, sort_keys=True)


if "__main__" == __name__:
    args = parse_eval_args() 
    makedirs(args.output, exist_ok=True)
    save_arguments_to_json(args, join(args.output, EDGE_EVAL_ARGS))

    input_dir = args.input
    output_dir = args.output
    alignment = args.alignment
    alignment_max_res = args.alignment_max_res
    no_cuda = args.no_cuda

    cuda_avail = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    preprocessing = partial(getattr_from_string(args.eval_preprocessing), args)
    dataset = getattr_from_string(args.eval_dataset)(args.eval_dataset_path, args.eval_dataset_subset, preprocessing)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    metrics_names = ["dbe_acc", "dbe_com"]
    tracker = EdgeMetricTracker(metrics=metrics_names)
    per_sample_filename = os.path.join(output_dir, EDGE_CSV_RESULTS)

    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join(metrics_names))
        f.write("\n")

    for sample in tqdm(dataloader, desc=f"Evaluating Edges {type(dataset).__name__}"):
        sample_id = sample[SAMPLE_ID_][0]
        pred_path = join(input_dir, f"{sample_id}.npy")

        if not exists(pred_path):
            print(f"sample {sample_id} was not found and won't be evaluated")
            continue

        depth_pred = numpy.load(pred_path)

        if args.alignment is not None and args.align_with is not None:
            if args.align_with == "groundtruth":
                align_target_ts = sample[GT_METRIC].squeeze()
                align_mask_ts = sample[GT_METRIC_MASK].squeeze()
            else:
                align_target_ts = sample[DEPTH_METRIC].squeeze()
                align_mask_ts = sample[DEPTH_METRIC_MASK].squeeze()
            
            align_target = align_target_ts.numpy()
            align_mask = align_mask_ts.numpy()

            align_target_ts = align_target_ts.to(device)
            align_mask_ts = align_mask_ts.to(device)

            if "least_square" == alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=align_target,
                    pred_arr=depth_pred,
                    valid_mask_arr=align_mask,
                    return_scale_shift=True,
                    max_resolution=alignment_max_res,
                )
            
            elif "least_square_disparity" == alignment:
                gt_disparity, gt_non_neg_mask = depth2disparity(
                    depth=align_target, return_mask=True
                )
                pred_non_neg_mask = depth_pred > 0
                valid_nonnegative_mask = align_mask & gt_non_neg_mask & pred_non_neg_mask

                disparity_pred, scale, shift = align_depth_least_square(
                    gt_arr=gt_disparity,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_nonnegative_mask,
                    return_scale_shift=True,
                    max_resolution=alignment_max_res,
                )
                disparity_pred = numpy.clip(
                    disparity_pred, a_min=1e-3, a_max=None
                )
                depth_pred = disparity2depth(disparity_pred)
    
        if args.min_depth is not None or args.max_depth is not None:
            depth_pred = numpy.clip(depth_pred, a_min=args.min_depth, a_max=args.max_depth)

        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)

        gt_ts = sample[GT_METRIC].squeeze().to(device)
        gt_mask_ts = sample[GT_METRIC_MASK].squeeze().to(device)

        if args.postprocessing:
            postprocessing_func = getattr_from_string(args.postprocessing)
            depth_pred_ts, gt_ts, gt_mask_ts = postprocessing_func(depth_pred_ts, gt_ts, gt_mask_ts)

        edges = sample[EDGES_MASK_].squeeze().numpy()
        depth_pred = depth_pred_ts.detach().cpu().numpy()
        dbe_acc, dbe_com = compute_depth_boundary_error(edges, depth_pred)

        sample_metric.append(str(dbe_acc))
        sample_metric.append(str(dbe_com))

        tracker.update("dbe_acc", dbe_acc)
        tracker.update("dbe_com", dbe_com)

        with open(per_sample_filename, "a+") as f:
            f.write(sample_id + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    tracker.save_average_to_json(join(output_dir, EDGE_JSON_RESULTS))
    tracker.save_count_to_json(join(output_dir, EDGE_JSON_COUNTS))
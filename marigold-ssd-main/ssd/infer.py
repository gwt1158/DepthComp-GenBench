# this code was adopted from:
# https://github.com/VisualComputingInstitute/diffusion-e2e-ft/blob/main/Marigold/infer.py

# Last modified: 2024-03-30
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

# @GonzaloMartinGarcia
# The following code is built upon Marigold's infer.py, and was adapted to include some new settings.
# All additions made are marked with # add.

import logging
from os import makedirs
from os.path import join, exists

import numpy
import torch
from os.path import join
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
from diffusers import (
    DDIMScheduler, 
    UNet2DConditionModel
)
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer,
)
from ssd.utils import (
    seed_all, 
    getattr_from_string
)
from ssd.args import (
    save_arguments_to_json, 
    ARGS_JSON
)
from ssd.datasets.constants import SAMPLE_ID_
from ssd.preprocessing.constants import (
    RGB,
    DEPTH_METRIC,
    DEPTH_METRIC_MASK,
    DEPTH_RELATIVE
)
from ssd.preprocessing.utils.callables import InterpolateLidar
from ssd.model.pipe import VaePipeline, DepthOutput
from ssd.model.depth_autoencoder_mid import DepthAutoencoderMid
from ssd.utils import visualization_depthfloat

from ssd.args import (
    OPT_INTERPOLATION, 
    OPT_VAE_MID,
    parse_infer_args
)

VIS_DEPTH_DIR = "depth_visualizations"
TIMERS_JSON = "timers.json"
INFERENCE_TIMER = "inference"

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    args = parse_infer_args()

    # add
    noise = args.noise
    timestep_spacing = args.timestep_spacing
    model_type = args.model_type

    model = args.model
    unet_subfolder = "unet" if args.unet_subfolder is None else args.unet_subfolder 
    vae_subfolder = "vae" if args.vae_subfolder is None else args.vae_subfolder

    dataset_str = args.eval_dataset
    dataset_dir = args.eval_dataset_path
    output_dir = args.output_dir

    half_precision = args.half_precision

    seed = args.seed

    # Save arguments
    makedirs(args.output_dir, exist_ok=True)
    save_arguments_to_json(args, join(args.output_dir, ARGS_JSON))

    depth_vis_dir = join(args.output_dir, VIS_DEPTH_DIR)
    if args.visualize_depth:
        makedirs(depth_vis_dir, exist_ok=True)

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")

    # -------------------- Data --------------------
    preprocessing = partial(getattr_from_string(args.eval_preprocessing), args)
    dataset = getattr_from_string(dataset_str)(dataset_dir, args.eval_dataset_subset, preprocessing, args.limit_dataset_size)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    enable_xformers = True
    enable_grad = False

    if model_type == OPT_INTERPOLATION:
        interpolation = InterpolateLidar(fill_value=0.0)
        enable_xformers = False

    elif model_type == OPT_VAE_MID:
        if args.model_vae is not None:
            vae = DepthAutoencoderMid.from_pretrained(args.model_vae)
        else:
            vae = DepthAutoencoderMid.from_pretrained(model, subfolder=vae_subfolder)
        
        unet         = UNet2DConditionModel.from_pretrained(model, subfolder=unet_subfolder)   
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")  
        tokenizer    = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer") 
        scheduler    = DDIMScheduler.from_pretrained(model, timestep_spacing=timestep_spacing, subfolder="scheduler") 
        pipe = VaePipeline.from_pretrained(pretrained_model_name_or_path = model, unet=unet, vae=vae, 
            scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer, variant=variant, torch_dtype=dtype)

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            logging.debug("run without xformers")

    try:
        pipe = pipe.to(device)
        pipe.unet.eval()
    except NameError:
        # do nothing, we are running model without pipeline
        pass

    # -------------------- Inference and saving --------------------
    with torch.set_grad_enabled(enable_grad):
        for batch in tqdm(loader, desc=f"Infering {type(dataset).__name__}", leave=True):
            sample_id = batch[SAMPLE_ID_][0]
            result_path = join(output_dir, f"{sample_id}.npy")
            visualization_path = join(depth_vis_dir, f"{sample_id}.png")

            if exists(result_path):
                continue

            if model_type == OPT_INTERPOLATION:
                depth_metric = batch[DEPTH_METRIC].squeeze().numpy()
                depth_mask = batch[DEPTH_METRIC_MASK].squeeze().numpy()
                prediction = interpolation(depth_metric, depth_mask).astype(numpy.float32)
                colored = Image.fromarray(visualization_depthfloat(prediction, prediction > 0.0))
                pipe_out = DepthOutput(depth_np=prediction, depth_colored=colored)

            elif model_type == OPT_VAE_MID:
                input_image = batch[RGB].squeeze(dim=0)
                depth_image = batch[DEPTH_RELATIVE].squeeze(dim=0)
                pipe_out = pipe(
                    input_image,
                    depth_image,
                    color_map="Spectral", 
                    show_progress_bar=False, 
                    noise=noise
                )

            # Save predictions
            numpy.save(result_path, pipe_out.depth_np)
            if args.visualize_depth:
                pipe_out.depth_colored.save(visualization_path)

# This is modified version of 
# https://github.com/VisualComputingInstitute/diffusion-e2e-ft/blob/main/Marigold/marigold/marigold_pipeline.py
# license notices:

# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-04-16
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
# This file is a modified version of the original Marigold pipeline file. 
# Based on GeoWizard, we added the option to sample surface normals, marked with # add.

import torch
import numpy as np
import matplotlib
from PIL import Image
from typing import Union
from tqdm.auto import tqdm
from torch.utils.data import (
    DataLoader, 
    TensorDataset
)
from diffusers import (
    DiffusionPipeline, 
    DDIMScheduler, 
    DDPMScheduler, 
    LCMScheduler,
    UNet2DConditionModel
)
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers.utils import BaseOutput
from ssd.model.utils import pyramid_noise_like
from ssd.model.depth_autoencoder_mid import DepthAutoencoderMid


class DepthOutput(BaseOutput):
    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


class VaePipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: DepthAutoencoderMid,
        scheduler: Union[DDIMScheduler, DDPMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: torch.Tensor,
        depth_condition: torch.Tensor,
        batch_size: int = 1,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        noise="gaussian"
    ) -> DepthOutput:
        assert input_image.min() >= -1.0 and input_image.max() <= 1.0

        # ----------------- Image Preprocess -----------------

        input_size = input_image.shape
        assert (3 == input_image.dim() and 3 == input_size[0]), f"Wrong input shape {input_size}, expected [rgb, H, W]"

        # ----------------- Predicting depth/normal --------------

        # Batch repeated input image
        ensemble_rgb = torch.stack([input_image] * 1)
        ensemble_depth = torch.stack([depth_condition] * 1)
        dataset = TensorDataset(ensemble_rgb, ensemble_depth)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # load iterator
        pred_ls  = []
        if show_progress_bar:
            iterable = tqdm(
                loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = loader

        # inference (batched)
        for batch in iterable:
            (batched_img, batched_depth) = batch
            pred_raw = self.single_infer(
                rgb_in=batched_img,
                depth_in=batched_depth,
                num_inference_steps=1,
                show_pbar=show_progress_bar,
                noise=noise
            )
            pred_ls.append(pred_raw.detach())
        preds = torch.concat(pred_ls, dim=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        pred = preds
        # ----------------- Post processing -----------------

        # Scale relative prediction to [0, 1]
        min_d = torch.min(pred)
        max_d = torch.max(pred)
        if max_d == min_d:
            pred = torch.zeros_like(pred)
        else:
            pred = (pred - min_d) / (max_d - min_d)

        # Convert to numpy
        pred = pred.cpu().numpy()

        # Process prediction for visualization

        # add
        pred = pred.clip(0, 1)
        if color_map is not None:
            colored = colorize_depth_maps(
                pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            colored = (colored * 255).astype(np.uint8)
            colored_hwc = chw2hwc(colored)
            colored_img = Image.fromarray(colored_hwc)
        else:
            colored_img = None
        
        return DepthOutput(
            depth_np       = pred,
            depth_colored  = colored_img
        )


    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        depth_in: torch.Tensor,
        num_inference_steps: int,
        show_pbar: bool,
        noise="gaussian"
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            noise (`str`, *optional*, defaults to `gaussian`):
                Type of noise to be used for the initial depth map.
                Can be one of `gaussian`, `pyramid`, `zeros`.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        depth_in = depth_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        
        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # add
        # Initial prediction
        latent_shape = rgb_latent.shape
        if noise == "gaussian":
            latent = torch.randn(
                latent_shape,
                device=device,
                dtype=self.dtype,
            )
        elif noise == "pyramid":
            latent = pyramid_noise_like(rgb_latent).to(device) # [B, 4, h, w]
        elif noise == "zeros":
            latent = torch.zeros(
                latent_shape,
                device=device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unknown noise type: {noise}")

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_step = self.scheduler.step(
                noise_pred, t, latent
            )
        
            latent = scheduler_step.prev_sample

            # add
            if i == num_inference_steps-1:
                latent = scheduler_step.pred_original_sample 
             
        # decode and normalize depth map          
        depth = self.decode_depth(latent, depth_in)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        return depth


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent
    

    def decode_depth(self, latent: torch.Tensor, depth_condition: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        latent = latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(latent)
        stacked = self.vae.decoder(z, depth_condition)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

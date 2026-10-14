import torch
from os.path import join
from diffusers import UNet2DConditionModel
from ssd.model.depth_autoencoder_mid import DepthAutoencoderMid
from ssd.utils import encode_image

class UnetDecMid(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, vae: DepthAutoencoderMid):
        super().__init__()
        self.unet = unet
        self.vae = vae


    def forward(self, rgb, depth_condition, noise_scheduler, empty_encoding, alpha_prod, beta_prod):
        # RGB latent
        rgb_latents = encode_image(self.vae, rgb)
        rgb_latents = rgb_latents * self.vae.config.scaling_factor

        # Set timesteps to the first denoising step
        timesteps = torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) * (noise_scheduler.config.num_train_timesteps-1) # 999
        timesteps = timesteps.long()

        # Sample noisy latent
        noisy_latents = torch.zeros_like(rgb_latents).to(rgb_latents.device)

        # Generate UNet prediction
        encoder_hidden_states = empty_encoding.repeat(len(rgb), 1, 1)
        unet_input = torch.cat((rgb_latents, noisy_latents), dim=1).to(rgb_latents.device)
        model_pred = self.unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0]

        # Convert parameterized prediction into latent prediction.
        # Code is based on the DDIM code from diffusers,
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py.
        alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t  =  beta_prod[timesteps].view(-1, 1, 1, 1)
        if noise_scheduler.config.prediction_type == "v_prediction":
            current_latent_estimate = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
        elif noise_scheduler.config.prediction_type == "epsilon":
            current_latent_estimate = (noisy_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
        elif noise_scheduler.config.prediction_type == "sample":
            current_latent_estimate = model_pred
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        current_latent_estimate = current_latent_estimate / self.vae.config.scaling_factor
        current_estimate = self.vae.decode_with_depth(current_latent_estimate, depth_condition)
        return current_estimate


    def initialize_fusion(self, initialization: str):
        self.vae.initialize_fusion(initialization)


    def initialize_condition_encoder(self):
        self.vae.initialize_condition_encoder()


    def save_pretrained(self, path: str):
        self.unet.save_pretrained(join(path, "unet"))
        self.vae.save_pretrained(join(path, "vae"))

    @staticmethod
    def from_pretrained(path: str):
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")
        vae = DepthAutoencoderMid.from_pretrained(path, subfolder="vae")
        return UnetDecMid(unet, vae)

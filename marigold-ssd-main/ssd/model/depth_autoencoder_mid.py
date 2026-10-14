import torch
from typing import Tuple, Optional
from torch import nn
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import Encoder
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import register_to_config
from ssd.model.depth_decoder_mid import DepthDecoderMid
from ssd.model.condition_encoder_mid import ConditionEncoderMid


class DepthAutoencoderMid(AutoencoderKL):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super(ModelMixin, self).__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = DepthDecoderMid(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
            condition_encoder_cls=ConditionEncoderMid,
            condition_encoder_params={
                "in_channels": 1, 
                "down_block_types": down_block_types, 
                "block_out_channels": block_out_channels, 
                "layers_per_block": layers_per_block,
                "norm_num_groups": norm_num_groups,
                "act_fn": act_fn,
                "mid_block_add_attention": mid_block_add_attention
            }
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25


    def initialize_fusion(self, initialization: str) -> None:
        if initialization is None or initialization == "none":
            return
        
        if initialization == "zeros":
            for block in self.decoder.fusion_blocks:
                torch.nn.init.zeros_(block.bias)
                torch.nn.init.zeros_(block.weight)

                with torch.no_grad():
                    for idx in range(block.weight.shape[0]):
                        block.weight[idx, idx, 0, 0] = 1.0

            return
        
        raise Exception(f"initialization {initialization} not implemented")


    def initialize_condition_encoder(self):
        # mid
        self.decoder.condition_encoder.mid_block.load_state_dict(self.encoder.mid_block.state_dict())

        # down
        for idx, down_block in enumerate(self.encoder.down_blocks):
            target_block = self.decoder.condition_encoder.down_blocks[idx]
            target_block.load_state_dict(down_block.state_dict(), strict=False)


    def decode_with_depth(self, latent, depth):
        z = self.post_quant_conv(latent)
        image = self.decoder(z, depth)
        return image

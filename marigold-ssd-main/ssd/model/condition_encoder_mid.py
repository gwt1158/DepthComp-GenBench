import torch
from torch import nn
from typing import Tuple
from diffusers.models.resnet import Downsample2D
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.models.unets.unet_2d_blocks import get_down_block


class ConditionEncoderMid(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            add_downsampling = i > 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=add_downsampling,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.downsample = Downsample2D(block_out_channels[-1], use_conv=True, padding=1, name="op")

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )


    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)

        features = (sample, )

        for down_block in self.down_blocks:
            sample = down_block(sample)
            features = features + (sample, )

        sample = self.downsample(sample)
        sample = self.mid_block(sample)
        features = features + (sample, )

        return features
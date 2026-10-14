import torch
import random
from torch.nn import Conv2d, Parameter

# function taken from https://github.com/VisualComputingInstitute/diffusion-e2e-ft/blob/e94db732161be6021322afc3688481f23071f5b2/training/util/unet_prep.py#L6
# @GonzaloMartinGarcia
# Function is based on an early commit of the Marigold GitHub repository.
def replace_unet_conv_in(unet, repeat=2):
    _weight = unet.conv_in.weight.clone() 
    _bias = unet.conv_in.bias.clone() 
    _weight = _weight.repeat((1, repeat, 1, 1))  
    # scale the activation magnitude
    _weight /= repeat
    _bias /= repeat
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(4*repeat, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    # replace config
    unet.config['in_channels'] = 4*repeat
    return

# Multiresolution nosie from
# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31.
def pyramid_noise_like(x, discount=0.9):
    b, c, w, h = x.shape 
    u = torch.nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random()*2+2  
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
        if w==1 or h==1: 
            break 
    return noise / noise.std()

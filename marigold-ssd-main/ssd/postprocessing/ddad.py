import torch
from torchvision.transforms import Resize, InterpolationMode


def ddad_larger_res2(prediction: torch.Tensor, gt: torch.Tensor, gt_mask:torch.Tensor):
    prediction = Resize((1216, 1936), interpolation=InterpolationMode.NEAREST_EXACT)(prediction.unsqueeze(0)).squeeze(0)
    return prediction, gt, gt_mask
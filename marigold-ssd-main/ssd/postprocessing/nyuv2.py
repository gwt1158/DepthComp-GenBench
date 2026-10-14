import torch
from torchvision.transforms import Resize, InterpolationMode, CenterCrop


def nyuv2_small_res(prediction: torch.Tensor, gt: torch.Tensor, gt_mask:torch.Tensor):
    resize = Resize((240, 320), interpolation=InterpolationMode.NEAREST_EXACT)
    crop = CenterCrop((228, 304))
    
    prediction = resize(prediction.unsqueeze(0))
    prediction = crop(prediction).squeeze(0)
    
    gt = resize(gt.unsqueeze(0))
    gt = crop(gt).squeeze(0)

    gt_mask = resize(gt_mask.unsqueeze(0))
    gt_mask = crop(gt_mask).squeeze(0)

    return prediction, gt, gt_mask

import torch
from torch.nn.functional import l1_loss


class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "L1"

    def forward(self, prediction, target, mask):
        return l1_loss(prediction[mask], target[mask])



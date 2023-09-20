import torch
import torch.nn as nn

class KTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth):
        loss = torch.nn.functional.binary_cross_entropy(
            pred, truth.float(), reduction="mean"
        )
        return loss
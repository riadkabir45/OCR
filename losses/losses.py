import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        # ... dice loss implementation ...
        pass

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_pull, delta_push):
        super().__init__()
        self.delta_pull = delta_pull
        self.delta_push = delta_push
    def forward(self, embedding_maps, instance_masks):
        # ... discriminative loss implementation ...
        pass

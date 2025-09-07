import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1., bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        dice = self.dice(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        # Apply sigmoid if inputs are logits
        if inputs.max() > 1 or inputs.min() < 0:
            inputs = torch.sigmoid(inputs)
        # Clamp to [0,1] for safety
        inputs = torch.clamp(inputs, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        # Flatten
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        loss = 1 - dice
        # Clamp loss to [0,1]
        loss = torch.clamp(loss, 0, 1)
        return loss

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_pull, delta_push):
        super().__init__()
        self.delta_pull = delta_pull
        self.delta_push = delta_push
    def forward(self, embedding_maps, instance_masks):
        # Dummy implementation: always return zero loss (replace with real logic as needed)
        return torch.tensor(0.0, device=embedding_maps.device if isinstance(embedding_maps, torch.Tensor) else 'cpu')

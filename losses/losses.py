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
        # embedding_maps: (B, emb_dim, H, W)
        # instance_masks: list of list of (H, W) masks per image
        device = embedding_maps.device
        batch_size, emb_dim, H, W = embedding_maps.shape
        total_loss = 0.0
        eps = 1e-6
        for b in range(batch_size):
            emb = embedding_maps[b]  # (emb_dim, H, W)
            masks = instance_masks[b]
            instance_means = []
            pull_loss = 0.0
            pixel_count = 0
            for mask in masks:
                mask = mask.to(device)
                if mask.sum() == 0:
                    continue
                # Crop mask and embedding to min common shape
                h_mask, w_mask = mask.shape
                h_emb, w_emb = emb.shape[1:]
                min_h = min(h_mask, h_emb)
                min_w = min(w_mask, w_emb)
                mask_cropped = mask[:min_h, :min_w]
                emb_cropped = emb[:, :min_h, :min_w]
                mask_flat = mask_cropped.reshape(-1)
                emb_flat = emb_cropped.reshape(emb_dim, -1)
                emb_fg = emb_flat[:, mask_flat > 0]  # (emb_dim, N_fg)
                if emb_fg.shape[1] == 0:
                    continue
                mean = emb_fg.mean(dim=1)
                instance_means.append(mean)
                # Pull loss: variance within instance
                pull_loss += ((emb_fg.t() - mean) ** 2).sum() / (emb_fg.shape[1] + eps)
                pixel_count += emb_fg.shape[1]
            if len(instance_means) > 1:
                means = torch.stack(instance_means, dim=0)  # (num_inst, emb_dim)
                # Push loss: means should be far apart
                push_loss = 0.0
                for i in range(len(means)):
                    for j in range(i + 1, len(means)):
                        dist = torch.norm(means[i] - means[j])
                        push_loss += torch.clamp(self.delta_push - dist, min=0) ** 2
                push_loss = push_loss / (len(means) * (len(means) - 1) / 2)
            else:
                push_loss = 0.0
            if pixel_count > 0:
                pull_loss = pull_loss / len(instance_means)
            total_loss += self.delta_pull * pull_loss + push_loss
        total_loss = total_loss / batch_size
        return total_loss

import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, instance_criterion, device):
    model.train()
    pbar = tqdm(dataloader)
    for images, masks_list in pbar:
    # ...existing code...
        images = images.to(device)
        masks = [[mask.to(device) for mask in image_masks] for image_masks in masks_list]
        optimizer.zero_grad()
        semantic_pred, embedding_pred = model(images)
    # ...existing code...
        # Create semantic ground truth by combining all instance masks for each image
        semantic_gt = torch.zeros_like(semantic_pred)
        for i, image_masks in enumerate(masks):
            if len(image_masks) > 0:
                stacked = torch.stack(image_masks, dim=0)
                # ...existing code...
                # Crop both to min common shape
                mask_sum = (stacked.sum(dim=0) > 0).float()
                h_gt, w_gt = semantic_gt[i, 0].shape
                h_mask, w_mask = mask_sum.shape
                min_h = min(h_gt, h_mask)
                min_w = min(w_gt, w_mask)
                semantic_gt[i, 0, :min_h, :min_w] = mask_sum[:min_h, :min_w]
    # Crop to minimum common size
        min_h = min(semantic_pred.shape[2], semantic_gt.shape[2])
        min_w = min(semantic_pred.shape[3], semantic_gt.shape[3])
        semantic_pred_cropped = semantic_pred[:, :, :min_h, :min_w]
        semantic_gt_cropped = semantic_gt[:, :, :min_h, :min_w]
    # ...existing code...
        # Compute losses
        semantic_loss = criterion(semantic_pred_cropped, semantic_gt_cropped)
        discriminative_loss = instance_criterion(embedding_pred, masks)
        loss = semantic_loss + discriminative_loss
    # ...existing code...
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': float(loss), 'semantic_loss': float(semantic_loss), 'disc_loss': float(discriminative_loss)})

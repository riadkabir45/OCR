import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, instance_criterion, device):
    model.train()
    for images, masks_list in tqdm(dataloader):
        images = images.to(device)
        masks = [[mask.to(device) for mask in image_masks] for image_masks in masks_list]
        optimizer.zero_grad()
        semantic_pred, embedding_pred = model(images)
        # ... crop, loss, backward, step ...
        # (Copy your training logic here)
        pass

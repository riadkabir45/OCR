import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import LocalizationDataset
from torch.utils.data import random_split
from models.unet import InstanceAwareUNet
from losses.losses import BCEDiceLoss, DiscriminativeLoss
from utils.collate import custom_collate_fn
from trainers.train import train_one_epoch
from visualize_predictions import visualize_predictions
import torch.optim as optim
import os


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset and DataLoader with train/val split
DATA_DIR = 'converted'
dataset = LocalizationDataset(DATA_DIR)
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

# Model, Loss, Optimizer
model = InstanceAwareUNet(in_channels=3, out_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = BCEDiceLoss()
instance_criterion = DiscriminativeLoss(delta_pull=0.1, delta_push=0.5)

# Training Loop
num_epochs = 50

def evaluate(model, dataloader, criterion, instance_criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, masks_list in dataloader:
            images = images.to(device)
            masks = [[mask.to(device) for mask in image_masks] for image_masks in masks_list]
            semantic_pred, embedding_pred = model(images)
            semantic_gt = torch.zeros_like(semantic_pred)
            for i, image_masks in enumerate(masks):
                if len(image_masks) > 0:
                    stacked = torch.stack(image_masks, dim=0)
                    mask_sum = (stacked.sum(dim=0) > 0).float()
                    h_gt, w_gt = semantic_gt[i, 0].shape
                    h_mask, w_mask = mask_sum.shape
                    min_h = min(h_gt, h_mask)
                    min_w = min(w_gt, w_mask)
                    semantic_gt[i, 0, :min_h, :min_w] = mask_sum[:min_h, :min_w]
            min_h = min(semantic_pred.shape[2], semantic_gt.shape[2])
            min_w = min(semantic_pred.shape[3], semantic_gt.shape[3])
            semantic_pred_cropped = semantic_pred[:, :, :min_h, :min_w]
            semantic_gt_cropped = semantic_gt[:, :, :min_h, :min_w]
            semantic_loss = criterion(semantic_pred_cropped, semantic_gt_cropped)
            discriminative_loss = instance_criterion(embedding_pred, masks)
            loss = semantic_loss + discriminative_loss
            total_loss += float(loss)
            total_batches += 1
    return total_loss / max(total_batches, 1)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer, criterion, instance_criterion, device)
    val_loss = evaluate(model, val_loader, criterion, instance_criterion, device)
    print(f"Epoch {epoch+1}: Validation loss = {val_loss:.4f}")

    # Save the model after each epoch
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'instance_aware_unet_epoch{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# Save the trained model
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'instance_aware_unet.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Visualize predictions on one sample from the validation set
print("Visualizing prediction on one validation sample...")
visualize_predictions(model, val_loader, device, num_samples=1)
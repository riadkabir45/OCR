import torch
from torch.utils.data import DataLoader
from data.dataset import LocalizationDataset
from utils.collate import custom_collate_fn
from models.unet import InstanceAwareUNet
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# Load model
save_dir = 'saved_models'
model_path = os.path.join(save_dir, 'instance_aware_unet.pth')
model = InstanceAwareUNet(in_channels=3, out_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load data
DATA_DIR = 'converted'
dataset = LocalizationDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

# Visualize predictions for a few samples
num_samples = 5
count = 0
for images, masks_list in dataloader:
    images = images.to(device)
    with torch.no_grad():
        semantic_pred, embedding_pred = model(images)
        img = images[0].cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        embedding = embedding_pred[0].cpu().numpy()  # shape: (embedding_dim, H, W)
        H, W = embedding.shape[1:]
        # Flatten spatial dims
        emb_flat = embedding.reshape(embedding.shape[0], -1).T  # (H*W, embedding_dim)
        # Only cluster foreground (semantic > 0.5)
        sem_mask = (semantic_pred[0, 0].cpu().numpy() > 0.5).flatten()
        emb_fg = emb_flat[sem_mask]
        coords = np.column_stack(np.where(semantic_pred[0, 0].cpu().numpy() > 0.5))
        if len(emb_fg) > 0:
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=20).fit(emb_fg)
            labels = clustering.labels_
            instance_mask = np.full((H*W), -1, dtype=np.int32)
            instance_mask[sem_mask] = labels
            instance_mask = instance_mask.reshape(H, W)
            unique_labels = [l for l in np.unique(labels) if l != -1][:5]
            plt.figure(figsize=(15, 3))
            plt.subplot(1, 6, 1)
            plt.imshow(img)
            plt.title('Image')
            plt.axis('off')
            for i, l in enumerate(unique_labels):
                mask = (instance_mask == l).astype(np.uint8)
                plt.subplot(1, 6, i+2)
                plt.imshow(mask, cmap='gray')
                plt.title(f'Pred Inst {i+1}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            count += 1
        else:
            print("No foreground found for this image. Skipping visualization.")
    if count >= num_samples:
        break

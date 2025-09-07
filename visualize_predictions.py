
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def visualize_predictions(model, dataloader, device, num_samples=1):
    model.eval()
    count = 0
    for images, masks_list in dataloader:
        images = images.to(device)
        with torch.no_grad():
            semantic_pred, embedding_pred = model(images)
            img = images[0].cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
            embedding = embedding_pred[0].cpu()
            semantic = semantic_pred[0, 0].cpu()
            H, W = embedding.shape[1:]
            emb_flat = embedding.reshape(embedding.shape[0], -1).T  # (H*W, emb_dim)
            sem_mask = (semantic > 0.5).flatten()
            emb_fg = emb_flat[sem_mask]
            # Limit number of points for DBSCAN
            max_points = 5000
            if len(emb_fg) > max_points:
                idx = np.random.choice(len(emb_fg), max_points, replace=False)
                emb_fg_sampled = emb_fg[idx]
                mask_idx = np.where(sem_mask)[0][idx]
            else:
                emb_fg_sampled = emb_fg
                mask_idx = np.where(sem_mask)[0]
            if len(emb_fg_sampled) > 0:
                clustering = DBSCAN(eps=0.5, min_samples=20).fit(emb_fg_sampled)
                labels = clustering.labels_
                instance_mask = np.full((H*W), -1, dtype=np.int32)
                instance_mask[mask_idx] = labels
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

# If run as a script, do a quick test visualization on a random sample
if __name__ == "__main__":
    from data.dataset import LocalizationDataset
    from utils.collate import custom_collate_fn
    from models.unet import InstanceAwareUNet
    import os
    from torch.utils.data import DataLoader
    save_dir = 'saved_models'
    model_path = os.path.join(save_dir, 'instance_aware_unet.pth')
    model = InstanceAwareUNet(in_channels=3, out_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    DATA_DIR = 'converted'
    dataset = LocalizationDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    visualize_predictions(model, dataloader, device, num_samples=1)

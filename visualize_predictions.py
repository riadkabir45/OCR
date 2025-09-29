
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def visualize_predictions(model, dataloader, device, num_samples=1, dbscan_eps=0.5, dbscan_min_samples=20, semantic_thresh=0.5):
    model.eval()
    count = 0
    import random
    for images, masks_list in dataloader:
        # Crop a random quadrant from the image
        img = images[0].cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        H, W = img.shape[:2]
        quadrant = random.choice([0, 1, 2, 3])
        if quadrant == 0:  # top-left
            y0, y1 = 0, H // 2
            x0, x1 = 0, W // 2
        elif quadrant == 1:  # top-right
            y0, y1 = 0, H // 2
            x0, x1 = W // 2, W
        elif quadrant == 2:  # bottom-left
            y0, y1 = H // 2, H
            x0, x1 = 0, W // 2
        else:  # bottom-right
            y0, y1 = H // 2, H
            x0, x1 = W // 2, W
        img_crop = img[y0:y1, x0:x1]
        image_tensor_crop = torch.from_numpy(img_crop).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            semantic_pred, embedding_pred = model(image_tensor_crop)
            embedding = embedding_pred[0].cpu()
            semantic = semantic_pred[0, 0].cpu()
            h, w = embedding.shape[1:]
            emb_flat = embedding.reshape(embedding.shape[0], -1).T  # (h*w, emb_dim)
            sem_mask = (semantic > semantic_thresh).flatten()
            print(f"Foreground pixels in semantic mask (thresh={semantic_thresh}): {sem_mask.sum()} / {sem_mask.shape[0]}")
            # Prepare semantic mask and instance mask for side-by-side display
            semantic_mask_img = (semantic > semantic_thresh).cpu().numpy()
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
                clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(emb_fg_sampled)
                labels = clustering.labels_
                instance_mask = np.full((h*w), -1, dtype=np.int32)
                instance_mask[mask_idx] = labels
                instance_mask = instance_mask.reshape(h, w)
                # Use bright, saturated colors for foreground, black for background
                bright_colors = [
                    (255, 255, 0),    # Yellow
                    (255, 128, 0),    # Orange
                    (255, 0, 255),    # Magenta
                    (0, 255, 255),    # Cyan
                    (255, 255, 255),  # White
                    (128, 255, 0),    # Chartreuse
                    (255, 0, 128),    # Pink
                ]
                all_labels = [l for l in np.unique(labels) if l != -1]
                color_instance_img = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)  # black background
                for i, l in enumerate(all_labels):
                    mask = (instance_mask == l)
                    rgb = bright_colors[i % len(bright_colors)]
                    for c in range(3):
                        color_instance_img[..., c][mask] = rgb[c]
                print(f"Number of instance segments found: {len(all_labels)} (DBSCAN eps={dbscan_eps}, min_samples={dbscan_min_samples})")
                # Show original image, semantic mask, instance mask, and embedding channels
                num_emb_channels = embedding.shape[0]
                plt.figure(figsize=(20, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(img_crop)
                plt.title('Original Image (Quadrant)')
                plt.axis('off')
                plt.subplot(1, 4, 2)
                plt.imshow(semantic_mask_img, cmap='gray')
                plt.title(f'Semantic Mask (>{semantic_thresh})')
                plt.axis('off')
                plt.subplot(1, 4, 3)
                plt.imshow(color_instance_img)
                plt.title('Instance Segments')
                plt.axis('off')
                # Plot the first embedding channel (or both if 2)
                if num_emb_channels == 1:
                    plt.subplot(1, 4, 4)
                    plt.imshow(embedding[0], cmap='viridis')
                    plt.title('Embedding Channel 0')
                    plt.axis('off')
                else:
                    plt.subplot(1, 4, 4)
                    plt.imshow(embedding[0], cmap='viridis')
                    plt.title('Embedding Channel 0')
                    plt.axis('off')
                    plt.figure(figsize=(5, 5))
                    plt.imshow(embedding[1], cmap='viridis')
                    plt.title('Embedding Channel 1')
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
    # Pick the latest saved model
    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError('No saved model found in saved_models directory.')
    # Sort by epoch number if present
    def extract_epoch(filename):
        import re
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1
    model_files.sort(key=extract_epoch)
    model_path = os.path.join(save_dir, model_files[-1])
    print(f'Using latest model: {model_path}')
    model = InstanceAwareUNet(in_channels=3, out_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    DATA_DIR = 'converted'
    dataset = LocalizationDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    visualize_predictions(model, dataloader, device, num_samples=1, dbscan_eps=0.2, dbscan_min_samples=5, semantic_thresh=0.3)

import torch
from torch.utils.data import DataLoader
from data.dataset import LocalizationDataset
from utils.collate import custom_collate_fn
import matplotlib.pyplot as plt
import numpy as np

# Load data
DATA_DIR = 'converted'
dataset = LocalizationDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

# Get one sample
images, masks_list = next(iter(dataloader))
img = images[0].cpu().numpy()
if img.shape[0] == 3:
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

masks = masks_list[0]

plt.figure(figsize=(15, 3))
plt.subplot(1, 6, 1)
plt.imshow(img)
plt.title('Image')
plt.axis('off')

for i in range(min(5, len(masks))):
    mask = masks[i].cpu().numpy()
    plt.subplot(1, 6, i+2)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

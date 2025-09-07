import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_comparison(image, pred_mask, gt_mask, idx=0):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.astype(np.uint8))
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title('Predicted Mask')
    axs[1].axis('off')
    axs[2].imshow(gt_mask, cmap='gray')
    axs[2].set_title('Ground Truth Mask')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

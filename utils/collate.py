import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    # Find max height and width in this batch
    heights = [img.shape[1] for img, _ in batch]
    widths = [img.shape[2] for img, _ in batch]
    max_h = max(heights)
    max_w = max(widths)

    padded_images = []
    padded_masks = []
    for img, mask_list in batch:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), 'constant', 0)
        padded_images.append(padded_img)
        padded_mask_list = []
        if len(mask_list) == 0:
            # Add a dummy mask if none exist
            padded_mask_list.append(torch.zeros((max_h, max_w), dtype=img.dtype))
        else:
            for mask in mask_list:
                pad_hm = max_h - mask.shape[0]
                pad_wm = max_w - mask.shape[1]
                padded_mask = F.pad(mask, (0, pad_wm, 0, pad_hm), 'constant', 0)
                padded_mask_list.append(padded_mask)
        padded_masks.append(padded_mask_list)
    images = torch.stack(padded_images, 0)
    return images, padded_masks

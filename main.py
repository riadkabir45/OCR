from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
import json
import torch
import os
from PIL import Image
import numpy as np

def custom_collate_fn(batch):
    # 'batch' is a list of tuples: [(image1, masks1), (image2, masks2), ...]
    images = [F.pad(item[0], (0, 8, 0, 1), 'constant',0) for item in batch]
    masks = [item[1] for item in batch] # This will be a list of lists of masks

    images = torch.stack(images, 0) # Stack images into a single tensor

    return images, masks

class LocalizationDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.image_files = [f for f in self.files if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.max_h = 0
        self.max_w = 0
        self.scaling_factor = 0.2
        self.max_boxes = 0

        self.cal_max_size()

    def __len__(self):
        return len(self.image_files)

    def cal_max_size(self):
        for img_file in self.image_files:
            img_path = os.path.join(self.img_dir, img_file)
            image_name = os.path.splitext(img_file)[0]
            json_path = os.path.join(self.img_dir, f"{image_name}.json")
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    self.max_h = max(self.max_h, h)
                    self.max_w = max(self.max_w, w)
            except Exception as e:
                print(f"Could not read image header for {img_file}: {e}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image_name = os.path.splitext(self.image_files[idx])[0]
        json_path = os.path.join(self.img_dir, f"{image_name}.json")

        image = read_image(img_path).float()/255

        h, w = image.shape[1], image.shape[2]
        padding = (0, self.max_w - w, 0 , self.max_h - h)
        image = F.pad(image, padding, mode='constant',value=1)

        image = F.interpolate(image.unsqueeze(0), scale_factor=self.scaling_factor, mode='bilinear', align_corners=False).squeeze(0)
        final_h, final_w = image.shape[1], image.shape[2]

        masks = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    points = shape.get('points', [])
                    if len(points) == 2:
                        mask = torch.zeros((self.max_h, self.max_w), dtype=torch.float32) # Create mask with max padded size
                        start_pos = np.array(points[0])
                        end_pos = np.array(points[1])
                        x1, y1 = start_pos
                        x2, y2 = end_pos
                        mask[int(y1):int(y2), int(x1):int(x2)] = 1

                        # Pad and scale the mask
                        mask = F.pad(mask.unsqueeze(0), padding, mode='constant', value=0).squeeze(0) # Add channel dim for padding
                        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(final_h, final_w), mode='nearest').squeeze(0).squeeze(0) # Add batch and channel dim for interpolate
                        masks.append(mask)

        return image, masks

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class InstanceAwareUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, embedding_channels=2):
        super(InstanceAwareUNet, self).__init__()

        # ... (all your existing encoder, decoder, and helper methods) ...
        # (These are unchanged)
        self.enc_conv1 = self.double_conv(in_channels, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(128, 64)

        # Output layers for both semantic mask and embeddings
        self.semantic_out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.embedding_out_conv = nn.Conv2d(64, embedding_channels, kernel_size=1)

    # double_conv and center_crop methods are unchanged
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, layer, target_size):
        _, _, layer_h, layer_w = layer.size()
        diff_h = (layer_h - target_size[0]) // 2
        diff_w = (layer_w - target_size[1]) // 2
        return layer[:, :, diff_h:(diff_h + target_size[0]), diff_w:(diff_w + target_size[1])]


    def forward(self, x):
        # ... (all the existing encoder and decoder logic) ...
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.maxpool(enc1))
        enc3 = self.enc_conv3(self.maxpool(enc2))
        enc4 = self.enc_conv4(self.maxpool(enc3))

        dec1 = self.upconv1(enc4)
        enc3_cropped = self.center_crop(enc3, (dec1.size(2), dec1.size(3)))
        dec1 = torch.cat([dec1, enc3_cropped], dim=1)
        dec1 = self.dec_conv1(dec1)

        dec2 = self.upconv2(dec1)
        enc2_cropped = self.center_crop(enc2, (dec2.size(2), dec2.size(3)))
        dec2 = torch.cat([dec2, enc2_cropped], dim=1)
        dec2 = self.dec_conv2(dec2)

        dec3 = self.upconv3(dec2)
        enc1_cropped = self.center_crop(enc1, (dec3.size(2), dec3.size(3)))
        dec3 = torch.cat([dec3, enc1_cropped], dim=1)
        dec3 = self.dec_conv3(dec3)

        # Predict both the semantic mask and the embeddings
        semantic_mask = torch.sigmoid(self.semantic_out_conv(dec3))
        embedding_map = self.embedding_out_conv(dec3)

        # Return both outputs for training and post-processing
        return semantic_mask, embedding_map

# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

# Instance Embedding loss

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_pull, delta_push):
        super(DiscriminativeLoss, self).__init__()
        self.delta_pull = delta_pull  # Margin for intra-instance loss
        self.delta_push = delta_push  # Margin for inter-instance loss

    def forward(self, embedding_maps, instance_masks):
        # embedding_maps: model's output, shape (N, C, H, W)
        # instance_masks: a list of lists of masks, where each inner list corresponds to an image
        #   e.g., [[mask1_img1, mask2_img1], [mask1_img2, mask2_img2, mask3_img2]]

        batch_size = embedding_maps.size(0)
        total_pull_loss = 0
        total_push_loss = 0
        total_regularize_loss = 0

        for i in range(batch_size):
            embeddings = embedding_maps[i]
            masks = instance_masks[i]

            num_instances = len(masks)
            if num_instances == 0:
                continue

            instance_means = []
            valid_mask_indices = []

            # Step 1: Calculate intra-instance loss and instance means
            pull_loss_per_image = 0
            for k in range(num_instances):
                mask = masks[k]
                mask_bool = mask.bool()

                # Use the boolean mask to index the embeddings
                instance_embeddings = embeddings[:, mask_bool]

                if instance_embeddings.numel() == 0:
                    continue

                mean_embedding = torch.mean(instance_embeddings, dim=1)
                instance_means.append(mean_embedding)
                valid_mask_indices.append(k)

                # Calculate L_pull for this instance
                pull_loss = torch.clamp(
                    torch.norm(instance_embeddings - mean_embedding.unsqueeze(1), dim=0) - self.delta_pull,
                    min=0
                )
                pull_loss_per_image += torch.sum(pull_loss ** 2) / instance_embeddings.shape[1]

            num_valid = len(instance_means)
            if num_valid == 0:
                continue
            total_pull_loss += pull_loss_per_image / num_valid

            # Step 2: Calculate inter-instance loss
            if num_valid > 1:
                push_loss_per_image = 0
                for k in range(num_valid):
                    for l in range(k + 1, num_valid):
                        mean_k = instance_means[k]
                        mean_l = instance_means[l]

                        # Calculate L_push between two instances
                        push_loss = torch.clamp(
                            2 * self.delta_push - torch.norm(mean_k - mean_l),
                            min=0
                        )
                        push_loss_per_image += push_loss ** 2

                total_push_loss += push_loss_per_image / (num_valid * (num_valid - 1) / 2)

            # Step 3: Regularization loss (optional but good practice)
            regularize_loss_per_image = 0
            for k in range(num_valid):
                mean_k = instance_means[k]
                regularize_loss_per_image += torch.norm(mean_k)

            total_regularize_loss += regularize_loss_per_image / num_valid

        return (total_pull_loss / batch_size) + (total_push_loss / batch_size) + (total_regularize_loss / batch_size)

# Define the loss function (using Dice Loss)
criterion = DiceLoss()
instance_criterion = DiscriminativeLoss(delta_pull=0.1, delta_push=0.5)

# Define the model
model = InstanceAwareUNet(in_channels=3, out_channels=1)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = LocalizationDataset('converted')

print(f"Using device: {device}")

def single_run(images,masks):
      # Determine original dimensions from the first image's first mask
      # Assuming masks list is not empty and the first image has at least one mask
      original_h, original_w = masks[0][0].shape

      optimizer.zero_grad()

      # Pass the images through your model
      semantic_pred, embedding_pred = model(images)


      # Crop model output to original mask size
      semantic_pred = semantic_pred[:, :, :original_h, :original_w]
      embedding_pred = embedding_pred[:, :, :original_h, :original_w]


      # Convert instance_masks (list of lists) to a single semantic mask for DiceLoss
      # Iterate through the batch
      semantic_ground_truth_list = []
      for image_masks in masks: # image_masks is a list of masks for a single image
          if image_masks: # Check if there are masks for this image
              # Stack instance masks for the current image and sum them
              stacked_masks = torch.stack(image_masks, dim=0)
              semantic_mask = torch.sum(stacked_masks, dim=0).clamp(0, 1)
              semantic_ground_truth_list.append(semantic_mask)
          else:
              # If no masks, create an empty semantic mask of the correct size
              semantic_ground_truth_list.append(torch.zeros((original_h, original_w), dtype=torch.float32, device=images.device))

      semantic_ground_truth = torch.stack(semantic_ground_truth_list, dim=0)


      # Calculate losses
      semantic_loss = criterion(semantic_pred, semantic_ground_truth)
      # The instance_criterion expects the original list of lists of masks
      discriminative_loss = instance_criterion(embedding_pred, masks)

      # Total loss is a weighted sum
      total_loss = semantic_loss + discriminative_loss

      # Backpropagate and update weights
      total_loss.backward()
      optimizer.step()

      return total_loss, semantic_loss, discriminative_loss


def test_run():
    # No custom collate_fn, this will fail if the data is not batchable
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=2,collate_fn=custom_collate_fn)

    # Try to retrieve a batch
    for images, masks in data_loader:

        loss, semantic_loss, discriminative_loss = single_run(images, masks)

        print(f"Total Loss: {loss.item():.4f}")
        print(f"Semantic Loss: {semantic_loss.item():.4f}")
        print(f"Discriminative Loss: {discriminative_loss.item():.4f}")
        break


from tqdm import tqdm
from torch.utils.data import DataLoader

# Create a DataLoader for the dataset
# Using a batch size of 1 for simplicity, adjust as needed
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn)

num_epochs = 1 # Assuming 10 epochs based on previous turn

for epoch in range(num_epochs):
    # Wrap the dataloader with tqdm for a progress bar
    tqdm_loader = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, masks_list in tqdm_loader:
        images = images.to(device)
        # masks_list is already a list of lists of masks from the collate_fn
        masks = [ [mask.to(device) for mask in image_masks] for image_masks in masks_list]


        loss, semantic_loss, discriminative_loss = single_run(images, masks)

        tqdm_loader.set_postfix(loss=loss.item())

print("Training finished.")
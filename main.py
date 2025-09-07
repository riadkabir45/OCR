import torch
from torch.utils.data import DataLoader
from data.dataset import LocalizationDataset
from models.unet import InstanceAwareUNet
from losses.losses import DiceLoss, DiscriminativeLoss
from utils.collate import custom_collate_fn
from trainers.train import train_one_epoch
from utils.visualize import visualize_comparison
import torch.optim as optim
import os

# Dataset and DataLoader
DATA_DIR = 'converted'
dataset = LocalizationDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

# Model, Loss, Optimizer
model = InstanceAwareUNet(in_channels=3, out_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = DiceLoss()
instance_criterion = DiscriminativeLoss(delta_pull=0.1, delta_push=0.5)

# Training Loop
num_epochs = 1
for epoch in range(num_epochs):

    train_one_epoch(model, dataloader, optimizer, criterion, instance_criterion, device)

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
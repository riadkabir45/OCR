import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceAwareUNet(nn.Module):
    def center_crop(self, enc_feat, target):
        # enc_feat: (N, C, H, W), target: (N, C, H2, W2)
        _, _, h, w = enc_feat.shape
        _, _, th, tw = target.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return enc_feat[:, :, dh:dh+th, dw:dw+tw]
    def __init__(self, in_channels=3, out_channels=1, embedding_channels=2):
        super().__init__()
        # Minimal UNet-like structure
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        # Two output heads: semantic and embedding
        self.final_semantic = nn.Conv2d(16, out_channels, 1)
        self.final_embedding = nn.Conv2d(16, embedding_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        # Decoder
        u2 = self.up2(b)
        e2_crop = self.center_crop(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_crop], dim=1))
        u1 = self.up1(d2)
        e1_crop = self.center_crop(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1_crop], dim=1))
        # Output heads
        semantic = self.final_semantic(d1)
        embedding = self.final_embedding(d1)
        return semantic, embedding

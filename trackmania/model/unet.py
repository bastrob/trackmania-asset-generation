import torch
import torch.nn as nn

from ..embedding import TimeEmbedding
from .block import ResidualBlock


class MiniUnet(nn.Module):
    def __init__(self, num_channels=64, time_dim=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, num_channels),
            nn.ReLU()
        )

        # encoder
        self.enc1 = ResidualBlock(3, num_channels, time_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(num_channels, num_channels * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ResidualBlock(num_channels * 2, num_channels * 4, time_dim)

        # decoder
        self.up1 = nn.ConvTranspose2d(num_channels * 4, num_channels * 2, 2, stride=2)
        self.dec1 = ResidualBlock(num_channels * 4, num_channels * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(num_channels * 2, num_channels, 2, stride=2)
        self.dec2 = ResidualBlock(num_channels * 2, num_channels, time_dim)

        self.final = nn.Conv2d(num_channels, 3, 1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # encoder
        x1 = self.enc1(x, t_emb)

        x2 = self.enc2(self.pool1(x1), t_emb)

        # bottleneck
        x3 = self.bottleneck(self.pool2(x2), t_emb)

        # decoder
        x = self.up1(x3)

        # skip connection
        x = torch.cat([x, x2], dim=1)
        
        x = self.dec1(x, t_emb)

        x = self.up2(x)

        # skip connection
        x = torch.cat([x, x1], dim=1)
        
        x = self.dec2(x, t_emb)

        return self.final(x)
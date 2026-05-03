import torch
import torch.nn as nn

from ..embedding import TimeEmbedding


class DiffusionModel(nn.Module):
    def __init__(self, num_channels=64, time_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, num_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, 3, 3, padding=1)
    
    def forward(self, x, t):
        # time embedding projection
        t_emb_proj = self.time_mlp(t) # (B, time_dim)

        # prepare to intect into feature map
        t_emb_proj = t_emb_proj[:, :, None, None]

        # broadcast and add
        x = self.conv1(x) # (B, num_channels, _, _)
        x = x + t_emb_proj
        x = torch.relu(x)
        x = self.conv2(x)
        return x

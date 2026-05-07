import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()

        # Project time embedding to feature channels
        self.time_mlp = nn.Linear(time_dim, out_channels)
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)

        # inject timestep embedding
        time_emb = self.time_mlp(time_emb)

        # (B, C) → (B, C, 1, 1)
        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        h = self.activation(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h
import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        t: (B,)
        returns: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1))
        )

        emb = t[:, None] * emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb
        
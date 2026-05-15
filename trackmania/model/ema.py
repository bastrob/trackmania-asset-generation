import copy

import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def to(self, device):
        self.model.to(device)
        return self

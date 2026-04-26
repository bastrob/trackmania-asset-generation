import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def normalize_image(image):
    return np.array(image).astype(np.float32) / 255.0

def to_channel_first(image):
    return np.transpose(image, (2, 0, 1))

def normalize_diffusion(image):
    # assume image is normalized as [0,1] values and even a np array.
    return image * 2.0 - 1.0

def to_tensor(image):
    return torch.from_numpy(image)
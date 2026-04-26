from PIL import Image
from torch.utils.data import Dataset


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, image_size=(512, 512)):
        """
        Simple image dataset for preprocessing Trackmania assets.

        Args:
            image_paths (List[Path]): list of image file paths
            transform
            image_size (tuple): target resize resolution (H, W)
        """
        self.image_paths = image_paths
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize(self.image_size)
        
        if self.transform:
            img = self.transform(img)


        return img, str(path)
        
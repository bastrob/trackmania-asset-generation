from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from ..dataset import Compose, SimpleImageDataset
from ..dataset.transform import (
    normalize_diffusion,
    normalize_image,
    to_channel_first,
    to_tensor,
)

DATA_PATTERNS = ["*.jpg", "*.png", "*.jpeg"]

class BaseTask:
    def __init__(self, config=None):
        self.config = config or {}
        self.image_paths = None
        self.train_data_dir = self.config.get("train_data_dir", "data/base")
        self.batch_size = self.config.get("batch_size", 4)

        self.dataset: SimpleImageDataset
        self.dataloader: DataLoader

        self.transform = Compose([
            normalize_image,
            normalize_diffusion,
            to_channel_first,
            to_tensor
        ])
    
    def load_data(self):
        """
        Load image paths from train directory.
        """
        if self.train_data_dir is None:
            raise ValueError("Data directory is undefined")

        logger.info(f"[BaseTask] Loading data from: {self.train_data_dir}")

        if not Path(self.train_data_dir).exists():
            raise FileNotFoundError(f"Dataset path not found: {self.train_data_dir}")
        
        base_path = Path(self.train_data_dir)
        self.image_paths = [
            p for pattern in DATA_PATTERNS for p in base_path.glob(pattern)
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.train_data_dir}")

        logger.info(f"[BaseTask] Found {len(self.image_paths)} images")
        logger.debug(f"Sample: {self.image_paths[0]}")

    def load_model(self):
        """
        Load model from configuration.
        """
        logger.info("[BaseTask] Loading model (placeholder)")
    
    def build_dataset(self):
        """
        Build a dataset from image paths.
        """
        logger.info("[BaseTask] Building dataset...")

        if not self.image_paths:
            raise ValueError("Cannot build dataset: no images loaded")
        
        self.dataset = SimpleImageDataset(image_paths=self.image_paths, transform=self.transform)
        
        #logger.debug(f"Sample: {self.dataset[0]['image'].shape}")

    def build_dataloader(self):
        """
        Build a dataloader from a dataset
        """
        logger.info("[BaseTask] Building dataloader...")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True

        )

    def train(self):
        """
        Train a model.
        """
        logger.info("[BaseTask] Training model (placeholder)")

    def run(self):
        logger.info("[BaseTask] running...")
        self.load_data()
        self.build_dataset()
        self.build_dataloader()
        self.load_model()
        self.train()
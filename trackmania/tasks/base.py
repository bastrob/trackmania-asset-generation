from pathlib import Path

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..dataset import Compose, SimpleImageDataset
from ..dataset.transform import (
    normalize_diffusion,
    normalize_image,
    to_channel_first,
    to_tensor,
)
from ..diffusion.utils import reconstruct
from ..model import DiffusionModel
from ..viz.image import show_triplet

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

        self.device: torch.device
        self.model: nn.Module
        self.optimizer: Optimizer
        self.criterion: nn.Module

    
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
        logger.info("[BaseTask] Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DiffusionModel().to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-4))
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_start=0.001, beta_end=0.02
        )

        self.criterion = nn.MSELoss()
    
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
        logger.info("[BaseTask] Training model...")

        self.model.train()
        epochs = self.config.get("epochs", 5)
        for epoch in range(epochs):
            total_loss = 0

            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                
                # Sample noise to add to the image
                noise = torch.randn_like(images)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (images.shape[0],),
                    device=self.device,
                ).long()

                # Add noise to the base images according
                # to the noise magnitude at each timestep
                noisy_images = self.scheduler.add_noise(images, noise, timesteps)

                # Get the model prediction for the noise
                noise_pred = self.model(noisy_images, timesteps)

                # fake target (autoencoder-style training)
                loss = self.criterion(noise_pred, noise)

                # backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                # viz part
                recon = reconstruct(noisy_images, noise_pred)
                show_triplet(
                    images[0],
                    noisy_images[0],
                    recon[0]
                )
            
            logger.info(f"Epoch {epoch}: loss = {total_loss:.4f}")

                


    def run(self):
        logger.info("[BaseTask] running...")
        self.load_data()
        self.build_dataset()
        self.build_dataloader()
        self.load_model()
        self.train()
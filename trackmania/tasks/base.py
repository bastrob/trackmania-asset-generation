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
from ..diffusion.sampling import sample
from ..diffusion.utils import reconstruct
from ..model import EMA, MiniUnet
from ..viz.image import show_triplet, tensor_to_pil

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
        self.ema : EMA
        self.optimizer: Optimizer
        self.criterion: nn.Module

        self.output_dir = Path(
            self.config.get("output_dir", "outputs")
        )

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
    
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

        max_samples = self.config.get("max_train_samples")

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        logger.info(f"[BaseTask] Found {len(self.image_paths)} images")
        logger.debug(f"Sample: {self.image_paths[0]}")

    def load_model(self):
        """
        Load model from configuration.
        """
        logger.info("[BaseTask] Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MiniUnet().to(self.device)
        self.ema = EMA(self.model, decay=0.95).to(self.device)
        
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
        
        self.dataset = SimpleImageDataset(image_paths=self.image_paths, transform=self.transform, image_size=(256, 256))
        
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
        save_every = self.config.get("save_every_epoch", 5)
        sample_every = self.config.get("sample_every_epoch", 5)
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

                # ema update
                self.ema.update(self.model)

                self.optimizer.zero_grad()

                total_loss += loss.item()
                
                # viz part
                if (epoch + 1) % 50 == 0:
                    recon = reconstruct(noisy_images, noise_pred)
                    show_triplet(
                        images[0],
                        noisy_images[0],
                        recon[0]
                    )
                
            logger.info(f"Epoch {epoch}: loss = {total_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
            
            if (epoch + 1) % sample_every == 0:
                self.generate_samples(epoch + 1)

    def generate(self):
        generated = sample(
            model=self.ema.model,
            scheduler=self.scheduler,
            device=self.device,
            image_size=(3, 256, 256),
            num_inference_steps=1000
        )

        image = tensor_to_pil(generated)

        return image
    
    def generate_samples(self, epoch, num_samples=4):
        logger.info("[Sampling] Generating samples...")
        epoch_dir = self.sample_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(num_samples):
            image = self.generate()

            output_path = epoch_dir / f"sample_{idx}.png"
            image.save(output_path)

            logger.info(f"[BaseTask] Generated sample saved to {output_path}")


    def save_checkpoint(self, epoch):
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

        logger.info(f"[BaseTask] Checkpoint saved: {checkpoint_path}")

    def run(self):
        logger.info("[BaseTask] running...")
        self.load_data()
        self.build_dataset()
        self.build_dataloader()
        self.load_model()
        self.train()
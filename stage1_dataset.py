"""
Stage 1 — Dataset Loader
Loads clean (shadow-free) TRI X-ray images for diffusion model pretraining.
Supports grayscale PCB X-ray images. Handles augmentation and normalization.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch


class CleanXRayDataset(Dataset):
    """
    Loads ONLY clean (shadow-free) X-ray images.
    Used in Stage 1 to train the diffusion prior.

    Expected folder structure:
        data/
          clean/          ← shadow-free TRI X-rays go here
            img001.png
            img002.png
            ...
          shadowed/       ← used in Stage 2 inference only
            img001.png
            ...
    """

    def __init__(self, clean_dir, image_size=256, augment=True):
        """
        Args:
            clean_dir   : path (str) or list of paths to clean X-ray image folders
            image_size  : resize all images to (image_size x image_size)
            augment     : apply random flips for data augmentation
        """
        self.image_size = image_size
        dirs = [clean_dir] if isinstance(clean_dir, str) else list(clean_dir)
        self.paths = []
        for d in dirs:
            self.paths.extend(sorted([
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.tiff', '.bmp'))
            ]))

        if len(self.paths) == 0:
            raise ValueError(f"No images found in {dirs}")

        print(f"[Dataset] Found {len(self.paths)} clean X-ray images.")

        # Build transform pipeline
        # X-rays are grayscale → single channel
        # Normalize to [-1, 1] which is what diffusion models expect
        transforms = [
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),               # [0, 1]
            T.Normalize([0.5], [0.5])   # → [-1, 1]
        ]

        if augment:
            transforms.insert(2, T.RandomHorizontalFlip())
            transforms.insert(3, T.RandomVerticalFlip())

        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')   # grayscale
        return self.transform(img)


class ShadowedXRayDataset(Dataset):
    """
    Loads shadowed X-ray images for Stage 2 SDEdit inference.
    No augmentation — we want exact images for shadow removal.
    """

    def __init__(self, shadowed_dir, image_size=256):
        self.image_size = image_size
        dirs = [shadowed_dir] if isinstance(shadowed_dir, str) else list(shadowed_dir)
        self.paths = []
        for d in dirs:
            self.paths.extend(sorted([
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.tiff', '.bmp'))
            ]))

        if len(self.paths) == 0:
            raise ValueError(f"No images found in {dirs}")

        print(f"[Dataset] Found {len(self.paths)} shadowed X-ray images.")

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self.filenames = [os.path.basename(p) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img), self.filenames[idx]


def get_clean_dataloader(clean_dir, image_size=256, batch_size=8,
                         num_workers=0, augment=True):
    """Returns DataLoader for training the diffusion prior."""
    dataset = CleanXRayDataset(clean_dir, image_size, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def get_shadowed_dataloader(shadowed_dir, image_size=256,
                            batch_size=1, num_workers=0):
    """Returns DataLoader for SDEdit inference."""
    dataset = ShadowedXRayDataset(shadowed_dir, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage1_dataset.py <path_to_clean_dir>")
        sys.exit(0)

    loader = get_clean_dataloader(sys.argv[1], image_size=256, batch_size=4)
    batch = next(iter(loader))
    print(f"Batch shape : {batch.shape}")          # [4, 1, 256, 256]
    print(f"Value range : [{batch.min():.2f}, {batch.max():.2f}]")  # [-1, 1]
    print("Dataset loader OK.")

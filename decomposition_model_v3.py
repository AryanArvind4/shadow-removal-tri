"""
=============================================================================
INTRINSIC DECOMPOSITION PIPELINE V3 — X-RAY SHADOW REMOVAL
=============================================================================

V3 Changes from V2:
    1. L variance loss (anti-collapse): target_std=0.15, weight=3.0
    2. Stronger synthetic shadows: strength (0.25, 0.75), stronger gradients/seams
    3. L_var logged during training
    
    Use with global L inference (test_global_L.py) for best results.

Core Idea:
    Learn to decompose any X-ray image I into:
        R = Reflectance (structure, shadow-free detail)
        L = Illumination (smooth shadow/shading field)
    Such that: I ≈ R * L  (in linear space)
    Or equivalently: log(I) = log(R) + log(L)

    Then the shadow-free output is simply R (rescaled).

Training Strategy (Semi-supervised):
    1. UNPAIRED shadowed TRI images (10-30+):
       - Self-supervised reconstruction: R * L ≈ I
       - Smoothness prior on L (total variation)
       - Structure prior on R (edge-aware smoothness)
       - Illumination range prior (L should be near 1.0 on average)

    2. SYNTHETIC shadow pairs (from NTU solder ball data):
       - We have clean NTU images, we add synthetic shadows
       - Supervised loss: R ≈ clean, L ≈ synthetic shadow field
       - This anchors the decomposition and prevents trivial solutions

    3. ISTD grayscale pairs (optional, for general shadow priors):
       - Supervised loss on natural image shadow removal
       - Helps with shadow edge handling

Why This Works Without Clean TRI Data:
    - The model never needs to see a "clean TRI image"
    - It learns what shadows look like (smooth, low-frequency) from priors
    - It learns what structure looks like from the images themselves
    - The synthetic pairs prevent trivial solutions (R=I, L=1)
    - Different component geometries are handled automatically because
      the priors are geometry-agnostic

=============================================================================
"""

import os
import cv2
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from math import log10
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda")  # Force CUDA GPU usage
print(f"Using device: {device}")


###############################################################################
#                         SECTION 1: MODEL ARCHITECTURE                       #
###############################################################################

class ConvBlock(nn.Module):
    """Standard double-conv block with ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """
    Shared encoder for both R and L decoders.
    Identical to original U-Net encoder — can load Phase 2 weights.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.d1 = ConvBlock(in_channels, 32);  self.p1 = nn.MaxPool2d(2)
        self.d2 = ConvBlock(32, 64);           self.p2 = nn.MaxPool2d(2)
        self.d3 = ConvBlock(64, 128);          self.p3 = nn.MaxPool2d(2)
        self.d4 = ConvBlock(128, 256);         self.p4 = nn.MaxPool2d(2)
        self.bn = ConvBlock(256, 512)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        bn = self.bn(self.p4(d4))
        return bn, [d1, d2, d3, d4]  # bottleneck + skip connections


class ReflectanceDecoder(nn.Module):
    """
    Full-resolution decoder for reflectance R.
    Standard U-Net decoder with skip connections — preserves all structural detail.
    Output: R in [0, 1] via sigmoid.
    """
    def __init__(self, out_channels=1):
        super().__init__()
        self.u4 = nn.ConvTranspose2d(512, 256, 2, 2); self.c4 = ConvBlock(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2); self.c3 = ConvBlock(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2);  self.c2 = ConvBlock(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2);   self.c1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, bn, skips):
        d1, d2, d3, d4 = skips
        x = self.c4(torch.cat([self.u4(bn), d4], 1))
        x = self.c3(torch.cat([self.u3(x), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return torch.sigmoid(self.out(x))  # R in [0, 1]


class IlluminationDecoder(nn.Module):
    """
    Illumination decoder for L — V2 with more capacity.

    V1 was too constrained (L collapsed to ~0.978 constant).
    V2 changes:
    - Uses d4 skip connection (1/8 res) for more spatial awareness
    - More conv layers to learn shadow field patterns
    - Explicit bias initialization to start near L=1.0
    - Output range [0.3, 1.8] via scaled sigmoid

    Still smooth by design: operates at 1/4 res max, final bilinear upsample.
    But now has enough capacity to actually represent shadow gradients.
    """
    def __init__(self, out_channels=1):
        super().__init__()
        # Reduce bottleneck channels
        self.reduce_bn = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
        )
        # Reduce d4 skip connection channels
        self.reduce_d4 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
        )
        # Process combined features at 1/8 resolution
        self.block1 = ConvBlock(256, 128)  # 128 from bn + 128 from d4
        self.block2 = ConvBlock(128, 64)
        self.block3 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

        # Initialize output bias so sigmoid starts near 0.5 → L starts near 1.05
        # This prevents the initial L from being too far from 1.0
        nn.init.constant_(self.out.bias, 0.2)

    def forward(self, bn, skips, target_size):
        """
        Args:
            bn: bottleneck features [B, 512, H/16, W/16]
            skips: [d1, d2, d3, d4] — we use d4 [B, 256, H/8, W/8]
            target_size: (H, W) of the full-resolution input
        """
        d4 = skips[3]  # [B, 256, H/8, W/8]

        # Upsample bottleneck to match d4
        bn_up = F.interpolate(bn, size=d4.shape[-2:], mode='bilinear', align_corners=False)
        bn_reduced = self.reduce_bn(bn_up)   # [B, 128, H/8, W/8]
        d4_reduced = self.reduce_d4(d4)       # [B, 128, H/8, W/8]

        # Combine and process
        x = torch.cat([bn_reduced, d4_reduced], dim=1)  # [B, 256, H/8, W/8]
        x = self.block1(x)                               # [B, 128, H/8, W/8]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # H/4
        x = self.block2(x)                               # [B, 64, H/4, W/4]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # H/2
        x = self.block3(x)                               # [B, 32, H/2, W/2]
        x = self.out(x)                                  # [B, 1, H/2, W/2]

        # Upsample to full resolution (bilinear ensures smoothness)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # Scale sigmoid to [0.3, 1.8] range
        # L=1.0 = no shadow, L<1 = shadow, L>1 = brighter region
        return 0.3 + 1.5 * torch.sigmoid(x)


class IntrinsicDecompositionNet(nn.Module):
    """
    Full decomposition network.

    Input:  I (shadowed image, [B, 1, H, W], values in [0, 1])
    Output: R (reflectance, [B, 1, H, W], values in [0, 1])
            L (illumination, [B, 1, H, W], values in [0.2, 2.0])

    Relationship: I ≈ R * L

    The shadow-free output is R (rescaled to full dynamic range).
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.reflectance_decoder = ReflectanceDecoder(in_channels)
        self.illumination_decoder = IlluminationDecoder(in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]
        bn, skips = self.encoder(x)
        R = self.reflectance_decoder(bn, skips)
        L = self.illumination_decoder(bn, skips, target_size=(h, w))
        return R, L

    def load_phase2_encoder(self, checkpoint_path):
        """
        Load Phase 2 (ISTD grayscale) weights into the shared encoder.
        The reflectance decoder also gets initialized from Phase 2's decoder
        since it has the same architecture.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Map Phase 2 U-Net keys to our encoder keys
        encoder_mapping = {}
        for key in ckpt:
            # Encoder layers: d1, p1, d2, p2, d3, p3, d4, p4, bn
            for layer in ['d1', 'd2', 'd3', 'd4', 'bn', 'p1', 'p2', 'p3', 'p4']:
                if key.startswith(f'{layer}.'):
                    encoder_mapping[key] = ckpt[key]

        # Load encoder weights
        enc_state = self.encoder.state_dict()
        loaded = 0
        for key, val in encoder_mapping.items():
            if key in enc_state and enc_state[key].shape == val.shape:
                enc_state[key] = val
                loaded += 1
        self.encoder.load_state_dict(enc_state)
        print(f"  Encoder: loaded {loaded}/{len(enc_state)} parameters from Phase 2")

        # Map decoder weights to reflectance decoder
        decoder_mapping = {}
        for key in ckpt:
            for layer in ['u4', 'c4', 'u3', 'c3', 'u2', 'c2', 'u1', 'c1', 'out']:
                if key.startswith(f'{layer}.'):
                    decoder_mapping[key] = ckpt[key]

        ref_state = self.reflectance_decoder.state_dict()
        loaded = 0
        for key, val in decoder_mapping.items():
            if key in ref_state and ref_state[key].shape == val.shape:
                ref_state[key] = val
                loaded += 1
        self.reflectance_decoder.load_state_dict(ref_state)
        print(f"  Reflectance decoder: loaded {loaded}/{len(ref_state)} parameters from Phase 2")
        print(f"  Illumination decoder: initialized randomly (intentional)")


###############################################################################
#                         SECTION 2: LOSS FUNCTIONS                           #
###############################################################################

class DecompositionLoss(nn.Module):
    """
    Combined loss for intrinsic decomposition training.

    For ALL images (paired + unpaired):
        L_recon:    |R * L - I|         Reconstruction fidelity
        L_smooth:   TV(L)               Illumination must be smooth
        L_struct:   EdgeAwareTV(R, I)   Reflectance: smooth except at edges
        L_illum_reg: |mean(L) - 1|      Illumination should average to ~1.0

    For PAIRED images only (synthetic shadow pairs):
        L_R_sup:    |R - clean|         Reflectance should match clean GT
        L_L_sup:    |L - shadow_field|  Illumination should match shadow GT

    Weight balancing is critical — see comments in __init__.
    """
    def __init__(
        self,
        w_recon=10.0,       # Reconstruction: must be high (primary constraint)
        w_smooth=0.3,       # L smoothness: REDUCED from 2.0 (was killing L variation)
        w_struct=0.3,       # R edge-aware smoothness: low (subtle regularization)
        w_illum_reg=0.5,    # L mean regularization: REDUCED from 1.0
        w_R_supervised=5.0, # Supervised R loss: strong anchor on synthetic pairs
        w_L_supervised=5.0, # Supervised L loss: INCREASED from 2.0 (critical for non-trivial L)
        w_L_variance=3.0,   # V3: increased from 2.0 — stronger anti-collapse pressure
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_smooth = w_smooth
        self.w_struct = w_struct
        self.w_illum_reg = w_illum_reg
        self.w_R_supervised = w_R_supervised
        self.w_L_supervised = w_L_supervised
        self.w_L_variance = w_L_variance

    def total_variation(self, x):
        """Isotropic total variation: sum of absolute spatial gradients."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def edge_aware_tv(self, x, guide):
        """
        Total variation on x, weighted by edge strength in guide image.
        Where guide has strong edges, allow x to also have edges.
        Where guide is smooth, force x to be smooth too.
        """
        # Gradient magnitude of guide
        guide_grad_h = torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])
        guide_grad_w = torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])

        # Edge weight: high gradient in guide → low penalty on x
        weight_h = torch.exp(-10.0 * guide_grad_h)
        weight_w = torch.exp(-10.0 * guide_grad_w)

        # Weighted TV of x
        x_grad_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        x_grad_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        return (weight_h * x_grad_h).mean() + (weight_w * x_grad_w).mean()

    def forward(self, I, R, L, clean_gt=None, shadow_field_gt=None):
        """
        Args:
            I: input shadowed image [B, 1, H, W] in [0, 1]
            R: predicted reflectance [B, 1, H, W] in [0, 1]
            L: predicted illumination [B, 1, H, W] in [0.2, 2.0]
            clean_gt: (optional) clean ground truth for supervised pairs
            shadow_field_gt: (optional) illumination GT for supervised pairs

        Returns:
            total_loss, loss_dict (for logging)
        """
        losses = {}

        # --- Self-supervised losses (applied to ALL images) ---

        # 1. Reconstruction: R * L should reproduce the input
        I_recon = R * L
        losses['recon'] = self.w_recon * F.l1_loss(I_recon, I)

        # 2. Illumination smoothness (total variation on L)
        # This is THE key constraint that makes L capture only shadows
        losses['L_smooth'] = self.w_smooth * self.total_variation(L)

        # 3. Reflectance structure prior (edge-aware TV)
        # R should be piecewise smooth, with edges only where input has edges
        losses['R_struct'] = self.w_struct * self.edge_aware_tv(R, I)

        # 4. Illumination regularization: mean(L) should be near 1.0
        # Prevents the model from making L uniformly too bright/dark
        losses['L_reg'] = self.w_illum_reg * torch.abs(L.mean() - 1.0)

        # 5. NEW: L variance loss — penalize L for being too flat
        # This is the KEY fix for the L-collapse problem.
        # If L has no spatial variation, it's not capturing shadows.
        # We want L to have meaningful variation (std > 0.15 for real TRI shadows).
        L_std = torch.std(L)
        # Penalize when std is too low: max(0, target_std - actual_std)
        target_std = 0.15  # V3: increased from 0.08 — TRI shadows are strong
        losses['L_var'] = self.w_L_variance * F.relu(target_std - L_std)

        # --- Supervised losses (only for paired synthetic data) ---

        if clean_gt is not None:
            losses['R_sup'] = self.w_R_supervised * F.l1_loss(R, clean_gt)

        if shadow_field_gt is not None:
            losses['L_sup'] = self.w_L_supervised * F.l1_loss(L, shadow_field_gt)

        total = sum(losses.values())
        return total, {k: v.item() for k, v in losses.items()}


###############################################################################
#                     SECTION 3: SYNTHETIC SHADOW GENERATION                  #
###############################################################################

class SyntheticShadowGenerator:
    """
    Generates realistic X-ray shadow fields and applies them to clean images.

    Returns BOTH the shadowed image AND the shadow field itself,
    so we can supervise both R and L branches.
    """
    def __init__(self, strength_range=(0.25, 0.75)):
        """V3: increased from (0.15, 0.55) to match real TRI shadow intensity."""
        self.strength_min, self.strength_max = strength_range

    def _smooth_random_field(self, h, w, n_basis=None):
        if n_basis is None:
            n_basis = np.random.randint(2, 7)
        field = np.zeros((h, w), dtype=np.float32)
        for _ in range(n_basis):
            cx = np.random.uniform(-w * 0.3, w * 1.3)
            cy = np.random.uniform(-h * 0.3, h * 1.3)
            sx = np.random.uniform(w * 0.2, w * 1.5)
            sy = np.random.uniform(h * 0.2, h * 1.5)
            amp = np.random.uniform(-1.0, 1.0)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            field += amp * np.exp(-((xx-cx)**2/(2*sx**2) + (yy-cy)**2/(2*sy**2)))
        if np.abs(field).max() > 1e-6:
            field /= np.abs(field).max() + 1e-6
        return field

    def _linear_gradient(self, h, w):
        angle = np.random.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5)

    def _seam_artifact(self, h, w):
        seam = np.zeros((h, w), dtype=np.float32)
        n = np.random.randint(1, 4)
        for _ in range(n):
            vertical = np.random.random() > 0.5
            if vertical:
                pos = np.random.randint(w // 6, 5 * w // 6)
                bw = np.random.randint(3, max(4, w // 15))
                intensity = np.random.uniform(-0.5, 0.5)
                for col in range(max(0, pos - bw), min(w, pos + bw)):
                    falloff = max(0, 1.0 - abs(col - pos) / (bw + 1e-6))
                    seam[:, col] += intensity * falloff
            else:
                pos = np.random.randint(h // 6, 5 * h // 6)
                bw = np.random.randint(3, max(4, h // 15))
                intensity = np.random.uniform(-0.5, 0.5)
                for row in range(max(0, pos - bw), min(h, pos + bw)):
                    falloff = max(0, 1.0 - abs(row - pos) / (bw + 1e-6))
                    seam[row, :] += intensity * falloff
        return seam

    def generate(self, clean_img):
        """
        Args:
            clean_img: np.float32 [H, W] in [0, 1]
        Returns:
            shadowed: np.float32 [H, W] in [0, 1]
            shadow_field: np.float32 [H, W] — the L field (≈1.0 where clean, <1 in shadow)
        """
        h, w = clean_img.shape
        strength = np.random.uniform(self.strength_min, self.strength_max)

        field = np.zeros((h, w), dtype=np.float32)
        field += self._smooth_random_field(h, w) * np.random.uniform(0.5, 1.0)
        # V3: increased gradient probability and strength
        if np.random.random() < 0.85:
            field += self._linear_gradient(h, w) * np.random.uniform(0.5, 1.2)
        # V3: increased seam probability and strength
        if np.random.random() < 0.6:
            field += self._seam_artifact(h, w) * np.random.uniform(0.5, 1.0)

        blur_sigma = max(h, w) * np.random.uniform(0.05, 0.15)
        field = gaussian_filter(field, sigma=blur_sigma)

        if np.abs(field).max() > 1e-6:
            field /= np.abs(field).max() + 1e-6

        # Create illumination field L: centered at 1.0
        # V3: shadow regions go darker (down to 0.4), bright regions up to 1.3
        L_field = 1.0 + field * strength * np.where(field < 0, -1.5, 0.5)
        L_field = np.clip(L_field, 0.3, 1.8)

        # Apply: shadowed = clean * L_field
        shadowed = np.clip(clean_img * L_field, 0.0, 1.0).astype(np.float32)

        return shadowed, L_field.astype(np.float32)


###############################################################################
#                         SECTION 4: DATASETS                                 #
###############################################################################

class UnpairedTRIDataset(Dataset):
    """
    Unpaired shadowed TRI images — used with self-supervised losses only.
    Extracts random patches from full TRI images.
    """
    def __init__(self, image_dir, patch_size=256, patches_per_image=30, augment=True):
        self.patch_size = patch_size
        self.augment = augment
        self.patches = []

        image_files = sorted(glob.glob(os.path.join(image_dir, "*")))
        for path in image_files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            h, w = img.shape

            if h < patch_size or w < patch_size:
                # Resize up if too small
                scale = max(patch_size / h, patch_size / w) + 0.1
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape

            for _ in range(patches_per_image):
                y = np.random.randint(0, h - patch_size + 1)
                x = np.random.randint(0, w - patch_size + 1)
                self.patches.append(img[y:y+patch_size, x:x+patch_size])

        print(f"  UnpairedTRI: {len(self.patches)} patches from {len(image_files)} images")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].copy()

        if self.augment:
            if np.random.random() < 0.5: patch = np.fliplr(patch).copy()
            if np.random.random() < 0.5: patch = np.flipud(patch).copy()
            if np.random.random() < 0.5:
                k = np.random.choice([1, 2, 3])
                patch = np.rot90(patch, k).copy()

        tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        # For unpaired data we have no clean/L ground truth; provide shape-compatible dummy tensors
        dummy_clean = torch.zeros_like(tensor)
        dummy_L = torch.zeros_like(tensor)
        return tensor, dummy_clean, dummy_L, torch.tensor(False)


class SyntheticPairedDataset(Dataset):
    """
    Clean images with synthetic shadows — provides full supervision for R and L.
    Uses NTU solder ball data and/or any cleanish patches you have.
    """
    def __init__(self, clean_image_dir, patch_size=256, patches_per_image=50,
                 augment=True, shadow_strength=(0.15, 0.55)):
        self.patch_size = patch_size
        self.augment = augment
        self.shadow_gen = SyntheticShadowGenerator(shadow_strength)
        self.clean_patches = []

        image_files = sorted(glob.glob(os.path.join(clean_image_dir, "*")))
        for path in image_files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            h, w = img.shape

            if h < patch_size or w < patch_size:
                scale = max(patch_size / h, patch_size / w) + 0.1
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape

            for _ in range(patches_per_image):
                y = np.random.randint(0, h - patch_size + 1)
                x = np.random.randint(0, w - patch_size + 1)
                self.clean_patches.append(img[y:y+patch_size, x:x+patch_size])

        print(f"  SyntheticPaired: {len(self.clean_patches)} clean patches "
              f"from {len(image_files)} images")

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx):
        clean = self.clean_patches[idx].copy()

        # Generate synthetic shadow
        shadowed, L_field = self.shadow_gen.generate(clean)

        if self.augment:
            if np.random.random() < 0.5:
                clean = np.fliplr(clean).copy()
                shadowed = np.fliplr(shadowed).copy()
                L_field = np.fliplr(L_field).copy()
            if np.random.random() < 0.5:
                clean = np.flipud(clean).copy()
                shadowed = np.flipud(shadowed).copy()
                L_field = np.flipud(L_field).copy()
            if np.random.random() < 0.5:
                k = np.random.choice([1, 2, 3])
                clean = np.rot90(clean, k).copy()
                shadowed = np.rot90(shadowed, k).copy()
                L_field = np.rot90(L_field, k).copy()

            # Intensity augmentation (both)
            if np.random.random() < 0.3:
                gamma = np.random.uniform(0.8, 1.2)
                clean = np.clip(clean ** gamma, 0, 1).astype(np.float32)
                shadowed = np.clip(shadowed ** gamma, 0, 1).astype(np.float32)

            # Sensor noise on shadowed
            if np.random.random() < 0.5:
                sigma = np.random.uniform(0.005, 0.02)
                shadowed = np.clip(
                    shadowed + np.random.randn(*shadowed.shape).astype(np.float32) * sigma,
                    0, 1
                )

        shadowed_t = torch.tensor(shadowed, dtype=torch.float32).unsqueeze(0)
        clean_t = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        L_t = torch.tensor(L_field, dtype=torch.float32).unsqueeze(0)

        return shadowed_t, clean_t, L_t, torch.tensor(True)


class ISTDGrayscaleDataset(Dataset):
    """
    ISTD dataset converted to grayscale — provides shadow removal supervision.
    Generates approximate L field by dividing shadow by clean.
    """
    def __init__(self, root_dir, mode='train', size=256):
        from PIL import Image as PILImage
        self.size = size
        shadow_dir = os.path.join(root_dir, mode, f'{mode}_A')
        clean_dir = os.path.join(root_dir, mode, f'{mode}_C')

        self.pairs = []
        if os.path.exists(shadow_dir):
            names = sorted(os.listdir(shadow_dir))
            for name in names:
                sp = os.path.join(shadow_dir, name)
                cp = os.path.join(clean_dir, name)
                if os.path.exists(cp):
                    self.pairs.append((sp, cp))
            print(f"  ISTD ({mode}): {len(self.pairs)} pairs")
        else:
            print(f"  ISTD: directory not found at {shadow_dir}, skipping")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sp, cp = self.pairs[idx]
        shadow = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)

        shadow = cv2.resize(shadow, (self.size, self.size)).astype(np.float32) / 255.0
        clean = cv2.resize(clean, (self.size, self.size)).astype(np.float32) / 255.0

        # Approximate L field: L = shadow / clean (where clean > 0)
        L_approx = np.ones_like(shadow)
        valid = clean > 0.01
        L_approx[valid] = shadow[valid] / clean[valid]
        L_approx = np.clip(L_approx, 0.2, 2.0)
        # Smooth it (L should be smooth even if pixel-level division is noisy)
        L_approx = gaussian_filter(L_approx, sigma=5.0)

        shadow_t = torch.tensor(shadow, dtype=torch.float32).unsqueeze(0)
        clean_t = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        L_t = torch.tensor(L_approx, dtype=torch.float32).unsqueeze(0)

        return shadow_t, clean_t, L_t, torch.tensor(True)


###############################################################################
#                    SECTION 5: MIXED DATALOADER                              #
###############################################################################

def create_mixed_dataloader(
    unpaired_tri_dir,          # Directory with shadowed TRI images
    ntu_clean_dir,             # Directory with NTU solder ball images
    istd_root=None,            # Optional: ISTD dataset root
    patch_size=256,
    batch_size=4,
    tri_patches_per_image=30,
    ntu_patches_per_image=50,
    shadow_strength=(0.15, 0.55),
):
    """
    Creates a mixed dataloader combining:
    1. Unpaired TRI patches (self-supervised)
    2. NTU + synthetic shadows (fully supervised)
    3. ISTD grayscale pairs (fully supervised, optional)
    """
    print("\n[Data] Creating mixed dataset...")

    datasets = []

    # 1. Unpaired TRI data
    if unpaired_tri_dir and os.path.exists(unpaired_tri_dir):
        tri_ds = UnpairedTRIDataset(
            unpaired_tri_dir, patch_size=patch_size,
            patches_per_image=tri_patches_per_image
        )
        datasets.append(tri_ds)

    # 2. NTU with synthetic shadows
    if ntu_clean_dir and os.path.exists(ntu_clean_dir):
        ntu_ds = SyntheticPairedDataset(
            ntu_clean_dir, patch_size=patch_size,
            patches_per_image=ntu_patches_per_image,
            shadow_strength=shadow_strength
        )
        datasets.append(ntu_ds)

    # 3. ISTD (optional)
    if istd_root and os.path.exists(istd_root):
        istd_ds = ISTDGrayscaleDataset(istd_root, mode='train', size=patch_size)
        datasets.append(istd_ds)

    combined = ConcatDataset(datasets)
    print(f"  Total: {len(combined)} samples")

    loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


###############################################################################
#                         SECTION 6: TRAINING                                 #
###############################################################################

def train_decomposition(
    # Data paths
    unpaired_tri_dir,
    ntu_clean_dir,
    istd_root=None,
    phase2_checkpoint=None,
    save_path="decomposition_model.pth",

    # Training config
    epochs=150,
    batch_size=4,
    learning_rate=2e-4,
    patch_size=256,

    # Shadow generation
    shadow_strength=(0.15, 0.55),

    # Loss weights
    w_recon=10.0,
    w_smooth=2.0,
    w_struct=0.5,
    w_illum_reg=1.0,
    w_R_supervised=5.0,
    w_L_supervised=2.0,

    # Early stopping
    patience=30,
):
    """
    Main training function for the decomposition model.

    Args:
        unpaired_tri_dir: Folder containing shadowed TRI images (any type)
        ntu_clean_dir: Folder containing NTU solder ball images
        istd_root: (Optional) ISTD dataset root for additional supervision
        phase2_checkpoint: (Optional) Path to phase2_istd_grayscale.pth
        save_path: Where to save the trained model
    """
    print("=" * 70)
    print("INTRINSIC DECOMPOSITION TRAINING")
    print("=" * 70)

    # --- Model ---
    model = IntrinsicDecompositionNet(in_channels=1).to(device)

    if phase2_checkpoint and os.path.exists(phase2_checkpoint):
        print(f"\n[Model] Loading Phase 2 weights from {phase2_checkpoint}")
        model.load_phase2_encoder(phase2_checkpoint)
    else:
        print("\n[Model] Training from scratch (no Phase 2 weights)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    print(f"  Trainable: {trainable:,}")

    # --- Data ---
    train_loader = create_mixed_dataloader(
        unpaired_tri_dir=unpaired_tri_dir,
        ntu_clean_dir=ntu_clean_dir,
        istd_root=istd_root,
        patch_size=patch_size,
        batch_size=batch_size,
        shadow_strength=shadow_strength,
    )

    # --- Loss & Optimizer ---
    criterion = DecompositionLoss(
        w_recon=w_recon, w_smooth=w_smooth, w_struct=w_struct,
        w_illum_reg=w_illum_reg, w_R_supervised=w_R_supervised,
        w_L_supervised=w_L_supervised,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- Training Loop ---
    best_loss = float('inf')
    no_improve = 0
    history = {
        'total_loss': [], 'recon': [], 'L_smooth': [],
        'R_sup': [], 'L_sup': [], 'R_struct': []
    }

    print(f"\n[Training] {epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  Batch size: {batch_size}, LR: {learning_rate}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {}
        n_batches = 0

        for I_batch, clean_batch, L_batch, has_gt_batch in tqdm(
            train_loader, desc=f"Epoch {epoch:03d}", leave=False
        ):
            I_batch = I_batch.to(device)
            clean_batch = clean_batch.to(device)
            L_batch = L_batch.to(device)
            has_gt_batch = has_gt_batch.to(device)

            # Forward pass
            R, L = model(I_batch)

            # Determine which samples have GT
            has_gt = has_gt_batch.bool()

            # Prepare supervised targets (only for samples with GT)
            clean_gt = None
            L_gt = None
            if has_gt.any():
                # Extract GT samples
                gt_indices = torch.where(has_gt)[0]
                if len(gt_indices) > 0:
                    clean_gt = clean_batch[gt_indices]
                    L_gt = L_batch[gt_indices]
                    R_gt_subset = R[gt_indices]
                    L_gt_subset = L[gt_indices]

            # Compute loss
            # Self-supervised losses on ALL samples
            total_loss, loss_dict = criterion(I_batch, R, L)

            # Add supervised losses on GT samples only
            if clean_gt is not None and len(gt_indices) > 0:
                sup_R_loss = criterion.w_R_supervised * F.l1_loss(R_gt_subset, clean_gt)
                sup_L_loss = criterion.w_L_supervised * F.l1_loss(L_gt_subset, L_gt)
                total_loss = total_loss + sup_R_loss + sup_L_loss
                loss_dict['R_sup'] = sup_R_loss.item()
                loss_dict['L_sup'] = sup_L_loss.item()

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            epoch_losses['total'] = epoch_losses.get('total', 0) + total_loss.item()
            n_batches += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Record history
        for k in history:
            if k == 'total_loss':
                history[k].append(epoch_losses.get('total', 0))
            elif k in epoch_losses:
                history[k].append(epoch_losses[k])

        # Save best
        improved = ""
        current_loss = epoch_losses['total']
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), save_path)
            improved = " ✓ BEST"
            no_improve = 0
        else:
            no_improve += 1

        # Print
        if epoch % 5 == 0 or improved or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            parts = [f"Epoch {epoch:03d}"]
            parts.append(f"Total: {epoch_losses['total']:.4f}")
            parts.append(f"Recon: {epoch_losses.get('recon', 0):.4f}")
            parts.append(f"L_sm: {epoch_losses.get('L_smooth', 0):.4f}")
            if 'L_var' in epoch_losses:
                parts.append(f"L_var: {epoch_losses['L_var']:.4f}")
            if 'R_sup' in epoch_losses:
                parts.append(f"R_sup: {epoch_losses['R_sup']:.4f}")
            parts.append(f"LR: {lr:.2e}{improved}")
            print(" | ".join(parts))

        if no_improve >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    print("\n" + "=" * 70)
    print(f"✓ Training complete! Best loss: {best_loss:.4f}")
    print(f"  Saved to: {save_path}")
    print("=" * 70)

    return model, history


###############################################################################
#              SECTION 7: INFERENCE (No post-processing!)                     #
###############################################################################

def decompose_image(model, image_path, save_prefix="result", tile_size=512,
                    overlap=64, show=True):
    """
    Run decomposition on a full TRI image.
    Returns R (reflectance = shadow-free) and L (illumination = shadow field).

    NO post-processing needed. R is the final output.
    """
    model.eval()

    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Cannot read: {image_path}")
        return None, None

    original_h, original_w = img.shape
    print(f"Image: {original_h}x{original_w} — {os.path.basename(image_path)}")

    # Pad
    pad_h = (tile_size - original_h % tile_size) % tile_size
    pad_w = (tile_size - original_w % tile_size) % tile_size
    img_padded = cv2.copyMakeBorder(
        img, pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_REFLECT
    )
    padded_h, padded_w = img_padded.shape
    img_norm = img_padded.astype(np.float32) / 255.0

    # Accumulators for R and L
    R_acc = np.zeros_like(img_norm)
    L_acc = np.zeros_like(img_norm)
    weight_map = np.zeros_like(img_norm)

    stride = tile_size - overlap
    feather = overlap // 2

    for y_start in range(0, padded_h - overlap, stride):
        for x_start in range(0, padded_w - overlap, stride):
            y_end = min(y_start + tile_size, padded_h)
            x_end = min(x_start + tile_size, padded_w)

            tile = img_norm[y_start:y_end, x_start:x_end]
            tile_h, tile_w = tile.shape

            if tile_h < tile_size or tile_w < tile_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, tile_size - tile_h, 0, tile_size - tile_w,
                    cv2.BORDER_REFLECT
                )

            tile_t = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                R_tile, L_tile = model(tile_t)

            R_np = R_tile.squeeze().cpu().numpy()[:tile_h, :tile_w]
            L_np = L_tile.squeeze().cpu().numpy()[:tile_h, :tile_w]

            # Feather weight
            weight = np.ones((tile_h, tile_w), dtype=np.float32)
            if feather > 0:
                for f in range(min(feather, tile_h)):
                    weight[f, :] *= f / feather
                for f in range(min(feather, tile_w)):
                    weight[:, f] *= f / feather

            R_acc[y_start:y_start+tile_h, x_start:x_start+tile_w] += R_np * weight
            L_acc[y_start:y_start+tile_h, x_start:x_start+tile_w] += L_np * weight
            weight_map[y_start:y_start+tile_h, x_start:x_start+tile_w] += weight

    # Normalize
    R_full = R_acc / (weight_map + 1e-8)
    L_full = L_acc / (weight_map + 1e-8)

    # Crop padding
    top, left = pad_h // 2, pad_w // 2
    R_full = R_full[top:top+original_h, left:left+original_w]
    L_full = L_full[top:top+original_h, left:left+original_w]

    # Save
    R_uint8 = np.clip(R_full * 255, 0, 255).astype(np.uint8)
    L_vis = np.clip((L_full - L_full.min()) / (L_full.max() - L_full.min() + 1e-8) * 255,
                    0, 255).astype(np.uint8)

    cv2.imwrite(f"{save_prefix}_reflectance.png", R_uint8)
    cv2.imwrite(f"{save_prefix}_illumination.png", L_vis)
    print(f"✓ Saved: {save_prefix}_reflectance.png (shadow-free)")
    print(f"✓ Saved: {save_prefix}_illumination.png (shadow field)")

    if show:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original (Shadowed)', fontsize=13)
        axes[0].axis('off')

        axes[1].imshow(R_uint8, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('R: Reflectance (Shadow-Free)', fontsize=13)
        axes[1].axis('off')

        axes[2].imshow(L_vis, cmap='hot')
        axes[2].set_title('L: Illumination (Shadow Field)', fontsize=13)
        axes[2].axis('off')

        # Reconstruction check: R * L should ≈ I
        recon = np.clip(R_full * L_full * 255, 0, 255).astype(np.uint8)
        axes[3].imshow(recon, cmap='gray', vmin=0, vmax=255)
        axes[3].set_title('R × L (Reconstruction Check)', fontsize=13)
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

    return R_uint8, L_vis


###############################################################################
#                     SECTION 8: VISUALIZATION                                #
###############################################################################

def plot_history(history):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.get('total_loss', []), label='Total', color='black', linewidth=2)
    axes[0].plot(history.get('recon', []), label='Reconstruction', color='blue', alpha=0.7)
    axes[0].set_title('Overall Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history.get('L_smooth', []), label='L Smoothness', color='red')
    axes[1].plot(history.get('R_struct', []), label='R Structure', color='green')
    axes[1].set_title('Prior Losses')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')

    if history.get('R_sup'):
        axes[2].plot(history['R_sup'], label='R Supervised', color='purple')
    if history.get('L_sup'):
        axes[2].plot(history['L_sup'], label='L Supervised', color='orange')
    axes[2].set_title('Supervised Losses')
    axes[2].legend()
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()


def visualize_decomposition_examples(model, image_dir, n=3):
    """Show decomposition results on random patches."""
    model.eval()
    files = sorted(glob.glob(os.path.join(image_dir, "*")))[:n]

    for path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Take center crop
        h, w = img.shape
        cs = min(h, w, 512)
        y, x = (h - cs) // 2, (w - cs) // 2
        crop = img[y:y+cs, x:x+cs].astype(np.float32) / 255.0

        tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            R, L = model(tensor)
        R_np = R.squeeze().cpu().numpy()
        L_np = L.squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(crop, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Input: {os.path.basename(path)}')
        axes[0].axis('off')
        axes[1].imshow(R_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('R (Shadow-Free)')
        axes[1].axis('off')
        axes[2].imshow(L_np, cmap='hot')
        axes[2].set_title(f'L (Shadow Field) [{L_np.min():.2f} - {L_np.max():.2f}]')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()


###############################################################################
#                         SECTION 9: USAGE                                    #
###############################################################################

model, history = train_decomposition(
    unpaired_tri_dir=r"C:\Users\tingchen\Documents\Aryan_Projects\Shadow_Removal\Shadow Removal Remote Access\Shadow_Removal\TRI_Dataset",       
    ntu_clean_dir=r"C:\Users\tingchen\Documents\Aryan_Projects\Shadow_Removal\Shadow Removal Remote Access\Shadow_Removal\NTU_Cropped",      
    istd_root=r"C:\Users\tingchen\Documents\Aryan_Projects\Shadow_Removal\Shadow Removal Remote Access\Shadow_Removal\ISTD_Dataset",         
    phase2_checkpoint="phase2_istd_grayscale.pth",      # Phase 2 weights

    epochs=50,
    batch_size=4,
    learning_rate=2e-4,
    patch_size=256,
    shadow_strength=(0.15, 0.55),

    # Loss weights (start with these defaults, tune if needed)
    w_recon=10.0,
    w_smooth=2.0,
    w_struct=0.5,
    w_illum_reg=1.0,
    w_R_supervised=5.0,
    w_L_supervised=2.0,
)
plot_history(history)




r"""
=============================================================================
HOW TO USE
=============================================================================

STEP 1: Organize  data
---------------------------
Put files in these directories:

    unpaired_tri/          ← ALL  shadowed TRI images (J1, U29, U30, etc.)
        J1_154.jpg
        U29_xxx.jpg
        U30_031.jpg
        ...any other shadowed X-ray images...

    ntu_solder_ball/       ←  NTU solder ball images (used for synthetic pairs)
        0000.tif
        0001.tif
        ...

    (Optional) ISTD dataset in its standard folder structure


STEP 2: Train
--------------
model, history = train_decomposition(
    unpaired_tri_dir=r"C:\Users\tingchen\Documents\Aryan_Projects\Shadow_Removal\Shadow Removal Remote Access\Shadow_Removal\TRI_Dataset",           #  PATH
    ntu_clean_dir=r"C:\Users\tingchen\Documents\Aryan_Projects\Shadow_Removal\Shadow Removal Remote Access\Shadow_Removal\NTU_Solder_Ball",           #  PATH
    istd_root=r"C:\Users\tingchen\Documents\Aryan_Projects\Shadow_Removal\Shadow Removal Remote Access\Shadow_Removal\ISTD_Dataset",                   #  PATH (optional)
    phase2_checkpoint="phase2_istd_grayscale.pth",      # Phase 2 weights

    epochs=150,
    batch_size=4,
    learning_rate=2e-4,
    patch_size=256,
    shadow_strength=(0.15, 0.55),

    # Loss weights (start with these defaults, tune if needed)
    w_recon=10.0,
    w_smooth=2.0,
    w_struct=0.5,
    w_illum_reg=1.0,
    w_R_supervised=5.0,
    w_L_supervised=2.0,
)
plot_history(history)


STEP 3: Test on any TRI image (no post-processing!)
----------------------------------------------------
model = IntrinsicDecompositionNet(in_channels=1).to(device)
model.load_state_dict(torch.load("decomposition_model.pth", map_location=device))

# Works on ANY component type with the same model + same settings:
R_j1, L_j1 = decompose_image(model, "J1_154.jpg", save_prefix="J1")
R_u30, L_u30 = decompose_image(model, "U30_031.jpg", save_prefix="U30")


TUNING GUIDE
-------------
If R still shows shadow residuals:
    → Increase w_smooth (3.0 → 5.0): forces L to capture more of the shadow
    → Increase shadow_strength max (0.55 → 0.70): train on heavier shadows
    → Increase epochs (150 → 300)

If R looks blurry / loses detail:
    → Decrease w_smooth (2.0 → 1.0): gives L less freedom
    → Decrease w_struct (0.5 → 0.1): less smoothing on R
    → Check that w_recon is high enough (should be highest weight)

If L looks noisy / captures structure:
    → Increase w_smooth significantly (5.0+)
    → Check the illumination decoder — it should only produce smooth fields

If reconstruction R*L doesn't match input well:
    → Increase w_recon (10 → 20)
    → This is the most important constraint — should always be the dominant loss

If model collapses (R=I, L=1 everywhere):
    → Increase w_R_supervised: stronger anchor from synthetic pairs
    → Make sure NTU data is being loaded correctly
    → Check that synthetic shadows are actually visible (run visualize)
=============================================================================
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nReady to train. See USAGE section at bottom of file.")
    print("Key paths to configure:")
    print("  1. unpaired_tri_dir  — shadowed TRI images")
    print("  2. ntu_clean_dir     — NTU solder ball images")
    print("  3. istd_root         — ISTD dataset (optional)")
    print("  4. phase2_checkpoint — phase2_istd_grayscale.pth")
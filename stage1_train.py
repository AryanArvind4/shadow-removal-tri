"""
Stage 1 — VP-SDE Schedule + Training Loop
Trains the diffusion prior on clean TRI X-ray images using denoising score matching.
After this stage you have a model that "knows" what clean X-rays look like.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from stage1_dataset import get_clean_dataloader
from stage1_model import UNet


# ── VP-SDE noise schedule ─────────────────────────────────────────────────────

class VPSDESchedule:
    """
    Variance-Preserving SDE:
        dx = -½ β(t) x dt + √β(t) dw
    where β(t) linearly increases from β_min to β_max.

    At time t, the marginal distribution is:
        x(t) = √α(t) · x(0) + √(1-α(t)) · z,   z ~ N(0,I)

    This satisfies α²(t) + σ²(t) = 1 (variance preserving).
    """

    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t):
        """Linear schedule: β(t) = β_min + t(β_max - β_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def log_alpha(self, t):
        """log α(t) = -½ ∫₀ᵗ β(s)ds"""
        return -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)

    def alpha(self, t):
        """α(t) — controls how much original image remains"""
        return torch.exp(self.log_alpha(t))

    def sigma(self, t):
        """σ(t) — noise level, satisfies α²+σ²=1"""
        return torch.sqrt(torch.clamp(1 - self.alpha(t)**2, min=1e-5))

    def marginal(self, x0, t):
        """
        Sample x(t) from the forward process:
            x(t) = α(t)·x(0) + σ(t)·z,  z ~ N(0,I)
        Returns (x_t, z, alpha_t, sigma_t)
        """
        t = t.view(-1, 1, 1, 1)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        z = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * z
        return x_t, z, alpha_t, sigma_t


# ── Loss function ─────────────────────────────────────────────────────────────

def score_matching_loss(model, x0, schedule):
    """
    Denoising score matching loss:
        L = E[|| σ(t) · sθ(x(t), t) - z ||²]
    where z is the noise added to x(0) to get x(t).

    Intuition: the model learns to predict the noise, which is equivalent
    to learning the score ∇ₓ log p_t(x).
    """
    B = x0.shape[0]
    device = x0.device

    # Sample random time steps uniformly in (0, 1)
    t = torch.rand(B, device=device) * 0.999 + 0.001  # avoid t=0 exactly

    # Forward process: add noise
    x_t, z, alpha_t, sigma_t = schedule.marginal(x0, t)

    # Predict score
    score_pred = model(x_t, t)

    # Epsilon-prediction loss: predict the noise z directly.
    # Equivalent to score matching but avoids dividing by σ(t),
    # which explodes at small t and causes loss spikes.
    #   score = -z / σ(t)  →  -σ(t) * score = z  →  predict z
    eps_pred = -sigma_t * score_pred       # convert score output to noise prediction
    loss = ((eps_pred - z) ** 2).mean()

    return loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config):
    """
    Main training function for Stage 1.

    config dict keys:
        clean_dir     : path to clean X-ray images
        save_dir      : where to save checkpoints
        image_size    : spatial resolution (default 256)
        batch_size    : (default 8, fits RTX 4070 Ti)
        epochs        : number of training epochs
        lr            : learning rate
        base_channels : UNet width (default 64)
        resume        : path to checkpoint to resume from (optional)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)

    # DataLoader
    loader = get_clean_dataloader(
        config['clean_dir'],
        image_size=config.get('image_size', 256),
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 0),
        augment=True
    )
    print(f"[Train] Steps per epoch: {len(loader)}")

    # Model
    model = UNet(
        image_channels=1,
        base_channels=config.get('base_channels', 64),
        image_size=config.get('image_size', 256)
    ).to(device)

    # Noise schedule
    schedule = VPSDESchedule(beta_min=0.1, beta_max=20.0)

    # Optimizer + AMP scaler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 2e-4),
        weight_decay=1e-4
    )
    scaler = GradScaler('cuda')

    # Cosine LR scheduler
    total_steps = config.get('epochs', 100) * len(loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Exponential moving average of model weights (improves sample quality)
    ema_model = UNet(
        image_channels=1,
        base_channels=config.get('base_channels', 64),
        image_size=config.get('image_size', 256)
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.9999

    start_epoch = 0

    # Resume from checkpoint if provided
    if config.get('resume'):
        ckpt = torch.load(config['resume'], map_location=device)
        model.load_state_dict(ckpt['model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[Train] Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.get('epochs', 100)):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.get('epochs',100)}")
        for step, x0 in enumerate(pbar):
            x0 = x0.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                loss = score_matching_loss(model, x0, schedule)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update EMA weights
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}] avg loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs (or more frequently for short runs)
        save_every = max(1, min(10, config.get('epochs', 30) // 3))
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(
                config['save_dir'], f'ckpt_epoch{epoch+1}.pt'
            )
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss'     : avg_loss
            }, ckpt_path)
            print(f"[Train] Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(config['save_dir'], 'diffusion_prior_final.pt')
    torch.save({
        'epoch'    : config.get('epochs', 100) - 1,
        'model'    : model.state_dict(),
        'ema_model': ema_model.state_dict(),
    }, final_path)
    print(f"[Train] Final model saved: {final_path}")
    return ema_model, schedule


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    config = {
        'clean_dir'    : 'data/clean',          # ← PUT YOUR CLEAN X-RAYS HERE
        'save_dir'     : 'checkpoints/',
        'image_size'   : 128,   # 4× faster than 256; shadows are low-frequency so 128 is fine
        'batch_size'   : 8,
        'epochs'       : 100,   # ~15-30 min with 85 images on RTX 5050
        'lr'           : 2e-4,
        'base_channels': 64,
        'num_workers'  : 0,                     # Windows-safe
        # 'resume'     : 'checkpoints/ckpt_epoch50.pt'  # uncomment to resume
    }

    print("=" * 60)
    print("STAGE 1 — Training Diffusion Prior on Clean X-rays")
    print("=" * 60)
    train(config)

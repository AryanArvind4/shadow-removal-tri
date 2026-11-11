"""
Stage 2 — SDEdit Inference Engine
Shadow removal on TRI X-ray PCB images using the trained diffusion prior.
Implements both plain SDEdit and masked SDEdit (structure-preserving).
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

from stage1_model import UNet
from stage1_train import VPSDESchedule
from stage1_dataset import get_shadowed_dataloader


# ── Core SDEdit algorithm (VP-SDE version) ────────────────────────────────────

class SDEditRemover:
    """
    SDEdit-based shadow removal for PCB X-ray images.

    How it works:
        1. Take shadowed X-ray image as input
        2. Add Gaussian noise up to level t₀
           → This partially destroys the low-frequency shadow artifacts
           → But preserves the high-frequency PCB structure (traces, BGA balls)
        3. Run the reverse SDE from t₀ back to t=0
           → Diffusion model rebuilds the image guided by its clean prior
           → Output is realistic + faithful to original PCB layout

    Key parameter t₀:
        - Small t₀ (0.2–0.3) : subtle correction, shadows may survive
        - Medium t₀ (0.35–0.5): good shadow removal, structure preserved   ← your target
        - Large t₀ (0.6+)    : aggressive correction, may alter PCB features
    """

    def __init__(self, model, schedule, device='cuda'):
        self.model    = model.eval()
        self.schedule = schedule
        self.device   = device

    @torch.no_grad()
    def remove_shadow(self, x_shadowed, t0=0.4, N=500):
        """
        Plain SDEdit — no mask.

        Args:
            x_shadowed : shadowed image tensor [B, 1, H, W] in [-1, 1]
            t0         : starting noise level (0 to 1)
            N          : number of reverse denoising steps

        Returns:
            x_clean : shadow-removed image [B, 1, H, W] in [-1, 1]
        """
        x = x_shadowed.to(self.device).clone()
        B = x.shape[0]

        # ── Step 1: Forward — add noise up to t₀ ────────────────────────────
        t0_tensor = torch.full((B,), t0, device=self.device)
        _, z, alpha_t0, sigma_t0 = self.schedule.marginal(x, t0_tensor)
        x = alpha_t0 * x + sigma_t0 * z   # partially noisy image

        # ── Step 2: Reverse — denoise from t₀ to 0 ──────────────────────────
        delta_t = t0 / N

        for n in tqdm(range(N, 0, -1), desc=f"  SDEdit t₀={t0:.2f}", leave=False):
            t_val = t0 * n / N
            t_tensor = torch.full((B,), t_val, device=self.device)

            # Score model prediction: ∇ₓ log p_t(x)
            score = self.model(x, t_tensor)

            # VP-SDE reverse step (Euler-Maruyama)
            beta_t = self.schedule.beta(torch.tensor(t_val, device=self.device))
            z_step = torch.randn_like(x)

            x = (1 / (1 - beta_t * delta_t).sqrt()) * (
                x + beta_t * delta_t * score
            ) + (beta_t * delta_t).sqrt() * z_step

        return x.clamp(-1, 1)

    @torch.no_grad()
    def remove_shadow_masked(self, x_shadowed, mask, t0=0.4, N=500):
        """
        Masked SDEdit — preserves clean regions exactly.
        Based on Algorithm 3 from the SDEdit paper.

        Use this when:
        - You have regions that are already shadow-free (from your v3 L map)
        - You want to guarantee those regions are untouched

        Args:
            x_shadowed : shadowed image [B, 1, H, W] in [-1, 1]
            mask       : binary mask [B, 1, H, W]
                         1 = shadow region   (will be corrected)
                         0 = clean region    (will be preserved exactly)
            t0         : noise level
            N          : denoising steps

        Returns:
            x_clean : shadow-removed image [B, 1, H, W]
        """
        x0 = x_shadowed.to(self.device).clone()   # original reference
        x  = x_shadowed.to(self.device).clone()
        mask = mask.to(self.device)
        B = x.shape[0]

        # Forward: add noise to entire image
        t0_tensor = torch.full((B,), t0, device=self.device)
        _, z, alpha_t0, sigma_t0 = self.schedule.marginal(x, t0_tensor)
        x = alpha_t0 * x + sigma_t0 * z

        delta_t = t0 / N

        for n in tqdm(range(N, 0, -1), desc=f"  Masked SDEdit t₀={t0:.2f}", leave=False):
            t_val = t0 * n / N
            t_tensor = torch.full((B,), t_val, device=self.device)

            # Current sigma for noise level maintenance in preserved regions
            sigma_t = self.schedule.sigma(t_tensor.view(-1, 1, 1, 1))

            # Score prediction
            score = self.model(x, t_tensor)
            beta_t = self.schedule.beta(torch.tensor(t_val, device=self.device))
            z_step = torch.randn_like(x)

            # Reverse step for shadow regions (mask=1)
            x_edited = (1 / (1 - beta_t * delta_t).sqrt()) * (
                x + beta_t * delta_t * score
            ) + (beta_t * delta_t).sqrt() * z_step

            # Preserved region: original image with decaying noise
            # As t→0, sigma_t→0, so this converges back to original x0
            z_preserve = torch.randn_like(x)
            x_preserved = x0 + sigma_t * z_preserve

            # Combine: shadow regions use diffusion, clean regions use original
            x = mask * x_edited + (1 - mask) * x_preserved

        return x.clamp(-1, 1)


# ── t₀ sweep for tuning ───────────────────────────────────────────────────────

def t0_sweep(remover, x_shadowed, x_clean_ref=None,
             t0_values=None, save_dir='t0_sweep'):
    """
    Runs SDEdit at multiple t₀ values on one image.
    Saves results for visual inspection + computes metrics if clean ref available.

    Use this BEFORE full inference to find the optimal t₀ for your data.
    """
    if t0_values is None:
        t0_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for t0 in t0_values:
        print(f"[Sweep] t₀ = {t0:.2f}")
        x_out = remover.remove_shadow(x_shadowed, t0=t0, N=300)

        # Denormalize [-1,1] → [0,255]
        img_np = ((x_out.squeeze().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, mode='L')
        img_pil.save(os.path.join(save_dir, f't0_{t0:.2f}.png'))

        metrics = {'t0': t0}

        # Compute metrics if clean reference available
        if x_clean_ref is not None:
            x_out_01   = (x_out + 1) / 2
            x_clean_01 = (x_clean_ref.to(x_out.device) + 1) / 2

            # PSNR
            mse  = ((x_out_01 - x_clean_01) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            metrics['psnr'] = psnr

            # Left/Right brightness uniformity (your v3 metric)
            h_mid = x_out_01.shape[-1] // 2
            left  = x_out_01[..., :h_mid].mean().item()
            right = x_out_01[..., h_mid:].mean().item()
            metrics['lr_diff'] = abs(left - right)

            print(f"  PSNR={psnr:.2f}dB  LR_diff={metrics['lr_diff']:.4f}")

        results[t0] = metrics

    return results


# ── Full inference pipeline ───────────────────────────────────────────────────

def run_inference(config):
    """
    Runs SDEdit shadow removal on all shadowed X-ray images.

    config keys:
        shadowed_dir  : folder with shadowed TRI X-rays
        output_dir    : where to save shadow-free results
        checkpoint    : path to trained diffusion model checkpoint
        t0            : chosen t0 (from sweep)
        N             : denoising steps (500 recommended)
        use_mask      : whether to use masked SDEdit
        mask_dir      : folder with binary shadow masks (if use_mask=True)
        image_size    : must match training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)

    # Load model
    model = UNet(
        image_channels=1,
        base_channels=config.get('base_channels', 64),
        image_size=config.get('image_size', 256)
    ).to(device)

    ckpt = torch.load(config['checkpoint'], map_location=device)
    # Use EMA weights for inference (better quality)
    state_dict = ckpt.get('ema_model', ckpt.get('model'))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Inference] Loaded model from {config['checkpoint']}")

    schedule = VPSDESchedule(beta_min=0.1, beta_max=20.0)
    remover  = SDEditRemover(model, schedule, device)

    # DataLoader
    loader = get_shadowed_dataloader(
        config['shadowed_dir'],
        image_size=config.get('image_size', 256),
        batch_size=1   # process one image at a time for stability
    )

    t0 = config.get('t0', 0.4)
    N  = config.get('N', 500)

    print(f"[Inference] t₀={t0}, N={N}, use_mask={config.get('use_mask', False)}")
    print(f"[Inference] Processing {len(loader)} images...")

    for x_shadowed, filename in tqdm(loader, desc="Shadow Removal"):
        x_shadowed = x_shadowed.to(device)
        fname = filename[0]

        if config.get('use_mask', False) and config.get('mask_dir'):
            # Load corresponding shadow mask
            mask_path = os.path.join(config['mask_dir'], fname)
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path).convert('L')
                mask_pil = mask_pil.resize(
                    (config.get('image_size', 256),) * 2,
                    Image.NEAREST
                )
                mask = TF.to_tensor(mask_pil).unsqueeze(0).to(device)
                mask = (mask > 0.5).float()
                x_out = remover.remove_shadow_masked(x_shadowed, mask, t0=t0, N=N)
            else:
                print(f"  [Warning] Mask not found for {fname}, using plain SDEdit")
                x_out = remover.remove_shadow(x_shadowed, t0=t0, N=N)
        else:
            x_out = remover.remove_shadow(x_shadowed, t0=t0, N=N)

        # Save result
        img_np = ((x_out.squeeze().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np, mode='L').save(
            os.path.join(config['output_dir'], fname)
        )

    print(f"[Inference] Done. Results saved to: {config['output_dir']}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    config = {
        'shadowed_dir' : 'data/shadowed',         # ← shadowed TRI X-rays
        'output_dir'   : 'results/shadow_removed',
        'checkpoint'   : 'checkpoints/diffusion_prior_final.pt',
        'image_size'   : 256,
        'base_channels': 64,
        't0'           : 0.4,    # tune this with t0_sweep first!
        'N'            : 500,
        'use_mask'     : False,  # set True if you have shadow masks from v3 L map
        # 'mask_dir'   : 'data/masks',
    }

    print("=" * 60)
    print("STAGE 2 — SDEdit Shadow Removal Inference")
    print("=" * 60)
    run_inference(config)

"""
Stage 3 — Evaluation
Measures shadow removal quality using:
  - PSNR  (pixel-level fidelity)
  - SSIM  (structural similarity)
  - Left/Right brightness uniformity (your existing v3 metric)
  - Shadow intensity reduction ratio
  - t₀ sweep visualization
"""

import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ── Metric functions ──────────────────────────────────────────────────────────

def psnr(img1, img2):
    """
    Peak Signal-to-Noise Ratio.
    img1, img2: numpy arrays in [0, 1]
    Higher = better (typically want > 30dB for good results)
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def ssim(img1, img2, window_size=11):
    """
    Structural Similarity Index.
    img1, img2: numpy arrays in [0, 1]
    Closer to 1.0 = better structural preservation
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, window_size) - mu2_sq
    sigma12   = uniform_filter(img1 * img2, window_size) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def uniform_filter(img, size):
    """Simple uniform (box) filter for SSIM computation."""
    from scipy.ndimage import uniform_filter as sp_uniform
    return sp_uniform(img.astype(float), size=size)


def lr_brightness_uniformity(img):
    """
    Left/Right brightness uniformity — your existing v3 metric.
    img: numpy array [H, W] in [0, 1]

    Returns:
        lr_diff  : absolute mean brightness difference L vs R (lower = better)
        lr_ratio : ratio of left to right mean brightness (closer to 1.0 = better)
    """
    w = img.shape[1]
    mid = w // 2
    left_mean  = img[:, :mid].mean()
    right_mean = img[:, mid:].mean()
    lr_diff  = abs(left_mean - right_mean)
    lr_ratio = left_mean / (right_mean + 1e-8)
    return lr_diff, lr_ratio


def shadow_reduction_ratio(img_shadowed, img_output):
    """
    Measures how much shadow intensity has been reduced.
    Computes std of local brightness across image quadrants.
    Lower std after = better uniformity.

    img_shadowed, img_output: numpy arrays [H, W] in [0, 1]
    """
    def quadrant_std(img):
        h, w = img.shape
        hm, wm = h // 2, w // 2
        quadrants = [
            img[:hm, :wm].mean(),
            img[:hm, wm:].mean(),
            img[hm:, :wm].mean(),
            img[hm:, wm:].mean(),
        ]
        return np.std(quadrants)

    std_before = quadrant_std(img_shadowed)
    std_after  = quadrant_std(img_output)
    reduction  = (std_before - std_after) / (std_before + 1e-8)
    return reduction, std_before, std_after


# ── Load image utility ────────────────────────────────────────────────────────

def load_img(path, image_size=256):
    """Load image as numpy float32 in [0, 1]."""
    img = Image.open(path).convert('L')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate(config):
    """
    Evaluates shadow removal results against clean references (if available)
    and always computes uniformity metrics.

    config keys:
        shadowed_dir  : original shadowed X-rays
        output_dir    : SDEdit shadow-removed results
        clean_dir     : clean reference images (optional, for PSNR/SSIM)
        image_size    : spatial resolution
        save_report   : path to save CSV report
    """
    _sd = config['shadowed_dir']
    shadowed_dirs = [_sd] if isinstance(_sd, str) else list(_sd)
    output_dir   = config['output_dir']
    clean_dir    = config.get('clean_dir', None)
    image_size   = config.get('image_size', 256)

    filenames = sorted([
        f for f in os.listdir(output_dir)
        if f.lower().endswith(('.png', '.jpg', '.tiff'))
    ])

    if not filenames:
        print(f"[Eval] No output images found in {output_dir}")
        return

    print(f"[Eval] Evaluating {len(filenames)} images...")

    all_metrics = []

    for fname in tqdm(filenames):
        out_path    = os.path.join(output_dir, fname)
        shadow_path = None
        for sd in shadowed_dirs:
            candidate = os.path.join(sd, fname)
            if os.path.exists(candidate):
                shadow_path = candidate
                break

        if shadow_path is None:
            print(f"  [Skip] No shadowed counterpart for {fname}")
            continue

        img_shadow = load_img(shadow_path, image_size)
        img_output = load_img(out_path, image_size)

        m = {'filename': fname}

        # Always compute uniformity metrics
        lr_diff_in,  lr_ratio_in  = lr_brightness_uniformity(img_shadow)
        lr_diff_out, lr_ratio_out = lr_brightness_uniformity(img_output)
        m['lr_diff_before'] = lr_diff_in
        m['lr_diff_after']  = lr_diff_out
        m['lr_ratio_before']= lr_ratio_in
        m['lr_ratio_after'] = lr_ratio_out
        m['lr_improvement'] = lr_diff_in - lr_diff_out  # positive = improvement

        reduction, std_before, std_after = shadow_reduction_ratio(img_shadow, img_output)
        m['shadow_reduction'] = reduction
        m['quad_std_before']  = std_before
        m['quad_std_after']   = std_after

        # If clean reference is available
        if clean_dir:
            clean_path = os.path.join(clean_dir, fname)
            if os.path.exists(clean_path):
                img_clean = load_img(clean_path, image_size)
                m['psnr_before'] = psnr(img_shadow, img_clean)
                m['psnr_after']  = psnr(img_output, img_clean)

                try:
                    m['ssim_before'] = ssim(img_shadow, img_clean)
                    m['ssim_after']  = ssim(img_output, img_clean)
                except Exception:
                    m['ssim_before'] = m['ssim_after'] = None

        all_metrics.append(m)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    lr_diffs_before = [m['lr_diff_before'] for m in all_metrics]
    lr_diffs_after  = [m['lr_diff_after']  for m in all_metrics]
    improvements    = [m['lr_improvement'] for m in all_metrics]
    reductions      = [m['shadow_reduction'] for m in all_metrics]

    print(f"L/R diff before : {np.mean(lr_diffs_before):.4f} ± {np.std(lr_diffs_before):.4f}")
    print(f"L/R diff after  : {np.mean(lr_diffs_after):.4f} ± {np.std(lr_diffs_after):.4f}")
    print(f"L/R improvement : {np.mean(improvements):.4f} (positive = better)")
    print(f"Shadow reduction: {np.mean(reductions)*100:.1f}%")

    if clean_dir and 'psnr_after' in all_metrics[0]:
        psnrs_before = [m['psnr_before'] for m in all_metrics if m.get('psnr_before')]
        psnrs_after  = [m['psnr_after']  for m in all_metrics if m.get('psnr_after')]
        ssims_before = [m['ssim_before'] for m in all_metrics if m.get('ssim_before')]
        ssims_after  = [m['ssim_after']  for m in all_metrics if m.get('ssim_after')]

        if psnrs_before:
            print(f"PSNR before     : {np.mean(psnrs_before):.2f} dB")
            print(f"PSNR after      : {np.mean(psnrs_after):.2f} dB")
        if ssims_before:
            print(f"SSIM before     : {np.mean(ssims_before):.4f}")
            print(f"SSIM after      : {np.mean(ssims_after):.4f}")

    # ── Save CSV report ───────────────────────────────────────────────────
    if config.get('save_report'):
        import csv
        keys = all_metrics[0].keys()
        with open(config['save_report'], 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[Eval] Report saved: {config['save_report']}")

    return all_metrics


# ── t₀ sweep evaluation (use before full inference) ──────────────────────────

def evaluate_t0_sweep(sweep_dir, shadowed_img_path,
                      clean_img_path=None, image_size=256):
    """
    Evaluates all t₀ sweep outputs to find the best t₀ for your data.
    Run this before stage2_inference.py to choose your t₀.
    """
    img_shadow = load_img(shadowed_img_path, image_size)

    results = []
    for fname in sorted(os.listdir(sweep_dir)):
        if not fname.startswith('t0_'):
            continue
        t0_val = float(fname.replace('t0_', '').replace('.png', ''))
        img_out = load_img(os.path.join(sweep_dir, fname), image_size)

        lr_diff, lr_ratio = lr_brightness_uniformity(img_out)
        reduction, _, _   = shadow_reduction_ratio(img_shadow, img_out)

        row = {
            't0'             : t0_val,
            'lr_diff'        : lr_diff,
            'lr_ratio'       : lr_ratio,
            'shadow_reduction': reduction
        }

        if clean_img_path and os.path.exists(clean_img_path):
            img_clean = load_img(clean_img_path, image_size)
            row['psnr'] = psnr(img_out, img_clean)
            try:
                row['ssim'] = ssim(img_out, img_clean)
            except Exception:
                row['ssim'] = None

        results.append(row)
        print(f"t₀={t0_val:.2f} | LR_diff={lr_diff:.4f} | "
              f"Shadow_red={reduction*100:.1f}%"
              + (f" | PSNR={row.get('psnr', 0):.2f}dB" if 'psnr' in row else ""))

    # Recommend best t₀
    # Criterion: maximize shadow reduction while keeping LR diff low
    best = min(results, key=lambda r: r['lr_diff'])
    print(f"\n[Eval] Recommended t₀ = {best['t0']:.2f} (lowest L/R diff = {best['lr_diff']:.4f})")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    config = {
        'shadowed_dir': 'data/shadowed',
        'output_dir'  : 'results/shadow_removed',
        'clean_dir'   : 'data/clean',            # optional, remove if no pairs
        'image_size'  : 256,
        'save_report' : 'results/eval_report.csv'
    }

    print("=" * 60)
    print("STAGE 3 — Evaluation")
    print("=" * 60)
    evaluate(config)

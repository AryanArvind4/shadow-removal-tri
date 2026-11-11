"""
Classical Shadow Removal for TRI PCB X-ray Images
==================================================
Handles two distinct artifact types seen in these images:
  1. Horizontal banding  — per-row illumination variation (std ~15% of mean)
  2. Large-scale gradient — smooth L/R or 2D brightness drift

Algorithm (separable, runs in milliseconds per image):
  Step 1: Row-wise correction  — smooth the row-median profile, divide each row
  Step 2: Column-wise correction — smooth the col-median profile, divide each col
  Step 3: 2D residual correction — Gaussian flat-field for any remaining gradient
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ── Core algorithm ────────────────────────────────────────────────────────────

def remove_shadow(img_np,
                  row_sigma=20,    # controls banding removal: smaller = more aggressive
                  col_sigma=200,   # controls L/R gradient: fraction of image width
                  final_sigma=500, # final 2D correction for residual gradient
                  ):
    """
    Separable shadow removal for PCB X-ray images.

    Args:
        img_np      : [H, W] numpy uint8 array
        row_sigma   : Gaussian sigma for row-profile smoothing.
                      Should be > solder-ball-row spacing but < shadow period.
                      Typical range: 10–40 for 2000px tall images.
        col_sigma   : Gaussian sigma for col-profile smoothing.
                      Should capture only the large-scale L/R gradient.
                      Typical range: 150–400 for 2000px wide images.
        final_sigma : Final 2D Gaussian for any remaining gradient.

    Returns:
        corrected [H, W] numpy uint8 array
    """
    H, W = img_np.shape
    img_f = img_np.astype(np.float64)
    target = np.mean(img_f)

    # ── Step 1: Row-wise banding correction ──────────────────────────────────
    # Use trimmed mean (ignore darkest 5% and brightest 5% per row)
    # This is robust to bright solder balls and dark PCB edges
    row_profile = _trimmed_row_mean(img_f, trim=0.05)
    row_smooth  = gaussian_filter(row_profile, sigma=row_sigma)
    # Avoid amplifying noise in very dark rows (clamp to 10% of target)
    row_smooth  = np.maximum(row_smooth, target * 0.1)
    img_f = img_f * (target / row_smooth[:, np.newaxis])

    # ── Step 2: Column-wise gradient correction ───────────────────────────────
    col_profile = _trimmed_col_mean(img_f, trim=0.05)
    col_smooth  = gaussian_filter(col_profile, sigma=col_sigma)
    col_smooth  = np.maximum(col_smooth, target * 0.1)
    img_f = img_f * (target / col_smooth[np.newaxis, :])

    # ── Step 3: 2D residual correction ───────────────────────────────────────
    background = gaussian_filter(img_f, sigma=final_sigma)
    background = np.maximum(background, target * 0.1)
    img_f = img_f / background * target

    # Rescale to original mean brightness to keep absolute intensity
    img_f = img_f * (np.mean(img_np) / (np.mean(img_f) + 1e-8))

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _trimmed_row_mean(img, trim=0.05):
    """Mean of each row after trimming extreme pixels (robust to balls/edges)."""
    H, W = img.shape
    sorted_rows = np.sort(img, axis=1)
    lo = int(W * trim)
    hi = int(W * (1 - trim))
    if hi <= lo:
        return img.mean(axis=1)
    return sorted_rows[:, lo:hi].mean(axis=1)


def _trimmed_col_mean(img, trim=0.05):
    """Mean of each column after trimming extreme pixels."""
    H, W = img.shape
    sorted_cols = np.sort(img, axis=0)
    lo = int(H * trim)
    hi = int(H * (1 - trim))
    if hi <= lo:
        return img.mean(axis=0)
    return sorted_cols[lo:hi, :].mean(axis=0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(img_np):
    w = img_np.shape[1]
    mid = w // 2
    left  = float(img_np[:, :mid].mean())
    right = float(img_np[:, mid:].mean())
    lr_diff = abs(left - right)

    h, w = img_np.shape
    hm, wm = h // 2, w // 2
    quads = [img_np[:hm, :wm].mean(), img_np[:hm, wm:].mean(),
             img_np[hm:, :wm].mean(), img_np[hm:, wm:].mean()]
    q_std = float(np.std(quads))

    row_meds = np.median(img_np.astype(float), axis=1)
    banding  = float(row_meds.std())

    return {'lr_diff': lr_diff, 'quad_std': q_std, 'row_banding_std': banding}


def compare_side_by_side(img_before, img_after):
    """Return single image: [before | divider | after]."""
    divider = np.full((img_before.shape[0], 6), 200, dtype=np.uint8)
    return np.concatenate([img_before, divider, img_after], axis=1)


# ── Sigma sweep ───────────────────────────────────────────────────────────────

def sigma_sweep(img_path, output_dir,
                row_sigmas=(5, 10, 20, 30, 50),
                col_sigmas=(100, 200, 400)):
    """
    Try all combinations of row_sigma × col_sigma on one image.
    Saves named result images so you can visually pick the best settings.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(img_path).convert('L')
    img_np = np.array(img)

    m_in = metrics(img_np)
    print(f"\nInput  — LR_diff={m_in['lr_diff']:.2f}  "
          f"row_banding={m_in['row_banding_std']:.2f}  "
          f"quad_std={m_in['quad_std']:.2f}")
    print(f"\n{'row_σ':>6} {'col_σ':>6} | {'LR_diff':>8} {'row_band':>9} {'quad_std':>9}")
    print("-" * 46)

    best = {'score': 1e9, 'row_sigma': 20, 'col_sigma': 200}

    for rs in row_sigmas:
        for cs in col_sigmas:
            corrected = remove_shadow(img_np, row_sigma=rs, col_sigma=cs)
            m = metrics(corrected)

            score = m['lr_diff'] + m['row_banding_std'] * 0.1
            if score < best['score']:
                best = {'score': score, 'row_sigma': rs, 'col_sigma': cs}

            side = compare_side_by_side(img_np, corrected)
            fname = f'row{rs:03d}_col{cs:04d}.png'
            Image.fromarray(side, mode='L').save(os.path.join(output_dir, fname))

            print(f"{rs:>6} {cs:>6} | {m['lr_diff']:>8.3f} {m['row_banding_std']:>9.3f} "
                  f"{m['quad_std']:>9.3f}")

    print(f"\n★ Best: row_sigma={best['row_sigma']}, col_sigma={best['col_sigma']}")
    return best['row_sigma'], best['col_sigma']


# ── Main batch processing ─────────────────────────────────────────────────────

def process_directory(input_dir, output_dir, row_sigma=20, col_sigma=200,
                      final_sigma=500, save_comparison=True):
    """
    Process all images in input_dir, save results + side-by-side comparisons.
    """
    os.makedirs(output_dir, exist_ok=True)
    comp_dir = os.path.join(output_dir, '_comparisons')
    if save_comparison:
        os.makedirs(comp_dir, exist_ok=True)

    exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif')
    fnames = sorted(f for f in os.listdir(input_dir)
                    if f.lower().endswith(exts))

    if not fnames:
        print(f"[Error] No images found in {input_dir}")
        return

    print(f"\n[Shadow Removal] {len(fnames)} images")
    print(f"  row_sigma={row_sigma}, col_sigma={col_sigma}, final_sigma={final_sigma}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}\n")

    results = []

    for fname in tqdm(fnames):
        img_np = np.array(Image.open(os.path.join(input_dir, fname)).convert('L'))
        corrected = remove_shadow(img_np, row_sigma, col_sigma, final_sigma)

        Image.fromarray(corrected, mode='L').save(os.path.join(output_dir, fname))

        if save_comparison:
            side = compare_side_by_side(img_np, corrected)
            Image.fromarray(side, mode='L').save(
                os.path.join(comp_dir, fname.rsplit('.', 1)[0] + '_compare.png'))

        m_in  = metrics(img_np)
        m_out = metrics(corrected)
        results.append({
            'file'            : fname,
            'lr_diff_before'  : round(m_in['lr_diff'],   3),
            'lr_diff_after'   : round(m_out['lr_diff'],  3),
            'banding_before'  : round(m_in['row_banding_std'],  3),
            'banding_after'   : round(m_out['row_banding_std'], 3),
            'quad_std_before' : round(m_in['quad_std'],  3),
            'quad_std_after'  : round(m_out['quad_std'], 3),
        })

    # ── Print summary ─────────────────────────────────────────────────────────
    lr_b = np.mean([r['lr_diff_before']  for r in results])
    lr_a = np.mean([r['lr_diff_after']   for r in results])
    bb   = np.mean([r['banding_before']  for r in results])
    ba   = np.mean([r['banding_after']   for r in results])

    print(f"\n{'='*55}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"{'':20s} {'Before':>10} {'After':>10} {'Target':>10}")
    print(f"{'L/R brightness diff':20s} {lr_b:>10.3f} {lr_a:>10.3f} {'< 0.02':>10}")
    print(f"{'Row banding std':20s} {bb:>10.3f} {ba:>10.3f} {'lower':>10}")
    print(f"\nComparisons saved: {comp_dir}")
    print(f"{'='*55}")

    import csv
    csv_path = os.path.join(output_dir, '_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Metrics CSV: {csv_path}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))

    CONCISED = os.path.join(
        BASE,
        'TRI_Dataset-20260511T190518Z-3-001',
        'TRI_Dataset_Concised'
    )
    OUTPUT = os.path.join(BASE, 'results', 'classical_TRI_concised')

    # ── STEP 1: Sigma sweep on one J1 image to find best settings ────────────
    # Saves side-by-side comparisons for every row/col sigma combo.
    # Review results/sigma_sweep/ and note the best row_sigma + col_sigma.
    # Then set them in STEP 2 below and comment this block out.

    sample = os.path.join(CONCISED, 'J1_000.jpg')
    best_row, best_col = sigma_sweep(
        sample,
        output_dir=os.path.join(BASE, 'results', 'sigma_sweep'),
        row_sigmas=[5, 10, 20, 40],
        col_sigmas=[100, 200, 400],
    )

    # ── STEP 2: Process all 32 images with the best settings ─────────────────
    process_directory(
        input_dir=CONCISED,
        output_dir=OUTPUT,
        row_sigma=best_row,    # override manually if sweep result looks wrong
        col_sigma=best_col,
        final_sigma=600,
        save_comparison=True,  # saves before|after images in _comparisons/
    )

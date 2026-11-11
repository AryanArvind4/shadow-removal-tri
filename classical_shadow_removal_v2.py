"""
Classical Shadow Removal V2 for TRI PCB X-ray Components
========================================================

This is an upgraded, non-generative shadow removal testbench for the
concised TRI dataset only:

    TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset_Concised

It does NOT modify classical_shadow_removal.py.

Goal:
    Remove low-frequency metal-sheet / illumination shadows from J1, J4,
    U29, and U30 X-ray component images while preserving solder balls,
    pads, traces, and component structure.

Why V2:
    The first Gaussian flat-field script was too simple. It estimated the
    shadow field directly from the image, so bright solder balls and pads
    leaked into the background estimate. V2 adds:

    - component-aware parameters
    - structure masking before background estimation
    - log-domain homomorphic correction
    - row/column band correction
    - seam-aware piecewise left/right balancing
    - per-image comparison sheets

Run:
    python classical_shadow_removal_v2.py

Outputs:
    results/classical_v2/
      gaussian_flatfield/
      masked_log_flatfield/
      row_col_balanced/
      hybrid/
      comparison_sheets/
      metrics.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    median_filter,
    zoom,
    sobel,
)
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class ComponentParams:
    """Tunable parameters for one component family."""

    sigma_frac: float
    mask_percentile: float
    median_frac: float
    row_sigma_frac: float
    col_sigma_frac: float
    seam_blend_frac: float
    correction_strength: float = 1.0
    aggressive_sigma_frac: float = 0.055
    contrast_percentiles: Tuple[float, float] = (0.5, 99.5)
    w_mean: float = 0.45
    w_zscore: float = 0.35
    w_retinex: float = 0.20


COMPONENT_PARAMS: Dict[str, ComponentParams] = {
    # Large connector / grid-like images. Need very broad illumination fields.
    "J1": ComponentParams(
        sigma_frac=0.070,
        mask_percentile=78.0,
        median_frac=0.035,
        row_sigma_frac=0.060,
        col_sigma_frac=0.060,
        seam_blend_frac=0.020,
        correction_strength=1.65,
        aggressive_sigma_frac=0.040,
        w_mean=0.10, w_zscore=0.80, w_retinex=0.10,
    ),
    "J4": ComponentParams(
        sigma_frac=0.070,
        mask_percentile=78.0,
        median_frac=0.035,
        row_sigma_frac=0.060,
        col_sigma_frac=0.060,
        seam_blend_frac=0.020,
        correction_strength=1.65,
        aggressive_sigma_frac=0.040,
        w_mean=0.10, w_zscore=0.80, w_retinex=0.10,
    ),
    # Smaller component crops. Slightly tighter background estimation.
    "U29": ComponentParams(
        sigma_frac=0.090,
        mask_percentile=76.0,
        median_frac=0.045,
        row_sigma_frac=0.080,
        col_sigma_frac=0.080,
        seam_blend_frac=0.030,
        correction_strength=1.85,
        aggressive_sigma_frac=0.060,
        w_mean=0.50, w_zscore=0.40, w_retinex=0.10,
    ),
    "U30": ComponentParams(
        sigma_frac=0.090,
        mask_percentile=76.0,
        median_frac=0.045,
        row_sigma_frac=0.080,
        col_sigma_frac=0.080,
        seam_blend_frac=0.030,
        correction_strength=1.85,
        aggressive_sigma_frac=0.060,
        w_mean=0.50, w_zscore=0.40, w_retinex=0.10,
    ),
}

DEFAULT_PARAMS = ComponentParams(
    sigma_frac=0.100,
    mask_percentile=82.0,
    median_frac=0.040,
    row_sigma_frac=0.070,
    col_sigma_frac=0.070,
    seam_blend_frac=0.025,
    correction_strength=1.60,
    aggressive_sigma_frac=0.055,
 )

MAX_BACKGROUND_SIZE = 420


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for(filename: str) -> ComponentParams:
    return COMPONENT_PARAMS.get(component_from_name(filename), DEFAULT_PARAMS)


def read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def save_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8(img), mode="L").save(path)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def odd_kernel(size: float, minimum: int = 9) -> int:
    k = int(round(size))
    k = max(k, minimum)
    if k % 2 == 0:
        k += 1
    return k


def resize_float(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a float image using scipy zoom, preserving numeric range."""
    h, w = img.shape
    out_h, out_w = shape
    if (h, w) == (out_h, out_w):
        return img.astype(np.float32, copy=True)
    factors = (out_h / h, out_w / w)
    return zoom(img, factors, order=1, mode="reflect").astype(np.float32)


def downsample_for_background(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Shadow fields are very low-frequency, so estimating them at full resolution
    is unnecessary and slow. Large J1/J4 images become fast after this step.
    """
    h, w = img.shape
    longest = max(h, w)
    if longest <= MAX_BACKGROUND_SIZE:
        return img.astype(np.float32, copy=True), 1.0

    scale = MAX_BACKGROUND_SIZE / float(longest)
    small_shape = (max(32, int(round(h * scale))), max(32, int(round(w * scale))))
    return resize_float(img, small_shape), scale


def robust_rescale_to_input(
    corrected: np.ndarray,
    original: np.ndarray,
    percentiles: Tuple[float, float],
) -> np.ndarray:
    """Match robust contrast to the input without amplifying outliers."""
    lo_o, hi_o = np.percentile(original, percentiles)
    lo_c, hi_c = np.percentile(corrected, percentiles)
    if hi_c - lo_c < 1e-6:
        return np.clip(corrected, 0, 255)

    out = (corrected - lo_c) / (hi_c - lo_c)
    out = out * (hi_o - lo_o) + lo_o

    # Match mean gently so the result does not look washed out.
    out += original.mean() - out.mean()
    return np.clip(out, 0, 255)


def robust_stretch(img: np.ndarray, percentiles: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Stretch one image to [0, 255] using robust percentiles."""
    lo, hi = np.percentile(img, percentiles)
    if hi - lo < 1e-6:
        return np.clip(img, 0, 255)
    out = (img - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255)


def mean_std_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match global mean/std only, without restoring spatial shadows."""
    img_mean = float(np.mean(img))
    img_std = float(np.std(img)) + 1e-6
    ref_mean = float(np.mean(reference))
    ref_std = float(np.std(reference)) + 1e-6
    return np.clip((img - img_mean) / img_std * ref_std + ref_mean, 0, 255)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def structure_mask(img: np.ndarray, percentile: float) -> np.ndarray:
    """
    Mask bright/dark structure so solder balls, pads, and edges do not become
    part of the smooth shadow-field estimate.

    Returns True for structure pixels to ignore.
    """
    smooth = gaussian_filter(img, sigma=max(min(img.shape) * 0.008, 2.0))
    highpass = np.abs(img - smooth)
    grad = gradient_magnitude(img)

    hp_thr = np.percentile(highpass, percentile)
    grad_thr = np.percentile(grad, percentile)
    mask = (highpass > hp_thr) | (grad > grad_thr)

    # Expand around solder-ball rims and component edges.
    dilations = 2 if min(img.shape) < 900 else 3
    for _ in range(dilations):
        mask = binary_dilation(mask)
    return mask


def masked_gaussian(img: np.ndarray, ignore_mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Normalized convolution. Ignores structure pixels while estimating a smooth
    background. This avoids solder-ball rings contaminating the illumination.
    """
    valid = (~ignore_mask).astype(np.float32)
    weighted = gaussian_filter(img * valid, sigma=sigma, mode="reflect")
    weights = gaussian_filter(valid, sigma=sigma, mode="reflect")
    fallback = gaussian_filter(img, sigma=sigma, mode="reflect")
    estimate = weighted / np.maximum(weights, 1e-4)
    estimate[weights < 1e-3] = fallback[weights < 1e-3]
    return estimate


def estimate_shadow_field(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    h, w = img.shape
    small, scale = downsample_for_background(img)
    sh, sw = small.shape

    sigma = max(min(sh, sw) * params.sigma_frac, 4.0)
    med_k = min(odd_kernel(min(sh, sw) * params.median_frac, minimum=5), 17)

    mask = structure_mask(small, params.mask_percentile)

    # Median prefilter suppresses small structures before masked blur.
    softened = median_filter(small, size=med_k, mode="reflect")
    field = masked_gaussian(softened, mask, sigma=sigma)

    # Make sure the field stays very low-frequency.
    field = gaussian_filter(field, sigma=max(sigma * 0.35, 6.0), mode="reflect")
    if scale != 1.0:
        field = resize_float(field, (h, w))
        field = gaussian_filter(field, sigma=max(min(h, w) * 0.004, 2.0), mode="reflect")
    return np.maximum(field, 1.0)


def estimate_shadow_field_with_sigma(
    img: np.ndarray,
    params: ComponentParams,
    sigma_frac: float,
) -> np.ndarray:
    """Same estimator as estimate_shadow_field, but with a chosen scale."""
    h, w = img.shape
    small, scale = downsample_for_background(img)
    sh, sw = small.shape

    sigma = max(min(sh, sw) * sigma_frac, 3.0)
    med_k = min(odd_kernel(min(sh, sw) * params.median_frac, minimum=5), 15)
    mask = structure_mask(small, params.mask_percentile)
    softened = median_filter(small, size=med_k, mode="reflect")
    field = masked_gaussian(softened, mask, sigma=sigma)
    field = gaussian_filter(field, sigma=max(sigma * 0.20, 3.0), mode="reflect")

    if scale != 1.0:
        field = resize_float(field, (h, w))
        field = gaussian_filter(field, sigma=max(min(h, w) * 0.0025, 1.5), mode="reflect")
    return np.maximum(field, 1.0)


def gaussian_flatfield(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    h, w = img.shape
    small, scale = downsample_for_background(img)
    sh, sw = small.shape
    sigma = max(min(sh, sw) * params.sigma_frac, 4.0)
    field = gaussian_filter(small, sigma=sigma, mode="reflect")
    if scale != 1.0:
        field = resize_float(field, (h, w))
        field = gaussian_filter(field, sigma=max(min(h, w) * 0.004, 2.0), mode="reflect")
    corrected = img / np.maximum(field, 1.0) * np.mean(field)
    return robust_rescale_to_input(corrected, img, params.contrast_percentiles)


def masked_log_flatfield(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Homomorphic correction:
        log(output) = log(input) - log(shadow_field) + mean(log(shadow_field))
    """
    eps = 1.0
    log_img = np.log(img + eps)
    log_field = np.log(estimate_shadow_field(img, params) + eps)
    corrected = np.exp(log_img - log_field + np.mean(log_field)) - eps
    return robust_rescale_to_input(corrected, img, params.contrast_percentiles)


def strong_log_flatfield(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Stronger log-domain correction. If the visible shadow survives V2's
    conservative correction, this deliberately over-corrects the estimated
    illumination field.
    """
    eps = 1.0
    log_img = np.log(img + eps)
    field = estimate_shadow_field_with_sigma(img, params, params.aggressive_sigma_frac)
    log_field = np.log(field + eps)
    shadow = log_field - np.mean(log_field)
    corrected = np.exp(log_img - params.correction_strength * shadow) - eps
    return robust_rescale_to_input(corrected, img, params.contrast_percentiles)


def additive_flatfield(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Some reconstructed X-ray images behave more like additive shading than
    pure multiplicative illumination. This removes the smooth field by
    subtraction:

        output = image - strength * (field - mean(field))
    """
    field = estimate_shadow_field_with_sigma(img, params, params.aggressive_sigma_frac)
    shadow = field - np.mean(field)
    corrected = img - params.correction_strength * shadow
    return robust_rescale_to_input(corrected, img, params.contrast_percentiles)


def additive_log_combo(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Blend additive and log corrections. This is intentionally stronger than
    the original hybrid and should visibly change the metal-sheet shadow.
    """
    add = additive_flatfield(img, params)
    log = strong_log_flatfield(img, params)
    corrected = 0.55 * add + 0.45 * log
    corrected = seam_balance(corrected, params)
    corrected = smooth_profile_correction(corrected, params, correct_rows=True, correct_cols=True)
    return robust_rescale_to_input(corrected, img, params.contrast_percentiles)


def local_mean_equalized(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Strong local background removal:

        output = image - local_mean(image) + global_mean

    This treats the metal-sheet pattern as a local additive background. It is
    intentionally more aggressive than flat-field division and should visibly
    reduce rectangular shadows.
    """
    h, w = img.shape
    sigma = max(min(h, w) * params.aggressive_sigma_frac, 10.0)
    background = gaussian_filter(img, sigma=sigma, mode="reflect")
    corrected = img - params.correction_strength * (background - np.mean(background))
    corrected = seam_balance(corrected, params)
    return mean_std_match(corrected, img)


def local_zscore_equalized(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Local brightness and contrast normalization. This is often better for X-ray
    stacks where the shadow darkens a region and also suppresses local contrast.

        output = (image - local_mean) / local_std

    Then match global mean/std back to the input.
    """
    h, w = img.shape
    sigma = max(min(h, w) * params.aggressive_sigma_frac, 10.0)
    local_mean = gaussian_filter(img, sigma=sigma, mode="reflect")
    local_sq_mean = gaussian_filter(img * img, sigma=sigma, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 4.0)
    local_std = np.sqrt(local_var)

    z = (img - local_mean) / local_std
    corrected = z * np.std(img) + np.mean(img)
    corrected = seam_balance(corrected, params)
    return np.clip(corrected, 0, 255)


def multi_scale_retinex_like(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Multi-scale log high-pass correction. Uses several background scales so it
    can attack both broad gradients and medium rectangular sheet shadows.
    """
    h, w = img.shape
    eps = 1.0
    log_img = np.log(img + eps)
    scales = [
        max(min(h, w) * params.aggressive_sigma_frac * 0.45, 5.0),
        max(min(h, w) * params.aggressive_sigma_frac * 0.90, 10.0),
        max(min(h, w) * params.aggressive_sigma_frac * 1.80, 18.0),
    ]

    detail_sum = np.zeros_like(log_img)
    for sigma in scales:
        background = gaussian_filter(log_img, sigma=sigma, mode="reflect")
        detail_sum += log_img - background
    detail = detail_sum / len(scales)

    corrected = np.exp(np.mean(log_img) + params.correction_strength * detail) - eps
    corrected = seam_balance(corrected, params)
    return mean_std_match(corrected, img)


def aggressive_local_combo(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Current experimental best candidate:
    local mean removal + local z-score normalization. This should make a visible
    dent in the sheet shadow. If it damages solder balls, reduce strength later.
    """
    mean_eq = local_mean_equalized(img, params)
    z_eq = local_zscore_equalized(img, params)
    ret = multi_scale_retinex_like(img, params)
    corrected = params.w_mean * mean_eq + params.w_zscore * z_eq + params.w_retinex * ret
    corrected = smooth_profile_correction(corrected, params, correct_rows=True, correct_cols=True)
    return mean_std_match(corrected, img)


def smooth_profile_correction(
    img: np.ndarray,
    params: ComponentParams,
    correct_rows: bool = True,
    correct_cols: bool = True,
) -> np.ndarray:
    """
    Correct scanner-like row bands and column gradients using smooth 1D
    profiles. This is intentionally low-frequency only.
    """
    out = img.astype(np.float64).copy()
    global_mean = float(np.mean(out))
    h, w = out.shape

    if correct_rows:
        row_sigma = max(h * params.row_sigma_frac, 8.0)
        row_profile = gaussian_filter(out.mean(axis=1), sigma=row_sigma, mode="reflect")
        out *= global_mean / np.maximum(row_profile[:, None], 1.0)

    if correct_cols:
        col_sigma = max(w * params.col_sigma_frac, 8.0)
        col_profile = gaussian_filter(out.mean(axis=0), sigma=col_sigma, mode="reflect")
        out *= global_mean / np.maximum(col_profile[None, :], 1.0)

    return robust_rescale_to_input(out, img, params.contrast_percentiles)
    


def detect_vertical_seam(img: np.ndarray) -> Optional[int]:
    """
    Finds a strong vertical panel boundary, common in J1/J4 stitched images.
    Returns an x coordinate or None.
    """
    h, w = img.shape
    if w < 900:
        return None

    col_profile = gaussian_filter(img.mean(axis=0), sigma=max(w * 0.01, 8.0))
    grad = np.abs(np.gradient(col_profile))

    # Ignore outer margins where border darkening is common.
    lo, hi = int(w * 0.20), int(w * 0.80)
    if hi <= lo:
        return None
    x = lo + int(np.argmax(grad[lo:hi]))

    if grad[x] < np.percentile(grad[lo:hi], 97.0):
        return None
    return x


def seam_balance(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    If a vertical seam exists, balance the left and right halves before the
    final profile correction. This targets the obvious left/right metal sheet
    boundary without moving structure.
    """
    seam = detect_vertical_seam(img)
    if seam is None:
        return img

    out = img.astype(np.float64).copy()
    h, w = out.shape
    blend = max(int(w * params.seam_blend_frac), 8)

    left = out[:, :seam]
    right = out[:, seam:]
    if left.size == 0 or right.size == 0:
        return img

    target = float(np.mean(out))
    left_scale = target / max(float(left.mean()), 1.0)
    right_scale = target / max(float(right.mean()), 1.0)
    out[:, :seam] *= left_scale
    out[:, seam:] *= right_scale

    # Feather across seam to avoid a new hard boundary.
    x0 = max(seam - blend, 0)
    x1 = min(seam + blend, w)
    if x1 > x0:
        alpha = np.linspace(0.0, 1.0, x1 - x0, dtype=np.float64)[None, :]
        original_band = img[:, x0:x1].astype(np.float64)
        corrected_band = out[:, x0:x1]
        out[:, x0:x1] = 0.35 * original_band + 0.65 * corrected_band
        out[:, x0:x1] = (
            (1 - alpha) * out[:, x0:x1]
            + alpha * out[:, x0:x1]
        )

    return robust_rescale_to_input(out, img, params.contrast_percentiles)


def row_col_balanced(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    out = seam_balance(img, params)
    out = smooth_profile_correction(out, params, correct_rows=True, correct_cols=True)
    return robust_rescale_to_input(out, img, params.contrast_percentiles)


def hybrid(img: np.ndarray, params: ComponentParams) -> np.ndarray:
    """
    Recommended candidate:
    1. log-domain masked flat-field removes smooth metal-sheet shadow
    2. seam balancing handles panel boundary
    3. row/column profiles remove residual bands/gradients
    """
    out = masked_log_flatfield(img, params)
    out = seam_balance(out, params)
    out = smooth_profile_correction(out, params, correct_rows=True, correct_cols=True)
    return robust_rescale_to_input(out, img, params.contrast_percentiles)


METHODS = {
    "gaussian_flatfield": gaussian_flatfield,
    "masked_log_flatfield": masked_log_flatfield,
    "row_col_balanced": row_col_balanced,
    "hybrid": hybrid,
    "strong_log": strong_log_flatfield,
    "additive_flatfield": additive_flatfield,
    "additive_log_combo": additive_log_combo,
    "local_mean_eq": local_mean_equalized,
    "local_zscore_eq": local_zscore_equalized,
    "multi_retinex": multi_scale_retinex_like,
    "aggressive_local": aggressive_local_combo,
}

FOCUSED_METHODS = [
    "local_mean_eq",
    "local_zscore_eq",
    "multi_retinex",
    "aggressive_local",
]


def lr_diff(img: np.ndarray) -> float:
    mid = img.shape[1] // 2
    return float(abs(img[:, :mid].mean() - img[:, mid:].mean()))


def quadrant_std(img: np.ndarray) -> float:
    h, w = img.shape
    hm, wm = h // 2, w // 2
    means = [
        img[:hm, :wm].mean(),
        img[:hm, wm:].mean(),
        img[hm:, :wm].mean(),
        img[hm:, wm:].mean(),
    ]
    return float(np.std(means))


def row_band_std(img: np.ndarray) -> float:
    row_means = img.mean(axis=1)
    smooth = gaussian_filter(row_means, sigma=max(img.shape[0] * 0.03, 4.0))
    return float(np.std(smooth))


def edge_energy_ratio(original: np.ndarray, corrected: np.ndarray) -> float:
    e0 = np.mean(gradient_magnitude(original)) + 1e-6
    e1 = np.mean(gradient_magnitude(corrected)) + 1e-6
    return float(e1 / e0)


def metric_row(filename: str, method: str, original: np.ndarray, corrected: np.ndarray) -> Dict[str, object]:
    q_before = quadrant_std(original)
    q_after = quadrant_std(corrected)
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "lr_diff_before": lr_diff(original),
        "lr_diff_after": lr_diff(corrected),
        "quad_std_before": q_before,
        "quad_std_after": q_after,
        "quad_reduction": (q_before - q_after) / (q_before + 1e-8),
        "row_band_std_before": row_band_std(original),
        "row_band_std_after": row_band_std(corrected),
        "edge_energy_ratio": edge_energy_ratio(original, corrected),
    }


def label_image(img: np.ndarray, label: str, width: int = 260) -> Image.Image:
    pil = Image.fromarray(to_uint8(img), mode="L").convert("RGB")
    scale = width / pil.width
    height = max(1, int(round(pil.height * scale)))
    pil = pil.resize((width, height), Image.BICUBIC)

    label_h = 28
    canvas = Image.new("RGB", (width, height + label_h), (20, 20, 20))
    canvas.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(235, 235, 235))
    return canvas


def make_contact_sheet(
    output_path: str,
    filename: str,
    original: np.ndarray,
    results: Dict[str, np.ndarray],
) -> None:
    panels = [label_image(original, f"input: {filename}")]
    for name, img in results.items():
        panels.append(label_image(img, name))

    width = panels[0].width
    height = max(p.height for p in panels)
    sheet = Image.new("RGB", (width * len(panels), height), (30, 30, 30))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * width, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)


def make_focused_contact_sheet(
    output_path: str,
    filename: str,
    original: np.ndarray,
    results: Dict[str, np.ndarray],
) -> None:
    focused = {name: results[name] for name in FOCUSED_METHODS if name in results}
    make_contact_sheet(output_path, filename, original, focused)


def image_paths(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def ensure_dirs(base_output: str, method_names: Iterable[str]) -> None:
    os.makedirs(base_output, exist_ok=True)
    for method in method_names:
        os.makedirs(os.path.join(base_output, method), exist_ok=True)
    os.makedirs(os.path.join(base_output, "comparison_sheets"), exist_ok=True)
    os.makedirs(os.path.join(base_output, "comparison_sheets_focused"), exist_ok=True)


def process_concised(input_dir: str, output_dir: str) -> List[Dict[str, object]]:
    paths = image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    ensure_dirs(output_dir, METHODS.keys())
    all_metrics: List[Dict[str, object]] = []

    print(f"[V2] Input : {input_dir}")
    print(f"[V2] Output: {output_dir}")
    print(f"[V2] Images: {len(paths)}")
    print("[V2] Methods: " + ", ".join(METHODS.keys()))

    for path in tqdm(paths, desc="Classical V2"):
        filename = os.path.basename(path)
        params = params_for(filename)
        original = read_gray(path)
        results: Dict[str, np.ndarray] = {}

        for method_name, method_fn in METHODS.items():
            corrected = method_fn(original, params)
            results[method_name] = corrected
            save_gray(os.path.join(output_dir, method_name, filename), corrected)
            all_metrics.append(metric_row(filename, method_name, original, corrected))

        sheet_path = os.path.join(
            output_dir,
            "comparison_sheets",
            os.path.splitext(filename)[0] + "_comparison.jpg",
        )
        make_contact_sheet(sheet_path, filename, original, results)

        focused_sheet_path = os.path.join(
            output_dir,
            "comparison_sheets_focused",
            os.path.splitext(filename)[0] + "_focused.jpg",
        )
        make_focused_contact_sheet(focused_sheet_path, filename, original, results)

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    print_summary(all_metrics)
    print(f"[V2] Metrics saved: {metrics_path}")
    print(f"[V2] Review sheets: {os.path.join(output_dir, 'comparison_sheets')}")
    print(f"[V2] Focused sheets: {os.path.join(output_dir, 'comparison_sheets_focused')}")
    return all_metrics


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 72)
    print("CLASSICAL V2 SUMMARY")
    print("=" * 72)

    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        lr_after = np.mean([float(r["lr_diff_after"]) for r in subset])
        q_red = np.mean([float(r["quad_reduction"]) for r in subset]) * 100.0
        band_after = np.mean([float(r["row_band_std_after"]) for r in subset])
        edge_ratio = np.mean([float(r["edge_energy_ratio"]) for r in subset])
        print(
            f"{method:22s} | LR after: {lr_after:7.3f} | "
            f"quad red: {q_red:7.1f}% | row band: {band_after:7.3f} | "
            f"edge ratio: {edge_ratio:5.3f}"
        )

    print("=" * 72)
    print("Use visual sheets first. Metrics help, but solder-ball shape is king.")


def default_paths() -> Tuple[str, str]:
    base = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(
        base,
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )
    output_dir = os.path.join(base, "results", "classical_v2")
    return input_dir, output_dir


if __name__ == "__main__":
    INPUT_DIR, OUTPUT_DIR = default_paths()
    process_concised(INPUT_DIR, OUTPUT_DIR)

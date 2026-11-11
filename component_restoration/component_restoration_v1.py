"""
Component-Aware PCB X-ray Restoration V1
=======================================

This script starts the thesis-stage direction:

  - J4: remove horizontal seams / row banding.
  - U29: improve square-grid visibility and suppress inter-square noise.
  - J1/U30: apply a conservative generic shadow/band correction.

It intentionally lives in component_restoration/ so the newer experiments
are separated from the older root-level classical scripts.

Run from repo root:

    python component_restoration/component_restoration_v1.py

Input:

    TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset_Concised

Output:

    component_restoration/result/component_restoration_v1/
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    grey_closing,
    grey_opening,
    median_filter,
    sobel,
)
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class RestoreParams:
    row_artifact_strength: float = 1.15
    row_artifact_sigma: float = 10.0
    row_baseline_sigma: float = 58.0
    column_balance_strength: float = 0.35
    denoise_sigma: float = 1.35
    square_enhance_strength: float = 0.55
    background_smooth_sigma: float = 2.0
    generic_shadow_sigma_frac: float = 0.055


PARAMS: Dict[str, RestoreParams] = {
    "J4": RestoreParams(
        row_artifact_strength=0.65,
        row_artifact_sigma=5.0,
        row_baseline_sigma=72.0,
        column_balance_strength=0.0,
        generic_shadow_sigma_frac=0.050,
    ),
    "J1": RestoreParams(
        row_artifact_strength=1.00,
        row_artifact_sigma=9.0,
        row_baseline_sigma=70.0,
        column_balance_strength=0.35,
        generic_shadow_sigma_frac=0.050,
    ),
    "U29": RestoreParams(
        denoise_sigma=1.55,
        square_enhance_strength=0.85,
        background_smooth_sigma=2.35,
        generic_shadow_sigma_frac=0.075,
    ),
    "U30": RestoreParams(
        row_artifact_strength=0.70,
        row_artifact_sigma=8.0,
        row_baseline_sigma=45.0,
        column_balance_strength=0.30,
        generic_shadow_sigma_frac=0.075,
    ),
}

DEFAULT_PARAMS = RestoreParams()


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_output_dir() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "result",
        "component_restoration_v1",
    )


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for(filename: str) -> RestoreParams:
    return PARAMS.get(component_from_name(filename), DEFAULT_PARAMS)


def read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def save_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8(img), mode="L").save(path)


def robust_stretch_like(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match robust global contrast without restoring spatial artifacts."""
    lo_r, hi_r = np.percentile(reference, (0.4, 99.6))
    lo_i, hi_i = np.percentile(img, (0.4, 99.6))
    if hi_i - lo_i < 1e-6:
        return np.clip(img, 0, 255)
    out = (img - lo_i) / (hi_i - lo_i) * (hi_r - lo_r) + lo_r
    out += np.mean(reference) - np.mean(out)
    return np.clip(out, 0, 255)


def mean_std_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    img_std = float(np.std(img)) + 1e-6
    ref_std = float(np.std(reference)) + 1e-6
    out = (img - np.mean(img)) / img_std * ref_std + np.mean(reference)
    return np.clip(out, 0, 255)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def structure_mask(img: np.ndarray, percentile: float = 78.0, dilations: int = 2) -> np.ndarray:
    smooth = gaussian_filter(img, sigma=max(min(img.shape) * 0.01, 2.0))
    highpass = np.abs(img - smooth)
    grad = gradient_magnitude(img)
    mask = (highpass > np.percentile(highpass, percentile)) | (
        grad > np.percentile(grad, percentile)
    )
    for _ in range(dilations):
        mask = binary_dilation(mask)
    return mask


def correct_column_balance(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape
    profile = np.median(img, axis=0)
    smooth = gaussian_filter(profile, sigma=max(w * 0.035, 10.0), mode="reflect")
    target = float(np.median(smooth))
    gain = target / np.maximum(smooth, 1.0)
    corrected = img * (1.0 + strength * (gain[None, :] - 1.0))
    return np.clip(corrected, 0, 255)


def remove_horizontal_seams(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    """
    J4-focused row artifact removal.

    The horizontal seams are modeled as row-wise additive brightness offsets.
    We estimate a robust row profile with medians, separate seam-scale variation
    from broad vertical shading, and subtract only that row artifact.
    """
    row_profile = np.median(img, axis=1)
    seam_scale = gaussian_filter(
        row_profile,
        sigma=params.row_artifact_sigma,
        mode="reflect",
    )
    broad_baseline = gaussian_filter(
        row_profile,
        sigma=params.row_baseline_sigma,
        mode="reflect",
    )
    row_artifact = seam_scale - broad_baseline

    corrected = img - params.row_artifact_strength * row_artifact[:, None]
    corrected = correct_column_balance(corrected, params.column_balance_strength)
    return robust_stretch_like(corrected, img)


def remove_horizontal_seams_safe(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    """
    Safer J4 correction.

    The previous J4 outputs looked worse because broad contrast restoration and
    column balancing made the existing vertical shadow more visible. This
    method only estimates a 1D horizontal seam profile and subtracts a limited
    amount from each row. It deliberately avoids any left/right or global
    contrast manipulation.
    """
    row_profile = np.median(img, axis=1)

    # Medium-scale row variation is the seam; very broad variation is real
    # exposure/shading and should be left alone in this J4-specific pass.
    medium = gaussian_filter(row_profile, sigma=params.row_artifact_sigma, mode="reflect")
    broad = gaussian_filter(row_profile, sigma=params.row_baseline_sigma, mode="reflect")
    seam = medium - broad

    # Prevent one bad row from causing a visible over-correction.
    limit = np.percentile(np.abs(seam), 92.0)
    seam = np.clip(seam, -limit, limit)

    corrected = img - params.row_artifact_strength * seam[:, None]
    return mean_std_match(corrected, img)


def remove_horizontal_seams_frequency(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    """
    Alternative J4 branch using a row profile high-pass. This is not a 2D FFT
    notch yet; it is intentionally explainable and less likely to distort pads.
    """
    row_mean = np.mean(img, axis=1)
    row_med = np.median(img, axis=1)
    profile = 0.35 * row_mean + 0.65 * row_med
    low = gaussian_filter(profile, sigma=params.row_baseline_sigma, mode="reflect")
    high = profile - low
    high = gaussian_filter(high, sigma=params.row_artifact_sigma * 0.55, mode="reflect")
    corrected = img - params.row_artifact_strength * high[:, None]
    corrected = correct_column_balance(corrected, params.column_balance_strength)
    return robust_stretch_like(corrected, img)


def remove_horizontal_seams_safe_blend(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    """
    Ultra-conservative variant: apply the safe seam removal and blend it with
    the input. This is useful when reviewers prefer a small clean-up over a
    visibly processed image.
    """
    safe = remove_horizontal_seams_safe(img, params)
    blended = 0.65 * img + 0.35 * safe
    return np.clip(blended, 0, 255)


def square_pad_mask(img: np.ndarray) -> np.ndarray:
    """
    U29 square pads are relatively bright, compact structures. We combine
    top-hat response and gradients, then dilate so square edges are protected.
    """
    size = max(9, int(round(min(img.shape) * 0.045)))
    if size % 2 == 0:
        size += 1

    opened = grey_opening(img, size=(size, size), mode="reflect")
    white_tophat = img - opened
    grad = gradient_magnitude(img)

    mask = (white_tophat > np.percentile(white_tophat, 83.0)) | (
        grad > np.percentile(grad, 82.0)
    )
    for _ in range(2):
        mask = binary_dilation(mask)
    return mask


def restore_u29_square_grid(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    """
    U29-focused restoration:

    - Smooth inter-square background/noise.
    - Preserve square pad edges using a structure mask.
    - Add a controlled square-pad enhancement via top-hat response.
    """
    mask = square_pad_mask(img)

    background_smooth = gaussian_filter(
        img,
        sigma=params.background_smooth_sigma,
        mode="reflect",
    )
    median_smooth = median_filter(img, size=5, mode="reflect")
    denoised_background = 0.55 * background_smooth + 0.45 * median_smooth

    size = max(9, int(round(min(img.shape) * 0.05)))
    if size % 2 == 0:
        size += 1
    opened = grey_opening(img, size=(size, size), mode="reflect")
    square_response = img - opened
    square_response = np.clip(square_response, 0, np.percentile(square_response, 99.0))

    enhanced_structure = img + params.square_enhance_strength * square_response
    corrected = np.where(mask, enhanced_structure, denoised_background)

    # Remove very broad shadow after structure-aware cleanup.
    corrected = generic_shadow_correction(corrected, params, strength=0.45)
    return mean_std_match(corrected, img)


def restore_u29_background_clean(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    """
    More conservative U29 branch: suppress background texture first, then gently
    sharpen square/pad edges.
    """
    mask = square_pad_mask(img)
    clean_bg = gaussian_filter(img, sigma=params.denoise_sigma, mode="reflect")
    detail = img - gaussian_filter(img, sigma=1.1, mode="reflect")
    corrected = np.where(mask, img + 0.45 * detail, clean_bg)
    corrected = generic_shadow_correction(corrected, params, strength=0.35)
    return mean_std_match(corrected, img)


def generic_shadow_correction(
    img: np.ndarray,
    params: RestoreParams,
    strength: float = 0.65,
) -> np.ndarray:
    h, w = img.shape
    sigma = max(min(h, w) * params.generic_shadow_sigma_frac, 14.0)
    background = gaussian_filter(img, sigma=sigma, mode="reflect")
    corrected = img - strength * (background - np.mean(background))
    corrected = correct_column_balance(corrected, params.column_balance_strength)
    return np.clip(corrected, 0, 255)


def restore_generic(img: np.ndarray, params: RestoreParams) -> np.ndarray:
    corrected = generic_shadow_correction(img, params, strength=0.65)
    corrected = remove_horizontal_seams(corrected, params)
    return mean_std_match(corrected, img)


def dispatch_component_restoration(img: np.ndarray, filename: str) -> Dict[str, np.ndarray]:
    component = component_from_name(filename)
    params = params_for(filename)

    if component == "J4":
        return {
            "j4_horizontal_safe": remove_horizontal_seams_safe(img, params),
            "j4_safe_blend": remove_horizontal_seams_safe_blend(img, params),
            "j4_row_seam": remove_horizontal_seams(img, params),
            "j4_profile_highpass": remove_horizontal_seams_frequency(img, params),
        }
    if component == "U29":
        return {
            "u29_square_grid": restore_u29_square_grid(img, params),
            "u29_background_clean": restore_u29_background_clean(img, params),
            "generic_shadow": restore_generic(img, params),
        }

    return {
        "generic_shadow": restore_generic(img, params),
        "row_seam": remove_horizontal_seams(img, params),
    }


def lr_diff(img: np.ndarray) -> float:
    mid = img.shape[1] // 2
    return float(abs(np.mean(img[:, :mid]) - np.mean(img[:, mid:])))


def row_band_score(img: np.ndarray) -> float:
    row_profile = np.median(img, axis=1)
    low = gaussian_filter(row_profile, sigma=max(img.shape[0] * 0.10, 20.0))
    seam = row_profile - low
    return float(np.std(seam))


def edge_energy(img: np.ndarray) -> float:
    return float(np.mean(gradient_magnitude(img)))


def square_contrast_score(img: np.ndarray) -> float:
    """
    Proxy for U29 square quality: high compact top-hat response compared with
    residual background texture. Higher is usually better, but visual review
    still wins.
    """
    size = max(9, int(round(min(img.shape) * 0.05)))
    if size % 2 == 0:
        size += 1
    opened = grey_opening(img, size=(size, size), mode="reflect")
    closed = grey_closing(img, size=(size, size), mode="reflect")
    bright = np.percentile(img - opened, 95.0)
    bg_texture = np.std(closed - opened) + 1e-6
    return float(bright / bg_texture)


def metric_row(
    filename: str,
    method: str,
    original: np.ndarray,
    corrected: np.ndarray,
) -> Dict[str, object]:
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "lr_diff_before": lr_diff(original),
        "lr_diff_after": lr_diff(corrected),
        "row_band_before": row_band_score(original),
        "row_band_after": row_band_score(corrected),
        "edge_ratio": edge_energy(corrected) / (edge_energy(original) + 1e-6),
        "square_score_before": square_contrast_score(original),
        "square_score_after": square_contrast_score(corrected),
    }


def label_image(img: np.ndarray, label: str, width: int = 260) -> Image.Image:
    pil = Image.fromarray(to_uint8(img), mode="L").convert("RGB")
    scale = width / pil.width
    height = max(1, int(round(pil.height * scale)))
    pil = pil.resize((width, height), Image.BICUBIC)

    label_h = 28
    canvas = Image.new("RGB", (width, height + label_h), (18, 18, 18))
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
    sheet = Image.new("RGB", (width * len(panels), height), (28, 28, 28))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (idx * width, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)


def image_paths(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def ensure_output_dirs(output_dir: str, method_names: Iterable[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison_sheets"), exist_ok=True)
    for name in method_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def process(input_dir: str, output_dir: str) -> List[Dict[str, object]]:
    paths = image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    method_names = [
        "j4_horizontal_safe",
        "j4_safe_blend",
        "j4_row_seam",
        "j4_profile_highpass",
        "u29_square_grid",
        "u29_background_clean",
        "generic_shadow",
        "row_seam",
    ]
    ensure_output_dirs(output_dir, method_names)

    print(f"[Component V1] Input : {input_dir}")
    print(f"[Component V1] Output: {output_dir}")
    print(f"[Component V1] Images: {len(paths)}")

    rows: List[Dict[str, object]] = []
    for path in tqdm(paths, desc="Component restoration"):
        filename = os.path.basename(path)
        original = read_gray(path)
        results = dispatch_component_restoration(original, filename)

        for method, corrected in results.items():
            save_gray(os.path.join(output_dir, method, filename), corrected)
            rows.append(metric_row(filename, method, original, corrected))

        sheet_path = os.path.join(
            output_dir,
            "comparison_sheets",
            os.path.splitext(filename)[0] + "_component_v1.jpg",
        )
        make_contact_sheet(sheet_path, filename, original, results)

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"[Component V1] Metrics: {metrics_path}")
    print(f"[Component V1] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 78)
    print("COMPONENT RESTORATION V1 SUMMARY")
    print("=" * 78)

    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        row_band = np.mean([float(row["row_band_after"]) for row in subset])
        edge = np.mean([float(row["edge_ratio"]) for row in subset])
        square = np.mean([float(row["square_score_after"]) for row in subset])
        print(
            f"{method:22s} | row band: {row_band:7.3f} | "
            f"edge ratio: {edge:5.3f} | square score: {square:7.3f}"
        )
    print("=" * 78)
    print("Use sheets first. J4 should lose horizontal seams; U29 squares should read cleaner.")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)

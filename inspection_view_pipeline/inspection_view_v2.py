"""
Inspection-View PCB X-ray Enhancement V2
========================================

V2 responds to TRI feedback on inspection_dark V1:

- Solder regions should have more uniform grayscale after shadow removal.
- The V1 dark view can saturate solder regions, causing "no contrast".
- Boosted texture can look like false solder fragments.

This script keeps V1's useful inspection direction, but adds:

1. Gentler shadow flattening for a balanced grayscale output.
2. Softer high-contrast inspection rendering to avoid pure-white saturation.
3. Connected-component cleanup to remove tiny false fragments.
4. Simple metrics for saturation, fragment count, and feature uniformity.

Run from repo root:

    .\\.venv312\\Scripts\\python.exe .\\inspection_view_pipeline\\inspection_view_v2.py

Outputs:

    inspection_view_pipeline/result/inspection_view_v2/
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class InspectParams:
    background_frac: float
    local_frac: float
    top_hat_frac: float
    denoise_size: int
    threshold_percentile: float
    gain: float
    gamma: float
    min_fragment_area: int
    max_saturation: float
    gray_strength: float


PARAMS: Dict[str, InspectParams] = {
    "J1": InspectParams(0.095, 0.026, 0.012, 5, 75.0, 2.25, 0.82, 12, 0.88, 0.58),
    "J4": InspectParams(0.092, 0.024, 0.012, 5, 75.0, 2.30, 0.82, 12, 0.88, 0.60),
    "U29": InspectParams(0.135, 0.040, 0.025, 5, 73.0, 2.05, 0.88, 16, 0.86, 0.52),
    "U30": InspectParams(0.125, 0.038, 0.022, 5, 73.0, 1.95, 0.90, 14, 0.86, 0.50),
}

DEFAULT_PARAMS = InspectParams(0.110, 0.032, 0.020, 5, 74.0, 2.10, 0.86, 12, 0.87, 0.55)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "inspection_view_v2")


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for(filename: str) -> InspectParams:
    return PARAMS.get(component_from_name(filename), DEFAULT_PARAMS)


def read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def save_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8(img), mode="L").save(path)


def image_paths(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def odd_size(value: float, minimum: int = 5) -> int:
    size = max(minimum, int(round(value)))
    if size % 2 == 0:
        size += 1
    return size


def robust01(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.4) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_p, hi_p))
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def estimate_background(img: np.ndarray, params: InspectParams) -> np.ndarray:
    h, w = img.shape
    min_dim = min(h, w)
    sigma = max(min_dim * params.background_frac, 20.0)
    med_size = min(31, odd_size(min_dim * params.top_hat_frac, minimum=9))

    softened = cv2.medianBlur(to_uint8(img), med_size).astype(np.float32)
    background = cv2.GaussianBlur(softened.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.maximum(background, 1.0)


def balanced_flatten(img: np.ndarray, params: InspectParams) -> Tuple[np.ndarray, np.ndarray]:
    background = estimate_background(img, params)
    target = np.median(background)

    additive = img - (background - target)
    divisive = img / background * target
    flat = (1.0 - params.gray_strength) * additive + params.gray_strength * divisive

    # Keep the grayscale image in the same visual family as the original while
    # compressing broad shadow variation.
    lo, hi = np.percentile(img, (0.5, 99.5))
    f_lo, f_hi = np.percentile(flat, (0.5, 99.5))
    if f_hi > f_lo + 1e-6:
        flat = (flat - f_lo) / (f_hi - f_lo) * (hi - lo) + lo
    return np.clip(flat, 0, 255).astype(np.float32), background


def local_background_flatten(img: np.ndarray, params: InspectParams) -> np.ndarray:
    h, w = img.shape
    sigma = max(min(h, w) * params.local_frac, 6.0)
    local_mean = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_sq = cv2.GaussianBlur((img * img).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 4.0))
    return (img - local_mean) / local_std


def top_hat_features(img: np.ndarray, params: InspectParams) -> np.ndarray:
    h, w = img.shape
    size = min(41, odd_size(min(h, w) * params.top_hat_frac, minimum=7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    img32 = img.astype(np.float32)
    opened = cv2.morphologyEx(img32, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(img32, cv2.MORPH_CLOSE, kernel)
    white = np.clip(img32 - opened, 0, None)
    black = np.clip(closed - img32, 0, None)
    return white + 0.28 * black


def feature_score(flat: np.ndarray, params: InspectParams) -> np.ndarray:
    local = local_background_flatten(flat, params)
    top = top_hat_features(flat, params)
    grad = gradient_magnitude(flat)

    base01 = robust01(flat, 2.0, 99.2)
    local01 = robust01(local, 2.0, 99.4)
    top01 = robust01(top, 2.0, 99.3)
    grad01 = robust01(grad, 5.0, 99.4)

    score = 0.26 * base01 + 0.34 * local01 + 0.30 * top01 + 0.10 * grad01
    smooth = cv2.medianBlur((score * 255.0).astype(np.uint8), odd_size(params.denoise_size, minimum=3)).astype(np.float32) / 255.0
    return np.clip(0.70 * score + 0.30 * smooth, 0.0, 1.0).astype(np.float32)


def clean_feature_mask(score: np.ndarray, params: InspectParams) -> np.ndarray:
    threshold = np.percentile(score, params.threshold_percentile)
    mask = score > threshold

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(1, min(width, height))

        # Keep tiny round dots, but reject tiny irregular speckles and very small
        # line fragments that reviewers called out as solder-like artifacts.
        if area >= params.min_fragment_area or (area >= 5 and aspect <= 2.2):
            cleaned[labels == label] = 255

    return cleaned > 0


def soft_knee(x: np.ndarray, max_value: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    compressed = 1.0 - np.exp(-2.8 * x)
    compressed = compressed / (1.0 - np.exp(-2.8))
    return np.minimum(compressed, max_value)


def inspection_clean_view(flat: np.ndarray, params: InspectParams) -> Tuple[np.ndarray, np.ndarray]:
    score = feature_score(flat, params)
    mask = clean_feature_mask(score, params)

    threshold = np.percentile(score, params.threshold_percentile)
    lifted = np.clip((score - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    lifted = np.power(lifted, params.gamma)
    lifted = np.clip(params.gain * lifted, 0.0, 1.0)

    cleaned = np.where(mask, lifted, lifted * 0.18)
    cleaned = cv2.medianBlur((cleaned * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
    cleaned = soft_knee(cleaned, params.max_saturation)
    return cleaned * 255.0, mask.astype(np.float32) * 255.0


def balanced_gray_view(flat: np.ndarray, params: InspectParams) -> np.ndarray:
    score = feature_score(flat, params)
    feature = np.power(np.clip(score, 0.0, 1.0), 0.82)
    base = robust01(flat, 1.0, 99.2)
    # This is the "shadow-free grayscale" companion to the dark inspection view.
    # It keeps context and avoids turning solder regions into saturated blobs.
    out = 0.66 * base + 0.34 * feature
    out = np.power(np.clip(out, 0.0, 1.0), 0.92)
    return out * 255.0


def process_image_at_work_resolution(img: np.ndarray, params: InspectParams) -> Dict[str, np.ndarray]:
    flat, background = balanced_flatten(img, params)
    clean, mask = inspection_clean_view(flat, params)
    gray = balanced_gray_view(flat, params)
    residual = robust01(flat - np.median(flat), 1.0, 99.2) * 255.0

    return {
        "inspection_clean": clean,
        "balanced_gray": gray,
        "feature_mask_clean": mask,
        "background_field": robust01(background, 1.0, 99.0) * 255.0,
        "flattened_residual": residual,
    }


def process_image(img: np.ndarray, params: InspectParams) -> Dict[str, np.ndarray]:
    h, w = img.shape
    max_dim = max(h, w)
    work_max_dim = 1150

    if max_dim <= work_max_dim:
        return process_image_at_work_resolution(img, params)

    scale = work_max_dim / max_dim
    work_w = max(32, int(round(w * scale)))
    work_h = max(32, int(round(h * scale)))
    work_img = cv2.resize(img.astype(np.float32), (work_w, work_h), interpolation=cv2.INTER_AREA)
    work_outputs = process_image_at_work_resolution(work_img, params)

    outputs: Dict[str, np.ndarray] = {}
    for name, output in work_outputs.items():
        interp = cv2.INTER_NEAREST if "mask" in name else cv2.INTER_CUBIC
        outputs[name] = cv2.resize(output.astype(np.float32), (w, h), interpolation=interp)
    return outputs


def feature_density(img: np.ndarray) -> float:
    norm = robust01(img, 1.0, 99.5)
    return float(np.mean(norm > 0.25))


def contrast_score(img: np.ndarray) -> float:
    norm = robust01(img, 1.0, 99.5)
    bright = norm[norm > np.percentile(norm, 85.0)]
    dark = norm[norm < np.percentile(norm, 50.0)]
    if bright.size == 0 or dark.size == 0:
        return 0.0
    return float((bright.mean() - dark.mean()) / (dark.std() + 1e-6))


def saturation_ratio(img: np.ndarray) -> float:
    return float(np.mean(img >= 245.0))


def small_fragment_count(mask_img: np.ndarray) -> int:
    mask_u8 = (mask_img > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    count = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            count += 1
    return count


def feature_uniformity(img: np.ndarray, mask_img: np.ndarray) -> float:
    mask = mask_img > 0
    vals = img[mask]
    if vals.size < 20:
        return 0.0
    return float(np.std(vals) / (np.mean(vals) + 1e-6))


def metric_row(filename: str, method: str, output: np.ndarray, feature_mask: np.ndarray) -> Dict[str, object]:
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "feature_density": feature_density(output),
        "contrast_score": contrast_score(output),
        "saturation_ratio": saturation_ratio(output),
        "small_fragment_count": small_fragment_count(feature_mask),
        "feature_uniformity_cv": feature_uniformity(output, feature_mask),
    }


def label_image(img: np.ndarray, label: str, width: int = 300) -> Image.Image:
    pil = Image.fromarray(to_uint8(img), mode="L").convert("RGB")
    scale = width / pil.width
    height = max(1, int(round(pil.height * scale)))
    pil = pil.resize((width, height), Image.BICUBIC)

    label_h = 30
    canvas = Image.new("RGB", (width, height + label_h), (18, 18, 18))
    canvas.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=(235, 235, 235))
    return canvas


def make_contact_sheet(path: str, filename: str, original: np.ndarray, outputs: Dict[str, np.ndarray]) -> None:
    panels = [label_image(original, f"input: {filename}")]
    panels += [label_image(img, name) for name, img in outputs.items()]

    width = panels[0].width
    height = max(p.height for p in panels)
    sheet = Image.new("RGB", (width * len(panels), height), (28, 28, 28))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * width, 0))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def ensure_dirs(output_dir: str, method_names: Iterable[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison_sheets"), exist_ok=True)
    for name in method_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def process(input_dir: str, output_dir: str) -> List[Dict[str, object]]:
    paths = image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    method_names = [
        "inspection_clean",
        "balanced_gray",
        "feature_mask_clean",
        "background_field",
        "flattened_residual",
    ]
    ensure_dirs(output_dir, method_names)

    print(f"[Inspection V2] Input : {input_dir}")
    print(f"[Inspection V2] Output: {output_dir}")
    print(f"[Inspection V2] Images: {len(paths)}")

    rows: List[Dict[str, object]] = []
    for path in tqdm(paths, desc="Inspection V2"):
        filename = os.path.basename(path)
        img = read_gray(path)
        outputs = process_image(img, params_for(filename))
        feature_mask = outputs["feature_mask_clean"]

        for method, output in outputs.items():
            save_gray(os.path.join(output_dir, method, filename), output)
            rows.append(metric_row(filename, method, output, feature_mask))

        make_contact_sheet(
            os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_inspection_v2.jpg"),
            filename,
            img,
            outputs,
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Inspection V2] Metrics: {metrics_path}")
    print(f"[Inspection V2] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)

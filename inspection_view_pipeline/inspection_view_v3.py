"""
Inspection-View PCB X-ray Enhancement V3
========================================

V3 is defect-focused. V1/V2 made features easy to see, but the dark inspection
views were still too binary: solder regions could become flat white, hiding
voids and internal defects.

V3 keeps the inspection-style dark background, but preserves grayscale inside
detected solder/pad regions and adds a separate void-candidate map.

Run from repo root:

    .\\.venv312\\Scripts\\python.exe .\\inspection_view_pipeline\\inspection_view_v3.py

Outputs:

    inspection_view_pipeline/result/inspection_view_v3/
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
class Params:
    bg_sigma_frac: float
    local_sigma_frac: float
    feature_percentile: float
    min_area: int
    max_area_frac: float
    mask_dilate: int
    bg_suppression: float
    feature_floor: float
    feature_gain: float
    void_gain: float
    void_sigma_frac: float
    row_suppression: float
    gray_clahe_blend: float
    gray_gamma: float


PARAMS: Dict[str, Params] = {
    "J1": Params(0.095, 0.018, 77.0, 10, 0.045, 1, 0.12, 0.22, 0.86, 0.46, 0.010, 0.22, 0.30, 0.86),
    "J4": Params(0.092, 0.017, 77.0, 10, 0.045, 1, 0.12, 0.22, 0.86, 0.46, 0.010, 0.25, 0.30, 0.86),
    "U29": Params(0.135, 0.026, 76.0, 20, 0.060, 2, 0.07, 0.28, 0.76, 0.74, 0.018, 0.08, 0.24, 0.88),
    "U30": Params(0.125, 0.024, 77.0, 20, 0.065, 2, 0.07, 0.30, 0.74, 0.78, 0.018, 0.06, 0.22, 0.90),
}

DEFAULT_PARAMS = Params(0.110, 0.022, 75.0, 12, 0.060, 1, 0.10, 0.25, 0.80, 0.52, 0.014, 0.12, 0.25, 0.88)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "inspection_view_v3")


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for(filename: str) -> Params:
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


def odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    if size % 2 == 0:
        size += 1
    return size


def robust01(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.2) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_p, hi_p))
    if hi <= lo + 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_for_work(img: np.ndarray, work_max_dim: int = 1200) -> Tuple[np.ndarray, float]:
    h, w = img.shape
    scale = min(1.0, work_max_dim / max(h, w))
    if scale >= 1.0:
        return img.astype(np.float32), 1.0
    work = cv2.resize(
        img.astype(np.float32),
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return work, scale


def restore_size(img: np.ndarray, shape: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    h, w = shape
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=interp)


def estimate_background(img: np.ndarray, params: Params) -> np.ndarray:
    h, w = img.shape
    sigma = max(min(h, w) * params.bg_sigma_frac, 18.0)
    med = min(35, odd_size(min(h, w) * 0.010, 9))
    softened = cv2.medianBlur(to_uint8(img), med).astype(np.float32)
    bg = cv2.GaussianBlur(softened, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.maximum(bg, 1.0)


def shadow_flatten(img: np.ndarray, params: Params) -> Tuple[np.ndarray, np.ndarray]:
    bg = estimate_background(img, params)
    target = float(np.median(bg))
    additive = img - (bg - target)
    divisive = img / bg * target
    flat = 0.62 * additive + 0.38 * divisive
    return np.clip(flat, 0, 255).astype(np.float32), bg


def suppress_row_bands(flat: np.ndarray, params: Params) -> np.ndarray:
    if params.row_suppression <= 0:
        return flat

    h, _ = flat.shape
    row_profile = np.median(flat, axis=1).astype(np.float32)
    sigma = max(h * 0.055, 12.0)
    smooth = cv2.GaussianBlur(row_profile[:, None], (0, 0), sigmaX=0, sigmaY=sigma).ravel()
    band = row_profile - smooth
    corrected = flat - params.row_suppression * band[:, None]
    return np.clip(corrected, 0, 255).astype(np.float32)


def local_scores(flat: np.ndarray, params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = flat.shape
    sigma = max(min(h, w) * params.local_sigma_frac, 5.0)
    local_mean = cv2.GaussianBlur(flat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_sq = cv2.GaussianBlur(flat * flat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 4.0))
    z = (flat - local_mean) / local_std

    k = min(43, odd_size(min(h, w) * 0.016, 7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    top_hat = cv2.morphologyEx(flat.astype(np.float32), cv2.MORPH_TOPHAT, kernel)

    gx = cv2.Sobel(flat.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(flat.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    feature = 0.40 * robust01(z, 2.0, 99.3) + 0.42 * robust01(top_hat, 1.0, 99.2) + 0.18 * robust01(grad, 5.0, 99.2)
    feature = cv2.GaussianBlur(feature.astype(np.float32), (0, 0), sigmaX=0.7, sigmaY=0.7)
    return np.clip(feature, 0.0, 1.0), local_mean, local_std


def clean_feature_mask(feature: np.ndarray, params: Params) -> np.ndarray:
    h, w = feature.shape
    threshold = np.percentile(feature, params.feature_percentile)
    mask = (feature > threshold).astype(np.uint8) * 255

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros((h, w), dtype=np.uint8)
    max_area = int(h * w * params.max_area_frac)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < params.min_area or area > max_area:
            continue
        aspect = max(width, height) / max(1, min(width, height))
        if area < params.min_area * 2 and aspect > 3.2:
            continue
        cleaned[labels == label] = 255

    if params.mask_dilate > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * params.mask_dilate + 1, 2 * params.mask_dilate + 1),
        )
        cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=1)
    return cleaned > 0


def void_candidates(flat: np.ndarray, mask: np.ndarray, local_mean: np.ndarray, local_std: np.ndarray, params: Params) -> np.ndarray:
    h, w = flat.shape
    sigma = max(min(h, w) * params.void_sigma_frac, 4.0)
    smooth = cv2.GaussianBlur(flat, (0, 0), sigmaX=sigma, sigmaY=sigma)

    dark_residual = np.maximum(smooth - flat, 0.0)
    z_dark = np.maximum((local_mean - flat) / np.maximum(local_std, 2.0), 0.0)
    void = 0.62 * robust01(dark_residual, 45.0, 99.5) + 0.38 * robust01(z_dark, 45.0, 99.5)
    void = np.where(mask, void, void * 0.08)

    # Remove isolated single-pixel noise, but do not close dark holes away.
    void_u8 = to_uint8(void * 255.0)
    void_u8 = cv2.medianBlur(void_u8, 3)
    void = void_u8.astype(np.float32) / 255.0
    return np.power(np.clip(void, 0.0, 1.0), 0.68)


def render_defect_focused(flat: np.ndarray, feature: np.ndarray, mask: np.ndarray, void: np.ndarray, params: Params) -> np.ndarray:
    base = robust01(flat, 1.0, 99.2)
    feature_tone = 0.58 * base + 0.42 * feature
    feature_tone = np.power(np.clip(feature_tone, 0.0, 1.0), 0.84)

    background = params.bg_suppression * np.power(base, 1.25)
    solder = params.feature_floor + params.feature_gain * feature_tone
    solder = solder - params.void_gain * void

    out = np.where(mask, solder, background)
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigmaX=0.35, sigmaY=0.35)
    return np.clip(out, 0.0, 0.92) * 255.0


def render_inspection_clean(flat: np.ndarray, feature: np.ndarray, mask: np.ndarray, params: Params) -> np.ndarray:
    base = robust01(flat, 1.0, 99.2)
    feature_tone = 0.42 * base + 0.58 * np.power(np.clip(feature, 0.0, 1.0), 0.86)

    background = 0.055 * np.power(base, 1.35)
    solder = 0.18 + 0.70 * feature_tone
    out = np.where(mask, solder, background)
    out = cv2.bilateralFilter(out.astype(np.float32), 5, 0.045, 5)
    return np.clip(out, 0.0, 0.88) * 255.0


def render_balanced_gray(flat: np.ndarray, feature: np.ndarray, void: np.ndarray, params: Params) -> np.ndarray:
    base = robust01(flat, 1.0, 99.2)

    base_u8 = to_uint8(base * 255.0)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(base_u8).astype(np.float32) / 255.0
    denoised = cv2.bilateralFilter(base.astype(np.float32), 5, 0.055, 5)

    # Keep grayscale dominant. Feature boost is mild so internal void/defect
    # texture is not crushed into a binary white solder blob.
    out = (
        (0.70 - params.gray_clahe_blend * 0.35) * denoised
        + params.gray_clahe_blend * clahe
        + 0.16 * np.power(np.clip(feature, 0.0, 1.0), 0.90)
    )
    out = np.power(np.clip(out, 0.0, 1.0), params.gray_gamma)
    return np.clip(out, 0.0, 1.0) * 255.0


def process_work_image(img: np.ndarray, params: Params) -> Dict[str, np.ndarray]:
    flat, bg = shadow_flatten(img, params)
    flat = suppress_row_bands(flat, params)
    feature, local_mean, local_std = local_scores(flat, params)
    mask = clean_feature_mask(feature, params)
    void = void_candidates(flat, mask, local_mean, local_std, params)

    return {
        "inspection_clean": render_inspection_clean(flat, feature, mask, params),
        "balanced_gray_clean": render_balanced_gray(flat, feature, void, params),
        "feature_mask": mask.astype(np.float32) * 255.0,
        "background_field": robust01(bg, 1.0, 99.0) * 255.0,
    }


def process_image(img: np.ndarray, params: Params) -> Dict[str, np.ndarray]:
    original_shape = img.shape
    work, scale = resize_for_work(img)
    outputs = process_work_image(work, params)
    if scale >= 1.0:
        return outputs
    return {
        name: restore_size(value, original_shape, is_mask=("mask" in name))
        for name, value in outputs.items()
    }


def feature_density(img: np.ndarray) -> float:
    norm = robust01(img, 1.0, 99.5)
    return float(np.mean(norm > 0.25))


def saturation_ratio(img: np.ndarray) -> float:
    return float(np.mean(img >= 245.0))


def background_leak(img: np.ndarray, feature_mask: np.ndarray) -> float:
    bg = feature_mask <= 0
    if not np.any(bg):
        return 0.0
    return float(np.mean(img[bg] > 80.0))


def metric_row(filename: str, method: str, outputs: Dict[str, np.ndarray]) -> Dict[str, object]:
    img = outputs[method]
    mask = outputs["feature_mask"]
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "feature_density": feature_density(img),
        "saturation_ratio": saturation_ratio(img),
        "background_leak": background_leak(img, mask),
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
    order = [
        "inspection_clean",
        "balanced_gray_clean",
        "feature_mask",
        "background_field",
    ]
    panels = [label_image(original, f"input: {filename}")]
    panels += [label_image(outputs[name], name) for name in order]

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
        "balanced_gray_clean",
        "feature_mask",
        "background_field",
    ]
    ensure_dirs(output_dir, method_names)

    print(f"[Inspection V3] Input : {input_dir}")
    print(f"[Inspection V3] Output: {output_dir}")
    print(f"[Inspection V3] Images: {len(paths)}")

    rows: List[Dict[str, object]] = []
    for path in tqdm(paths, desc="Inspection V3"):
        filename = os.path.basename(path)
        img = read_gray(path)
        outputs = process_image(img, params_for(filename))

        for method, output in outputs.items():
            save_gray(os.path.join(output_dir, method, filename), output)
            rows.append(metric_row(filename, method, outputs))

        make_contact_sheet(
            os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_inspection_v3.jpg"),
            filename,
            img,
            outputs,
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Inspection V3] Metrics: {metrics_path}")
    print(f"[Inspection V3] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)

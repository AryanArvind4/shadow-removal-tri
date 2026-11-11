"""
Masked background-field restoration for TRI PCB X-ray components.

This is the first implementation of the new research direction:
estimate shadows/background only from pixels that do NOT look like real PCB
structure. The goal is to avoid the failure mode where solder balls, square pads,
and traces leak into the shadow field.

Outputs are written inside:
    masked_background_research/result/masked_background_v1

Default input is the concised TRI dataset.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Params:
    small_bg_frac: float
    large_bg_frac: float
    mask_percentile: float
    mask_dilate_frac: float
    inpaint_radius_frac: float
    correction_strength: float
    strong_strength: float
    row_strength: float
    inspection_gain: float


DEFAULT_PARAMS = Params(
    small_bg_frac=0.020,
    large_bg_frac=0.115,
    mask_percentile=84.0,
    mask_dilate_frac=0.010,
    inpaint_radius_frac=0.010,
    correction_strength=0.72,
    strong_strength=0.92,
    row_strength=0.28,
    inspection_gain=2.6,
)


PARAMS: Dict[str, Params] = {
    "J1": Params(0.018, 0.120, 84.0, 0.009, 0.010, 0.62, 0.82, 0.22, 2.4),
    "J4": Params(0.018, 0.125, 83.0, 0.008, 0.011, 0.70, 0.95, 0.36, 2.7),
    "U29": Params(0.026, 0.150, 81.5, 0.014, 0.012, 0.55, 0.72, 0.18, 2.2),
    "U30": Params(0.024, 0.135, 82.5, 0.012, 0.011, 0.58, 0.76, 0.20, 2.2),
}


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "masked_background_v1")


def component_from_name(filename: str) -> str:
    return os.path.basename(filename).split("_")[0].upper()


def params_for(filename: str) -> Params:
    return PARAMS.get(component_from_name(filename), DEFAULT_PARAMS)


def odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    if size % 2 == 0:
        size += 1
    return size


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def robust01(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.3) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_p, hi_p])
    if hi <= lo + 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def local_contrast(img: np.ndarray, params: Params) -> np.ndarray:
    h, w = img.shape
    min_dim = min(h, w)
    small = odd_size(min_dim * params.small_bg_frac, 9)
    large = odd_size(min_dim * params.large_bg_frac, 31)

    small_bg = cv2.GaussianBlur(img, (small, small), 0)
    large_bg = cv2.GaussianBlur(img, (large, large), 0)
    detail = np.abs(img - small_bg)
    mid_detail = np.abs(small_bg - large_bg)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    score = 0.52 * robust01(detail) + 0.30 * robust01(grad) + 0.18 * robust01(mid_detail)
    return robust01(score, 0.5, 99.5)


def structure_mask(img: np.ndarray, params: Params) -> np.ndarray:
    score = local_contrast(img, params)
    threshold = np.percentile(score, params.mask_percentile)
    mask = score >= threshold

    h, w = img.shape
    min_dim = min(h, w)
    close_size = odd_size(min_dim * 0.004, 3)
    dilate_size = odd_size(min_dim * params.mask_dilate_frac, 3)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.dilate(mask_u8, dilate_kernel, iterations=1)
    return mask_u8 > 0


def estimate_background_field(img: np.ndarray, mask: np.ndarray, params: Params) -> np.ndarray:
    h, w = img.shape
    max_dim = max(h, w)
    scale = min(1.0, 850.0 / max_dim)

    if scale < 1.0:
        small_w = max(32, int(round(w * scale)))
        small_h = max(32, int(round(h * scale)))
        work_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        work_mask = cv2.resize(
            mask.astype(np.uint8),
            (small_w, small_h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        work_img = img
        work_mask = mask

    wh, ww = work_img.shape
    min_dim = min(wh, ww)
    inpaint_radius = min(9, max(3, int(round(min_dim * params.inpaint_radius_frac))))
    large = min(151, odd_size(min_dim * params.large_bg_frac, 41))
    huge = min(251, odd_size(min_dim * (params.large_bg_frac * 1.8), 61))

    mask_u8 = (work_mask.astype(np.uint8) * 255)
    inpainted_u8 = cv2.inpaint(to_uint8(work_img), mask_u8, inpaint_radius, cv2.INPAINT_TELEA)
    inpainted = inpainted_u8.astype(np.float32) / 255.0

    median_size = min(15, odd_size(min_dim * 0.020, 7))
    denoised = cv2.medianBlur(to_uint8(inpainted), median_size).astype(np.float32) / 255.0
    field1 = cv2.GaussianBlur(denoised, (large, large), 0)
    field2 = cv2.GaussianBlur(denoised, (huge, huge), 0)
    field = 0.65 * field1 + 0.35 * field2

    floor = max(0.03, float(np.percentile(field, 1)))
    field = np.clip(field, floor, 1.0).astype(np.float32)
    if field.shape != img.shape:
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, floor, 1.0).astype(np.float32)


def remove_row_profile_from_background(img: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Suppress horizontal seams using row statistics from background pixels only."""
    h, _ = img.shape
    row_values = np.full(h, np.nan, dtype=np.float32)
    background = ~mask

    for y in range(h):
        vals = img[y, background[y]]
        if vals.size > 12:
            row_values[y] = np.median(vals)

    valid = np.flatnonzero(np.isfinite(row_values))
    if valid.size < 4:
        return img

    filled = np.interp(np.arange(h), valid, row_values[valid]).astype(np.float32)
    smooth_size = odd_size(h * 0.10, 21)
    smooth = cv2.GaussianBlur(filled[:, None], (1, smooth_size), 0).ravel()
    row_noise = filled - smooth
    corrected = img - strength * row_noise[:, None]
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def masked_flatten(img: np.ndarray, field: np.ndarray, strength: float) -> np.ndarray:
    target = float(np.median(field))
    corrected = img / np.maximum(field, 1e-4) * target
    corrected = robust01(corrected, 0.6, 99.4)
    original_scaled = robust01(img, 0.6, 99.4)
    return np.clip((1.0 - strength) * original_scaled + strength * corrected, 0.0, 1.0)


def natural_render(img: np.ndarray, mask: np.ndarray, field: np.ndarray, params: Params, strength: float) -> np.ndarray:
    flat = masked_flatten(img, field, strength)
    flat = remove_row_profile_from_background(flat, mask, params.row_strength)
    return robust01(flat, 0.5, 99.2)


def inspection_render(img: np.ndarray, mask: np.ndarray, field: np.ndarray, params: Params) -> np.ndarray:
    flat = natural_render(img, mask, field, params, params.strong_strength)

    h, w = img.shape
    min_dim = min(h, w)
    feature_kernel = odd_size(min_dim * 0.020, 9)
    bg_kernel = odd_size(min_dim * 0.075, 31)

    local_bg = cv2.GaussianBlur(flat, (bg_kernel, bg_kernel), 0)
    bright_features = np.maximum(flat - local_bg, 0.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feature_kernel, feature_kernel))
    top_hat = cv2.morphologyEx(to_uint8(flat), cv2.MORPH_TOPHAT, kernel).astype(np.float32) / 255.0

    gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    feature = 0.46 * robust01(bright_features) + 0.36 * robust01(top_hat) + 0.18 * robust01(grad)
    feature = np.clip(feature * params.inspection_gain, 0.0, 1.0)

    dark = 1.0 - np.exp(-3.2 * feature)
    dark = cv2.bilateralFilter(dark.astype(np.float32), 5, 0.05, 5)
    return robust01(dark, 1.0, 99.7)


def panel_step_metric(img: np.ndarray) -> float:
    h, w = img.shape
    left = img[:, int(w * 0.42):int(w * 0.49)]
    right = img[:, int(w * 0.51):int(w * 0.58)]
    return float(abs(np.median(left) - np.median(right)) * 255.0)


def row_band_metric(img: np.ndarray) -> float:
    row = np.median(img, axis=1)
    smooth = cv2.GaussianBlur(row[:, None].astype(np.float32), (1, odd_size(len(row) * 0.12, 21)), 0).ravel()
    return float(np.std(row - smooth) * 255.0)


def feature_contrast_metric(img: np.ndarray) -> float:
    score = local_contrast(img, DEFAULT_PARAMS)
    high = img[score > np.percentile(score, 90)]
    low = img[score < np.percentile(score, 45)]
    if high.size == 0 or low.size == 0:
        return 0.0
    return float((np.median(high) - np.median(low)) * 255.0)


def process_image(img: np.ndarray, filename: str) -> Dict[str, np.ndarray]:
    params = params_for(filename)
    mask = structure_mask(img, params)
    field = estimate_background_field(img, mask, params)

    natural = natural_render(img, mask, field, params, params.correction_strength)
    strong = natural_render(img, mask, field, params, params.strong_strength)
    inspection = inspection_render(img, mask, field, params)

    return {
        "masked_flat_natural": natural,
        "masked_flat_strong": strong,
        "inspection_masked": inspection,
        "debug_structure_mask": mask.astype(np.float32),
        "debug_background_field": robust01(field, 0.5, 99.5),
    }


def image_paths(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def save_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, to_uint8(img))


def label_image(img: np.ndarray, label: str, width: int = 300) -> Image.Image:
    arr = to_uint8(img)
    pil = Image.fromarray(arr, mode="L").convert("RGB")
    scale = width / pil.width
    height = max(1, int(round(pil.height * scale)))
    pil = pil.resize((width, height), Image.Resampling.BICUBIC)

    label_h = 28
    out = Image.new("RGB", (width, height + label_h), (20, 20, 20))
    out.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), label, fill=(235, 235, 235), font=font)
    return out


def make_contact_sheet(path: str, filename: str, original: np.ndarray, outputs: Dict[str, np.ndarray]) -> None:
    panels = [label_image(original, f"input {filename}")]
    for name in [
        "masked_flat_natural",
        "masked_flat_strong",
        "inspection_masked",
        "debug_structure_mask",
        "debug_background_field",
    ]:
        panels.append(label_image(outputs[name], name))

    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    sheet = Image.new("RGB", (width, height), (12, 12, 12))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def ensure_dirs(output_dir: str, method_names: Iterable[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison_sheets"), exist_ok=True)
    for name in method_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def metric_row(filename: str, method: str, original: np.ndarray, output: np.ndarray) -> Dict[str, object]:
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "panel_step": panel_step_metric(output),
        "row_band": row_band_metric(output),
        "feature_contrast": feature_contrast_metric(output),
        "input_panel_step": panel_step_metric(original),
        "input_row_band": row_band_metric(original),
    }


def process(input_dir: str, output_dir: str) -> List[Dict[str, object]]:
    paths = image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    method_names = [
        "masked_flat_natural",
        "masked_flat_strong",
        "inspection_masked",
        "debug_structure_mask",
        "debug_background_field",
    ]
    ensure_dirs(output_dir, method_names)

    print(f"[Masked Background V1] Input : {input_dir}")
    print(f"[Masked Background V1] Output: {output_dir}")
    print(f"[Masked Background V1] Images: {len(paths)}")
    print(f"[Masked Background V1] Methods: {', '.join(method_names)}")

    rows: List[Dict[str, object]] = []
    for path in tqdm(paths, desc="Masked background V1", ncols=90):
        filename = os.path.basename(path)
        img = read_gray(path)
        outputs = process_image(img, filename)

        for method, output in outputs.items():
            save_gray(os.path.join(output_dir, method, filename), output)
            if not method.startswith("debug_"):
                rows.append(metric_row(filename, method, img, output))

        make_contact_sheet(
            os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_masked_v1.jpg"),
            filename,
            img,
            outputs,
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Masked Background V1] Metrics: {metrics_path}")
    print(f"[Masked Background V1] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)

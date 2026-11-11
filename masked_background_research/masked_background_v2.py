"""
Masked background-field restoration V2.

Why V2 exists:
V1 used a hard structure mask plus inpainting. On TRI images the mask became too
large and inpainting produced a background field that still contained component
ghosts. V2 uses soft confidence weights instead:

    low local-detail pixels get high background confidence
    high local-detail pixels get low background confidence
    background = Gaussian(img * weight) / Gaussian(weight)

This avoids binary mask holes and makes the natural view much less aggressive.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Params:
    detail_frac: float
    field_frac: float
    detail_percentile: float
    weight_gamma: float
    additive_strength: float
    strong_strength: float
    row_strength: float
    natural_contrast: float
    inspection_gain: float


DEFAULT_PARAMS = Params(0.018, 0.120, 94.0, 2.8, 0.55, 0.76, 0.18, 1.08, 2.1)


PARAMS: Dict[str, Params] = {
    "J1": Params(0.016, 0.130, 94.0, 2.8, 0.48, 0.68, 0.16, 1.06, 2.0),
    "J4": Params(0.015, 0.145, 94.5, 2.6, 0.56, 0.82, 0.30, 1.08, 2.25),
    "U29": Params(0.024, 0.155, 93.0, 3.0, 0.42, 0.58, 0.12, 1.05, 1.9),
    "U30": Params(0.022, 0.140, 93.5, 3.0, 0.42, 0.60, 0.12, 1.05, 1.9),
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
    return os.path.join(os.path.dirname(__file__), "result", "masked_background_v2")


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


def robust01(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_p, hi_p])
    if hi <= lo + 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gentle_rescale_like_original(corrected: np.ndarray, original: np.ndarray, contrast: float) -> np.ndarray:
    orig_lo, orig_hi = np.percentile(original, [0.5, 99.5])
    corr_lo, corr_hi = np.percentile(corrected, [0.5, 99.5])
    if corr_hi <= corr_lo + 1e-6:
        return np.clip(corrected, 0.0, 1.0)

    out = (corrected - corr_lo) / (corr_hi - corr_lo)
    out = out * (orig_hi - orig_lo) + orig_lo
    mean = float(np.mean(out))
    out = mean + contrast * (out - mean)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def detail_score(img: np.ndarray, params: Params) -> np.ndarray:
    h, w = img.shape
    min_dim = min(h, w)
    detail_size = odd_size(min_dim * params.detail_frac, 7)
    larger_size = odd_size(min_dim * params.field_frac * 0.45, 21)

    local = cv2.GaussianBlur(img, (detail_size, detail_size), 0)
    broad = cv2.GaussianBlur(img, (larger_size, larger_size), 0)
    detail = np.abs(img - local)
    medium = np.abs(local - broad)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    return robust01(0.50 * robust01(detail) + 0.30 * robust01(grad) + 0.20 * robust01(medium), 0.5, 99.5)


def soft_background_weight(img: np.ndarray, params: Params) -> np.ndarray:
    score = detail_score(img, params)
    cutoff = np.percentile(score, params.detail_percentile)
    normalized = np.clip(score / max(cutoff, 1e-4), 0.0, 1.0)
    weight = (1.0 - normalized) ** params.weight_gamma

    # Keep a small floor so broad areas never become impossible to estimate.
    weight = np.clip(weight, 0.035, 1.0).astype(np.float32)

    # Do not let very dark borders dominate the background fit.
    dark = img < np.percentile(img, 0.8)
    weight[dark] *= 0.15
    return weight


def weighted_blur_field(img: np.ndarray, weight: np.ndarray, params: Params) -> np.ndarray:
    h, w = img.shape
    max_dim = max(h, w)
    scale = min(1.0, 900.0 / max_dim)

    if scale < 1.0:
        sw, sh = int(round(w * scale)), int(round(h * scale))
        work_img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        work_weight = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        work_img = img
        work_weight = weight

    wh, ww = work_img.shape
    min_dim = min(wh, ww)
    k1 = min(201, odd_size(min_dim * params.field_frac, 51))
    k2 = min(301, odd_size(min_dim * params.field_frac * 1.9, 81))

    def norm_blur(k: int) -> np.ndarray:
        num = cv2.GaussianBlur(work_img * work_weight, (k, k), 0)
        den = cv2.GaussianBlur(work_weight, (k, k), 0)
        return num / np.maximum(den, 1e-4)

    field = 0.70 * norm_blur(k1) + 0.30 * norm_blur(k2)
    field = cv2.GaussianBlur(field, (min(151, k1), min(151, k1)), 0)

    if field.shape != img.shape:
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)

    lo, hi = np.percentile(field, [0.5, 99.5])
    return np.clip(field, lo, hi).astype(np.float32)


def remove_row_profile(img: np.ndarray, weight: np.ndarray, strength: float) -> np.ndarray:
    h, _ = img.shape
    row_values = np.zeros(h, dtype=np.float32)
    for y in range(h):
        row_w = weight[y]
        row_sum = float(np.sum(row_w))
        if row_sum < 1e-4:
            row_values[y] = float(np.median(img[y]))
        else:
            row_values[y] = float(np.sum(img[y] * row_w) / row_sum)

    smooth_k = odd_size(h * 0.11, 31)
    smooth = cv2.GaussianBlur(row_values[:, None], (1, smooth_k), 0).ravel()
    seam = row_values - smooth
    return np.clip(img - strength * seam[:, None], 0.0, 1.0).astype(np.float32)


def additive_flatten(img: np.ndarray, field: np.ndarray, weight: np.ndarray, params: Params, strength: float) -> np.ndarray:
    target = float(np.average(field, weights=np.maximum(weight, 0.05)))
    corrected = img - strength * (field - target)
    corrected = remove_row_profile(corrected, weight, params.row_strength)
    return gentle_rescale_like_original(corrected, img, params.natural_contrast)


def inspection_view(natural: np.ndarray, field: np.ndarray, params: Params) -> np.ndarray:
    h, w = natural.shape
    min_dim = min(h, w)
    bg_k = odd_size(min_dim * 0.050, 25)
    top_k = odd_size(min_dim * 0.018, 7)

    local_bg = cv2.GaussianBlur(natural, (bg_k, bg_k), 0)
    residual = np.maximum(natural - local_bg, 0.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (top_k, top_k))
    top_hat = cv2.morphologyEx(to_uint8(natural), cv2.MORPH_TOPHAT, kernel).astype(np.float32) / 255.0

    gx = cv2.Sobel(natural, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(natural, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    feature = 0.46 * robust01(residual) + 0.38 * robust01(top_hat) + 0.16 * robust01(grad)
    feature = cv2.bilateralFilter(feature.astype(np.float32), 7, 0.08, 7)
    feature = np.clip(feature * params.inspection_gain, 0.0, 1.0)
    return robust01(1.0 - np.exp(-2.8 * feature), 0.5, 99.6)


def process_image(img: np.ndarray, filename: str) -> Dict[str, np.ndarray]:
    params = params_for(filename)
    weight = soft_background_weight(img, params)
    field = weighted_blur_field(img, weight, params)

    natural = additive_flatten(img, field, weight, params, params.additive_strength)
    strong = additive_flatten(img, field, weight, params, params.strong_strength)
    inspect = inspection_view(strong, field, params)

    return {
        "soft_flat_natural": natural,
        "soft_flat_strong": strong,
        "inspection_soft": inspect,
        "debug_background_weight": weight,
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
    pil = Image.fromarray(to_uint8(img), mode="L").convert("RGB")
    scale = width / pil.width
    height = max(1, int(round(pil.height * scale)))
    pil = pil.resize((width, height), Image.Resampling.BICUBIC)
    label_h = 28
    out = Image.new("RGB", (width, height + label_h), (18, 18, 18))
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
        "soft_flat_natural",
        "soft_flat_strong",
        "inspection_soft",
        "debug_background_weight",
        "debug_background_field",
    ]:
        panels.append(label_image(outputs[name], name))

    sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), (12, 12, 12))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def panel_step_metric(img: np.ndarray) -> float:
    _, w = img.shape
    left = img[:, int(w * 0.42):int(w * 0.49)]
    right = img[:, int(w * 0.51):int(w * 0.58)]
    return float(abs(np.median(left) - np.median(right)) * 255.0)


def row_band_metric(img: np.ndarray) -> float:
    row = np.median(img, axis=1)
    smooth = cv2.GaussianBlur(row[:, None].astype(np.float32), (1, odd_size(len(row) * 0.12, 21)), 0).ravel()
    return float(np.std(row - smooth) * 255.0)


def metric_row(filename: str, method: str, original: np.ndarray, output: np.ndarray) -> Dict[str, object]:
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "panel_step": panel_step_metric(output),
        "row_band": row_band_metric(output),
        "input_panel_step": panel_step_metric(original),
        "input_row_band": row_band_metric(original),
    }


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
        "soft_flat_natural",
        "soft_flat_strong",
        "inspection_soft",
        "debug_background_weight",
        "debug_background_field",
    ]
    ensure_dirs(output_dir, method_names)

    print(f"[Masked Background V2] Input : {input_dir}")
    print(f"[Masked Background V2] Output: {output_dir}")
    print(f"[Masked Background V2] Images: {len(paths)}")

    rows: List[Dict[str, object]] = []
    for path in tqdm(paths, desc="Masked background V2", ncols=90):
        filename = os.path.basename(path)
        img = read_gray(path)
        outputs = process_image(img, filename)
        for method, output in outputs.items():
            save_gray(os.path.join(output_dir, method, filename), output)
            if not method.startswith("debug_"):
                rows.append(metric_row(filename, method, img, output))
        make_contact_sheet(
            os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_masked_v2.jpg"),
            filename,
            img,
            outputs,
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Masked Background V2] Metrics: {metrics_path}")
    print(f"[Masked Background V2] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)

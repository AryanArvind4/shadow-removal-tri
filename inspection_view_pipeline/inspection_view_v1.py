"""
Inspection-View PCB X-ray Enhancement V1
=======================================

Mode 2 pipeline:
    aggressive background flattening + feature enhancement

This is intentionally different from "natural" shadow removal. It tries to
produce a TRI-inspection-style view with a dark, flattened background and
bright solder balls / pads / traces.

Run from repo root:

    py inspection_view_pipeline/inspection_view_v1.py

Outputs:

    inspection_view_pipeline/result/inspection_view_v1/
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, grey_closing, grey_opening, median_filter, percentile_filter, sobel
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class InspectParams:
    background_frac: float
    local_frac: float
    top_hat_frac: float
    denoise_size: int
    gain: float
    gamma: float
    threshold_percentile: float


PARAMS: Dict[str, InspectParams] = {
    "J1": InspectParams(0.085, 0.026, 0.030, 5, 2.8, 0.72, 72.0),
    "J4": InspectParams(0.080, 0.024, 0.028, 5, 3.0, 0.70, 71.0),
    "U29": InspectParams(0.120, 0.040, 0.060, 5, 2.5, 0.78, 68.0),
    "U30": InspectParams(0.115, 0.038, 0.052, 5, 2.4, 0.80, 69.0),
}

DEFAULT_PARAMS = InspectParams(0.100, 0.032, 0.040, 5, 2.6, 0.78, 70.0)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "inspection_view_v1")


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for(filename: str) -> InspectParams:
    return PARAMS.get(component_from_name(filename), DEFAULT_PARAMS)


def read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


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
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def estimate_background(img: np.ndarray, params: InspectParams) -> np.ndarray:
    h, w = img.shape
    sigma = max(min(h, w) * params.background_frac, 18.0)
    med_size = odd_size(min(h, w) * params.top_hat_frac, minimum=9)

    # Large median suppresses compact PCB features before broad smoothing.
    softened = median_filter(img, size=med_size, mode="reflect")
    background = gaussian_filter(softened, sigma=sigma, mode="reflect")
    return np.maximum(background, 1.0)


def background_flattened_foreground(img: np.ndarray, params: InspectParams) -> np.ndarray:
    background = estimate_background(img, params)
    subtractive = img - background + np.median(background)
    divisive = img / background * np.median(background)
    combined = 0.55 * subtractive + 0.45 * divisive
    return combined


def local_background_flatten(img: np.ndarray, params: InspectParams) -> np.ndarray:
    h, w = img.shape
    sigma = max(min(h, w) * params.local_frac, 6.0)
    local_mean = gaussian_filter(img, sigma=sigma, mode="reflect")
    local_sq = gaussian_filter(img * img, sigma=sigma, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 4.0))
    normalized = (img - local_mean) / local_std
    return normalized


def top_hat_features(img: np.ndarray, params: InspectParams) -> np.ndarray:
    h, w = img.shape
    size = odd_size(min(h, w) * params.top_hat_frac, minimum=9)
    opened = grey_opening(img, size=(size, size), mode="reflect")
    closed = grey_closing(img, size=(size, size), mode="reflect")
    white = np.clip(img - opened, 0, None)
    black = np.clip(closed - img, 0, None)

    # X-ray features can be bright rims or local dark centers. Keep both, with
    # stronger weight on bright structures for the TRI-style output.
    return white + 0.35 * black


def denoise_background_keep_features(feature01: np.ndarray, params: InspectParams) -> np.ndarray:
    denoise_size = odd_size(params.denoise_size, minimum=3)
    smooth = median_filter(feature01, size=denoise_size, mode="reflect")
    grad = robust01(gradient_magnitude(feature01), 5.0, 99.5)
    feature_mask = grad > np.percentile(grad, 78.0)
    return np.where(feature_mask, feature01, 0.55 * feature01 + 0.45 * smooth)


def inspection_dark_view(img: np.ndarray, params: InspectParams) -> np.ndarray:
    """
    Main TRI-style inspection rendering:
      1. flatten global background
      2. extract local/top-hat features
      3. suppress weak background
      4. render bright features over dark background
    """
    flat = background_flattened_foreground(img, params)
    local = local_background_flatten(flat, params)
    top_hat = top_hat_features(flat, params)

    flat01 = robust01(flat, 2.0, 99.4)
    local01 = robust01(local, 2.0, 99.6)
    top01 = robust01(top_hat, 2.0, 99.5)
    grad01 = robust01(gradient_magnitude(flat), 5.0, 99.6)

    feature = 0.38 * flat01 + 0.32 * local01 + 0.22 * top01 + 0.08 * grad01
    feature = denoise_background_keep_features(feature, params)

    threshold = np.percentile(feature, params.threshold_percentile)
    darkened = np.clip((feature - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    enhanced = np.power(darkened, params.gamma)
    enhanced = np.clip(params.gain * enhanced, 0.0, 1.0)
    return enhanced * 255.0


def inspection_soft_view(img: np.ndarray, params: InspectParams) -> np.ndarray:
    """
    Less aggressive view. Useful if the dark inspection rendering loses too
    much context.
    """
    flat = background_flattened_foreground(img, params)
    top = robust01(top_hat_features(flat, params), 1.0, 99.5)
    base = robust01(flat, 1.0, 99.4)
    soft = 0.62 * base + 0.38 * top
    soft = np.power(np.clip(soft, 0.0, 1.0), 0.82)
    return soft * 255.0


def background_field_view(img: np.ndarray, params: InspectParams) -> np.ndarray:
    bg = estimate_background(img, params)
    return robust01(bg, 1.0, 99.0) * 255.0


def residual_view(img: np.ndarray, params: InspectParams) -> np.ndarray:
    flat = background_flattened_foreground(img, params)
    return robust01(flat - np.median(flat), 1.0, 99.3) * 255.0


def process_image(img: np.ndarray, params: InspectParams) -> Dict[str, np.ndarray]:
    return {
        "inspection_dark": inspection_dark_view(img, params),
        "inspection_soft": inspection_soft_view(img, params),
        "background_field": background_field_view(img, params),
        "flattened_residual": residual_view(img, params),
    }


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


def metric_row(filename: str, method: str, output: np.ndarray) -> Dict[str, object]:
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "feature_density": feature_density(output),
        "contrast_score": contrast_score(output),
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

    method_names = ["inspection_dark", "inspection_soft", "background_field", "flattened_residual"]
    ensure_dirs(output_dir, method_names)

    print(f"[Inspection V1] Input : {input_dir}")
    print(f"[Inspection V1] Output: {output_dir}")
    print(f"[Inspection V1] Images: {len(paths)}")

    rows: List[Dict[str, object]] = []
    for path in tqdm(paths, desc="Inspection view"):
        filename = os.path.basename(path)
        img = read_gray(path)
        outputs = process_image(img, params_for(filename))

        for method, output in outputs.items():
            save_gray(os.path.join(output_dir, method, filename), output)
            rows.append(metric_row(filename, method, output))

        make_contact_sheet(
            os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_inspection_v1.jpg"),
            filename,
            img,
            outputs,
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Inspection V1] Metrics: {metrics_path}")
    print(f"[Inspection V1] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)


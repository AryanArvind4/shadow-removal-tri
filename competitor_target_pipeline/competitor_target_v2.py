"""
Competitor-Target PCB X-ray Inspection Pipeline V2
==================================================

V1 failed because it treated "high contrast" as "solder/pad". That promoted
seams, traces, background texture, and shadow boundaries into false white
objects.

V2 separates the problem:

    1. flatten broad shadow
    2. find bright candidates
    3. keep only geometry-plausible pad/solder regions
    4. protect dark void/defect candidates inside those regions
    5. darken the non-pad background

Run from repo root:

    .\\.venv312\\Scripts\\python.exe .\\competitor_target_pipeline\\competitor_target_v2.py
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
    work_max_dim: int
    bg_sigma_frac: float
    row_suppression: float
    local_sigma_frac: float
    tophat_frac: float
    candidate_percentile: float
    min_area_frac: float
    max_area_frac: float
    min_wh_frac: float
    max_wh_frac: float
    max_aspect: float
    min_extent: float
    min_solidity: float
    circularity_weight: float
    dilate: int
    void_percentile: float
    bg_level: float
    pad_base: float
    pad_gain: float
    edge_gain: float
    void_gain: float
    gamma: float


PARAMS: Dict[str, Params] = {
    # J1/J4: many small circular/ring solder balls plus horizontal seams.
    "J1": Params(700, 0.115, 0.45, 0.016, 0.018, 88.0, 0.000010, 0.0030, 0.0035, 0.050, 2.2, 0.16, 0.20, 0.70, 1, 84.0, 0.020, 0.60, 0.36, 0.18, 0.78, 0.72),
    "J4": Params(700, 0.115, 0.48, 0.016, 0.018, 88.0, 0.000010, 0.0030, 0.0035, 0.050, 2.2, 0.16, 0.20, 0.70, 1, 84.0, 0.020, 0.60, 0.36, 0.18, 0.78, 0.72),
    # U29/U30: larger square/rectangular solder regions and component bodies.
    "U29": Params(760, 0.150, 0.08, 0.030, 0.045, 80.0, 0.000050, 0.0300, 0.0100, 0.180, 3.0, 0.22, 0.28, 0.35, 2, 80.0, 0.025, 0.62, 0.34, 0.14, 0.86, 0.74),
    "U30": Params(768, 0.145, 0.07, 0.030, 0.042, 80.0, 0.000050, 0.0350, 0.0100, 0.190, 3.2, 0.22, 0.28, 0.35, 2, 80.0, 0.025, 0.62, 0.34, 0.14, 0.86, 0.74),
}

DEFAULT_PARAMS = Params(700, 0.130, 0.15, 0.024, 0.030, 84.0, 0.000020, 0.0150, 0.0060, 0.120, 2.8, 0.20, 0.25, 0.45, 1, 82.0, 0.025, 0.61, 0.35, 0.16, 0.82, 0.73)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(repo_root(), "TRI_Dataset-20260511T190518Z-3-001", "TRI_Dataset_Concised")


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "competitor_target_v2")


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


def odd_size(value: float, minimum: int = 3, maximum: int | None = None) -> int:
    size = max(minimum, int(round(value)))
    if maximum is not None:
        size = min(size, maximum)
    if size % 2 == 0:
        size += 1
    return size


def robust01(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.2) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_p, hi_p))
    if hi <= lo + 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_for_work(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img.astype(np.float32), 1.0
    work = cv2.resize(img.astype(np.float32), (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return work, scale


def restore_size(img: np.ndarray, shape: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    h, w = shape
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=interp)


def estimate_background(img: np.ndarray, params: Params) -> np.ndarray:
    h, w = img.shape
    med = odd_size(min(h, w) * 0.018, 9, 51)
    sigma = max(min(h, w) * params.bg_sigma_frac, 18.0)
    softened = cv2.medianBlur(to_uint8(img), med).astype(np.float32)
    return np.maximum(cv2.GaussianBlur(softened, (0, 0), sigmaX=sigma, sigmaY=sigma), 1.0)


def flatten_shadow(img: np.ndarray, params: Params) -> Tuple[np.ndarray, np.ndarray]:
    bg = estimate_background(img, params)
    target = float(np.median(bg))
    additive = img - (bg - target)
    divisive = img / bg * target
    flat = 0.42 * additive + 0.58 * divisive
    return np.clip(flat, 0, 255).astype(np.float32), bg


def suppress_rows(flat: np.ndarray, params: Params) -> np.ndarray:
    if params.row_suppression <= 0:
        return flat
    h, _ = flat.shape
    profile = np.median(flat, axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(profile[:, None], (0, 0), sigmaX=0, sigmaY=max(h * 0.060, 10)).ravel()
    return np.clip(flat - params.row_suppression * (profile - smooth)[:, None], 0, 255).astype(np.float32)


def feature_maps(flat: np.ndarray, params: Params) -> Dict[str, np.ndarray]:
    h, w = flat.shape
    base = robust01(flat, 1.0, 99.2)
    local_sigma = max(min(h, w) * params.local_sigma_frac, 4.0)
    mean = cv2.GaussianBlur(flat, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    sq = cv2.GaussianBlur(flat * flat, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    std = np.sqrt(np.maximum(sq - mean * mean, 4.0))
    z = (flat - mean) / np.maximum(std, 2.0)

    k = odd_size(min(h, w) * params.tophat_frac, 7, 71)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(flat.astype(np.float32), cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(flat.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    white = np.clip(flat - opened, 0, None)
    black = np.clip(closed - flat, 0, None)

    gx = cv2.Sobel(flat.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(flat.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    bright = 0.42 * robust01(np.maximum(z, 0), 4.0, 99.5) + 0.43 * robust01(white, 4.0, 99.5) + 0.15 * robust01(grad, 10.0, 99.5)
    bright = cv2.GaussianBlur(bright.astype(np.float32), (0, 0), sigmaX=0.55, sigmaY=0.55)
    dark = 0.55 * robust01(black, 45.0, 99.6) + 0.45 * robust01(np.maximum(-z, 0), 45.0, 99.6)
    dark = cv2.medianBlur(to_uint8(dark * 255.0), 3).astype(np.float32) / 255.0
    return {"base": base, "bright": np.clip(bright, 0, 1), "dark": np.clip(dark, 0, 1), "grad": robust01(grad, 10.0, 99.6)}


def remove_line_like_candidates(candidate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = candidate.shape
    cand_u8 = candidate.astype(np.uint8) * 255
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(17, w // 22), 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(17, h // 18)))
    h_lines = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, horizontal_kernel)
    v_lines = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, vertical_kernel)
    rejected = (h_lines > 0) | (v_lines > 0)
    cleaned = candidate & ~rejected
    return cleaned, rejected


def geometry_pad_mask(bright: np.ndarray, base: np.ndarray, params: Params, component: str) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bright.shape
    threshold = np.percentile(bright, params.candidate_percentile)
    candidate = bright >= threshold
    candidate |= (bright >= threshold * 0.78) & (base >= np.percentile(base, 62.0))

    candidate, rejected_lines = remove_line_like_candidates(candidate)
    cand_u8 = candidate.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, k3, iterations=1)
    close_size = 9 if component in {"U29", "U30"} else 5
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    keep = np.zeros((h, w), dtype=np.uint8)
    min_area = max(5, int(h * w * params.min_area_frac))
    max_area = max(min_area + 1, int(h * w * params.max_area_frac))
    min_wh = max(3, int(min(h, w) * params.min_wh_frac))
    max_wh = max(min_wh + 1, int(min(h, w) * params.max_wh_frac))

    for label in range(1, n):
        x, y, bw, bh, area = stats[label]
        area = int(area)
        if area < min_area or area > max_area:
            continue
        if bw < min_wh or bh < min_wh or bw > max_wh or bh > max_wh:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > params.max_aspect:
            continue
        region = labels == label
        rect_area = float(bw * bh)
        extent = area / max(rect_area, 1.0)
        if extent < params.min_extent:
            continue

        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = float(cv2.contourArea(contour)) / hull_area
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        circularity = float(4.0 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter))

        if solidity < params.min_solidity:
            continue
        if component in {"J1", "J4"}:
            score = params.circularity_weight * circularity + (1.0 - params.circularity_weight) * min(extent, 1.0)
            if score < 0.18:
                continue

        keep[region] = 255

    if params.dilate > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.dilate + 1, 2 * params.dilate + 1))
        keep = cv2.dilate(keep, dk, iterations=1)

    return keep > 0, rejected_lines


def soft_feature_support(base: np.ndarray, bright: np.ndarray, pad_mask: np.ndarray, component: str) -> np.ndarray:
    """Soft support keeps grayscale detail instead of forcing a binary mask look."""
    if component in {"U29", "U30"}:
        blended = 0.58 * cv2.GaussianBlur(base, (0, 0), sigmaX=1.2, sigmaY=1.2) + 0.42 * bright
        thresh = np.percentile(blended, 62.0)
        support = blended >= thresh
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        support_u8 = cv2.morphologyEx(support.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k, iterations=1)
        support_u8 = cv2.morphologyEx(support_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    else:
        support_u8 = pad_mask.astype(np.uint8) * 255
        support_u8 = cv2.dilate(support_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    soft = cv2.GaussianBlur(support_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=2.2, sigmaY=2.2)
    soft = np.maximum(soft, cv2.GaussianBlur(pad_mask.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0))
    return np.clip(soft, 0.0, 1.0)


def void_mask(dark: np.ndarray, pad_mask: np.ndarray, params: Params) -> np.ndarray:
    if not np.any(pad_mask):
        return np.zeros_like(pad_mask, dtype=bool)
    inside = dark[pad_mask]
    thr = np.percentile(inside, params.void_percentile)
    mask = (dark >= thr) & pad_mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k, iterations=1)
    return clean > 0


def reference_gray_render(base: np.ndarray, bright: np.ndarray, dark: np.ndarray, grad: np.ndarray, component: str) -> np.ndarray:
    base_u8 = to_uint8(base * 255.0)
    denoised = cv2.fastNlMeansDenoising(base_u8, None, h=7, templateWindowSize=7, searchWindowSize=21).astype(np.float32) / 255.0
    denoised = cv2.bilateralFilter(to_uint8(denoised * 255.0), 7, 22, 7).astype(np.float32) / 255.0

    h, w = base.shape
    local_sigma = max(min(h, w) * (0.030 if component in {"U29", "U30"} else 0.018), 6.0)
    local_bg = cv2.GaussianBlur(denoised, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    residual = denoised - local_bg
    positive = robust01(residual, 58.0, 99.6)

    k = odd_size(min(h, w) * (0.035 if component in {"U29", "U30"} else 0.018), 7, 55)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    top_hat = cv2.morphologyEx(to_uint8(denoised * 255.0), cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    top_hat = robust01(top_hat, 8.0, 99.6)

    feature = np.clip(0.48 * positive + 0.34 * bright + 0.18 * top_hat, 0.0, 1.0)
    gate_center = np.percentile(feature, 72.0 if component in {"U29", "U30"} else 78.0)
    gate = 1.0 / (1.0 + np.exp(-(feature - gate_center) * 18.0))
    gate = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)

    # Keep internal dark spots visible. We darken them only where local bright structure exists.
    defect_dark = cv2.GaussianBlur(dark.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)
    edge = cv2.GaussianBlur(grad.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)

    background = 0.025 + 0.070 * np.power(denoised, 1.8)
    foreground = 0.30 + 0.44 * denoised + 0.34 * feature + 0.10 * edge - 0.48 * defect_dark
    out = (1.0 - gate) * background + gate * foreground
    out = np.power(np.clip(out, 0.0, 1.0), 0.78)
    return np.clip(out, 0, 1) * 255.0


def render(base: np.ndarray, bright: np.ndarray, grad: np.ndarray, pad_mask: np.ndarray, voids: np.ndarray, params: Params, component: str) -> Dict[str, np.ndarray]:
    base_u8 = to_uint8(base * 255.0)
    smooth = cv2.bilateralFilter(base_u8, 7, 24, 7).astype(np.float32) / 255.0
    smooth = cv2.medianBlur(to_uint8(smooth * 255.0), 3).astype(np.float32) / 255.0

    support = soft_feature_support(base, bright, pad_mask, component)
    edge = grad * support
    void_strength = np.where(voids, 1.0, 0.0).astype(np.float32)
    dark_detail = cv2.GaussianBlur(void_strength, (0, 0), sigmaX=0.45, sigmaY=0.45)

    background = 0.030 + params.bg_level * np.power(base, 1.65)
    background = cv2.bilateralFilter(to_uint8(background * 255.0), 9, 18, 9).astype(np.float32) / 255.0

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(base_u8).astype(np.float32) / 255.0
    pad_tone = 0.33 + 0.48 * smooth + 0.28 * clahe + 0.22 * bright + params.edge_gain * edge
    pad_tone -= params.void_gain * dark_detail
    inspection = (1.0 - support) * background + support * pad_tone
    inspection = np.power(np.clip(inspection, 0, 1), params.gamma)

    preserving = np.where(
        support > 0.08,
        0.16 + 0.52 * smooth + 0.22 * clahe + 0.13 * bright + 0.06 * edge - 0.54 * dark_detail,
        0.08 * np.power(base, 1.35),
    )

    uniform = np.where(
        support > 0.08,
        0.24 + 0.58 * cv2.GaussianBlur(smooth, (0, 0), sigmaX=1.1, sigmaY=1.1) + 0.13 * bright + 0.04 * edge - 0.40 * dark_detail,
        0.13 * np.power(base, 1.20),
    )

    return {
        "inspection_grade": np.clip(inspection, 0, 1) * 255.0,
        "defect_preserving": np.clip(preserving, 0, 1) * 255.0,
        "pad_uniform_gray": np.clip(uniform, 0, 1) * 255.0,
        "soft_support": support * 255.0,
    }


def process_work_image(img: np.ndarray, params: Params, component: str) -> Dict[str, np.ndarray]:
    flat, bg = flatten_shadow(img, params)
    flat = suppress_rows(flat, params)
    maps = feature_maps(flat, params)
    pads, rejected = geometry_pad_mask(maps["bright"], maps["base"], params, component)
    voids = void_mask(maps["dark"], pads, params)
    outputs = render(maps["base"], maps["bright"], maps["grad"], pads, voids, params, component)
    outputs["reference_gray"] = reference_gray_render(maps["base"], maps["bright"], maps["dark"], maps["grad"], component)
    outputs.update(
        {
            "pad_mask": pads.astype(np.float32) * 255.0,
            "void_mask": voids.astype(np.float32) * 255.0,
            "rejected_features": rejected.astype(np.float32) * 255.0,
            "flattened_gray": maps["base"] * 255.0,
            "background_field": robust01(bg, 1.0, 99.0) * 255.0,
        }
    )
    return outputs


def process_image(img: np.ndarray, params: Params, component: str) -> Dict[str, np.ndarray]:
    original_shape = img.shape
    work, scale = resize_for_work(img, params.work_max_dim)
    outputs = process_work_image(work, params, component)
    if scale >= 1.0:
        return outputs
    return {name: restore_size(value, original_shape, is_mask=("mask" in name or name == "rejected_features")) for name, value in outputs.items()}


def pad_std_mean(img: np.ndarray, pad_mask: np.ndarray, voids: np.ndarray) -> float:
    healthy = (pad_mask > 0) & (voids <= 0)
    if not np.any(healthy):
        return 0.0
    norm = img.astype(np.float32) / 255.0
    return float(np.std(norm[healthy]))


def pad_background_contrast(img: np.ndarray, pad_mask: np.ndarray) -> float:
    pad = pad_mask > 0
    bg = ~pad
    if not np.any(pad) or not np.any(bg):
        return 0.0
    norm = img.astype(np.float32) / 255.0
    return float((np.mean(norm[pad]) - np.mean(norm[bg])) / (np.std(norm[bg]) + 1e-6))


def void_contrast(img: np.ndarray, pad_mask: np.ndarray, voids: np.ndarray) -> float:
    v = voids > 0
    healthy = (pad_mask > 0) & ~v
    if not np.any(v) or not np.any(healthy):
        return 0.0
    norm = img.astype(np.float32) / 255.0
    return float((np.mean(norm[healthy]) - np.mean(norm[v])) / (np.std(norm[healthy]) + 1e-6))


def background_noise(img: np.ndarray, pad_mask: np.ndarray) -> float:
    bg = pad_mask <= 0
    if not np.any(bg):
        return 0.0
    return float(np.std(img.astype(np.float32)[bg] / 255.0))


def metric_row(filename: str, method: str, outputs: Dict[str, np.ndarray]) -> Dict[str, object]:
    img = outputs[method]
    pad_mask = outputs["pad_mask"]
    voids = outputs["void_mask"]
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "pad_std_mean": pad_std_mean(img, pad_mask, voids),
        "pad_background_contrast": pad_background_contrast(img, pad_mask),
        "void_contrast": void_contrast(img, pad_mask, voids),
        "background_noise": background_noise(img, pad_mask),
        "saturation_ratio": float(np.mean(img >= 248.0)),
        "pad_mask_fraction": float(np.mean(pad_mask > 0)),
        "void_mask_fraction": float(np.mean(voids > 0)),
    }


def label_image(img: np.ndarray, label: str, width: int = 310) -> Image.Image:
    pil = Image.fromarray(to_uint8(img), mode="L").convert("RGB")
    scale = width / pil.width
    height = max(1, int(round(pil.height * scale)))
    pil = pil.resize((width, height), Image.BICUBIC)
    label_h = 32
    canvas = Image.new("RGB", (width, height + label_h), (18, 18, 18))
    canvas.paste(pil, (0, label_h))
    ImageDraw.Draw(canvas).text((8, 9), label, fill=(235, 235, 235))
    return canvas


def make_contact_sheet(path: str, filename: str, original: np.ndarray, outputs: Dict[str, np.ndarray]) -> None:
    order = ["reference_gray", "inspection_grade", "defect_preserving", "pad_uniform_gray", "soft_support", "pad_mask", "void_mask", "rejected_features", "flattened_gray"]
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
        "reference_gray",
        "inspection_grade",
        "defect_preserving",
        "pad_uniform_gray",
        "pad_mask",
        "soft_support",
        "void_mask",
        "rejected_features",
        "flattened_gray",
        "background_field",
    ]
    ensure_dirs(output_dir, method_names)
    print(f"[Competitor Target V2] Input : {input_dir}")
    print(f"[Competitor Target V2] Output: {output_dir}")
    print(f"[Competitor Target V2] Images: {len(paths)}")

    rows: List[Dict[str, object]] = []
    scored = ["reference_gray", "inspection_grade", "defect_preserving", "pad_uniform_gray"]
    for path in tqdm(paths, desc="Competitor Target V2"):
        filename = os.path.basename(path)
        component = component_from_name(filename)
        img = read_gray(path)
        outputs = process_image(img, params_for(filename), component)
        for method, output in outputs.items():
            save_gray(os.path.join(output_dir, method, filename), output)
            if method in scored:
                rows.append(metric_row(filename, method, outputs))
        make_contact_sheet(
            os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_competitor_target_v2.jpg"),
            filename,
            img,
            outputs,
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Competitor Target V2] Metrics: {metrics_path}")
    print(f"[Competitor Target V2] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    print("[Competitor Target V2] Priority: reference_gray -> defect_preserving -> pad_uniform_gray -> inspection_grade")
    return rows


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    process(input_dir, output_dir)

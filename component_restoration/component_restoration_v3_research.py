"""
Component-Aware Research Restoration V3
======================================

Research-library version of the stack restoration pipeline.

Main additions:
  - BaSiCPy flat-field/dark-field estimation from the full TRI component stack.
  - J4 panel-seam (piecewise gain/offset step) correction, applied before BaSiC,
    since a smooth flat-field cannot represent a hard discontinuity.
  - VSNR (Gabor-prior) destriping for J4 horizontal banding, preferred over
    wavelet/FFT band damping because the directional prior targets long
    horizontal stripes specifically and leaves solder-ball rows alone.
    Falls back to wavelet/Fourier destriping if PyWavelets is available and
    pyvsnr is not, else to row-profile destriping.
  - Conservative square-grid cleanup for U29 after BaSiC correction, followed
    by edge-preserving (bilateral) inter-pad mottle suppression restricted to
    non-structure pixels.
  - Structure-masked, artifact-specific metrics (row-band FFT power, panel
    step size, pad contrast / gap noise, residual structure leak) instead of
    just an index of which method ran.

Run from repo root:

    py component_restoration/component_restoration_v3_research.py

Install optional research dependencies:

    py -m pip install BaSiCPy PyWavelets pyvsnr scikit-image

Outputs:

    component_restoration/result/component_restoration_v3_research/
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, grey_opening, median_filter, sobel, zoom
from tqdm import tqdm

try:
    from pyvsnr import vsnr2d as _vsnr2d
except Exception:
    _vsnr2d = None

try:
    from skimage.restoration import denoise_bilateral as _denoise_bilateral
except Exception:
    _denoise_bilateral = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MAX_FIT_SIZE = 512
MAX_FIT_IMAGES = 80


@dataclass(frozen=True)
class V3Params:
    basic_strength: float
    row_strength: float
    square_enhance: float
    denoise_sigma: float
    panel_strength: float = 0.0
    vsnr_noise_level: float = 80.0
    vsnr_sigma: Tuple[float, float] = (60.0, 1.0)
    mottle_sigma_color: float = 12.0
    mottle_sigma_spatial: float = 4.0
    mottle_enable: bool = False


# panel_strength > 0  -> component has a J4-style piecewise panel seam: run
#                        correct_panel_step() before BaSiC, then vsnr_destripe().
# mottle_enable        -> component has no seam/grid but shows fine inter-
#                        component graininess: run mottle_suppress() after BaSiC,
#                        no grid top-hat enhancement (no BGA-style pad array).
PARAMS: Dict[str, V3Params] = {
    "J1": V3Params(0.55, 0.45, 0.0, 1.1, panel_strength=1.0, vsnr_noise_level=90.0, vsnr_sigma=(60.0, 1.0)),
    "J4": V3Params(0.60, 0.75, 0.0, 1.1, panel_strength=1.0, vsnr_noise_level=80.0, vsnr_sigma=(60.0, 1.0)),
    "U29": V3Params(0.45, 0.20, 0.45, 1.45, mottle_sigma_color=12.0, mottle_sigma_spatial=4.0),
    "U30": V3Params(0.45, 0.30, 0.0, 1.2, mottle_sigma_color=10.0, mottle_sigma_spatial=3.0, mottle_enable=True),
}

DEFAULT_PARAMS = V3Params(0.50, 0.35, 0.0, 1.2)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_target_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_template_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset",
    )


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "component_restoration_v3_research")


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for(component: str) -> V3Params:
    return PARAMS.get(component.upper(), DEFAULT_PARAMS)


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


def group_paths(paths: Iterable[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for path in paths:
        grouped.setdefault(component_from_name(os.path.basename(path)), []).append(path)
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def resize_float(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if img.shape == shape:
        return img.astype(np.float32, copy=True)
    factors = (shape[0] / img.shape[0], shape[1] / img.shape[1])
    return zoom(img, factors, order=1, mode="reflect").astype(np.float32)


def fit_shape_for(shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = shape
    scale = min(1.0, MAX_FIT_SIZE / float(max(h, w)))
    return max(64, int(round(h * scale))), max(64, int(round(w * scale)))


def normalize01(img: np.ndarray) -> np.ndarray:
    return np.clip(img / 255.0, 0.0, 1.0).astype(np.float32)


def robust_percentile_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    lo_i, hi_i = np.percentile(img, (0.4, 99.6))
    lo_r, hi_r = np.percentile(reference, (0.4, 99.6))
    if hi_i - lo_i < 1e-6:
        return np.clip(img, 0, 255)
    out = (img - lo_i) / (hi_i - lo_i) * (hi_r - lo_r) + lo_r
    out += np.mean(reference) - np.mean(out)
    return np.clip(out, 0, 255)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def structure_mask(img: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    smooth = gaussian_filter(img, sigma=max(min(img.shape) * 0.010, 2.0), mode="reflect")
    highpass = np.abs(img - smooth)
    grad = gradient_magnitude(img)
    mask = (highpass > np.percentile(highpass, percentile)) | (grad > np.percentile(grad, percentile))
    for _ in range(2):
        mask = binary_dilation(mask)
    return mask


def detect_panel_seam(img: np.ndarray) -> int:
    """Locate a hard vertical panel discontinuity via peak gradient of the column-median profile."""
    h, w = img.shape
    profile = gaussian_filter(np.median(img, axis=0), sigma=max(w * 0.008, 5.0), mode="reflect")
    grad = np.abs(np.gradient(profile))
    lo, hi = int(w * 0.25), int(w * 0.75)
    if hi <= lo:
        return w // 2
    return lo + int(np.argmax(grad[lo:hi]))


def correct_panel_step(img: np.ndarray, strength: float, structure_percentile: float = 80.0) -> np.ndarray:
    """
    Match mean/std on each side of a detected panel seam using only non-structure
    (background) pixels, feathered across the seam so the correction itself does
    not introduce a new edge. This targets the piecewise panel gain/offset step
    that a smooth flat-field (BaSiC/Gaussian) cannot represent.
    """
    if strength <= 0:
        return img

    h, w = img.shape
    seam = detect_panel_seam(img)
    mask = structure_mask(img, structure_percentile)
    valid = ~mask

    left_valid = img[:, :seam][valid[:, :seam]]
    right_valid = img[:, seam:][valid[:, seam:]]
    if left_valid.size == 0 or right_valid.size == 0:
        return img

    l_mean, l_std = float(left_valid.mean()), float(left_valid.std()) + 1e-6
    r_mean, r_std = float(right_valid.mean()), float(right_valid.std()) + 1e-6
    target_mean = 0.5 * (l_mean + r_mean)
    target_std = 0.5 * (l_std + r_std)

    def normed(block: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (block - mean) / std * target_std + target_mean

    out = img.copy()
    left_norm = normed(img[:, :seam], l_mean, l_std)
    right_norm = normed(img[:, seam:], r_mean, r_std)
    out[:, :seam] = img[:, :seam] + strength * (left_norm - img[:, :seam])
    out[:, seam:] = img[:, seam:] + strength * (right_norm - img[:, seam:])

    pad = max(int(w * 0.035), 12)
    x0, x1 = max(0, seam - pad), min(w, seam + pad)
    if x1 > x0:
        alpha = np.linspace(0.0, 1.0, x1 - x0, dtype=np.float32)[None, :]
        block = img[:, x0:x1]
        left_version = block + strength * (normed(block, l_mean, l_std) - block)
        right_version = block + strength * (normed(block, r_mean, r_std) - block)
        out[:, x0:x1] = (1.0 - alpha) * left_version + alpha * right_version

    return np.clip(out, 0, 255)


def try_import_basic():
    try:
        from basicpy import BaSiC  # type: ignore

        return BaSiC
    except Exception:
        return None


def get_basic_field(model: object, names: Iterable[str]) -> Optional[np.ndarray]:
    for name in names:
        if hasattr(model, name):
            value = getattr(model, name)
            if value is not None:
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim >= 2:
                    return np.squeeze(arr)
    return None


def fit_basicpy_fields(train_images: List[np.ndarray], target_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Fit BaSiCPy on a downsampled component stack and return full-resolution
    flatfield/darkfield estimates if possible.
    """
    BaSiC = try_import_basic()
    if BaSiC is None:
        return None, None, "BaSiCPy is not installed. Run: py -m pip install BaSiCPy"

    fit_shape = fit_shape_for(target_shape)
    subset = train_images[:MAX_FIT_IMAGES]
    stack = np.stack([normalize01(resize_float(img, fit_shape)) for img in subset], axis=0)

    init_options = [
        {"get_darkfield": True},
        {"estimate_darkfield": True},
        {},
    ]
    last_error = ""
    for kwargs in init_options:
        try:
            model = BaSiC(**kwargs)
            model.fit(stack)
            flat = get_basic_field(model, ["flatfield", "flatfield_", "flat_field", "flatfield_image"])
            dark = get_basic_field(model, ["darkfield", "darkfield_", "dark_field", "darkfield_image"])
            if flat is None:
                last_error = "BaSiCPy fit succeeded, but no flatfield attribute was found."
                continue

            flat_full = resize_float(flat, target_shape)
            if dark is None:
                dark_full = np.zeros(target_shape, dtype=np.float32)
            else:
                dark_full = resize_float(dark, target_shape)
            return flat_full, dark_full, "BaSiCPy"
        except Exception as exc:
            last_error = str(exc)

    return None, None, f"BaSiCPy could not fit this stack: {last_error}"


def basicpy_correct(img: np.ndarray, flat: np.ndarray, dark: np.ndarray, strength: float) -> np.ndarray:
    img01 = normalize01(img)
    flat = np.maximum(flat, 1e-4)
    corrected01 = (img01 - strength * dark) / ((1.0 - strength) + strength * flat)
    corrected = np.clip(corrected01 * 255.0, 0, 255)
    return robust_percentile_match(corrected, img)


def row_profile_destripe(img: np.ndarray, strength: float) -> np.ndarray:
    mask = structure_mask(img, 80.0)
    valid = (~mask).astype(np.float32)
    row_profile = np.sum(img * valid, axis=1) / np.maximum(np.sum(valid, axis=1), 1.0)
    medium = gaussian_filter(row_profile, sigma=4.0, mode="reflect")
    broad = gaussian_filter(row_profile, sigma=max(img.shape[0] * 0.10, 40.0), mode="reflect")
    stripe = medium - broad
    stripe = np.clip(stripe, -np.percentile(np.abs(stripe), 92.0), np.percentile(np.abs(stripe), 92.0))
    out = img - strength * stripe[:, None]
    return robust_percentile_match(out, img)


def wavelet_fft_destripe(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Muench-style wavelet/Fourier destriping approximation.

    If PyWavelets is installed, suppress horizontal stripe frequencies in
    vertical-detail wavelet bands. If unavailable, fall back to row-profile
    destriping.
    """
    try:
        import pywt  # type: ignore
    except Exception:
        return row_profile_destripe(img, strength)

    coeffs = pywt.wavedec2(img.astype(np.float32), wavelet="db4", level=3, mode="periodization")
    new_coeffs = [coeffs[0]]

    for detail in coeffs[1:]:
        cH, cV, cD = detail
        # Horizontal bands vary along y and are mostly represented in cV/cD for
        # our image orientation. We damp low horizontal-frequency stripe content.
        filtered = []
        for band in (cH, cV, cD):
            fft = np.fft.fftshift(np.fft.fft2(band))
            h, w = band.shape
            yy = np.arange(h)[:, None] - h // 2
            xx = np.arange(w)[None, :] - w // 2
            horizontal_stripe_mask = np.exp(-(yy ** 2) / (2 * max(h * 0.045, 2.0) ** 2))
            damping = 1.0 - strength * 0.35 * horizontal_stripe_mask
            restored = np.real(np.fft.ifft2(np.fft.ifftshift(fft * damping)))
            filtered.append(restored.astype(np.float32))
        new_coeffs.append(tuple(filtered))

    out = pywt.waverec2(new_coeffs, wavelet="db4", mode="periodization")
    out = out[: img.shape[0], : img.shape[1]]
    return robust_percentile_match(out, img)


def vsnr_destripe(
    img: np.ndarray,
    noise_level: float,
    sigma: Tuple[float, float],
    theta: float = 0.0,
    maxit: int = 30,
) -> np.ndarray:
    """
    Variational Stationary Noise Removal (Fehrenbach/Weiss) with a Gabor prior
    tuned for long horizontal stripes. Preferred over wavelet/FFT band-damping
    for J4 because the directional Gabor filter targets "long thin horizontal
    artifact" specifically, leaving round solder-ball rows largely untouched.
    Falls back to wavelet_fft_destripe if pyvsnr is not installed.
    """
    if _vsnr2d is None:
        return wavelet_fft_destripe(img, strength=min(1.0, noise_level / 80.0))

    filters = [{"name": "Gabor", "noise_level": noise_level, "sigma": sigma, "theta": theta}]
    out = _vsnr2d(img.astype(np.float32)[None, ...], filters, maxit=maxit, algo="auto")
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    return robust_percentile_match(out, img)


def mottle_suppress(img: np.ndarray, sigma_color: float, sigma_spatial: float, structure_percentile: float = 82.0) -> np.ndarray:
    """
    Edge-preserving smoothing of the inter-pad "mottle" texture on U29, applied
    only to non-structure (background/gap) pixels so pad edges stay sharp.
    Falls back to a light median blend if scikit-image is unavailable.
    """
    mask = structure_mask(img, structure_percentile)
    img01 = normalize01(img)

    if _denoise_bilateral is not None:
        smoothed01 = _denoise_bilateral(
            img01,
            sigma_color=sigma_color / 255.0,
            sigma_spatial=sigma_spatial,
        )
        smoothed = smoothed01 * 255.0
    else:
        smoothed = 0.5 * gaussian_filter(img, sigma=sigma_spatial, mode="reflect") + 0.5 * median_filter(
            img, size=3, mode="reflect"
        )

    out = np.where(mask, img, smoothed)
    return robust_percentile_match(out, img)


def square_cleanup(img: np.ndarray, strength: float, denoise_sigma: float) -> np.ndarray:
    size = max(9, int(round(min(img.shape) * 0.050)))
    if size % 2 == 0:
        size += 1
    opened = grey_opening(img, size=(size, size), mode="reflect")
    top_hat = np.clip(img - opened, 0, np.percentile(img - opened, 99.2))
    grad = gradient_magnitude(img)
    mask = (top_hat > np.percentile(top_hat, 82.0)) | (grad > np.percentile(grad, 82.0))
    for _ in range(2):
        mask = binary_dilation(mask)

    bg = 0.55 * median_filter(img, size=3, mode="reflect") + 0.45 * gaussian_filter(
        img, sigma=denoise_sigma, mode="reflect"
    )
    enhanced = img + strength * top_hat
    out = np.where(mask, enhanced, 0.70 * img + 0.30 * bg)
    return robust_percentile_match(out, img)


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


def make_sheet(path: str, filename: str, original: np.ndarray, results: Dict[str, np.ndarray]) -> None:
    panels = [label_image(original, f"input: {filename}")]
    panels.extend(label_image(img, name) for name, img in results.items())
    width = panels[0].width
    height = max(p.height for p in panels)
    sheet = Image.new("RGB", (width * len(panels), height), (28, 28, 28))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def row_band_power(img: np.ndarray, structure_percentile: float = 80.0) -> float:
    """FFT power of the masked row-mean profile outside the smooth low-frequency band -- proxy for J4 horizontal banding."""
    mask = structure_mask(img, structure_percentile)
    valid = (~mask).astype(np.float32)
    row_profile = np.sum(img * valid, axis=1) / np.maximum(np.sum(valid, axis=1), 1.0)
    spectrum = np.abs(np.fft.rfft(row_profile - row_profile.mean()))
    n = spectrum.shape[0]
    lo_cut = max(1, int(n * 0.02))
    return float(np.sum(spectrum[lo_cut:]) / (np.sum(spectrum) + 1e-6))


def panel_step_score(img: np.ndarray, structure_percentile: float = 80.0) -> float:
    """Median brightness gap across the detected J4 panel seam, using only background pixels."""
    seam = detect_panel_seam(img)
    w = img.shape[1]
    pad = max(int(w * 0.035), 12)
    mask = structure_mask(img, structure_percentile)
    valid = ~mask
    left = img[:, max(0, seam - 3 * pad) : max(1, seam - pad)]
    right = img[:, min(w - 1, seam + pad) : min(w, seam + 3 * pad)]
    left_valid = left[valid[:, max(0, seam - 3 * pad) : max(1, seam - pad)]]
    right_valid = right[valid[:, min(w - 1, seam + pad) : min(w, seam + 3 * pad)]]
    if left_valid.size == 0 or right_valid.size == 0:
        return 0.0
    return float(abs(np.median(left_valid) - np.median(right_valid)))


def pad_contrast_and_gap_std(img: np.ndarray, structure_percentile: float = 82.0) -> Tuple[float, float]:
    """U29 pad-visibility pair: (pad-vs-gap contrast, inter-pad gap noise). Higher contrast + lower gap_std = better."""
    mask = structure_mask(img, structure_percentile)
    if np.sum(mask) == 0 or np.sum(~mask) == 0:
        return 0.0, float(np.std(img))
    gap_std = float(np.std(img[~mask]))
    contrast = float((np.mean(img[mask]) - np.mean(img[~mask])) / (gap_std + 1e-6))
    return contrast, gap_std


def residual_structure_leak(input_img: np.ndarray, output_img: np.ndarray, structure_percentile: float = 80.0) -> float:
    """Ratio of removed-signal energy sitting on structure vs. background -- a hallucination/over-correction alarm."""
    residual = input_img - output_img
    mask = structure_mask(input_img, structure_percentile)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs(residual[mask])) / (np.mean(np.abs(residual)) + 1e-6))


def metric_row(filename: str, component: str, method: str, original: np.ndarray, output: np.ndarray) -> Dict[str, object]:
    row: Dict[str, object] = {
        "filename": filename,
        "component": component,
        "method": method,
        "residual_structure_leak": residual_structure_leak(original, output),
    }
    if component in ("J4", "J1"):
        row["row_band_power_before"] = row_band_power(original)
        row["row_band_power_after"] = row_band_power(output)
        row["panel_step_before"] = panel_step_score(original)
        row["panel_step_after"] = panel_step_score(output)
    elif component in ("U29", "U30"):
        contrast_before, gap_std_before = pad_contrast_and_gap_std(original)
        contrast_after, gap_std_after = pad_contrast_and_gap_std(output)
        row["pad_contrast_before"] = contrast_before
        row["pad_contrast_after"] = contrast_after
        row["gap_std_before"] = gap_std_before
        row["gap_std_after"] = gap_std_after
    return row


def ensure_dirs(output_dir: str, method_names: Iterable[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison_sheets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "research_fields"), exist_ok=True)
    for name in method_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def process(target_dir: str, output_dir: str, template_dir: str) -> None:
    target_groups = group_paths(image_paths(target_dir))
    template_groups = group_paths(image_paths(template_dir))
    method_names = [
        "basicpy_light",
        "basicpy_mid",
        "j4_panel_corrected",
        "j4_vsnr_destripe",
        "j4_basic_rowprofile",
        "j1_panel_corrected",
        "j1_vsnr_destripe",
        "j1_basic_rowprofile",
        "u29_basic_square",
        "u29_mottle_suppressed",
        "u30_mottle_suppressed",
        "u30_basic_rowprofile",
        "fallback_rowprofile",
    ]
    ensure_dirs(output_dir, method_names)

    print(f"[V3] Target input : {target_dir}")
    print(f"[V3] Template input: {template_dir}")
    print(f"[V3] Output       : {output_dir}")
    print(f"[V3] pyvsnr available: {_vsnr2d is not None} | scikit-image bilateral available: {_denoise_bilateral is not None}")

    rows = []
    for component, target_paths in target_groups.items():
        target_images = [read_gray(path) for path in target_paths]
        target_shape = target_images[0].shape
        train_paths = template_groups.get(component, target_paths)
        train_images_all = [read_gray(path) for path in train_paths]
        train_images = [img for img in train_images_all if img.shape == target_shape]
        if not train_images:
            train_images = target_images

        params = params_for(component)
        prefix = component.lower()

        if params.panel_strength > 0:
            # Correct each stack image's panel step before BaSiC sees it, so the
            # fitted flatfield reflects shading only, not the piecewise seam.
            train_images = [correct_panel_step(im, params.panel_strength) for im in train_images]

        print(f"[V3] {component}: fitting BaSiCPy on {min(len(train_images), MAX_FIT_IMAGES)} of {len(train_images)} images.")
        flat, dark, status = fit_basicpy_fields(train_images, target_shape)
        print(f"[V3] {component}: {status}")

        if flat is not None:
            flat_vis = (flat - np.percentile(flat, 1.0)) / (np.percentile(flat, 99.0) - np.percentile(flat, 1.0) + 1e-6) * 255.0
            save_gray(os.path.join(output_dir, "research_fields", f"{component}_basicpy_flatfield.png"), flat_vis)
            if dark is not None:
                dark_vis = (dark - np.percentile(dark, 1.0)) / (np.percentile(dark, 99.0) - np.percentile(dark, 1.0) + 1e-6) * 255.0
                save_gray(os.path.join(output_dir, "research_fields", f"{component}_basicpy_darkfield.png"), dark_vis)

        for path, img in tqdm(list(zip(target_paths, target_images)), desc=f"{component} V3"):
            filename = os.path.basename(path)
            results: Dict[str, np.ndarray] = {}

            basic_input = img
            if params.panel_strength > 0:
                basic_input = correct_panel_step(img, params.panel_strength)
                results[f"{prefix}_panel_corrected"] = basic_input

            if flat is not None and dark is not None:
                basic_light = basicpy_correct(basic_input, flat, dark, strength=min(params.basic_strength, 0.45))
                basic_mid = basicpy_correct(basic_input, flat, dark, strength=params.basic_strength)
                results["basicpy_light"] = basic_light
                results["basicpy_mid"] = basic_mid

                if params.panel_strength > 0:
                    # J4-style artifact profile: hard panel seam + horizontal banding.
                    results[f"{prefix}_vsnr_destripe"] = vsnr_destripe(
                        basic_light,
                        noise_level=params.vsnr_noise_level,
                        sigma=params.vsnr_sigma,
                        theta=0.0,
                    )
                    results[f"{prefix}_basic_rowprofile"] = row_profile_destripe(basic_light, params.row_strength)
                elif component == "U29":
                    # BGA-style artifact profile: square-grid visibility + inter-pad mottle.
                    square = square_cleanup(
                        basic_light,
                        strength=params.square_enhance,
                        denoise_sigma=params.denoise_sigma,
                    )
                    results["u29_basic_square"] = square
                    results["u29_mottle_suppressed"] = mottle_suppress(
                        square,
                        sigma_color=params.mottle_sigma_color,
                        sigma_spatial=params.mottle_sigma_spatial,
                    )
                elif params.mottle_enable:
                    # No seam, no grid -- smooth shading plus fine component-body
                    # graininess. Offer a light destripe and a mottle-suppression
                    # candidate side by side rather than assuming which helps.
                    results[f"{prefix}_basic_rowprofile"] = row_profile_destripe(basic_light, params.row_strength)
                    results[f"{prefix}_mottle_suppressed"] = mottle_suppress(
                        basic_light,
                        sigma_color=params.mottle_sigma_color,
                        sigma_spatial=params.mottle_sigma_spatial,
                    )
            else:
                results["fallback_rowprofile"] = row_profile_destripe(img, params.row_strength)

            for method, result in results.items():
                save_gray(os.path.join(output_dir, method, filename), result)
                rows.append(metric_row(filename, component, method, img, result))

            make_sheet(
                os.path.join(output_dir, "comparison_sheets", os.path.splitext(filename)[0] + "_v3_research.jpg"),
                filename,
                img,
                results,
            )

    metrics_path = os.path.join(output_dir, "metrics_index.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[V3] Done. Sheets: {os.path.join(output_dir, 'comparison_sheets')}")
    print(f"[V3] Research fields: {os.path.join(output_dir, 'research_fields')}")
    print(f"[V3] Metrics: {metrics_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else default_target_dir()
    output = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    template = sys.argv[3] if len(sys.argv) > 3 else default_template_dir()
    process(target, output, template)


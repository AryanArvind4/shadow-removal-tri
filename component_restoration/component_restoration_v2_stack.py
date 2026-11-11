"""
Component-Aware Stack Restoration V2
====================================

This version follows the newer thesis direction:

  - Use the full TRI stack per component family as the template signal.
  - Save restored outputs only for the concise 8-image-per-component set.
  - Estimate repeatable component-family artifact fields from the stack.
  - Apply component-specific cleanup:
      J4  : panel seam + horizontal stripe/banding suppression
      U29 : square-grid preservation + inter-square background cleanup
      J1/U30: stack shading correction + conservative row cleanup

No external packages beyond numpy / scipy / pillow / tqdm are required.

Run from repo root:

    py component_restoration/component_restoration_v2_stack.py

Optional:

    py component_restoration/component_restoration_v2_stack.py <target_dir> <output_dir> <template_dir>

Outputs:

    component_restoration/result/component_restoration_v2_stack/
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, grey_opening, median_filter, sobel
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
EPS = 1.0


@dataclass(frozen=True)
class StackParams:
    shading_sigma_frac: float
    structure_percentile: float
    row_sigma: float
    row_baseline_sigma: float
    row_strength: float
    seam_strength: float
    square_enhance: float = 0.0
    bg_denoise_sigma: float = 1.2


PARAMS: Dict[str, StackParams] = {
    "J1": StackParams(0.060, 80.0, 6.0, 74.0, 0.45, 0.60),
    "J4": StackParams(0.055, 80.0, 5.0, 78.0, 0.70, 0.85),
    "U29": StackParams(0.085, 76.0, 7.0, 48.0, 0.25, 0.35, square_enhance=0.60, bg_denoise_sigma=1.65),
    "U30": StackParams(0.085, 77.0, 7.0, 50.0, 0.30, 0.40),
}

DEFAULT_PARAMS = StackParams(0.075, 78.0, 7.0, 55.0, 0.35, 0.45)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset_Concised",
    )


def default_template_input_dir() -> str:
    return os.path.join(
        repo_root(),
        "TRI_Dataset-20260511T190518Z-3-001",
        "TRI_Dataset",
    )


def default_output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "result", "component_restoration_v2_stack")


def component_from_name(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def params_for_component(component: str) -> StackParams:
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


def group_paths_by_component(paths: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in paths:
        component = component_from_name(os.path.basename(path))
        groups.setdefault(component, []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def structure_mask(img: np.ndarray, percentile: float) -> np.ndarray:
    smooth = gaussian_filter(img, sigma=max(min(img.shape) * 0.010, 2.0), mode="reflect")
    highpass = np.abs(img - smooth)
    grad = gradient_magnitude(img)
    mask = (highpass > np.percentile(highpass, percentile)) | (
        grad > np.percentile(grad, percentile)
    )
    for _ in range(2):
        mask = binary_dilation(mask)
    return mask


def fill_structure_for_background(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace structure pixels with a smooth image before estimating shading."""
    smooth = gaussian_filter(img, sigma=max(min(img.shape) * 0.025, 5.0), mode="reflect")
    return np.where(mask, smooth, img)


def robust_mean_std_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    img_std = float(np.std(img)) + 1e-6
    ref_std = float(np.std(reference)) + 1e-6
    out = (img - np.mean(img)) / img_std * ref_std + np.mean(reference)
    return np.clip(out, 0, 255)


def robust_percentile_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    lo_i, hi_i = np.percentile(img, (0.4, 99.6))
    lo_r, hi_r = np.percentile(reference, (0.4, 99.6))
    if hi_i - lo_i < 1e-6:
        return robust_mean_std_match(img, reference)
    out = (img - lo_i) / (hi_i - lo_i) * (hi_r - lo_r) + lo_r
    out += np.mean(reference) - np.mean(out)
    return np.clip(out, 0, 255)


def estimate_component_shading_template(images: List[np.ndarray], params: StackParams) -> np.ndarray:
    """
    Estimate a shared log-domain artifact field for a component stack.

    The template is intentionally low-frequency. Structure is suppressed per
    image first, each log image is mean-centered, then the stack median gives a
    repeatable component-family field.
    """
    fields = []
    for img in images:
        mask = structure_mask(img, params.structure_percentile)
        bg_img = fill_structure_for_background(img, mask)
        log_bg = np.log(bg_img + EPS)
        sigma = max(min(img.shape) * params.shading_sigma_frac, 14.0)
        field = gaussian_filter(log_bg, sigma=sigma, mode="reflect")
        field = field - np.median(field)
        fields.append(field)

    template = np.median(np.stack(fields, axis=0), axis=0)
    template = gaussian_filter(template, sigma=max(min(images[0].shape) * 0.010, 3.0), mode="reflect")
    return template


def apply_stack_shading_correction(img: np.ndarray, template: np.ndarray, strength: float = 1.0) -> np.ndarray:
    log_img = np.log(img + EPS)
    corrected = np.exp(log_img - strength * template) - EPS
    return robust_percentile_match(corrected, img)


def detect_panel_seam(img: np.ndarray) -> int:
    h, w = img.shape
    profile = gaussian_filter(np.median(img, axis=0), sigma=max(w * 0.008, 5.0), mode="reflect")
    grad = np.abs(np.gradient(profile))
    lo, hi = int(w * 0.25), int(w * 0.75)
    if hi <= lo:
        return w // 2
    return lo + int(np.argmax(grad[lo:hi]))


def correct_panel_step(img: np.ndarray, strength: float) -> np.ndarray:
    seam = detect_panel_seam(img)
    h, w = img.shape
    pad = max(int(w * 0.035), 12)

    left = img[:, : max(seam - pad, 1)]
    right = img[:, min(seam + pad, w - 1) :]
    if left.size == 0 or right.size == 0:
        return img

    left_med = np.median(left)
    right_med = np.median(right)
    target = 0.5 * (left_med + right_med)
    left_gain = target / max(left_med, 1.0)
    right_gain = target / max(right_med, 1.0)

    out = img.copy()
    out[:, :seam] *= 1.0 + strength * (left_gain - 1.0)
    out[:, seam:] *= 1.0 + strength * (right_gain - 1.0)

    # Feather around seam so correction itself does not create a hard edge.
    x0, x1 = max(0, seam - pad), min(w, seam + pad)
    if x1 > x0:
        alpha = np.linspace(0.0, 1.0, x1 - x0, dtype=np.float32)[None, :]
        left_version = img[:, x0:x1] * (1.0 + strength * (left_gain - 1.0))
        right_version = img[:, x0:x1] * (1.0 + strength * (right_gain - 1.0))
        out[:, x0:x1] = (1.0 - alpha) * left_version + alpha * right_version

    return np.clip(out, 0, 255)


def remove_row_stripes_masked(img: np.ndarray, params: StackParams) -> np.ndarray:
    """
    Estimate row artifact from non-structure pixels, then subtract only a
    clipped 1D row profile. This targets J4 horizontal seams without using
    column balancing that made the previous result worse.
    """
    mask = structure_mask(img, params.structure_percentile)
    valid = (~mask).astype(np.float32)
    row_sum = np.sum(img * valid, axis=1)
    row_count = np.maximum(np.sum(valid, axis=1), 1.0)
    row_profile = row_sum / row_count

    medium = gaussian_filter(row_profile, sigma=params.row_sigma, mode="reflect")
    broad = gaussian_filter(row_profile, sigma=params.row_baseline_sigma, mode="reflect")
    stripe = medium - broad
    limit = np.percentile(np.abs(stripe), 92.0)
    stripe = np.clip(stripe, -limit, limit)

    corrected = img - params.row_strength * stripe[:, None]
    return robust_mean_std_match(corrected, img)


def square_pad_mask(img: np.ndarray) -> np.ndarray:
    size = max(9, int(round(min(img.shape) * 0.050)))
    if size % 2 == 0:
        size += 1
    opened = grey_opening(img, size=(size, size), mode="reflect")
    top_hat = img - opened
    grad = gradient_magnitude(img)
    mask = (top_hat > np.percentile(top_hat, 82.0)) | (grad > np.percentile(grad, 82.0))
    for _ in range(2):
        mask = binary_dilation(mask)
    return mask


def cleanup_u29_squares(img: np.ndarray, params: StackParams) -> np.ndarray:
    mask = square_pad_mask(img)
    bg = gaussian_filter(img, sigma=params.bg_denoise_sigma, mode="reflect")
    bg = 0.35 * bg + 0.65 * median_filter(img, size=3, mode="reflect")

    size = max(9, int(round(min(img.shape) * 0.050)))
    if size % 2 == 0:
        size += 1
    opened = grey_opening(img, size=(size, size), mode="reflect")
    square_response = np.clip(img - opened, 0, np.percentile(img - opened, 99.2))
    enhanced = img + 0.55 * params.square_enhance * square_response

    out = np.where(mask, enhanced, 0.65 * img + 0.35 * bg)
    return robust_percentile_match(out, img)


def restore_image(img: np.ndarray, component: str, template: np.ndarray) -> Dict[str, np.ndarray]:
    params = params_for_component(component)
    shaded_light = apply_stack_shading_correction(img, template, strength=0.35)
    shaded_mid = apply_stack_shading_correction(img, template, strength=0.60)
    shaded_strong = apply_stack_shading_correction(img, template, strength=0.85)
    results = {
        "stack_shading_light": shaded_light,
        "stack_shading_mid": shaded_mid,
        "stack_shading_strong": shaded_strong,
    }

    if component == "J4":
        panel_light = correct_panel_step(shaded_light, strength=0.35)
        panel_mid = correct_panel_step(shaded_mid, strength=0.55)
        stripe_light = remove_row_stripes_masked(panel_light, params)
        stripe_mid = remove_row_stripes_masked(panel_mid, params)
        results["j4_panel_light"] = panel_light
        results["j4_stack_panel_stripe_light"] = stripe_light
        results["j4_stack_panel_stripe_mid"] = stripe_mid
    elif component == "U29":
        square_light = cleanup_u29_squares(shaded_light, params)
        square_mid = cleanup_u29_squares(shaded_mid, params)
        results["u29_stack_square_light"] = square_light
        results["u29_stack_square_mid"] = square_mid
    else:
        stripes = remove_row_stripes_masked(shaded_mid, params)
        results["stack_row_cleanup"] = stripes

    return results


def row_band_score(img: np.ndarray) -> float:
    profile = np.median(img, axis=1)
    low = gaussian_filter(profile, sigma=max(img.shape[0] * 0.10, 20.0), mode="reflect")
    return float(np.std(profile - low))


def panel_step_score(img: np.ndarray) -> float:
    seam = detect_panel_seam(img)
    w = img.shape[1]
    pad = max(int(w * 0.035), 12)
    left = img[:, max(0, seam - 3 * pad) : max(1, seam - pad)]
    right = img[:, min(w - 1, seam + pad) : min(w, seam + 3 * pad)]
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(abs(np.median(left) - np.median(right)))


def square_score(img: np.ndarray) -> float:
    size = max(9, int(round(min(img.shape) * 0.050)))
    if size % 2 == 0:
        size += 1
    opened = grey_opening(img, size=(size, size), mode="reflect")
    response = img - opened
    return float(np.percentile(response, 95.0) / (np.std(response) + 1e-6))


def residual_structure_leak(input_img: np.ndarray, output_img: np.ndarray) -> float:
    residual = input_img - output_img
    mask = structure_mask(input_img, 80.0)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs(residual[mask])) / (np.mean(np.abs(residual)) + 1e-6))


def metric_row(filename: str, method: str, original: np.ndarray, output: np.ndarray) -> Dict[str, object]:
    return {
        "filename": filename,
        "component": component_from_name(filename),
        "method": method,
        "row_band_before": row_band_score(original),
        "row_band_after": row_band_score(output),
        "panel_step_before": panel_step_score(original),
        "panel_step_after": panel_step_score(output),
        "square_score_before": square_score(original),
        "square_score_after": square_score(output),
        "residual_structure_leak": residual_structure_leak(original, output),
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


def make_contact_sheet(path: str, filename: str, original: np.ndarray, results: Dict[str, np.ndarray]) -> None:
    panels = [label_image(original, f"input: {filename}")]
    panels += [label_image(img, name) for name, img in results.items()]
    width = panels[0].width
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width * len(panels), height), (28, 28, 28))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def ensure_dirs(output_dir: str, method_names: Iterable[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison_sheets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stack_templates"), exist_ok=True)
    for name in method_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def process(input_dir: str, output_dir: str, template_input_dir: str) -> List[Dict[str, object]]:
    paths = image_paths(input_dir)
    template_paths = image_paths(template_input_dir)
    if not paths:
        raise FileNotFoundError(f"No target images found in {input_dir}")
    if not template_paths:
        raise FileNotFoundError(f"No template images found in {template_input_dir}")

    groups = group_paths_by_component(paths)
    template_groups = group_paths_by_component(template_paths)
    method_names = [
        "stack_shading_light",
        "stack_shading_mid",
        "stack_shading_strong",
        "j4_panel_light",
        "j4_stack_panel_stripe_light",
        "j4_stack_panel_stripe_mid",
        "u29_stack_square_light",
        "u29_stack_square_mid",
        "stack_row_cleanup",
    ]
    ensure_dirs(output_dir, method_names)

    print(f"[Stack V2] Target input : {input_dir}")
    print(f"[Stack V2] Template input: {template_input_dir}")
    print(f"[Stack V2] Output       : {output_dir}")
    print("[Stack V2] Target components  : " + ", ".join(f"{k}({len(v)})" for k, v in groups.items()))
    print("[Stack V2] Template components: " + ", ".join(f"{k}({len(v)})" for k, v in template_groups.items()))

    rows: List[Dict[str, object]] = []
    for component, component_paths in groups.items():
        if component not in template_groups:
            print(f"[Stack V2] Warning: no template images for {component}; using target images only.")
            component_template_paths = component_paths
        else:
            component_template_paths = template_groups[component]

        params = params_for_component(component)
        template_images = [read_gray(path) for path in component_template_paths]
        target_images = [read_gray(path) for path in component_paths]

        target_shape = target_images[0].shape
        compatible_template_images = [
            img for img in template_images if img.shape == target_shape
        ]
        if not compatible_template_images:
            print(
                f"[Stack V2] Warning: no same-shape template images for {component}; "
                "using target images only."
            )
            compatible_template_images = target_images

        print(
            f"[Stack V2] {component}: estimating template from "
            f"{len(compatible_template_images)} images, applying to {len(target_images)} images."
        )

        template = estimate_component_shading_template(compatible_template_images, params)

        # Visualize the log template as an inspectable artifact map.
        template_vis = (template - np.percentile(template, 1.0)) / (
            np.percentile(template, 99.0) - np.percentile(template, 1.0) + 1e-6
        ) * 255.0
        save_gray(os.path.join(output_dir, "stack_templates", f"{component}_template.png"), template_vis)

        for path, img in tqdm(list(zip(component_paths, target_images)), desc=f"{component} stack restore"):
            filename = os.path.basename(path)
            results = restore_image(img, component, template)
            for method, output in results.items():
                save_gray(os.path.join(output_dir, method, filename), output)
                rows.append(metric_row(filename, method, img, output))

            make_contact_sheet(
                os.path.join(
                    output_dir,
                    "comparison_sheets",
                    os.path.splitext(filename)[0] + "_stack_v2.jpg",
                ),
                filename,
                img,
                results,
            )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"[Stack V2] Metrics: {metrics_path}")
    print(f"[Stack V2] Sheets : {os.path.join(output_dir, 'comparison_sheets')}")
    return rows


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 84)
    print("COMPONENT STACK RESTORATION V2 SUMMARY")
    print("=" * 84)
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        row_after = np.mean([float(row["row_band_after"]) for row in subset])
        step_after = np.mean([float(row["panel_step_after"]) for row in subset])
        square_after = np.mean([float(row["square_score_after"]) for row in subset])
        leak = np.mean([float(row["residual_structure_leak"]) for row in subset])
        print(
            f"{method:24s} | row band: {row_after:7.3f} | "
            f"panel step: {step_after:7.3f} | square: {square_after:6.3f} | "
            f"leak: {leak:5.2f}"
        )
    print("=" * 84)
    print("Inspect sheets first. Template maps should look like artifacts, not solder balls.")


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir()
    template_dir = sys.argv[3] if len(sys.argv) > 3 else default_template_input_dir()
    process(in_dir, out_dir, template_dir)

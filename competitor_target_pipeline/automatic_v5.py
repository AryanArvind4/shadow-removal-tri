"""Image-driven V5 correction. Proposals are diagnostics, never defect labels."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

import competitor_target_v3 as base


@dataclass(frozen=True)
class Settings:
    analysis_size: int = 768
    flatten_strength: float = .85
    correction_limit: float = 28.
    maximum_gain: float = 1.6
    denoise_strength: float = .85


def noise_sigma(raw):
    # A diagonal difference cancels constant and linear shading.
    diagonal = (raw[1:, 1:] - raw[1:, :-1] - raw[:-1, 1:] + raw[:-1, :-1]) / 2
    independent = float(np.median(np.abs(diagonal - np.median(diagonal))) / .67449)
    correlated = float(np.median(np.abs(raw - base.blur(raw, 2.))) / .67449)
    return max(.35, independent, correlated)


def band_field(image, diameter, candidates, noise):
    """Attenuate shared row fluctuations only in dense, small-feature scenes."""
    h, w = image.shape
    empty = np.zeros_like(image)
    if candidates < 100 or diameter / min(h, w) >= .025:
        return empty, dict(applied=False, reason='No dense small-feature evidence')
    labels, _ = diagnostic_regions(image, noise, diameter)
    excluded = cv2.dilate((labels > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    profiles, centers = [], []
    for indices in np.array_split(np.arange(w), 6):
        start, end = int(indices[0]), int(indices[-1]) + 1
        patch = image[:, start:end]
        valid = ~excluded[:, start:end]
        profile = np.full(h, np.nan, np.float32)
        for y in range(h):
            if valid[y].sum() >= max(8, patch.shape[1] // 4):
                profile[y] = np.median(patch[y, valid[y]])
        good = np.isfinite(profile)
        if good.sum() < h * .6:
            return empty, dict(applied=False, reason='Insufficient background support')
        profile = np.interp(np.arange(h), np.flatnonzero(good), profile[good]).astype(np.float32)
        profiles.append(base.blur(profile, 1.) - base.blur(profile, max(5., h * .035)))
        centers.append((start + end - 1) / 2)
    profiles = np.asarray(profiles)
    correlations = []
    for i in range(5):
        if min(np.std(profiles[i]), np.std(profiles[i + 1])) > noise * .5:
            correlations.append(float(np.corrcoef(profiles[i], profiles[i + 1])[0, 1]))
    agreement = float(np.median(correlations)) if correlations else 0.
    if agreement < .35:
        return empty, dict(applied=False, reason='Rows are not consistent across strips', agreement=agreement)
    delta = -6 * np.tanh(.75 * profiles / 6)
    field = np.stack([np.interp(np.arange(w), centers, delta[:, y]) for y in range(h)]).astype(np.float32)
    field = base.blur(field, (.8, 2.))
    return field, dict(applied=True, agreement=agreement, max_abs=float(np.abs(field).max()),
                       note='Consistent row structure is not necessarily an acquisition artifact.')


def smooth_noise(raw, noise, strength):
    # Patch comparisons suppress repeated grain more effectively than a single
    # small blur. Blend with the native source; the NLM estimate alone is uint8.
    filtered = cv2.fastNlMeansDenoising(base.u8(raw), None, max(2., 1.6 * noise), 7, 21).astype(np.float32)
    change = raw - filtered
    # Large signed differences are retained, including dark features. This is a
    # safeguard, not proof that small defects cannot be affected by denoising.
    weight = strength / (1 + (np.abs(change) / max(1., 2.5 * noise)) ** 4)
    return raw - weight * change, weight


def estimate_scale(small, noise):
    short = min(small.shape)
    residual = small - base.blur(small, max(2., short * .035))
    candidates = []
    for threshold in (max(2., 2 * noise), max(4., 4 * noise), max(8., 6 * noise)):
        binary = (residual > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (3 <= min(w, h) and max(w, h) < short * .23
                    and max(w, h) / min(w, h) < 4 and area > 8
                    and area / (w * h) > .5):
                candidates.append(float(max(w, h)))
    diameter = float(np.percentile(candidates, 65)) if candidates else short * .04
    return float(np.clip(diameter, 4, short * .15)), len(candidates)


def diagnostic_regions(image, noise, diameter):
    """Compact bright supports for inspection only; no pixels are replaced."""
    residual = image - base.blur(image, max(2., diameter))
    binary = (residual > max(3., 2.5 * noise)).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = np.zeros(image.shape, np.int32)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if min(w, h) < 4 or area < 16 or max(w, h) > min(image.shape) * .3:
            continue
        if area / (w * h) < .5 or max(w, h) / min(w, h) > 4:
            continue
        identifier = len(regions) + 1
        cv2.drawContours(labels, [contour], -1, identifier, -1)
        regions.append((identifier, x, y, w, h))
    return labels, regions


def tone(image, background_level, signal_span):
    # A single continuous increasing display curve, with headroom at both ends.
    span = max(24., float(signal_span))
    z = np.clip(4.2 * (image - background_level - .40 * span) / span, -30, 30)
    return (8 + 239 / (1 + np.exp(-z))).astype(np.float32)


def process(raw, component=None, settings=Settings()):
    """Same array gives the same result regardless of filename or component."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or min(raw.shape) < 32 or not np.isfinite(raw).all():
        raise ValueError('Expected a finite grayscale image with both dimensions >= 32.')
    if raw.min() < 0 or raw.max() > 255:
        raise ValueError('Automatic V5 expects grayscale intensities in [0, 255].')
    if not (0 <= settings.flatten_strength <= 1 and settings.correction_limit > 0
            and 1 <= settings.maximum_gain <= 2 and 0 <= settings.denoise_strength <= 1
            and settings.analysis_size >= 64):
        raise ValueError('Invalid automatic correction settings.')
    h, w = raw.shape
    noise = noise_sigma(raw)
    clean, smoothing = smooth_noise(raw, noise, settings.denoise_strength)
    scale = min(1., settings.analysis_size / max(h, w))
    size = (max(32, round(w * scale)), max(32, round(h * scale)))
    small = cv2.resize(clean, size, interpolation=cv2.INTER_AREA)
    small_noise = noise_sigma(small)
    diameter, evidence = estimate_scale(small, small_noise)
    bands_small, bands_info = band_field(small, diameter, evidence, small_noise)
    bands = cv2.resize(bands_small, (w, h), interpolation=cv2.INTER_LINEAR)
    small = small + bands_small
    radius = int(np.ceil(max(min(small.shape) * .04, diameter * 1.35)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Opening estimates a lower background envelope. Only this estimated field
    # is smoothed/upscaled; native-resolution image features are not resized.
    envelope = cv2.morphologyEx(base.blur(small, .8), cv2.MORPH_OPEN, kernel,
                              borderType=cv2.BORDER_REFLECT)
    background_small = base.blur(envelope, max(2., diameter * .45))
    background = cv2.resize(background_small, (w, h), interpolation=cv2.INTER_LINEAR)
    anchor = float(np.median(background))
    correction = settings.correction_limit * np.tanh(
        settings.flatten_strength * (anchor - background) / settings.correction_limit)
    flattened = clean + correction + bands

    residual = small - background_small
    # Estimate visibility from coarse tiles, not per-pad equalization. Tiles
    # without clear positive signal receive no extra contrast boost.
    ny, nx = max(2, round(small.shape[0] / (4 * diameter))), max(2, round(small.shape[1] / (4 * diameter)))
    levels = np.zeros((ny, nx), np.float32)
    for iy, ys in enumerate(np.array_split(np.arange(small.shape[0]), ny)):
        for ix, xs in enumerate(np.array_split(np.arange(small.shape[1]), nx)):
            levels[iy, ix] = np.percentile(residual[np.ix_(ys, xs)], 95)
    valid = levels > max(3., 3 * small_noise)
    target = float(np.percentile(levels[valid], 65)) if valid.any() else 0.
    confidence = np.clip((levels - 2 * small_noise) / max(2., 4 * small_noise), 0, 1)
    gain_small = 1 + confidence * (np.clip(target / np.maximum(levels, 1), 1, settings.maximum_gain) - 1)
    gain_small[~valid] = 1
    gain = cv2.resize(base.blur(gain_small, .65), (w, h), interpolation=cv2.INTER_LINEAR)
    # Bound additional amplification in noisy images and the actual intensity change.
    gain = np.minimum(gain, 1 + .6 / (1 + (noise / 4.) ** 2))
    corrected = flattened + 14 * np.tanh((gain - 1) * (flattened - anchor) / 14)
    span = max(24., float(np.percentile(corrected, 99.5) - anchor))
    reference = tone(corrected, anchor, span)
    detail = tone(raw + correction + bands, anchor, span)
    if float(np.ptp(raw)) < 1e-6:
        reference = raw.copy()
        detail = raw.copy()
    labels, regions = diagnostic_regions(clean, noise, diameter / scale)
    outputs = dict(reference_gray=reference, detail_gray=detail,
                   region_mask=(labels > 0).astype(np.float32) * 255,
                   gain_field=(gain - 1) / .6 * 255,
                   background_smoothing=smoothing * 255,
                   band_correction=128 + bands * 127 / 6,
                   correction_field=128 + correction * 127 / settings.correction_limit)
    diagnostics = dict(mode='automatic', settings=asdict(settings), noise_sigma=noise,
                       display_background=anchor, display_signal_span=span,
                       feature_diameter_native=diameter / scale, scale_candidates=evidence,
                       background_radius_native=radius / scale, max_gain=float(gain.max()),
                       bands=bands_info,
                       correction_max_abs=float(np.abs(correction).max()),
                       saturation_fraction=float(np.mean((reference <= 1) | (reference >= 254))),
                       estimated_healthy_pixels=False,
                       note='No manual labels, fixed pad grid, component routing or learned model. '
                            'Region statistics include possible defects and are not healthy-pad metrics.')
    return outputs, labels, regions, corrected, diagnostics

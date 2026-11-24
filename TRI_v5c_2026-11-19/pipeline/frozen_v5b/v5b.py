"""V5b: source-background compositing around source-supported components."""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'frozen_v5a'))
import competitor_target_v5a as v5a


@dataclass(frozen=True)
class Settings:
    feather_fraction: float = .07
    minimum_region_area: int = 16


def filled_contour(contour, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    return mask


def component_support(raw, labels, regions, info, settings=Settings()):
    """Estimate outer supports; filling mask holes never fills image cavities."""
    smooth = v5a.base.blur(raw, .8)
    noise = info['noise_sigma']
    diameter = info['feature_diameter_native']
    support = np.zeros(raw.shape, np.uint8)
    records = []
    if regions:
        for identifier, x, y, w, h in regions:
            margin = max(3, round(min(w, h) * .20))
            x0, y0 = max(0, x-margin), max(0, y-margin)
            x1, y1 = min(raw.shape[1], x+w+margin), min(raw.shape[0], y+h+margin)
            patch = smooth[y0:y1, x0:x1]
            ring = np.ones(patch.shape, bool)
            ring[y-y0:y-y0+h, x-x0:x-x0+w] = False
            level = float(np.percentile(patch[ring], 50)) if ring.any() else float(patch.min())
            peak = float(np.percentile(smooth[y:y+h, x:x+w], 80))
            contrast = peak-level
            if contrast < max(3., noise * 2):
                continue
            # Marker watershed follows source gradients between a pad-interior
            # seed and its surrounding background. Interior cavities share the
            # support but retain their original enhanced intensities.
            markers = np.zeros(patch.shape, np.int32)
            markers[:2] = markers[-2:] = 1
            markers[:, :2] = markers[:, -2:] = 1
            inset = max(2, round(min(w,h)*.22))
            markers[y-y0+inset:y-y0+h-inset, x-x0+inset:x-x0+w-inset] = 2
            normalized = np.clip((patch-level)/max(contrast, 1.)*170+40, 0, 255).astype(np.uint8)
            cv2.watershed(cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), markers)
            local = (markers == 2).astype(np.uint8)
            if local.sum() < max(settings.minimum_region_area, .30*w*h):
                continue
            support[y0:y1, x0:x1] |= local
            records.append(dict(kind='refined_pad', proposal_id=identifier, pixels=int(local.sum())))
        method = 'source_gradient_watershed_with_v5a_proposals'
    else:
        # Two context scales cover small pads and broader plate regions. Long
        # traces and image-border contours are rejected; uncertain areas stay raw.
        candidates = []
        short = min(raw.shape)
        detection = v5a.base.blur(raw, max(.8, min(2., diameter*.06)))
        detection_noise = v5a.auto.noise_sigma(detection)
        for sigma in (max(3., diameter*.9), max(diameter*2., short*.16)):
            signal = detection-v5a.base.blur(detection, sigma)
            for threshold in (max(2., 1.6*detection_noise), max(4., 3*detection_noise)):
                binary = (signal > threshold).astype(np.uint8)
                # Closing connects short breaks in a support mask only.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    solidity = area/max(1., cv2.contourArea(cv2.convexHull(contour)))
                    if (min(w,h) < max(4, diameter*.24) or max(w,h) > short*.58
                            or max(w,h)/min(w,h) > 4.5 or area < settings.minimum_region_area
                            or area/(w*h) < .36 or solidity < .62
                            or 4*np.pi*area/max(1., perimeter**2) < .12
                            or x <= 1 or y <= 1 or x+w >= raw.shape[1]-1 or y+h >= raw.shape[0]-1):
                        continue
                    candidates.append((area, contour))
        for area, contour in sorted(candidates, key=lambda item: -item[0]):
            local = filled_contour(contour, raw.shape)
            if np.count_nonzero(local & support) > .15 * local.sum():
                continue
            support |= local
            records.append(dict(kind='compact_source_contour', pixels=int(local.sum())))
        method = 'multiscale_source_contours'
    return support.astype(bool), dict(method=method, regions=records, count=len(records))


def composite(raw, enhanced, support, diameter, settings=Settings()):
    """Keep raw outside support and suppress additional darkening at its edge."""
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    width = float(np.clip(diameter*settings.feather_fraction, 1.5, 4.))
    t = np.clip(distance/width, 0, 1)
    alpha = t*t*(3-2*t)
    edge = 1-np.clip((distance-width)/max(width, 1.), 0, 1)
    delta = enhanced-raw
    restored = np.maximum(-delta, 0)*edge
    target = enhanced+restored
    output = raw+alpha*(target-raw)
    output[~support] = raw[~support]
    return output, alpha, target, alpha*restored


def render_original_mode(source, result):
    """Retain original RGB background and annotations when the source is RGB."""
    if source.ndim == 2:
        final = v5a.base.u8(result['output'])
    else:
        alpha = result['alpha'][..., None]
        final = v5a.base.u8(source.astype(np.float32)+alpha*(result['target'][..., None]-source))
    final[~result['support']] = source[~result['support']]
    return final


def process(raw, settings=Settings(), baseline=None, protected=None):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or min(raw.shape) < 32 or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
        raise ValueError('Expected finite 8-bit grayscale intensities, minimum size 32')
    if not 0 < settings.feather_fraction < .5 or settings.minimum_region_area < 1:
        raise ValueError('Invalid support settings')
    baseline = v5a.process(raw) if baseline is None else baseline
    images, labels, regions, _, diagnostics, _ = baseline
    enhanced = images['reference_gray']
    info = diagnostics['baseline_diagnostics']
    support, selection = component_support(raw, labels, regions, info, settings)
    if protected is not None:
        support[np.asarray(protected, bool)] = False
    output, alpha, target, halo = composite(raw, enhanced, support, info['feature_diameter_native'], settings)
    bg = ~support
    details = dict(settings=asdict(settings), selection=selection, v5a_layout=diagnostics['layout'],
                   support_fraction=float(support.mean()),
                   background_max_abs_change=float(np.max(np.abs(output[bg]-raw[bg]))) if bg.any() else None,
                   boundary_darkening_reduced_pixels=int(np.count_nonzero(halo > .01)),
                   boundary_darkening_reduction_max=float(halo.max()),
                   limitation='Supports are unreviewed. Original dark rims remain. V5a denoising and interior tone changes can affect faint features.')
    return dict(output=output, alpha=alpha, support=support, target=target, halo=halo,
                v5a=enhanced, details=details)

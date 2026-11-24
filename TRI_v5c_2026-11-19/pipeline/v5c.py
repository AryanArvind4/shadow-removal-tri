"""V5c: source-guided supports and monotone component lifting for review."""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent / 'frozen_v5b'))
import v5b


@dataclass(frozen=True)
class Settings:
    weak_fraction: float = .10
    strong_fraction: float = .40
    minimum_contrast: float = 3.
    noise_multiplier: float = 2.5
    maximum_slope: float = 6.
    white_level: float = 245.


def generic_proposals(raw, support):
    """Conservative compact-object filter; uncertain shapes remain unchanged."""
    count, cc, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    for i in range(1, count):
        x, y, w, h = map(int, stats[i, :4])
        local = (cc[y:y+h, x:x+w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        solidity = area / max(1., cv2.contourArea(cv2.convexHull(contour)))
        occupancy = area / (w * h)
        margin = max(5, round(min(w, h) * .24))
        complete_ring = x >= margin and y >= margin and x+w+margin <= raw.shape[1] and y+h+margin <= raw.shape[0]
        reason = None
        if not complete_ring:
            reason = 'incomplete background context'
        elif solidity < .85 or occupancy < .62:
            reason = 'uncertain fragmented or diffuse object'
        # Yield one full-size prior at a time, not one allocation per object.
        yield (x, y, w, h), (cc == i) if reason is None else None, reason


def background_plane(patch, ring):
    """Fit a slowly varying background without the component or bright neighbors."""
    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]].astype(np.float32)
    design = np.stack([np.ones_like(xx), xx / max(1, patch.shape[1]),
                       yy / max(1, patch.shape[0])], axis=-1)
    samples = patch[ring]
    lo, hi = np.percentile(samples, [10, 75])
    valid = ring & (patch >= lo) & (patch <= hi)
    if valid.sum() < 12:
        return np.full(patch.shape, np.median(samples), np.float32)
    a, values = design[valid], patch[valid]
    fit = np.linalg.lstsq(a, values, rcond=None)[0]
    for _ in range(3):
        error = values - a @ fit
        scale = max(1., 1.4826 * float(np.median(np.abs(error - np.median(error)))))
        weights = np.minimum(1., 1.5 * scale / np.maximum(np.abs(error), .01))
        fit = np.linalg.lstsq(a * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
    plane = design @ fit
    # Bound extrapolation through the component; no physical flat-field claim.
    return np.clip(plane, np.percentile(samples, 10) - 2,
                   np.percentile(samples, 90) + 2).astype(np.float32)


def source_region(raw, box, noise, settings=Settings(), prior=None):
    """Hysteresis keeps source-supported thin rims; no mask closing across gaps."""
    x, y, w, h = map(int, box)
    margin = max(5, round(min(w, h) * .24))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(raw.shape[1], x + w + margin), min(raw.shape[0], y + h + margin)
    patch = raw[y0:y1, x0:x1]
    smooth = cv2.GaussianBlur(patch, (0, 0), .45)
    interior = np.zeros(patch.shape, bool)
    interior[y-y0:y-y0+h, x-x0:x-x0+w] = True
    ring = ~cv2.dilate(interior.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if ring.sum() < 12:
        return None, dict(accepted=False, reason='insufficient background ring', box=list(box))
    plane = background_plane(smooth, ring)
    residual = smooth - plane
    values = residual[interior]
    contrast = float(np.percentile(values, 80))
    local_noise = max(.35, min(noise, v5b.v5a.auto.noise_sigma(patch)))
    record = dict(box=list(box), contrast=contrast, noise=local_noise)
    if contrast < max(settings.minimum_contrast, settings.noise_multiplier * local_noise):
        return None, dict(record, accepted=False, reason='weak source contrast')
    strong = max(settings.strong_fraction * contrast, 2 * local_noise)
    weak = max(settings.weak_fraction * contrast, .7 * local_noise)
    # A small margin recovers soft source edges beyond the gradient-centre box.
    expansion = max(2, round(min(w, h) * .12))
    envelope = cv2.dilate(interior.astype(np.uint8), np.ones((2*expansion+1, 2*expansion+1), np.uint8)) > 0
    if prior is not None:
        local_prior = prior[y0:y1, x0:x1].astype(np.uint8)
        envelope &= cv2.dilate(local_prior, np.ones((3, 3), np.uint8)) > 0
    seeds = (residual > strong) & interior
    if seeds.sum() < max(9, .04 * w * h):
        return None, dict(record, accepted=False, reason='too few source seeds')
    allowed = (residual > weak) & envelope
    count, components = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    ids, sizes = np.unique(components[seeds & allowed], return_counts=True)
    ids = ids[(ids != 0) & (sizes >= max(4, .015 * w * h))]
    selected = np.isin(components, ids)
    # Only truly enclosed holes join the support. Eight-connected background
    # leaves diagonal or narrow open notches connected to the exterior.
    selected = ndimage.binary_fill_holes(selected, structure=np.ones((3, 3), bool))
    if selected.sum() < max(16, .20 * w * h):
        return None, dict(record, accepted=False, reason='insufficient connected support')
    selected[[0, -1], :] = False
    selected[:, [0, -1]] = False
    distance_to_box = cv2.distanceTransform((~interior).astype(np.uint8), cv2.DIST_L2, 5)
    outer_taper = np.clip(1 - distance_to_box / (expansion + 1), 0, 1)
    # Confidence fades to zero at the weak threshold. Enclosed dark holes stay
    # dark through the intensity mapping, rather than through mask subtraction.
    q = np.clip((residual - weak) / max(1., .35 * contrast), 0, 1)
    alpha = q * q * (3 - 2 * q) * outer_taper
    alpha[~selected] = 0
    if prior is not None:
        distance = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 5)
        t = np.clip(distance / 3., 0, 1)
        alpha *= t * t * (3 - 2 * t)
    record.update(accepted=True, reason='source hysteresis', pixels=int(selected.sum()), weak_threshold=weak)
    return dict(slices=(slice(y0, y1), slice(x0, x1)), support=selected, alpha=alpha,
                background=plane, seeds=seeds & selected, contrast=contrast), record


def monotone_lift(raw, background, anchor_source, anchor_target, contrast, settings=Settings()):
    """Positive source contrast controls lift; dark cavities are never painted in."""
    headroom = np.maximum(settings.white_level - raw, 0)
    available = max(1., settings.white_level - anchor_source)
    target = np.clip(anchor_target, anchor_source, settings.white_level - 1)
    rate = -np.log(max(.001, 1 - (target - anchor_source) / available)) / max(contrast, 1)
    rate = min(rate, (settings.maximum_slope - 1) / max(1., settings.white_level - float(background.min())))
    return raw + headroom * (-np.expm1(-rate * np.maximum(raw - background, 0))), float(rate)


def process(raw, settings=Settings(), baseline=None, protected=None):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or min(raw.shape) < 32 or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
        raise ValueError('Expected finite grayscale values in [0,255], minimum size 32')
    if not (0 < settings.weak_fraction < settings.strong_fraction < 1 and
            settings.minimum_contrast > 0 and settings.noise_multiplier > 0 and
            1 <= settings.maximum_slope <= 12 and 128 <= settings.white_level <= 255):
        raise ValueError('Invalid V5c settings')
    if protected is not None and np.asarray(protected).shape != raw.shape:
        raise ValueError('Protected mask shape mismatch')
    baseline = v5b.v5a.process(raw) if baseline is None else baseline
    images, labels, regions, _, diagnostics, _ = baseline
    info = diagnostics['baseline_diagnostics']
    enhanced = np.asarray(images['reference_gray'], np.float32)
    legacy = v5b.process(raw, baseline=baseline, protected=protected)
    if regions:
        proposals = ((tuple(r[1:]), None, None) for r in regions)
        method = 'source_hysteresis_from_rectangular_proposals'
    else:
        proposals = generic_proposals(raw, legacy['support'])
        method = 'source_contrast_validation_of_v5b_proposals'
    output, target = raw.copy(), raw.copy()
    alpha = np.zeros_like(raw)
    support = np.zeros(raw.shape, bool)
    records = []
    for box, prior, rejected in proposals:
        if rejected:
            records.append(dict(box=list(box), accepted=False, reason=rejected))
            continue
        region, record = source_region(raw, box, info['noise_sigma'], settings, prior)
        records.append(record)
        if region is None:
            continue
        sl = region['slices']
        source = raw[sl]
        seed = region['seeds']
        anchor_source = float(np.percentile(source[seed], 65))
        anchor_target = float(np.percentile(enhanced[sl][seed], 65))
        mapped, rate = monotone_lift(source, region['background'], anchor_source,
                                     anchor_target, region['contrast'], settings)
        if rate <= 0:
            record.update(accepted=False, reason='no positive enhancement required', rate=rate)
            continue
        a = region['alpha'].copy()
        if protected is not None:
            a[np.asarray(protected, bool)[sl]] = 0
        take = a > alpha[sl]
        target[sl][take] = mapped[take]
        alpha[sl][take] = a[take]
        support[sl] |= a > 0
        record.update(rate=rate, source_anchor=anchor_source, v5a_anchor=anchor_target)
    output = raw + alpha * (target - raw)
    output[~support] = raw[~support]
    details = dict(settings=asdict(settings), selection=dict(method=method, regions=records,
                   accepted=sum(r['accepted'] for r in records), proposed=len(records)),
                   support_fraction=float(support.mean()),
                   background_exact=bool(np.array_equal(output[~support], raw[~support])),
                   limitation='Source-guided display only. No semantic defect or component labels. Weak rims may remain unresolved.')
    return dict(output=output, support=support, alpha=alpha, target=target, details=details,
                v5b=legacy['output'], v5b_support=legacy['support'], v5a=enhanced)


def render_original_mode(source, result):
    if source.ndim == 2:
        final = v5b.v5a.base.u8(result['output'])
    else:
        a = result['alpha'][..., None]
        final = v5b.v5a.base.u8(source.astype(np.float32) + a * (result['target'][..., None] - source))
    final[~result['support']] = source[~result['support']]
    return final

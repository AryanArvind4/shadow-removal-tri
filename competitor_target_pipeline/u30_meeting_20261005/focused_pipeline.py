"""Experimental U30 local-contrast view; no pad masks or defect synthesis."""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import automatic_v5 as auto
import competitor_target_v3 as base


@dataclass(frozen=True)
class Settings:
    context_fraction: float = .065
    detail_sigma: float = 1.2
    maximum_gain: float = 1.75
    extra_detail_limit: float = 10.
    lift_limit: float = 8.


def local_adjustment(numeric, noise, anchor, span, settings=Settings()):
    """Bounded signed detail gain with a smoothly varying context estimate."""
    if (numeric.ndim != 2 or min(numeric.shape) < 32
            or not np.isfinite(numeric).all() or not np.isfinite(noise)
            or not np.isfinite(anchor) or not np.isfinite(span) or noise <= 0 or span <= 0):
        raise ValueError('Expected finite grayscale data and positive noise/span.')
    if not (.02 <= settings.context_fraction <= .2 and .5 <= settings.detail_sigma <= 3
            and 1 <= settings.maximum_gain <= 2 and settings.extra_detail_limit > 0
            and settings.lift_limit > 0):
        raise ValueError('Invalid focus settings.')
    context_sigma = max(12., min(numeric.shape) * settings.context_fraction)
    # Only the added detail term is smoothed. Fine source detail stays in numeric.
    coarse = base.blur(numeric, settings.detail_sigma)
    context = base.blur(coarse, context_sigma)
    signed_detail = coarse - context
    spread = np.sqrt(np.maximum(0, base.blur(signed_detail**2, context_sigma)))
    floor = max(5., noise * 2.)
    desired = np.clip(.24 * span / np.maximum(spread, floor), 1., settings.maximum_gain)
    evidence = spread**2 / (spread**2 + floor**2)
    gain = 1 + evidence * (desired - 1)
    extra = settings.extra_detail_limit * np.tanh(
        (gain - 1) * signed_detail / settings.extra_detail_limit)
    # Lift dark contexts gently into the useful part of the display curve.
    lift = settings.lift_limit * np.tanh(
        np.maximum(0, anchor + .30 * span - context) / settings.lift_limit)
    return numeric + extra + lift, dict(context=context, gain=gain,
                                       added_detail=extra, midtone_lift=lift)


def process(raw, settings=Settings()):
    # Reuse V5a's automatic core, but omit optional rectangular fitting and dot
    # attenuation. The frozen V5a baseline is generated separately for comparison.
    outputs, _, _, numeric, diagnostics = auto.process(raw)
    if np.ptp(raw) < 1e-6:
        zeros = np.zeros_like(raw, dtype=np.float32)
        return outputs['reference_gray'], dict(corrected=numeric, added_detail=zeros,
            midtone_lift=zeros, context=numeric, gain=np.ones_like(numeric)), diagnostics
    corrected, fields = local_adjustment(numeric, diagnostics['noise_sigma'],
        diagnostics['display_background'], diagnostics['display_signal_span'], settings)
    view = auto.tone(corrected, diagnostics['display_background'], diagnostics['display_signal_span'])
    diagnostics = dict(baseline=diagnostics, settings=asdict(settings),
                       max_added_detail=float(np.abs(fields['added_detail']).max()),
                       max_midtone_lift=float(fields['midtone_lift'].max()),
                       note='Exploratory display, not recovered anatomy or defect labels. '
                            'Local contrast may amplify texture and introduce halos.')
    return view, dict(corrected=corrected, **fields), diagnostics

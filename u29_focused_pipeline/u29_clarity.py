"""U29 clarity pipeline (V3).

Goal
----
Make U29 BGA X-ray pads read as clearly as the reference inspection view:
flat dark background, evenly bright pads across the whole array, and voids
that show as solid dark blobs.

Why V1 and V2 fell short
------------------------
V1 painted a locally normalised pad rendering over a separately darkened
background using a brightness-dependent mask, which cut into pad edges.
V2 avoided that but only applied a broad brightness and contrast correction,
so the shadow gradient and the mid-scale board structure survived, and the
tone curve amplified them along with the pads.

V3 approach
-----------
The detector image is an attenuation map: solder is bright, voids inside a
pad are dark, and the board behind the array adds a slowly varying offset.
That makes the signal additive, so the correction is a two-field affine
normalisation rather than a blend:

    N = (I - B) / (P - B)

B is the local board level and P is the local healthy-solder level. Both are
sampled on the detected pad lattice and interpolated into smooth fields that
vary on the scale of the pad pitch. Because the same affine map applies to
every pixel in a neighbourhood and both fields are smooth, no pad boundary
sits on a discontinuity. After normalisation the board level is about 0 and
healthy solder is about 1 everywhere, regardless of the shadow.

Board structure finer than the lattice can follow, meaning traces, thermal
vias and the die edge, is then estimated by masking out the pads and smoothing
the rest by normalised convolution, which interpolates the board level across
each pad. Subtracting that is again a smooth additive correction.

What remains between pads is attenuated by a purely geometric gain field built
from the detected pad positions, never from pixel brightness. The field is 1
over every pad plus a guard ring wider than the pad edge, so the attenuation
only ever acts on background pixels. A strong excursion in a gap is exempted,
so a bridge or a stray ball keeps its contrast.

Finally a bounded dome correction evens out the brightness inside each pad, a
band pass confined to the pad body sharpens voids, and one monotone tone curve
maps normalised values to display levels.

Caveats
-------
pad_norm controls how much of the per-pad level difference is removed. At 1.0
every pad is driven to the same brightness, which looks cleanest but also
hides genuine pad-to-pad solder volume differences. The per-pad solder and
board levels are written to run.json so that information is not lost.

bg_gain deliberately discards weak board detail between pads. See
README_CLARITY_V3.md for the full list of limits, including the fact that a
2D projection cannot reproduce a tomographic slice.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator


DEFAULT_SRC = r"E:\Shadow_Removal_SDE\TRI_Dataset-20260511T190518Z-3-001\TRI_Dataset_Concised"
DEFAULT_OUT = r"E:\Shadow_Removal_SDE\u29_focused_pipeline\result\u29_clarity_v3"

# The full delivery is a laminography height stack, 68 frames for U29. Layers
# focus at different indices: package and die near 0, the solder balls near 28,
# board vias past 48. These two directories together hold the whole sweep.
STACK_DIRS = (
    r"E:\Shadow_Removal_SDE\TRI_Dataset-20260511T190518Z-3-001\TRI_Dataset",
    r"E:\Shadow_Removal_SDE\TRI_Dataset-20260511T190518Z-3-001\TRI_Dataset_Concised",
)


@dataclass
class Params:
    # lattice search
    pitch_guess_x: float = 73.5
    pitch_guess_y: float = 74.0
    pitch_search: float = 6.0
    border_margin: int = 18

    # level sampling, in pixels
    pad_core_r: int = 17          # solder-level sample radius, inside the pad
    pad_core_pct: float = 80.0    # percentile, so voids do not drag it down
    bg_ring_r0: int = 30          # board-level annulus, starts outside the edge
    bg_ring_r1: int = 46
    bg_ring_pct: float = 40.0

    # field smoothing, as a fraction of pad pitch
    field_sigma: float = 0.40

    # per-pad equalisation, 0 corrects shadow only, 1 makes every pad equal
    pad_norm: float = 0.85

    # fine board-structure removal by masked interpolation across the pads
    fine_sigma: float = 20.0      # scale of structure removed, must exceed the
                                  # pad edge width by a wide margin
    fine_mask_r: int = 21         # pad half-width excluded from the estimate
    fine_strength: float = 1.0

    # geometric background attenuation
    guard_r: int = 24             # gain stays 1.0 inside this half-width
    fade_r: int = 34              # gain reaches bg_gain at this distance
    guard_blur: float = 2.0       # must satisfy guard_r - 3*blur > pad half width
    bg_gain: float = 0.30         # fraction of weak board contrast that survives
    # a strong excursion in a gap is a bridge or a stray ball, not substrate
    # haze, so it is exempted from the attenuation
    bg_keep_lo: float = 0.55      # deviation where the exemption starts
    bg_keep_hi: float = 0.95      # deviation that passes at full contrast

    # within-pad dome removal
    dome_fill_r: int = 16         # closing radius, must exceed the largest void
    dome_inner_r: int = 13        # weight is 1.0 inside this half-width
    dome_blur: float = 1.6        # weight reaches 0 by dome_inner_r + 3*blur
    dome_strength: float = 0.35
    dome_clip: float = 0.30       # bound on the correction, in normalised units

    # void enhancement, confined to the pad body
    void_amount: float = 0.8      # 0 disables it
    void_sigma_lo: float = 1.2
    void_sigma_hi: float = 6.0
    void_inner_r: int = 13        # stays clear of the pad edge at r about 18

    # detail
    denoise_h: float = 3.0        # light, source noise sigma is about 0.3 DN
    # Richardson-Lucy deblur. The source is blur limited, not noise limited:
    # pad contrast is about 34 DN against 0.3 DN of noise, so inverting the
    # imaging blur is well conditioned here. 0 disables it.
    # The core imaging blur was measured from the pad edge-spread averaged over
    # all 72 pads: a clean Gaussian core of sigma about 1.8 px (X 1.96, Y 1.78),
    # not a heavy-tailed scatter PSF (only 5-8 % of the line-spread energy sits
    # beyond 3 px). So the model is Gaussian and the assumed sigma should sit a
    # little below the measured core: RL also sharpens the real solder-ball rim,
    # and pushing sigma to the measured 1.8+ over-emphasises that rim into a
    # ring. 1.6 is the ceiling that still sharpens edges without ringing; do not
    # raise it further. See README_CLARITY_V3.md, "Measured PSF".
    deconv_sigma: float = 1.6
    deconv_iters: int = 20
    unsharp_sigma: float = 1.6
    unsharp_amount: float = 0.20

    # Display tone curve, normalised units to 8 bit, monotone by construction.
    # Because the pad dome is removed upstream, this can stay steep right
    # through the pad body, so a shallow void keeps its contrast instead of
    # being squashed by a flat shoulder. The end knots carry soft shoulders so
    # a bright rim rolls off rather than clipping to 255, which would flatten
    # voids on the brightest pads.
    tone_x: tuple = (-1.50, -0.60, -0.10, 0.10, 0.30, 0.52,
                     0.72, 0.92, 1.10, 1.60, 2.40)
    tone_y: tuple = (0.0, 14.0, 34.0, 48.0, 68.0, 105.0,
                     150.0, 195.0, 222.0, 250.0, 255.0)


# ---------------------------------------------------------------------------
# lattice detection
# ---------------------------------------------------------------------------

def pad_response(img: np.ndarray, pitch: float) -> np.ndarray:
    """Centre-surround response that peaks on pads."""
    k_in = max(3, int(round(pitch * 0.47)) | 1)
    k_out = max(5, int(round(pitch * 1.29)) | 1)
    diff = cv2.blur(img, (k_in, k_in)) - cv2.blur(img, (k_out, k_out))
    return cv2.GaussianBlur(diff, (0, 0), 3)


def _fit_comb(profile: np.ndarray, guess: float, span: float, margin: int):
    n = len(profile)
    axis = np.arange(n, dtype=np.float32)
    best = None
    for pitch in np.arange(guess - span, guess + span, 0.1):
        for phase in np.arange(0.0, pitch, 0.25):
            xs = np.arange(phase, n - margin, pitch)
            xs = xs[xs >= margin]
            if len(xs) < 5:
                continue
            score = float(np.interp(xs, axis, profile).mean())
            if best is None or score > best[0]:
                best = (score, float(pitch), xs.astype(np.float32))
    if best is None:
        raise RuntimeError("pad lattice search found no candidate spacing")
    return best


def detect_grid(img: np.ndarray, p: Params):
    """Return centers[R, C, 2], col_x, row_y, pitch_x, pitch_y."""
    resp = pad_response(img, 0.5 * (p.pitch_guess_x + p.pitch_guess_y))
    _, pitch_x, col_x = _fit_comb(resp.mean(axis=0), p.pitch_guess_x,
                                  p.pitch_search, p.border_margin)
    _, pitch_y, row_y = _fit_comb(resp.mean(axis=1), p.pitch_guess_y,
                                  p.pitch_search, p.border_margin)

    h, w = img.shape
    half = int(round(min(pitch_x, pitch_y) * 0.28))
    centers = np.zeros((len(row_y), len(col_x), 2), np.float32)
    for i, cy in enumerate(row_y):
        for j, cx in enumerate(col_x):
            x0, x1 = max(0, int(cx - half)), min(w, int(cx + half + 1))
            y0, y1 = max(0, int(cy - half)), min(h, int(cy + half + 1))
            patch = resp[y0:y1, x0:x1]
            if patch.size < 16:
                centers[i, j] = (cx, cy)
                continue
            wgt = np.clip(patch - patch.min(), 0, None) + 1e-6
            ys, xs = np.mgrid[y0:y1, x0:x1]
            cxr = float((xs * wgt).sum() / wgt.sum())
            cyr = float((ys * wgt).sum() / wgt.sum())
            # refinement may never wander more than a third of a pitch
            cxr = float(np.clip(cxr, cx - pitch_x / 3.0, cx + pitch_x / 3.0))
            cyr = float(np.clip(cyr, cy - pitch_y / 3.0, cy + pitch_y / 3.0))
            centers[i, j] = (cxr, cyr)
    return centers, col_x, row_y, pitch_x, pitch_y


# ---------------------------------------------------------------------------
# level sampling and smooth fields
# ---------------------------------------------------------------------------

def _disc_pct(img, x, y, r, pct):
    h, w = img.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r + 1))
    y0, y1 = max(0, int(y - r)), min(h, int(y + r + 1))
    sub = img[y0:y1, x0:x1]
    if sub.size < 12:
        return np.nan
    return float(np.percentile(sub, pct))


def _ring_pct(img, xx, yy, x, y, r0, r1, pct):
    h, w = img.shape
    x0, x1 = max(0, int(x - r1)), min(w, int(x + r1 + 1))
    y0, y1 = max(0, int(y - r1)), min(h, int(y + r1 + 1))
    sub = img[y0:y1, x0:x1]
    d = np.hypot(xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y)
    m = (d >= r0) & (d <= r1)
    if m.sum() < 24:
        return np.nan
    return float(np.percentile(sub[m], pct))


def _fill_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    m = np.isnan(a)
    if m.any():
        a[m] = float(np.nanmedian(a))
    return a


def _field_from_lattice(vals, gx, gy, shape, sigma):
    """Separable lattice interpolation with edge hold, then Gaussian smoothing."""
    h, w = shape
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    rows = np.stack([np.interp(xs, gx, vals[i]) for i in range(vals.shape[0])], 0)
    out = np.stack([np.interp(ys, gy, rows[:, c]) for c in range(w)], 1)
    return cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma)


# ---------------------------------------------------------------------------
# geometric background attenuation
# ---------------------------------------------------------------------------

def fine_board_field(norm: np.ndarray, centers, p: Params) -> np.ndarray:
    """Board structure at a finer scale than the pad lattice.

    Pads are masked out and the remaining pixels are smoothed by normalised
    convolution, which interpolates the board level straight across each pad
    instead of letting the solder leak into the estimate. The result varies on
    a scale of roughly ``2 * fine_sigma`` pixels, far coarser than the three to
    four pixel pad edge transition, so subtracting it shifts board level but
    not pad geometry.
    """
    h, w = norm.shape
    mask = np.ones((h, w), np.float32)
    r = p.fine_mask_r
    for row in centers:
        for cx, cy in row:
            x, y = int(round(cx)), int(round(cy))
            cv2.rectangle(mask, (x - r, y - r), (x + r, y + r), 0.0, -1)
    num = cv2.GaussianBlur(norm * mask, (0, 0), p.fine_sigma)
    den = cv2.GaussianBlur(mask, (0, 0), p.fine_sigma)
    return num / np.maximum(den, 1e-3)


def richardson_lucy(img: np.ndarray, sigma: float, iters: int) -> np.ndarray:
    """Deblur against a Gaussian point spread function.

    The imaging blur was measured from the pad edge-spread averaged over all 72
    pads and is a clean Gaussian core of sigma about 1.8 px, so a Gaussian PSF
    is the right model. Deconvolving at sigma 1.6 narrows the 10 to 90 per cent
    rise from 4.39 px to about 3.7 px without adding a halo. The 20 per cent
    bright shoulder just inside the pad edge is the solder ball rim and is
    present in the source, so do not read it as ringing; deconvolution leaves it
    essentially unchanged. Do not push sigma past 1.6: RL also sharpens that
    real rim, and at the measured 1.8+ it rings the rim into a hard bright frame
    while a gradient metric keeps rewarding it (see README_CLARITY_V3.md).
    """
    obs = img.astype(np.float64)
    offset = float(obs.min()) - 1.0          # keep the data strictly positive
    obs = obs - offset
    est = obs.copy()
    for _ in range(max(0, iters)):
        blurred = cv2.GaussianBlur(est, (0, 0), sigma)
        ratio = obs / np.maximum(blurred, 1e-6)
        est = np.clip(est * cv2.GaussianBlur(ratio, (0, 0), sigma), 0.0, None)
    return (est + offset).astype(np.float32)


def _pad_weight(shape, centers, half, blur) -> np.ndarray:
    """Smooth geometric weight, 1.0 over each pad core and 0 outside."""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    r = int(half)
    for row in centers:
        for cx, cy in row:
            x, y = int(round(cx)), int(round(cy))
            cv2.rectangle(m, (x - r, y - r), (x + r, y + r), 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), blur)


def remove_dome(norm, centers, col_x, row_y, p: Params):
    """Flatten the smooth brightness dome inside each pad.

    A solder ball is thicker in the middle than at its rim, and board
    structure adds more slow shading on top. Without this the display curve
    has to stay gentle across the whole pad body so that shading does not
    read as a void, and that same gentleness then squashes real shallow
    voids. Removing the shading lets the curve stay steep everywhere.

    The envelope is a grayscale closing, which fills dark blobs smaller than
    its radius while leaving the pad outline where it is, so voids survive as
    the difference and pad geometry is untouched. The correction is applied
    only through a weight that reaches zero well inside the pad edge.
    """
    k = 2 * p.dome_fill_r + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    env = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, se)

    # per-pad envelope level, interpolated into a smooth field
    n_rows, n_cols = centers.shape[:2]
    lvl = np.empty((n_rows, n_cols), np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = centers[i, j]
            lvl[i, j] = _disc_pct(env, cx, cy, p.dome_inner_r, 50.0)
    lvl = _fill_nan(lvl)
    target = _field_from_lattice(lvl, col_x, row_y, norm.shape, 6.0)

    # Bounding the correction keeps two things safe. Near the pad edge the
    # envelope plunges, and an unbounded correction would add a bright halo
    # just inside the boundary. A void larger than the closing radius also
    # drags the envelope down with it, and the bound limits how much of such
    # a void can be filled back in.
    shade = np.clip(env - target, -p.dome_clip, p.dome_clip)
    wgt = _pad_weight(norm.shape, centers, p.dome_inner_r, p.dome_blur)
    return norm - p.dome_strength * wgt * shade, env, wgt


def guard_gain(shape, centers, p: Params) -> np.ndarray:
    """1.0 over every pad plus a guard ring, falling to bg_gain in the gaps.

    Derived only from detected pad geometry, never from pixel values, so it
    cannot carve into a pad edge the way a brightness-driven mask can.
    """
    h, w = shape
    core = np.zeros((h, w), np.uint8)
    r = p.guard_r
    for row in centers:
        for cx, cy in row:
            x, y = int(round(cx)), int(round(cy))
            # a square, not a disc: pads are square, so a disc of radius r
            # leaves the pad corners at r*sqrt(2) outside the protected zone
            # and the attenuation rounds them off
            cv2.rectangle(core, (x - r, y - r), (x + r, y + r), 255, -1)
    dist = cv2.distanceTransform(255 - core, cv2.DIST_L2, 5)
    t = np.clip(dist / max(1.0, float(p.fade_r - p.guard_r)), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)                       # smoothstep
    gain = 1.0 - (1.0 - p.bg_gain) * s
    gain = cv2.GaussianBlur(gain.astype(np.float32), (0, 0), p.guard_blur)
    # smoothing pulls the ramp inward, so restore exact unity over the pad and
    # its immediate margin; the maximum of two smooth fields stays continuous
    prot = np.zeros((h, w), np.float32)
    pr = p.guard_r - int(np.ceil(3.0 * p.guard_blur))
    for row in centers:
        for cx, cy in row:
            x, y = int(round(cx)), int(round(cy))
            cv2.rectangle(prot, (x - pr, y - pr), (x + pr, y + pr), 1.0, -1)
    prot = cv2.GaussianBlur(prot, (0, 0), 1.0)
    return np.maximum(gain, prot)


# ---------------------------------------------------------------------------
# main processing
# ---------------------------------------------------------------------------

def load_stack(component: str, dirs=STACK_DIRS):
    """Load every height slice for a component, ordered by frame index."""
    import re
    found = {}
    for d in dirs:
        for f in _glob.glob(os.path.join(d, component + "_*.jpg")):
            m = re.search(component + r"_(\d+)", os.path.basename(f))
            if m:
                found.setdefault(int(m.group(1)), f)
    if not found:
        raise FileNotFoundError("no stack frames for " + component)
    keys = sorted(found)
    imgs = np.stack([cv2.imread(found[k], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                     for k in keys])
    return imgs, keys


def solder_plane(imgs, p: Params, keep_frac: float = 0.93,
                 max_half_width: int = 4):
    """Average the height slices that sit on the solder-ball plane.

    Focus is measured as gradient magnitude around the pad boundary, not as
    pad-versus-gap brightness. Brightness contrast peaks about eight slices
    below the sharpest plane, because slices nearer the die carry more overall
    attenuation over the pads without resolving them any better; selecting on
    it picks a visibly worse plane.

    The depth response here is broad, so the plateau alone is not a tight
    enough criterion; ``max_half_width`` caps the window. Widening it past
    about four slices either side pulls in off-plane layers and the pads start
    to look fragmented.
    """
    mid = imgs[len(imgs) // 2]
    centers, _, _, _, _ = detect_grid(mid, p)
    h, w = mid.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dmin = np.full((h, w), 1e9, np.float32)
    for row in centers:
        for cx, cy in row:
            dmin = np.minimum(dmin, np.maximum(np.abs(xx - cx), np.abs(yy - cy)))
    boundary = (dmin > 15) & (dmin < 21)

    def sharpness(im):
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1)
        return float(np.sqrt(gx * gx + gy * gy)[boundary].mean())

    contrast = np.array([sharpness(im) for im in imgs])
    peak = int(np.argmax(contrast))
    thresh = contrast[peak] * keep_frac
    lo = peak
    while lo > 0 and contrast[lo - 1] >= thresh and peak - lo < max_half_width:
        lo -= 1
    hi = peak
    while (hi < len(imgs) - 1 and contrast[hi + 1] >= thresh
           and hi - peak < max_half_width):
        hi += 1
    return imgs[lo:hi + 1].mean(0), (lo, peak, hi), contrast


def process(path: str, p: Params, image: np.ndarray | None = None):
    """Process one image. ``image`` overrides reading from ``path``."""
    if image is None:
        src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if src is None:
            raise FileNotFoundError(path)
        src = src.astype(np.float32)
    else:
        src = np.clip(image, 0, 255).astype(np.float32)
    h, w = src.shape

    if p.denoise_h > 0:
        work = cv2.fastNlMeansDenoising(
            src.astype(np.uint8), None, h=p.denoise_h,
            templateWindowSize=7, searchWindowSize=21).astype(np.float32)
    else:
        work = src.copy()

    # invert the imaging blur before anything measures a level or an edge
    if p.deconv_sigma > 0 and p.deconv_iters > 0:
        work = richardson_lucy(work, p.deconv_sigma, p.deconv_iters)

    centers, col_x, row_y, pitch_x, pitch_y = detect_grid(work, p)
    n_rows, n_cols = centers.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    solder = np.empty((n_rows, n_cols), np.float32)
    board = np.empty((n_rows, n_cols), np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = centers[i, j]
            solder[i, j] = _disc_pct(work, cx, cy, p.pad_core_r, p.pad_core_pct)
            board[i, j] = _ring_pct(work, xx, yy, cx, cy,
                                    p.bg_ring_r0, p.bg_ring_r1, p.bg_ring_pct)
    solder, board = _fill_nan(solder), _fill_nan(board)

    # the board level must be a shadow estimate, so reject trace-contaminated
    # samples before it becomes a field
    board = cv2.GaussianBlur(cv2.medianBlur(board, 3), (0, 0), 0.9)

    # pad_norm blends the true per-pad level with its smoothed version, so 0
    # corrects the shadow only and 1 equalises every pad
    solder_smooth = cv2.GaussianBlur(cv2.medianBlur(solder, 3), (0, 0), 1.1)
    solder_used = p.pad_norm * solder + (1.0 - p.pad_norm) * solder_smooth

    sigma = p.field_sigma * 0.5 * (pitch_x + pitch_y)
    board_f = _field_from_lattice(board, col_x, row_y, (h, w), sigma)
    solder_f = _field_from_lattice(solder_used, col_x, row_y, (h, w), sigma)
    amp = np.maximum(solder_f - board_f, 4.0)
    norm_coarse = (work - board_f) / amp

    # remove board structure that is finer than the pad lattice can follow,
    # such as traces, thermal vias and the die edge
    fine = fine_board_field(norm_coarse, centers, p)
    norm = norm_coarse - p.fine_strength * (fine - float(np.median(fine)))

    # flatten the brightness dome inside each pad so the display curve can
    # stay steep across the pad body without shading reading as a void
    if p.dome_strength > 0:
        norm, env, dome_w = remove_dome(norm, centers, col_x, row_y, p)
    else:
        env, dome_w = norm, np.zeros_like(norm)

    # flatten what is left between pads
    gain = guard_gain((h, w), centers, p)
    inside = np.zeros((h, w), bool)
    inside[int(row_y[0]):int(row_y[-1]), int(col_x[0]):int(col_x[-1])] = True
    gap = inside & (gain < p.bg_gain + 0.05)
    base = float(np.median(norm[gap])) if gap.any() else 0.0
    dev = norm - base
    # exempt strong excursions so a bridge or a stray ball in a gap keeps its
    # contrast while weak substrate haze is flattened
    t = np.clip((np.abs(dev) - p.bg_keep_lo) /
                max(1e-6, p.bg_keep_hi - p.bg_keep_lo), 0.0, 1.0)
    keep = t * t * (3.0 - 2.0 * t)
    eff_gain = gain + (1.0 - gain) * keep
    norm_flat = base + dev * eff_gain

    # Void enhancement. A band pass tuned to void size is added back only over
    # the pad body. The weight reaches zero well inside the pad edge, so the
    # boundary itself is never band-pass amplified and cannot ring.
    if p.void_amount > 0:
        inner = np.zeros((h, w), np.float32)
        vr = p.void_inner_r
        for row in centers:
            for cx, cy in row:
                x, y = int(round(cx)), int(round(cy))
                cv2.rectangle(inner, (x - vr, y - vr), (x + vr, y + vr), 1.0, -1)
        inner = cv2.GaussianBlur(inner, (0, 0), 2.5)
        band = (cv2.GaussianBlur(norm_flat, (0, 0), p.void_sigma_lo)
                - cv2.GaussianBlur(norm_flat, (0, 0), p.void_sigma_hi))
        norm_flat = norm_flat + p.void_amount * inner * band

    if p.unsharp_amount > 0:
        blur = cv2.GaussianBlur(norm_flat, (0, 0), p.unsharp_sigma)
        norm_flat = norm_flat + p.unsharp_amount * (norm_flat - blur)

    # clamp into the knot range rather than extrapolating: a PCHIP end cubic
    # can turn over outside its knots, which would invert contrast. Clamp in
    # float64 so no float32 rounding lands just outside a knot and yields NaN.
    curve = PchipInterpolator(np.asarray(p.tone_x, float),
                              np.asarray(p.tone_y, float), extrapolate=False)
    clamped = np.clip(np.nan_to_num(norm_flat.astype(np.float64), nan=0.0),
                      float(p.tone_x[0]), float(p.tone_x[-1]))
    out = np.clip(np.nan_to_num(curve(clamped), nan=0.0), 0, 255).astype(np.uint8)

    diag = {
        "pitch_x": round(float(pitch_x), 2),
        "pitch_y": round(float(pitch_y), 2),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "board_level_min_dn": round(float(board.min()), 1),
        "board_level_max_dn": round(float(board.max()), 1),
        "shadow_span_dn": round(float(board.max() - board.min()), 1),
        "solder_level_min_dn": round(float(solder.min()), 1),
        "solder_level_max_dn": round(float(solder.max()), 1),
        "pad_contrast_min_dn": round(float(amp.min()), 1),
        "pad_contrast_max_dn": round(float(amp.max()), 1),
        "background_base_norm": round(base, 3),
        "solder_grid_dn": np.round(solder, 1).tolist(),
        "board_grid_dn": np.round(board, 1).tolist(),
    }
    stages = {
        "source": src, "work": work, "board_field": board_f,
        "solder_field": solder_f, "norm_coarse": norm_coarse,
        "fine_field": fine, "norm": norm, "gain": gain,
        "norm_flat": norm_flat, "out": out, "centers": centers, "env": env, "dome_w": dome_w,
        "amp": amp,
    }
    return out, diag, stages


def main():
    ap = argparse.ArgumentParser(description="U29 clarity pipeline (V3)")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--pattern", default="U29_*.jpg")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--pad-norm", type=float, default=Params.pad_norm)
    ap.add_argument("--bg-gain", type=float, default=Params.bg_gain)
    ap.add_argument("--unsharp", type=float, default=Params.unsharp_amount)
    ap.add_argument("--stack", metavar="COMPONENT", default="",
                    help="average the solder-plane slices of the full height "
                         "stack for this component, e.g. U29")
    args = ap.parse_args()

    p = Params(pad_norm=args.pad_norm, bg_gain=args.bg_gain,
               unsharp_amount=args.unsharp)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "diagnostic"), exist_ok=True)

    params_json = {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(p).items()}
    report = {"params": params_json, "images": {}}

    if args.stack:
        imgs, keys = load_stack(args.stack)
        merged, (lo, peak, hi), contrast = solder_plane(imgs, p)
        name = args.stack + "_solder_plane"
        out, diag, st = process(name, p, image=merged)
        diag["stack_frames"] = len(keys)
        diag["solder_plane_frames"] = [int(keys[lo]), int(keys[hi])]
        diag["solder_plane_peak"] = int(keys[peak])
        cv2.imwrite(os.path.join(args.out, name + ".png"), out)
        report["images"][name] = diag
        with open(os.path.join(args.out, "run.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        print("stack: {0} frames, solder plane {1}..{2}, peak {3}".format(
            len(keys), keys[lo], keys[hi], keys[peak]))
        print("wrote " + os.path.join(args.out, name + ".png"))
        return

    files = sorted(_glob.glob(os.path.join(args.src, args.pattern)))
    if not files:
        raise SystemExit("no inputs matched " + args.pattern + " in " + args.src)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        out, diag, st = process(f, p)
        cv2.imwrite(os.path.join(args.out, name + ".png"), out)
        side = np.hstack([
            np.clip(st["source"] / 160.0 * 255.0, 0, 255).astype(np.uint8),
            np.full((out.shape[0], 6), 255, np.uint8),
            out,
        ])
        cv2.imwrite(os.path.join(args.out, "diagnostic",
                                 name + "_before_after.png"), side)
        report["images"][name] = diag
        print("{0}: grid {1}x{2}, pitch {3}x{4}, shadow span {5} DN".format(
            name, diag["n_rows"], diag["n_cols"], diag["pitch_x"],
            diag["pitch_y"], diag["shadow_span_dn"]))

    with open(os.path.join(args.out, "run.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote " + args.out)


if __name__ == "__main__":
    main()

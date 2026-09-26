"""Component-aware background cleanup, with native-resolution solder detail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import competitor_target_v3 as v3


OUTPUT = v3.OUTPUT.parent / 'competitor_target_v4'


def fit_axis(profile, count, pitch_range, origin_range):
    """Fit a regular lattice to a one-dimensional image response, not to filenames."""
    smooth = v3.blur(profile, 2.)
    cap = np.percentile(smooth, 95)
    best = None
    for pitch in np.linspace(*pitch_range, 81):
        for origin in np.linspace(*origin_range, 81):
            positions = origin + pitch * np.arange(count)
            if positions[-1] >= len(profile) - 12:
                continue
            response = np.interp(positions, np.arange(len(profile)), smooth)
            # Capping prevents a single strong edge from deciding the entire grid.
            score = np.mean(np.minimum(response, cap))
            if best is None or score > best[0]:
                best = (score, positions)
    return best[1]


def square_grid(image, fallback_labels, fallback_regions, noise):
    h, w = image.shape
    side = int(round(min(h, w) * .055))
    inner = max(5, int(side * .68) | 1)
    outer = max(inner + 2, int(side * 1.65) | 1)
    response = cv2.boxFilter(image, -1, (inner, inner)) - cv2.boxFilter(image, -1, (outer, outer))
    xprofile = response[int(h * .06):int(h * .94)].mean(0)
    yprofile = response[:, int(w * .06):int(w * .94)].mean(1)
    columns = fit_axis(xprofile, 9, (w * .088, w * .106), (w * .08, w * .15))
    rows = fit_axis(yprofile, 8, (h * .099, h * .118), (h * .09, h * .145))
    labels = np.zeros_like(fallback_labels)
    regions, proposals = [], []
    for row, cy in enumerate(rows):
        for col, cx in enumerate(columns):
            radius = max(2, round(side * .14))
            x, y = int(round(cx)), int(round(cy))
            window = response[y-radius:y+radius+1, x-radius:x+radius+1]
            iy, ix = np.unravel_index(np.argmax(window), window.shape)
            x = int(x + ix - radius)
            y = int(y + iy - radius)
            half = side // 2
            x0, y0 = max(0, x-half), max(0, y-half)
            x1, y1 = min(w, x+half+1), min(h, y+half+1)
            patch = image[y0:y1, x0:x1]
            surround = image[max(0,y-side):min(h,y+side+1), max(0,x-side):min(w,x+side+1)]
            contrast = float(np.percentile(patch, 65) - np.percentile(surround, 30))
            accepted = bool(response[y, x] > max(1., noise * .7) and contrast > max(3., noise * 2.))
            proposals.append(dict(row=row, column=col, x=x, y=y, contrast=contrast, accepted=accepted))
            if accepted:
                identifier = len(regions) + 1
                labels[y0:y1, x0:x1] = identifier
                regions.append((identifier, x0, y0, x1-x0, y1-y0))
    # A poor lattice fit must not turn arbitrary texture into 72 solder regions.
    if len(regions) < 40:
        return fallback_labels, fallback_regions, dict(used=False, reason='insufficient supported cells', cells=proposals)
    return labels, regions, dict(used=True, cells=proposals, note='Rectangular inspection ROIs, not exact solder boundaries.')


def masked_bands(image, labels, component):
    if component not in {'J1', 'J4'}:
        return np.zeros_like(image)
    h, w = image.shape
    guard = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    profiles, centers = [], []
    for x0, x1 in zip(np.linspace(0, w, 7, dtype=int)[:-1], np.linspace(0, w, 7, dtype=int)[1:]):
        sample = np.where(guard[:, x0:x1], np.nan, image[:, x0:x1])
        valid = np.isfinite(sample).sum(1) > (x1-x0) * .15
        profile = np.zeros(h, np.float32)
        if valid.sum() < h * .25:
            profiles.append(profile)
            centers.append((x0+x1) / 2)
            continue
        profile[valid] = np.nanmedian(sample[valid], axis=1)
        profile = np.interp(np.arange(h), np.flatnonzero(valid), profile[valid])
        band = v3.blur(profile, 1.5) - v3.blur(profile, h * .035)
        profiles.append(-8 * np.tanh(.85 * band / 8))
        centers.append((x0+x1) / 2)
    # Interpolate within each panel, avoiding a false transition across the center.
    field = np.zeros_like(image)
    for panel in range(2):
        start, end = panel * (w // 2), w if panel else w // 2
        xp = centers[panel*3:panel*3+3]
        values = np.stack(profiles[panel*3:panel*3+3], axis=1)
        for y in range(h):
            field[y, start:end] = np.interp(np.arange(start, end), xp, values[y])
    return v3.blur(field, (0., 2.))


def clean_background(image, labels, noise):
    # Solder ROIs plus their boundary margin receive exactly zero denoising.
    guard = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    distance = cv2.distanceTransform(1-guard, cv2.DIST_L2, 5)
    weight = np.clip(distance / 6., 0, 1)
    smooth = v3.blur(image, 1.)
    detail = np.abs(smooth - v3.blur(image, 2.5))
    # Large unclassified transitions get less smoothing, retaining their edges.
    weight *= .85 / (1 + (detail / max(2., noise * 2)) ** 2)
    denoised = cv2.bilateralFilter(image.astype(np.float32), 7, max(2., noise * 2.5), 2.)
    delta = np.clip(denoised - image, -noise * 2, noise * 2) * weight
    return image + delta, weight


def process(raw, component):
    flattened, broad = v3.broad_correction(raw)
    labels, regions, _, noise = v3.propose_regions(flattened, component)
    grid = dict(used=False)
    if component == 'U29':
        labels, regions, grid = square_grid(flattened, labels, regions, noise)
    bands = masked_bands(flattened, labels, component)
    balanced = flattened + bands
    local = v3.pad_correction(balanced, labels, regions)
    corrected = balanced + local
    cleaned, weight = clean_background(corrected, labels, noise)
    low, high = np.percentile(flattened, (5, 99.5))
    outputs = dict(reference_gray=v3.tone(cleaned, low, high),
                   detail_gray=v3.tone(balanced, low, high),
                   region_mask=(labels > 0).astype(np.float32) * 255,
                   background_smoothing=weight / .85 * 255,
                   band_correction=128 + bands * 8)
    diagnostic = dict(noise_estimate=noise, grid=grid,
                      max_band_correction=float(np.max(np.abs(bands))),
                      background_smoothing_fraction=float(np.mean(weight > .1)))
    return outputs, labels, regions, cleaned, diagnostic


def make_sheet(path, raw, previous, outputs, labels, regions):
    overlay = cv2.cvtColor(v3.u8(raw), cv2.COLOR_GRAY2RGB)
    for _, x, y, w, h in regions:
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (30, 220, 100), 1)
    panels = [('Input', raw), ('Previous V3: same image', previous), ('V4 reference_gray', outputs['reference_gray']),
              ('Region proposals', overlay), ('Background smoothing', outputs['background_smoothing']), ('Band correction: gray = zero', outputs['band_correction'])]
    width = 620
    height = round(raw.shape[0] / raw.shape[1] * width)
    sheet = Image.new('RGB', (width*3, (height+30)*2), (24,24,24))
    draw = ImageDraw.Draw(sheet)
    for i, (title, array) in enumerate(panels):
        x, y = i%3*width, i//3*(height+30)
        sheet.paste(Image.fromarray(v3.u8(array)).convert('RGB').resize((width,height), Image.Resampling.LANCZOS), (x,y+30))
        draw.text((x+8,y+8), title, fill='white')
    sheet.save(path)


def background_metrics(image, labels):
    guard = cv2.dilate((labels>0).astype(np.uint8), np.ones((11,11),np.uint8))
    mask = guard == 0
    if mask.sum() < 100:
        return dict(background_high_frequency_rms=None, background_row_variation=None)
    residual = image - v3.blur(image, 1.)
    profile = np.zeros(image.shape[0])
    valid = mask.sum(1) > 10
    if not valid.any():
        return dict(background_high_frequency_rms=float(np.sqrt(np.mean(residual[mask]**2))), background_row_variation=None)
    profile[valid] = np.nanmedian(np.where(mask[valid], image[valid], np.nan), axis=1)
    profile = np.interp(np.arange(len(profile)), np.flatnonzero(valid), profile[valid])
    return dict(background_high_frequency_rms=float(np.sqrt(np.mean(residual[mask]**2))),
                background_row_variation=float(np.std(profile-v3.blur(profile, len(profile)*.035))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=v3.INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--files', nargs='+', help='Optional exact input filenames for a smoke run.')
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'})
    if args.files:
        paths = [p for p in paths if p.name in args.files]
        missing = set(args.files)-{p.name for p in paths}
        if missing:
            raise ValueError(f'Missing inputs: {sorted(missing)}')
    if not paths:
        raise ValueError('No input images.')
    args.output.mkdir(parents=True, exist_ok=True)
    all_regions, metrics, manifest = [], [], []
    for index, path in enumerate(paths, 1):
        print(f'[{index}/{len(paths)}] {path.name}', flush=True)
        raw = v3.read(path)
        component = path.stem.split('_')[0]
        outputs, labels, regions, cleaned, diagnostic = process(raw, component)
        prior_path = v3.OUTPUT / 'reference_gray' / (path.stem+'.png')
        previous = v3.read(prior_path) if prior_path.exists() else v3.process(raw, component)[0]['reference_gray']
        for method, image in outputs.items():
            folder = args.output / method
            folder.mkdir(exist_ok=True)
            Image.fromarray(v3.u8(image)).save(folder / (path.stem+'.png'))
        numeric = args.output/'numeric'
        numeric.mkdir(exist_ok=True)
        np.savez_compressed(numeric/(path.stem+'.npz'), corrected=cleaned, region_labels=labels)
        sheets = args.output/'comparison_sheets'
        sheets.mkdir(exist_ok=True)
        make_sheet(sheets/(path.stem+'.jpg'),raw,previous,outputs,labels,regions)
        # Fixed V4 proposals for both versions prevent changing masks from gaming comparison.
        records = v3.region_metrics(raw, dict(detail_gray=previous, reference_gray=outputs['reference_gray']), labels, regions)
        for record in records:
            record['method'] = {'detail_gray':'v3_reference_gray', 'reference_gray':'v4_reference_gray', 'input':'input'}[record['method']]
            record.update(filename=path.name,component=component)
        all_regions.extend(records)
        for method, image in [('input',raw),('v3_reference_gray',previous),('v4_reference_gray',outputs['reference_gray'])]:
            selected = [r for r in records if r['method']==method]
            metrics.append(dict(filename=path.name,component=component,method=method,proposed_regions=len(regions),
                                mean_region_interior_std=float(np.mean([r['interior_std'] for r in selected])) if selected else None,
                                **background_metrics(image,labels)))
        diagnostic.update(filename=path.name,proposed_regions=len(regions),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest.append(diagnostic)
        print(f'  regions={len(regions)}, smoothing coverage={diagnostic["background_smoothing_fraction"]:.1%}',flush=True)
    v3.write_csv(args.output/'metrics.csv',metrics)
    fields=['region_id','x','y','width','height','method','interior_pixels','interior_mean','interior_std','interior_iqr','input_correlation','clipping_fraction','filename','component']
    v3.write_csv(args.output/'per_region_metrics.csv',all_regions,fields)
    versions={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(v3.__file__)]}
    (args.output/'run.json').write_text(json.dumps(dict(pipeline='v4',sources=versions,images=manifest,
        warning='Automatic ROI and background proxies; not validated defect or healthy-solder measurements.'),indent=2),encoding='utf-8')
    print(f'Results: {args.output.resolve()}',flush=True)


if __name__ == '__main__':
    main()

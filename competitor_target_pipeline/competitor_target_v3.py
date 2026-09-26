"""Native-resolution grayscale correction with auditable solder-region proposals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'TRI_Dataset-20260511T190518Z-3-001' / 'TRI_Dataset_Concised'
OUTPUT = Path(__file__).resolve().parent / 'result' / 'competitor_target_v3'


def blur(a, sigma):
    return gaussian_filter(a.astype(np.float32), sigma, mode='reflect')


def read(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L'), dtype=np.float32)


def u8(a):
    return np.rint(np.clip(a, 0, 255)).astype(np.uint8)


def broad_correction(raw):
    # Only the smooth correction field is estimated at reduced resolution.
    h, w = raw.shape
    scale = min(1., 512 / max(h, w))
    small = cv2.resize(raw, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    field = blur(cv2.medianBlur(u8(small), 9), min(small.shape) * .12)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
    correction = 15 * np.tanh((np.median(field) - field) / 15)
    return raw + correction, correction


def row_correction(image, component):
    correction = np.zeros_like(image)
    if component not in {'J1', 'J4'}:
        return image.copy(), correction
    # Separate panels avoid imposing one panel's row brightness on the other.
    h, w = image.shape
    for start, end in ((0, w // 2), (w // 2, w)):
        panel = image[:, start:end]
        profile = np.median(panel, axis=1)
        residual = profile - blur(profile, h * .035)
        delta = -5 * np.tanh(.65 * residual / 5)
        correction[:, start:end] = delta[:, None]
    correction = blur(correction, (1., 8.))
    return image + correction, correction


def propose_regions(image, component):
    """Filled contours define support only; their interiors never replace image pixels."""
    h, w = image.shape
    short = min(h, w)
    small_balls = component in {'J1', 'J4'}
    diameter = short * (.013 if small_balls else .052)
    smooth = blur(image, .7)
    residual = smooth - blur(smooth, diameter * .85)
    noise = max(.6, float(np.median(np.abs(image - blur(image, 1.))) / .6745))
    candidates = []
    thresholds = (max(1.5, noise), max(3., noise * 1.7), max(5., noise * 2.5), max(8., noise * 4), max(12., noise * 6))
    for threshold in thresholds:
        binary = (residual > threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if small_balls:
                valid = .3 * diameter < min(bw, bh) and max(bw, bh) < 2.2 * diameter
                valid &= max(bw, bh) / max(1, min(bw, bh)) < 1.65
            else:
                valid = .35 * diameter < min(bw, bh) and max(bw, bh) < diameter * (2 if component == 'U29' else 4.5)
                valid &= max(bw, bh) / max(1, min(bw, bh)) < (1.6 if component == 'U29' else 4.)
            if not valid or area / (bw * bh) < .48 or area < 16:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if area / max(1., hull_area) < .78:
                continue
            candidates.append((area, contour))
    labels = np.zeros((h, w), np.int32)
    regions = []
    # Larger, complete contours take precedence over fragments at higher thresholds.
    for _, contour in sorted(candidates, key=lambda item: -item[0]):
        x, y, bw, bh = cv2.boundingRect(contour)
        local = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(local, [contour - (x, y)], -1, 1, -1)
        selected = local > 0
        patch = labels[y:y + bh, x:x + bw]
        if np.any(patch[selected]):
            continue
        identifier = len(regions) + 1
        patch[selected] = identifier
        regions.append((identifier, x, y, bw, bh))
    return labels, regions, residual, noise


def pad_correction(image, labels, regions):
    correction = np.zeros_like(image)
    pad_levels = []
    for identifier, x, y, w, h in regions:
        local = labels[y:y+h, x:x+w] == identifier
        pad_levels.append(np.percentile(image[y:y+h, x:x+w][local], 65))
    if not pad_levels:
        return correction
    target = float(np.median(pad_levels))
    for identifier, x, y, w, h in regions:
        mask = (labels[y:y+h, x:x+w] == identifier).astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        core = dist >= max(1.5, .15 * float(dist.max()))
        if np.count_nonzero(core) < 12:
            continue
        patch = image[y:y+h, x:x+w]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        xx = (xx - w / 2) / max(w / 2, 1)
        yy = (yy - h / 2) / max(h / 2, 1)
        design = np.stack([np.ones_like(xx), xx, yy], axis=-1)
        values = patch[core]
        # Robust plane fit limits influence of dark cavities; it does not label defects.
        a = design[core]
        fit = np.linalg.lstsq(a, values, rcond=None)[0]
        for _ in range(4):
            error = values - a @ fit
            mad = max(1., float(np.median(np.abs(error - np.median(error))) * 1.4826))
            weight = np.minimum(1., 1.5 * mad / np.maximum(np.abs(error), .01))
            fit = np.linalg.lstsq(a * np.sqrt(weight[:, None]), values * np.sqrt(weight), rcond=None)[0]
        slope = fit[1] * xx + fit[2] * yy
        slope *= min(1., 8. / max(float(np.max(np.abs(slope))), .01))
        offset = np.clip(.5 * (target - np.percentile(values, 65)), -10, 10)
        feather = np.clip(dist / max(2., .25 * float(dist.max())), 0, 1)
        correction[y:y+h, x:x+w] += feather * (offset - .65 * slope)
    return correction


def tone(image, low, high):
    """One global increasing curve preserves intensity ordering without hard clipping."""
    span = max(high - low, 30.)
    center = low + .62 * span
    return (12 + 231 / (1 + np.exp(np.clip(-4.2 * (image - center) / span, -30, 30)))).astype(np.float32)


def process(raw, component):
    flattened, broad = broad_correction(raw)
    flattened, rows = row_correction(flattened, component)
    labels, regions, residual, noise = propose_regions(flattened, component)
    local = pad_correction(flattened, labels, regions)
    corrected = flattened + local
    # Protect all visible structure, including structures rejected by the pad detector.
    smooth = blur(raw, 1.)
    detail = np.abs(smooth - blur(raw, 3))
    protected = ((labels > 0) | (np.abs(blur(residual, 1.)) > max(4., noise * 3)) | (detail > max(2., noise * 2))).astype(np.uint8)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - protected, cv2.DIST_L2, 5)
    background_weight = np.clip(distance / 5., 0, 1) * .65
    cleaned = corrected * (1 - background_weight) + blur(corrected, 1.5) * background_weight
    low, high = np.percentile(flattened, (5, 99.5))
    outputs = {
        'reference_gray': tone(cleaned, low, high),
        'detail_gray': tone(flattened, low, high),
        'region_mask': (labels > 0).astype(np.float32) * 255,
        'background_smoothing': background_weight / .65 * 255,
        'correction_field': 128 + 3 * (broad + rows + local),
    }
    return outputs, labels, regions, corrected


def region_metrics(raw, outputs, labels, regions):
    records = []
    for identifier, x, y, w, h in regions:
        mask = (labels[y:y+h, x:x+w] == identifier).astype(np.uint8)
        core = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if core.sum() < 8:
            continue
        source = raw[y:y+h, x:x+w][core]
        for method in ('input', 'detail_gray', 'reference_gray'):
            image = raw if method == 'input' else outputs[method]
            values = image[y:y+h, x:x+w][core]
            # All interior pixels are included: no claim that these are healthy pixels.
            correlation = None
            if np.std(source) > 1e-6 and np.std(values) > 1e-6:
                correlation = float(np.corrcoef(source, values)[0, 1])
            records.append(dict(region_id=identifier, x=x, y=y, width=w, height=h,
                                method=method, interior_pixels=int(core.sum()),
                                interior_mean=float(np.mean(values)), interior_std=float(np.std(values)),
                                interior_iqr=float(np.percentile(values, 75) - np.percentile(values, 25)),
                                input_correlation=correlation, clipping_fraction=float(np.mean((values <= 1) | (values >= 254)))))
    return records


def write_csv(path, records, fields=None):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def comparison(path, raw, output, labels, regions, old=None):
    overlay = cv2.cvtColor(u8(raw), cv2.COLOR_GRAY2RGB)
    for identifier, x, y, w, h in regions:
        contour_mask = (labels[y:y+h, x:x+w] == identifier).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay[y:y+h, x:x+w], contours, -1, (40, 230, 100), 1)
    panels = [('Input', raw), ('V2 reference_gray', old if old is not None else raw),
              ('V3 detail_gray', output['detail_gray']), ('V3 reference_gray', output['reference_gray']),
              ('Proposed regions: review required', overlay), ('Background smoothing support', output['background_smoothing'])]
    width = 520
    height = round(raw.shape[0] / raw.shape[1] * width)
    sheet = Image.new('RGB', (width * 3, (height + 30) * 2), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for index, (title, array) in enumerate(panels):
        left, top = (index % 3) * width, (index // 3) * (height + 30)
        panel = Image.fromarray(u8(array)).convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
        sheet.paste(panel, (left, top + 30))
        draw.text((left + 8, top + 8), title, fill='white')
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--sample', action='store_true', help='Process one image per component.')
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if args.sample:
        paths = [next(p for p in paths if p.name.startswith(c + '_')) for c in ('J1', 'J4', 'U29', 'U30')]
    if not paths:
        raise ValueError('No input images found.')
    all_records, summaries = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        print(f'[{index}/{len(paths)}] {path.name}: native-resolution correction', flush=True)
        raw = read(path)
        component = path.stem.split('_')[0]
        outputs, labels, regions, corrected = process(raw, component)
        for method, image in outputs.items():
            directory = args.output / method
            directory.mkdir(exist_ok=True)
            Image.fromarray(u8(image)).save(directory / (path.stem + '.png'))
        numeric = args.output / 'numeric'
        numeric.mkdir(exist_ok=True)
        np.savez_compressed(numeric / (path.stem + '.npz'), corrected=corrected, region_labels=labels)
        sheets = args.output / 'comparison_sheets'
        sheets.mkdir(exist_ok=True)
        previous = OUTPUT.parent / 'competitor_target_v2' / 'reference_gray' / path.name
        comparison(sheets / (path.stem + '.jpg'), raw, outputs, labels, regions, read(previous) if previous.exists() else None)
        records = region_metrics(raw, outputs, labels, regions)
        for record in records:
            record.update(filename=path.name, component=component)
        all_records.extend(records)
        for method in ('input', 'detail_gray', 'reference_gray'):
            selected = [r for r in records if r['method'] == method]
            image = raw if method == 'input' else outputs[method]
            summaries.append(dict(filename=path.name, component=component, method=method,
                                  proposed_regions=len(regions), measured_regions=len(selected),
                                  mean_region_interior_std=float(np.mean([r['interior_std'] for r in selected])) if selected else None,
                                  clipping_fraction=float(np.mean((image <= 1) | (image >= 254)))))
        print(f'  {len(regions)} proposed regions; review overlays before trusting region metrics.', flush=True)
    write_csv(args.output / 'metrics.csv', summaries)
    fields = ['region_id', 'x', 'y', 'width', 'height', 'method', 'interior_pixels', 'interior_mean', 'interior_std', 'interior_iqr', 'input_correlation', 'clipping_fraction', 'filename', 'component']
    write_csv(args.output / 'per_region_metrics.csv', all_records, fields)
    (args.output / 'run.json').write_text(json.dumps(dict(pipeline='competitor_target_v3', input=str(args.input.resolve()), files=[p.name for p in paths], sample=args.sample,
        note='Unvalidated region proposals. Interior statistics include defects. No defect accuracy or healthy-pad uniformity claim.'), indent=2), encoding='utf-8')
    print(f'Results: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()

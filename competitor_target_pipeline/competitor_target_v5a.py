"""V5a: evidence-gated rectangular-pad correction and repeated inter-pad spot attenuation."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks

import automatic_v5 as auto
import competitor_target_v3 as base

OUTPUT = base.OUTPUT.parent / 'competitor_target_v5a'


def axis_peaks(profile, diameter):
    profile = base.blur(profile, max(1., diameter * .07))
    peaks, _ = find_peaks(profile, distance=max(2, int(diameter * 1.35)),
                          prominence=max(.04, float(np.std(profile)) * .20))
    peaks = peaks[(peaks > diameter * .6) & (peaks < len(profile) - diameter * .6)]
    if len(peaks) < 4:
        return np.array([], dtype=int)
    gaps = np.diff(peaks)
    pitch = float(np.median(gaps))
    if abs(gaps[0] / pitch - 1) > .18:
        peaks = peaks[1:]
    if abs(gaps[-1] / pitch - 1) > .18:
        peaks = peaks[:-1]
    if len(peaks) < 4:
        return np.array([], dtype=int)
    gaps = np.diff(peaks)
    pitch = float(np.median(gaps))
    if not 1.35 * diameter < pitch < 3.2 * diameter:
        return np.array([], dtype=int)
    # Allow an occasional missing row/column but not an irregular collection.
    ratio = gaps / pitch
    if np.mean(np.abs(ratio - np.rint(ratio)) < .18) < .8:
        return np.array([], dtype=int)
    return peaks


def rectangular_layout(raw, diameter, noise):
    labels = np.zeros(raw.shape, np.int32)
    empty = dict(used=False, reason='No supported rectangular layout', accepted=0)
    if not .035 <= diameter / min(raw.shape) <= .12:
        return labels, [], empty
    smooth = base.blur(raw, 1.)
    gx, gy = np.gradient(smooth, axis=1), np.gradient(smooth, axis=0)
    side = max(3, round(diameter * .55)) | 1
    tolerance = max(3, round(diameter * .23)) | 1
    sx = cv2.boxFilter(gx, -1, (3, side), borderType=cv2.BORDER_REFLECT)
    sy = cv2.boxFilter(gy, -1, (side, 3), borderType=cv2.BORDER_REFLECT)
    half_shift = round(diameter / 2)
    # Four opposing edges must coincide. A bright trace alone cannot provide
    # both vertical sides and both horizontal sides of a rectangular pad.
    response = np.maximum(0, np.minimum.reduce([
        np.roll(cv2.dilate(sx, np.ones((1,tolerance),np.uint8)), half_shift, axis=1),
        np.roll(cv2.dilate(-sx, np.ones((1,tolerance),np.uint8)), -half_shift, axis=1),
        np.roll(cv2.dilate(sy, np.ones((tolerance,1),np.uint8)), half_shift, axis=0),
        np.roll(cv2.dilate(-sy, np.ones((tolerance,1),np.uint8)), -half_shift, axis=0)]))
    response[:half_shift] = 0
    response[-half_shift:] = 0
    response[:,:half_shift] = 0
    response[:,-half_shift:] = 0
    xs = axis_peaks(np.mean(response, axis=0), diameter)
    ys = axis_peaks(np.mean(response, axis=1), diameter)
    if len(xs) < 4 or len(ys) < 4:
        return labels, [], dict(empty, columns=xs.tolist(), rows=ys.tolist())
    regions = []
    strengths = []
    margin = max(2, round(diameter * .25))
    half = diameter / 2
    for cy in ys:
        for cx in xs:
            y0, y1 = max(0, round(cy - diameter * .23)), min(raw.shape[0], round(cy + diameter * .23))
            x0, x1 = max(0, round(cx - diameter * .23)), min(raw.shape[1], round(cx + diameter * .23))
            xp = np.median(gx[y0:y1], axis=0)
            yp = np.median(gy[:, x0:x1], axis=1)
            def edge(profile, position, sign):
                a, b = max(1, round(position) - margin), min(len(profile) - 1, round(position) + margin + 1)
                values = sign * profile[a:b]
                best = int(np.argmax(values))
                return a + best, float(values[best])
            left, ls = edge(xp, cx - half, 1)
            right, rs = edge(xp, cx + half, -1)
            top, ts = edge(yp, cy - half, 1)
            bottom, bs = edge(yp, cy + half, -1)
            w, h = right - left + 1, bottom - top + 1
            if not (.60 * diameter < min(w, h) and max(w, h) < 1.45 * diameter
                    and max(w, h) / min(w, h) < 1.4
                    and min(ls, rs, ts, bs) > max(.3, .28 * noise)):
                continue
            if labels[top:bottom + 1, left:right + 1].any():
                continue
            identifier = len(regions) + 1
            labels[top:bottom + 1, left:right + 1] = identifier
            regions.append((identifier, left, top, w, h))
            strengths.append(min(ls, rs, ts, bs))
    if len(regions) < max(12, .5 * len(xs) * len(ys)):
        return np.zeros_like(labels), [], dict(empty, reason='Too few four-sided pad boundaries',
                                               proposed=len(regions), columns=xs.tolist(), rows=ys.tolist())
    # A border trace can create an extra projected column. Keep only axes with
    # supported pads across most of the opposite axis, not isolated rectangles.
    assignments = [(int(np.argmin(np.abs(xs-(x+w/2)))), int(np.argmin(np.abs(ys-(y+h/2)))))
                   for _,x,y,w,h in regions]
    good_x = {i for i in range(len(xs)) if sum(ix == i for ix,_ in assignments) >= .65 * len(ys)}
    good_y = {i for i in range(len(ys)) if sum(iy == i for _,iy in assignments) >= .65 * len(xs)}
    if min(len(good_x),len(good_y)) < 4:
        return np.zeros_like(labels), [], dict(empty, reason='Insufficient repeated boundary support')
    selected = [region for region,(ix,iy) in zip(regions,assignments) if ix in good_x and iy in good_y]
    labels.fill(0)
    regions = []
    for _,x,y,w,h in selected:
        identifier = len(regions)+1
        labels[y:y+h,x:x+w] = identifier
        regions.append((identifier,x,y,w,h))
    xs,ys = xs[sorted(good_x)],ys[sorted(good_y)]
    return labels, regions, dict(used=True, accepted=len(regions), columns=xs.tolist(), rows=ys.tolist(),
                                 diameter=float(diameter), minimum_side_strength=float(min(strengths)),
                                 note='Rectangular supports are not verified solder segmentation.')


def pad_field(image, labels, regions):
    estimates = []
    for identifier, x, y, w, h in regions:
        margin = max(3, round(min(w, h) * .18))
        patch = image[y:y+h, x:x+w]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        xx = (xx - (w - 1) / 2) / max(w / 2, 1)
        yy = (yy - (h - 1) / 2) / max(h / 2, 1)
        core = np.zeros((h, w), bool)
        core[margin:h-margin, margin:w-margin] = True
        if core.sum() < 25:
            continue
        level = float(np.percentile(patch[core], 65))
        design = np.stack([np.ones_like(xx), xx, yy], axis=-1)
        values = patch[core]
        a = design[core]
        fit = np.linalg.lstsq(a, values, rcond=None)[0]
        for _ in range(5):
            error = values - a @ fit
            scale = max(1., float(np.median(np.abs(error - np.median(error))) * 1.4826))
            weights = np.minimum(1., scale / np.maximum(np.abs(error), .01))
            weights[error < -1.5 * scale] *= .1
            fit = np.linalg.lstsq(a * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
        slope = .5 * (fit[1] * xx + fit[2] * yy)
        slope *= min(1., 4. / max(float(np.abs(slope).max()), .001))
        estimates.append((identifier, x, y, w, h, level, slope))
    field, denominator = np.zeros_like(image), np.zeros_like(image)
    records = []
    if not estimates:
        return field, records
    target = float(np.percentile([e[5] for e in estimates], 85))
    for identifier, x, y, w, h, level, slope in estimates:
        shift = float(np.clip(.85 * (target - level), -6, 18))
        margin = max(3, round(min(w, h) * .12))
        x0, y0 = max(0, x-margin), max(0, y-margin)
        x1, y1 = min(image.shape[1], x+w+margin), min(image.shape[0], y+h+margin)
        local = labels[y0:y1, x0:x1]
        dist = cv2.distanceTransform((local != identifier).astype(np.uint8), cv2.DIST_L2, 5)
        weight = np.maximum(0, 1-dist/margin)
        weight[(local > 0) & (local != identifier)] = 0
        extended_slope = np.pad(slope, ((y-y0, y1-y-h), (x-x0, x1-x-w)), mode='edge')
        field[y0:y1, x0:x1] += weight * (shift - extended_slope)
        denominator[y0:y1, x0:x1] += weight
        records.append(dict(region_id=identifier, provisional_level=level, shift=shift,
                            slope_max_abs=float(np.abs(slope).max())))
    return field / np.maximum(1, denominator), records


def repeated_dots(raw, labels, layout, noise):
    mask = np.zeros(raw.shape, np.float32)
    if not layout['used']:
        return mask, []
    diameter = layout['diameter']
    xs, ys = np.asarray(layout['columns']), np.asarray(layout['rows'])
    px, py = float(np.median(np.diff(xs))), float(np.median(np.diff(ys)))
    guard = max(3, round(diameter * .23))
    protected = cv2.dilate((labels > 0).astype(np.uint8), np.ones((2*guard+1, 2*guard+1), np.uint8)) > 0
    signal = base.blur(raw, .7) - base.blur(raw, max(2., diameter * .14))
    binary = (signal > max(2.5, noise * 2.5)).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        cx, cy = x+(w-1)/2, y+(h-1)/2
        if not (3 <= min(w,h) and max(w,h) < diameter * .38 and max(w,h)/min(w,h) < 1.55
                and area > 4 and area / (w*h) > .45 and xs[0] < cx < xs[-1] and ys[0] < cy < ys[-1]):
            continue
        if protected[y:y+h, x:x+w].any():
            continue
        phase = np.array([((cx-xs[0])/px) % 1, ((cy-ys[0])/py) % 1])
        candidates.append((contour, cx, cy, phase))
    accepted = []
    for contour, cx, cy, phase in candidates:
        peers = []
        for _, ox, oy, other in candidates:
            distance = np.abs(phase-other)
            if np.all(np.minimum(distance, 1-distance) < .10):
                peers.append((ox, oy))
        # Repeated phase across several cells, not merely one isolated bright speck.
        cells = {(int((x-xs[0])/px), int((y-ys[0])/py)) for x,y in peers}
        if len(cells) < 4:
            continue
        cv2.drawContours(mask, [contour], -1, 1., -1)
        accepted.append(dict(x=cx, y=cy, repeated_cells=len(cells)))
    mask = np.clip(base.blur(mask, .8), 0, 1)
    mask[protected] = 0
    return mask, accepted


def process(raw, component=None):
    raw = np.asarray(raw,dtype=np.float32)
    outputs, _, _, baseline, info = auto.process(raw)
    labels, regions, layout = rectangular_layout(raw, info['feature_diameter_native'], info['noise_sigma'])
    field, adjustments = pad_field(baseline, labels, regions) if layout['used'] else (np.zeros_like(raw), [])
    dots, spots = repeated_dots(raw, labels, layout, info['noise_sigma'])
    difference = np.maximum(0, baseline-base.blur(baseline, max(2., info['feature_diameter_native'] * .14)))
    attenuation = dots * np.minimum(8., .65 * difference)
    corrected = baseline + field - attenuation
    reference = auto.tone(corrected, info['display_background'], info['display_signal_span']) if layout['used'] else outputs['reference_gray'].copy()
    pad_only = auto.tone(baseline+field, info['display_background'], info['display_signal_span']) if layout['used'] else reference.copy()
    images = dict(reference_gray=reference, pad_only_gray=pad_only, pad_mask=(labels > 0).astype(np.float32)*255,
                  dot_attenuation_mask=dots*255, pad_correction=128+field*5)
    details = dict(layout=layout, pad_adjustments=adjustments, attenuated_spots=spots,
                   dot_delta_max=float(attenuation.max()), baseline_diagnostics=info,
                   note='Automatic, unreviewed supports. Dot attenuation is an inspection display choice, not defect classification.')
    return images, labels, regions, corrected, details, outputs['reference_gray']


def save_sheet(path, panels, width=640):
    height = round(panels[0][1].shape[0] / panels[0][1].shape[1] * width)
    canvas = Image.new('RGB', (len(panels)*width, height+32), (24,24,24))
    draw = ImageDraw.Draw(canvas)
    for i, (title, array) in enumerate(panels):
        canvas.paste(Image.fromarray(base.u8(array)).convert('RGB').resize((width,height), Image.Resampling.LANCZOS), (i*width,32))
        draw.text((i*width+8,9), title, fill='white')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base.INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--files', nargs='+')
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if args.files:
        paths = [p for p in paths if p.name in args.files]
        if set(args.files) - {p.name for p in paths}:
            raise ValueError('Requested image missing')
    if not paths or len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError('No images or duplicate output stems')
    if args.output.resolve() == (base.OUTPUT.parent/'competitor_target_v5_auto').resolve():
        raise ValueError('Do not overwrite the V5 automatic baseline')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, metrics = [], []
    for i, path in enumerate(paths, 1):
        print(f'[{i}/{len(paths)}] {path.name}', flush=True)
        with Image.open(path) as source:
            if source.mode != 'L':
                raise ValueError('Expected 8-bit grayscale input')
            raw = np.asarray(source, dtype=np.float32)
        images, labels, regions, numeric, diagnostic, previous = process(raw)
        for name, array in images.items():
            (args.output/name).mkdir(exist_ok=True)
            Image.fromarray(base.u8(array)).save(args.output/name/(path.stem+'.png'))
        overlay = cv2.cvtColor(base.u8(raw),cv2.COLOR_GRAY2RGB)
        for _,x,y,w,h in regions:
            cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),(50,220,90),1)
        for spot in diagnostic['attenuated_spots']:
            cv2.circle(overlay,(round(spot['x']),round(spot['y'])),4,(255,170,30),1)
        (args.output/'selection_overlay').mkdir(exist_ok=True)
        Image.fromarray(overlay).save(args.output/'selection_overlay'/(path.stem+'.png'))
        for folder in ('numeric','input_output','comparison_sheets'):
            (args.output/folder).mkdir(exist_ok=True)
        np.savez_compressed(args.output/'numeric'/(path.stem+'.npz'), corrected=numeric, region_labels=labels)
        save_sheet(args.output/'input_output'/(path.stem+'.png'), [(f'Input: {path.name}',raw),('V5a: reference_gray',images['reference_gray'])],min(1000,raw.shape[1]))
        save_sheet(args.output/'comparison_sheets'/(path.stem+'.png'), [('Input',raw),('V5 automatic baseline',previous),('V5a',images['reference_gray'])])
        records = base.region_metrics(raw,dict(detail_gray=previous,reference_gray=images['reference_gray']),labels,regions)
        for row in records:
            row.update(filename=path.name)
            row['method'] = {'input':'input','detail_gray':'v5_auto','reference_gray':'v5a'}[row['method']]
        metrics.extend(records)
        diagnostic.update(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape=list(raw.shape))
        manifest.append(diagnostic)
        print(f"  layout={diagnostic['layout']['used']}, pads={len(regions)}, repeated spots={len(diagnostic['attenuated_spots'])}",flush=True)
    fields = ['region_id','x','y','width','height','method','interior_pixels','interior_mean','interior_std','interior_iqr','input_correlation','clipping_fraction','filename']
    base.write_csv(args.output/'per_region_metrics.csv',metrics,fields)
    code = [Path(__file__),Path(auto.__file__),Path(base.__file__)]
    (args.output/'run.json').write_text(json.dumps(dict(pipeline='v5a',input=str(args.input.resolve()),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code},images=manifest,
        validation='Unreviewed experimental results; no defect accuracy claim.'),indent=2),encoding='utf-8')
    print(args.output.resolve(),flush=True)


if __name__ == '__main__':
    main()

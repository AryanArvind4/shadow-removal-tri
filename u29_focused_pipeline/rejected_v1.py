"""U29-only, source-derived pad display enhancement at native resolution."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'competitor_target_pipeline'))
import automatic_v5 as auto
import competitor_target_v3 as base
import competitor_target_v5a as v5a

OUTPUT = Path(__file__).resolve().parent / 'result' / 'u29_focused_v1'


@dataclass(frozen=True)
class Settings:
    pad_level: float = 182.
    pad_gain: float = 3.6
    slope_limit: float = 8.
    background_level: float = 72.
    background_contrast: float = .42
    background_blur: float = 1.3
    dot_strength: float = .75


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def display_curve(delta, level):
    """Continuous increasing mapping with headroom for dark and bright features."""
    positive = (247. - level) * np.tanh(np.maximum(delta, 0.) / (247. - level))
    negative = (level - 8.) * np.tanh(np.minimum(delta, 0.) / (level - 8.))
    return level + positive + negative


def fit_pad_plane(patch, noise, slope_limit):
    h, w = patch.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    xx = (xx - (w - 1) / 2.) / (w / 2.)
    yy = (yy - (h - 1) / 2.) / (h / 2.)
    # A small central fitting window can be occupied almost entirely by a large
    # dark cavity. Include the surrounding interior so that cavity is not used
    # as the solder brightness reference.
    margin = max(3, round(min(h, w) * .12))
    core = np.zeros(patch.shape, bool)
    core[margin:h-margin, margin:w-margin] = True
    a = np.stack([np.ones_like(xx), xx, yy], axis=-1)[core]
    values = patch[core]
    fit = np.array([np.percentile(values, 65), 0., 0.], np.float32)
    for _ in range(8):
        error = values - a @ fit
        scale = max(noise, float(np.median(np.abs(error - np.median(error))) * 1.4826))
        weights = np.minimum(1., 1.5 * scale / np.maximum(np.abs(error), .01))
        weights[error < -1.5 * scale] *= .08
        fit = np.linalg.lstsq(a * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
    slope = fit[1] * xx + fit[2] * yy
    slope *= min(1., slope_limit / max(float(np.abs(slope).max()), .001))
    level = float(np.percentile((patch - slope)[core], 65))
    return level + slope, core, level


def pad_support(clean, regions):
    """Use source brightness at the perimeter to retain irregular pad corners."""
    support = np.zeros_like(clean)
    for _, x, y, w, h in regions:
        extension = 5
        x0, y0 = max(0, x-extension), max(0, y-extension)
        x1, y1 = min(clean.shape[1], x+w+extension), min(clean.shape[0], y+h+extension)
        patch = clean[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        signed_distance = np.minimum.reduce([xx-x, x+w-1-xx, yy-y, y+h-1-yy])
        outer = signed_distance < -2
        inner = signed_distance > 5
        low = float(np.percentile(patch[outer], 55))
        high = float(np.percentile(patch[inner], 70))
        intensity = smoothstep((patch-low) / max(5., high-low))
        # Interior dark marks receive the full pad transform, not a hole in the
        # mask. Only the boundary transition is guided by the observed signal.
        interior = smoothstep((signed_distance-1.) / 3.)
        geometric = smoothstep((signed_distance+3.) / 5.)
        alpha = np.maximum(interior, geometric * intensity)
        support[y0:y1, x0:x1] = np.maximum(support[y0:y1, x0:x1], alpha)
    return support


def clean_source(raw, noise):
    estimate = cv2.fastNlMeansDenoising(base.u8(raw), None, max(2.5, 2.*noise), 7, 21).astype(np.float32)
    change = raw-estimate
    weight = .85/(1.+(np.abs(change)/max(2., 2.*noise))**6)
    return raw-weight*change


def shared_row_shading(clean, regions, planes, layout, noise):
    """Estimate only row departures from the typical within-pad row profile."""
    profiles, row_ids = [], []
    axis = np.linspace(-.6, .6, 41)
    for (_,x,y,w,h), (plane,_,_) in zip(regions, planes):
        margin = max(3, round(w*.22))
        profile = np.percentile((clean[y:y+h,x:x+w]-plane)[:,margin:w-margin], 65, axis=1)
        profile = np.interp(axis, np.linspace(-1.,1.,h), profile).astype(np.float32)
        profile = base.blur(profile, 3.)
        profiles.append(profile-np.median(profile))
        row_ids.append(int(np.argmin(np.abs(np.asarray(layout['rows'])-(y+(h-1)/2.)))))
    profiles, row_ids = np.asarray(profiles), np.asarray(row_ids)
    groups = {i:np.median(profiles[row_ids==i],axis=0) for i in np.unique(row_ids)}
    common = np.median(list(groups.values()),axis=0)
    row_fields, records = {}, []
    for row, curve in groups.items():
        delta = curve-common
        shared = profiles[row_ids==row]-common
        active = np.abs(delta) > max(1.,noise)
        agreement = float(np.mean(np.sign(shared[:,active])==np.sign(delta[active]))) if active.any() else 0.
        amplitude = float(np.ptp(delta))
        used = amplitude > 2.5*noise and agreement >= .70 and len(shared) >= 5
        correction = -6.*np.tanh(.8*delta/6.) if used else np.zeros_like(delta)
        row_fields[row] = correction
        records.append(dict(row=int(row),used=bool(used),agreement=agreement,profile_range=amplitude,
                            max_abs=float(np.abs(correction).max())))
    field = np.zeros_like(clean)
    for (_,x,y,w,h), row in zip(regions,row_ids):
        curve = np.interp(np.linspace(-1.,1.,h),axis,row_fields[row])
        field[y:y+h,x:x+w] = curve[:,None]
    return field, records


def enhance(raw, previous, labels, regions, layout, noise, settings=Settings()):
    if not layout.get('used') or len(regions) < 12:
        raise ValueError('U29 pad boundaries were not supported; no focused result generated.')
    clean = clean_source(raw, noise)
    support = pad_support(clean, regions)
    planes = [fit_pad_plane(clean[y:y+h,x:x+w],noise,settings.slope_limit) for _,x,y,w,h in regions]
    row_field, row_records = shared_row_shading(clean,regions,planes,layout,noise)
    guard = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    dots, spots = v5a.repeated_dots(raw, labels, layout, noise)
    bg_reference = previous.copy()
    dot_detail = np.maximum(0., bg_reference-base.blur(bg_reference, 3.))
    bg_reference -= settings.dot_strength * dots * dot_detail
    background = base.blur(bg_reference, settings.background_blur)
    anchor = float(np.median(background[~guard]))
    background = display_curve(settings.background_contrast*(background-anchor), settings.background_level)
    reference = background.copy()
    detail_reference = display_curve(settings.background_contrast*(previous-anchor), settings.background_level)
    plane_map = np.zeros_like(raw)
    records = []
    for (identifier, x, y, w, h), (plane, core, level) in zip(regions,planes):
        # One gain for all pads avoids amplifying weak pad noise by an arbitrary
        # amount. Corrections contain only an offset and a bounded plane.
        extension = 5
        x0, y0 = max(0, x-extension), max(0, y-extension)
        x1, y1 = min(raw.shape[1], x+w+extension), min(raw.shape[0], y+h+extension)
        padded_plane = np.pad(plane, ((y-y0, y1-y-h), (x-x0, x1-x-w)), mode='edge')
        padded_row = np.pad(row_field[y:y+h,x:x+w], ((y-y0, y1-y-h), (x-x0, x1-x-w)), mode='edge')
        local = clean[y0:y1, x0:x1]+padded_row
        rendered = display_curve(settings.pad_gain*(local-padded_plane), settings.pad_level)
        detail = display_curve(settings.pad_gain*(raw[y0:y1, x0:x1]+padded_row-padded_plane), settings.pad_level)
        alpha = support[y0:y1, x0:x1]
        reference[y0:y1, x0:x1] = (1-alpha)*background[y0:y1, x0:x1] + alpha*rendered
        detail_reference[y0:y1, x0:x1] = (1-alpha)*detail_reference[y0:y1, x0:x1] + alpha*detail
        plane_map[y:y+h, x:x+w] = plane
        records.append(dict(region_id=identifier, source_level=level,
                            slope_max_abs=float(np.abs(plane-level).max()),
                            core_pixels=int(core.sum())))
    return dict(reference_gray=reference, detail_gray=detail_reference,
                pad_support=support*255, dot_attenuation_mask=dots*255), dict(
        pad_plane=plane_map, source=raw, enhanced=reference, region_labels=labels,
        pad_support=support, row_correction=row_field), dict(settings=asdict(settings), pad_adjustments=records,
                                  row_shading=row_records,
                                  attenuated_spots=spots, layout=layout, noise_sigma=noise)


def process(raw, settings=Settings()):
    raw = np.asarray(raw, dtype=np.float32)
    old, labels, regions, _, diagnostics, _ = v5a.process(raw)
    images, arrays, details = enhance(raw, old['reference_gray'], labels, regions,
                                    diagnostics['layout'], diagnostics['baseline_diagnostics']['noise_sigma'], settings)
    return images, arrays, details, old['reference_gray'], regions


def source_mark_checks(raw, previous, result, labels, noise):
    """Compare source-selected dark responses, not confirmed defects."""
    guard = cv2.erode((labels>0).astype(np.uint8),np.ones((13,13),np.uint8))>0
    residual = base.blur(raw,.65)-base.blur(raw,3.)
    selected = ((residual < -max(3.,2.5*noise)) & guard).astype(np.uint8)
    count, candidates, stats, centers = cv2.connectedComponentsWithStats(selected,8)
    records = []
    for i in range(1,count):
        if not 2 <= stats[i,cv2.CC_STAT_AREA] <= 160:
            continue
        mark = candidates==i
        pad_id = int(np.median(labels[mark]))
        ring = (cv2.dilate(mark.astype(np.uint8),np.ones((9,9),np.uint8))>0) & ~mark & (labels==pad_id)
        if ring.sum()<12:
            continue
        contrasts = {name:float(np.percentile(a[ring],65)-np.mean(a[mark]))
                     for name,a in [('input',raw),('v5a',previous),('focused',result)]}
        records.append(dict(mark_id=i,pad_id=pad_id,x=float(centers[i,0]),y=float(centers[i,1]),
                            pixels=int(mark.sum()),**contrasts))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base.INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--pad-level', type=float, default=Settings.pad_level)
    parser.add_argument('--pad-gain', type=float, default=Settings.pad_gain)
    args = parser.parse_args()
    paths = sorted(args.input.glob('U29_*.jpg'))
    if args.files:
        paths = [p for p in paths if p.name in args.files]
        if set(args.files)-{p.name for p in paths}:
            raise ValueError('Requested U29 images missing')
    if not paths:
        raise ValueError('No U29 input images')
    settings = Settings(pad_level=args.pad_level, pad_gain=args.pad_gain)
    if not 140 <= settings.pad_level <= 210 or not 1 <= settings.pad_gain <= 6:
        raise ValueError('Invalid display settings')
    if (ROOT/'competitor_target_pipeline'/'result') in args.output.resolve().parents:
        raise ValueError('Focused outputs must be separate from the V5a results')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, metrics, marks = [], [], []
    for i, path in enumerate(paths, 1):
        print(f'[{i}/{len(paths)}] {path.name}', flush=True)
        with Image.open(path) as image:
            if image.mode != 'L':
                raise ValueError('Expected 8-bit grayscale source')
            raw = np.asarray(image, dtype=np.float32)
        images, arrays, details, previous, regions = process(raw, settings)
        for name, array in images.items():
            (args.output/name).mkdir(exist_ok=True)
            Image.fromarray(base.u8(array)).save(args.output/name/(path.stem+'.png'))
        for name in ('numeric', 'input_output', 'comparison_sheets', 'pad_crops'):
            (args.output/name).mkdir(exist_ok=True)
        np.savez_compressed(args.output/'numeric'/(path.stem+'.npz'), **arrays)
        v5a.save_sheet(args.output/'input_output'/(path.stem+'.png'), [('Input',raw),('U29 focused',images['reference_gray'])],raw.shape[1])
        v5a.save_sheet(args.output/'comparison_sheets'/(path.stem+'.png'), [('Input',raw),('V5a',previous),('U29 focused',images['reference_gray'])],raw.shape[1])
        for title, y0, y1, x0, x1 in [('upper',40,190,55,405),('middle',200,405,55,405),('lower',425,630,55,405)]:
            v5a.save_sheet(args.output/'pad_crops'/(path.stem+'_'+title+'.png'),
                          [(name,a[y0:y1,x0:x1]) for name,a in [('Input',raw),('V5a',previous),('U29 focused',images['reference_gray'])]],700)
        rows = base.region_metrics(raw, {'detail_gray':previous,'reference_gray':images['reference_gray']}, arrays['region_labels'], regions)
        for row in rows:
            row['filename'] = path.name
            row['method'] = {'input':'input','detail_gray':'v5a','reference_gray':'u29_focused'}[row['method']]
        metrics.extend(rows)
        checks = source_mark_checks(raw,previous,images['reference_gray'],arrays['region_labels'],details['noise_sigma'])
        for record in checks:
            record['filename'] = path.name
        marks.extend(checks)
        details['source_mark_checks'] = dict(count=len(checks),
            positive_contrast_count=sum(r['focused']>0 for r in checks),
            note='Automatic source dark responses; may include shadows or texture. Not defect recall.')
        details.update(filename=path.name,shape=list(raw.shape),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest.append(details)
        print(f'  pads={len(regions)}; repeated dots={len(details["attenuated_spots"])}',flush=True)
    fields = ['region_id','x','y','width','height','method','interior_pixels','interior_mean','interior_std','interior_iqr','input_correlation','clipping_fraction','filename']
    base.write_csv(args.output/'per_region_metrics.csv',metrics,fields)
    base.write_csv(args.output/'source_mark_checks.csv',marks,['filename','mark_id','pad_id','x','y','pixels','input','v5a','focused'])
    code = [Path(__file__),Path(v5a.__file__),Path(auto.__file__),Path(base.__file__)]
    (args.output/'run.json').write_text(json.dumps(dict(pipeline='u29_focused_v1',input=str(args.input.resolve()),images=manifest,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        validation='Automatic pad supports and dark marks are unreviewed; no confirmed-defect accuracy or competitor-parity claim.'),indent=2),encoding='utf-8')
    print(args.output.resolve(),flush=True)


if __name__ == '__main__':
    main()

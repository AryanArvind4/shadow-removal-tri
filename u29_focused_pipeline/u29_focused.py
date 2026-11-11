"""U29 V2: smooth photometric fields with source boundaries protected."""
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
sys.path.insert(0, str(ROOT/'competitor_target_pipeline'))
import automatic_v5 as auto
import competitor_target_v3 as base
import competitor_target_v5a as v5a

OUTPUT = Path(__file__).resolve().parent/'result'/'u29_focused_v2'
REJECTED = OUTPUT.parent/'u29_focused_v1'


@dataclass(frozen=True)
class Settings:
    target_contrast: float = 26.
    maximum_gain: float = 1.8
    field_pitch_fraction: float = .65
    display_gain: float = 2.2
    display_background: float = 88.
    edge_guard_pixels: int = 7
    background_change_limit: float = 5.


def display_curve(linear):
    """Linear across the useful range; one common, monotonic curve everywhere."""
    result = np.asarray(linear,dtype=np.float32).copy()
    low,high = result<40.,result>210.
    result[low] = 8.+32./(1.+(40.-result[low])/32.)
    result[high] = 247.-37./(1.+(result[high]-210.)/37.)
    return result


def smooth_samples(shape, centers, values, sigma):
    numerator,denominator = np.zeros(shape,np.float32),np.zeros(shape,np.float32)
    for (x,y),value in zip(centers,values):
        numerator[round(y),round(x)] += value
        denominator[round(y),round(x)] += 1.
    numerator,denominator = base.blur(numerator,sigma),base.blur(denominator,sigma)
    return numerator/np.maximum(denominator,1e-12)


def estimate_fields(raw, labels, regions, layout, noise, settings):
    pitch = float(np.median(np.concatenate([np.diff(layout['rows']),np.diff(layout['columns'])])))
    centers,backgrounds,gains,records = [],[],[],[]
    for identifier,x,y,w,h in regions:
        margin = max(3,round(min(w,h)*.15))
        interior = raw[y+margin:y+h-margin,x+margin:x+w-margin]
        pad_level = float(np.percentile(interior,70))
        extent = max(7,round(pitch*.19))
        x0,y0 = max(0,x-extent),max(0,y-extent)
        x1,y1 = min(raw.shape[1],x+w+extent),min(raw.shape[0],y+h+extent)
        yy,xx = np.mgrid[y0:y1,x0:x1]
        distance = np.maximum.reduce([x-xx,xx-(x+w-1),y-yy,yy-(y+h-1)])
        ring = (distance>=5) & (distance<=extent) & (labels[y0:y1,x0:x1]==0)
        if ring.sum()<20:
            continue
        background = float(np.median(raw[y0:y1,x0:x1][ring]))
        contrast = pad_level-background
        gain = float(np.clip(settings.target_contrast/max(contrast,3.*noise),.65,settings.maximum_gain))
        centers.append((x+(w-1)/2.,y+(h-1)/2.))
        backgrounds.append(background)
        gains.append(gain)
        records.append(dict(region_id=identifier,source_level=pad_level,background=background,estimated_contrast=contrast,gain_sample=gain))
    if len(centers)<12:
        raise ValueError('Insufficient pad/background measurements')
    sigma = max(12.,pitch*settings.field_pitch_fraction)
    # Pad coordinates supply sparse measurements, never a rendering mask.
    background = smooth_samples(raw.shape,centers,backgrounds,sigma)
    gain = smooth_samples(raw.shape,centers,gains,sigma)
    return background,gain,records


def enhance(raw, labels, regions, layout, noise, settings=Settings()):
    if not layout.get('used') or len(regions)<12:
        raise ValueError('U29 pad boundaries were not supported; no focused result generated.')
    background,gain,records = estimate_fields(raw,labels,regions,layout,noise,settings)
    distance = cv2.distanceTransform((labels==0).astype(np.uint8),cv2.DIST_L2,5)
    edge_guard = distance<=settings.edge_guard_pixels
    # Denoising and repeated-dot suppression are forbidden throughout every pad
    # and its edge guard. Background changes fade in outside the boundary.
    fade = np.clip((distance-settings.edge_guard_pixels)/10.,0.,1.)
    fade = fade*fade*(3.-2.*fade)
    dots,spots = v5a.repeated_dots(raw,labels,layout,noise)
    residual = np.maximum(0.,raw-base.blur(raw,3.))
    delta = .75*(base.blur(raw,1.8)-raw)-.65*dots*residual
    delta = fade*np.clip(delta,-settings.background_change_limit,settings.background_change_limit)
    corrected = raw+delta
    linear = settings.display_background+settings.display_gain*gain*(corrected-background)
    detail_linear = settings.display_background+settings.display_gain*gain*(raw-background)
    reference,detail = display_curve(linear),display_curve(detail_linear)
    np.testing.assert_array_equal(corrected[edge_guard],raw[edge_guard])
    images = dict(reference_gray=reference,detail_gray=detail,
                  edge_guard=edge_guard.astype(np.float32)*255,
                  background_change=128.+delta*20.,gain_field=255.*(gain-.65)/(settings.maximum_gain-.65))
    arrays = dict(source=raw,enhanced=reference,detail=detail,region_labels=labels,
                  edge_guard=edge_guard,background_field=background,gain_field=gain,
                  background_delta=delta,linear_display=linear)
    details = dict(settings=asdict(settings),layout=layout,noise_sigma=noise,
        pad_measurements=records,attenuated_spots=spots,
        protected_source_change_max=float(np.abs(delta[edge_guard]).max()),
        background_gradient_max=float(max(np.abs(a).max() for a in np.gradient(background))),
        gain_gradient_max=float(max(np.abs(a).max() for a in np.gradient(gain))),
        note='All pixels use the same tone curve. No foreground alpha compositing, pad replacement, per-pad plane rendering or row-interior correction.')
    return images,arrays,details


def process(raw, settings=Settings()):
    raw = np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or min(raw.shape)<64 or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>255:
        raise ValueError('Expected finite 8-bit grayscale image with dimensions >= 64')
    noise = auto.noise_sigma(raw)
    diameter,_ = auto.estimate_scale(raw,noise)
    labels,regions,layout = v5a.rectangular_layout(raw,diameter,noise)
    images,arrays,details = enhance(raw,labels,regions,layout,noise,settings)
    return images,arrays,details,regions


def source_mark_checks(raw, previous, result, labels, noise):
    """Compare source-selected dark responses, not confirmed defects."""
    guard = cv2.erode((labels>0).astype(np.uint8),np.ones((13,13),np.uint8))>0
    residual = base.blur(raw,.65)-base.blur(raw,3.)
    selected = ((residual < -max(3.,2.5*noise)) & guard).astype(np.uint8)
    count,candidates,stats,centers = cv2.connectedComponentsWithStats(selected,8)
    records = []
    for i in range(1,count):
        if not 2<=stats[i,cv2.CC_STAT_AREA]<=160:
            continue
        mark = candidates==i
        pad_id = int(np.median(labels[mark]))
        ring = (cv2.dilate(mark.astype(np.uint8),np.ones((9,9),np.uint8))>0) & ~mark & (labels==pad_id)
        if ring.sum()<12:
            continue
        contrasts = {name:float(np.percentile(a[ring],65)-np.mean(a[mark]))
            for name,a in [('input',raw),('v5a',previous),('focused',result)]}
        records.append(dict(mark_id=i,pad_id=pad_id,x=float(centers[i,0]),y=float(centers[i,1]),pixels=int(mark.sum()),**contrasts))
    return records


def edge_profile_checks(raw, methods, regions):
    """Local edge locations from half-height crossings, selected on source only."""
    records = []
    offsets = np.arange(-7,8,dtype=np.float32)
    def crossing(profile):
        low,high = float(np.median(profile[:3])),float(np.median(profile[-3:]))
        if high-low<1e-4:
            return None
        p = (profile-low)/(high-low)
        locations = []
        for j in range(len(p)-1):
            if p[j]<=.5<p[j+1]:
                locations.append(float(offsets[j]+(.5-p[j])/(p[j+1]-p[j])))
        return min(locations,key=abs) if locations else None
    for identifier,x,y,w,h in regions:
        for side in ('left','right','top','bottom'):
            for fraction in np.linspace(.12,.88,9):
                if side in ('left','right'):
                    cy = y+fraction*(h-1)
                    cx = x if side=='left' else x+w-1
                    xs = cx+offsets*(1 if side=='left' else -1)
                    ys = np.full_like(xs,cy)
                else:
                    cx = x+fraction*(w-1)
                    cy = y if side=='top' else y+h-1
                    ys = cy+offsets*(1 if side=='top' else -1)
                    xs = np.full_like(ys,cx)
                if xs.min()<0 or ys.min()<0 or xs.max()>=raw.shape[1] or ys.max()>=raw.shape[0]:
                    continue
                def sample(image):
                    return cv2.remap(image.astype(np.float32),xs.reshape(1,-1),ys.reshape(1,-1),cv2.INTER_LINEAR).ravel()
                profile = sample(raw)
                contrast = float(np.median(profile[-3:])-np.median(profile[:3]))
                if contrast<6.:
                    continue
                source_crossing = crossing(profile)
                if source_crossing is None or abs(source_crossing)>3.:
                    continue
                for method,image in methods.items():
                    out_crossing = crossing(sample(image))
                    records.append(dict(region_id=identifier,side=side,fraction=float(fraction),method=method,
                        source_contrast=contrast,source_crossing=source_crossing,
                        output_crossing=out_crossing,
                        displacement=None if out_crossing is None else abs(out_crossing-source_crossing)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=base.INPUT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--files',nargs='+')
    args = parser.parse_args()
    paths = sorted(args.input.glob('U29_*.jpg'))
    if args.files:
        paths = [p for p in paths if p.name in args.files]
        if set(args.files)-{p.name for p in paths}:
            raise ValueError('Requested U29 images missing')
    if not paths:
        raise ValueError('No U29 input images')
    if args.output.resolve()==REJECTED.resolve() or (ROOT/'competitor_target_pipeline'/'result') in args.output.resolve().parents:
        raise ValueError('Use a separate output for the shape correction')
    args.output.mkdir(parents=True,exist_ok=True)
    def comparison_sources(folder):
        path = folder/'run.json'
        if not path.exists():
            return {}
        return {r['filename']:r.get('source_sha256',r.get('sha256')) for r in json.loads(path.read_text(encoding='utf-8'))['images']}
    baseline_sources,rejected_sources = comparison_sources(v5a.OUTPUT),comparison_sources(REJECTED)
    manifest,metrics,marks,edges = [],[],[],[]
    for i,path in enumerate(paths,1):
        print(f'[{i}/{len(paths)}] {path.name}',flush=True)
        with Image.open(path) as image:
            if image.mode!='L':
                raise ValueError('Expected 8-bit grayscale source')
            raw = np.asarray(image,dtype=np.float32)
        images,arrays,details,regions = process(raw)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        baseline = v5a.OUTPUT/'reference_gray'/(path.stem+'.png')
        previous = base.read(baseline) if baseline.exists() and baseline_sources.get(path.name)==source_hash else v5a.process(raw)[0]['reference_gray']
        rejected_path = REJECTED/'reference_gray'/(path.stem+'.png')
        rejected = base.read(rejected_path) if rejected_path.exists() and rejected_sources.get(path.name)==source_hash else None
        for name,array in images.items():
            (args.output/name).mkdir(exist_ok=True)
            Image.fromarray(base.u8(array)).save(args.output/name/(path.stem+'.png'))
        for name in ('numeric','input_output','comparison_sheets','shape_comparison','pad_crops'):
            (args.output/name).mkdir(exist_ok=True)
        np.savez_compressed(args.output/'numeric'/(path.stem+'.npz'),**arrays)
        result = images['reference_gray']
        v5a.save_sheet(args.output/'input_output'/(path.stem+'.png'), [('Input',raw),('U29 V2: shape correction',result)],raw.shape[1])
        v5a.save_sheet(args.output/'comparison_sheets'/(path.stem+'.png'), [('Input',raw),('V5a',previous),('U29 V2',result)],raw.shape[1])
        panels = [('Input',raw)]+([('Rejected U29 V1',rejected)] if rejected is not None else [])+[('U29 V2',result)]
        v5a.save_sheet(args.output/'shape_comparison'/(path.stem+'.png'),panels,raw.shape[1])
        for title,y0,y1,x0,x1 in [('upper',40,190,55,405),('middle',200,405,55,405),('lower',425,630,55,405)]:
            v5a.save_sheet(args.output/'pad_crops'/(path.stem+'_'+title+'.png'),[(name,a[y0:y1,x0:x1]) for name,a in panels],700)
        rows = base.region_metrics(raw,{'detail_gray':previous,'reference_gray':result},arrays['region_labels'],regions)
        for row in rows:
            row.update(filename=path.name)
            row['method'] = {'input':'input','detail_gray':'v5a','reference_gray':'u29_focused'}[row['method']]
        metrics.extend(rows)
        checks = source_mark_checks(raw,previous,result,arrays['region_labels'],details['noise_sigma'])
        for record in checks:
            record['filename'] = path.name
        marks.extend(checks)
        methods = {'v5a':previous,'v2':result}
        if rejected is not None:
            methods['rejected_v1'] = rejected
        checks = edge_profile_checks(raw,methods,regions)
        for record in checks:
            record['filename'] = path.name
        edges.extend(checks)
        details.update(filename=path.name,shape=list(raw.shape),source_sha256=source_hash)
        manifest.append(details)
        print(f'  pads={len(regions)}, protected source change={details["protected_source_change_max"]:.1f}',flush=True)
    base.write_csv(args.output/'per_region_metrics.csv',metrics,['region_id','x','y','width','height','method','interior_pixels','interior_mean','interior_std','interior_iqr','input_correlation','clipping_fraction','filename'])
    base.write_csv(args.output/'source_mark_checks.csv',marks,['filename','mark_id','pad_id','x','y','pixels','input','v5a','focused'])
    base.write_csv(args.output/'edge_profile_checks.csv',edges,['filename','region_id','side','fraction','method','source_contrast','source_crossing','output_crossing','displacement'])
    code = [Path(__file__),Path(v5a.__file__),Path(auto.__file__),Path(base.__file__)]
    (args.output/'run.json').write_text(json.dumps(dict(pipeline='u29_focused_v2',input=str(args.input.resolve()),images=manifest,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        validation='Edge profiles are a source-relative geometric proxy, not full contour or defect ground truth.'),indent=2),encoding='utf-8')
    print(args.output.resolve(),flush=True)


if __name__=='__main__':
    main()

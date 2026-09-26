"""Automatic grayscale enhancement, with the earlier component-specific V5 retained as legacy."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import competitor_target_v3 as v3
import competitor_target_v4 as v4
import automatic_v5

OUTPUT = v3.OUTPUT.parent / 'competitor_target_v5'
AUTO_OUTPUT = v3.OUTPUT.parent / 'competitor_target_v5_auto'


def refine_squares(raw, old_labels, regions, noise):
    """Find four consistent outer edge positions near each grid ROI."""
    smooth = v3.blur(raw, 1.)
    labels = np.zeros_like(old_labels)
    refined, decisions = [], []
    for identifier, x, y, w, h in regions:
        margin = max(4, round(max(w, h) * .3))
        x0, y0 = max(0, x-margin), max(0, y-margin)
        x1, y1 = min(raw.shape[1], x+w+margin), min(raw.shape[0], y+h+margin)
        patch = smooth[y0:y1, x0:x1]
        gx = np.gradient(patch, axis=1)
        gy = np.gradient(patch, axis=0)
        # Medians across most of each side reduce influence from isolated cavities.
        xp = np.median(gx[max(0,y-y0+h//5):min(patch.shape[0],y-y0+4*h//5)], axis=0)
        yp = np.median(gy[:,max(0,x-x0+w//5):min(patch.shape[1],x-x0+4*w//5)], axis=1)
        def edge(profile, center, positive):
            a, b = max(1, center-margin), min(len(profile)-1, center+margin+1)
            values = profile[a:b] * (1 if positive else -1)
            index = int(np.argmax(values))
            return a+index, float(values[index])
        left, ls = edge(xp, x-x0, True)
        right, rs = edge(xp, x+w-x0-1, False)
        top, ts = edge(yp, y-y0, True)
        bottom, bs = edge(yp, y+h-y0-1, False)
        bw, bh = right-left+1, bottom-top+1
        accepted = min(ls, rs, ts, bs) > max(.7, noise * .35) and .65*w < bw < 1.4*w and .65*h < bh < 1.4*h
        box = (x0+left, y0+top, bw, bh) if accepted else (x,y,w,h)
        bx, by, bw, bh = map(int, box)
        if np.any(labels[by:by+bh,bx:bx+bw]):
            bx,by,bw,bh = x,y,w,h
            accepted = False
        labels[by:by+bh,bx:bx+bw] = identifier
        refined.append((identifier,bx,by,bw,bh))
        decisions.append(dict(region_id=identifier, refined=bool(accepted), side_strengths=[ls,rs,ts,bs]))
    return labels, refined, decisions


def outer_feather_correction(image, labels, regions):
    """Constant per-pad shifts with transitions outside the pad, preserving its interior differences."""
    estimates = []
    for identifier,x,y,w,h in regions:
        mask = (labels[y:y+h,x:x+w] == identifier).astype(np.uint8)
        core = cv2.erode(mask,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        if core.sum() < 12:
            continue
        values = image[y:y+h,x:x+w][core]
        estimates.append((identifier,x,y,w,h,float(np.percentile(values,70))))
    field = np.zeros_like(image)
    denominator = np.zeros_like(image)
    if not estimates:
        return field
    target = np.median([row[-1] for row in estimates])
    for identifier,x,y,w,h,level in estimates:
        shift = float(np.clip(.65 * (target-level),-12,12))
        m = 5
        x0,y0 = max(0,x-m),max(0,y-m)
        x1,y1 = min(image.shape[1],x+w+m),min(image.shape[0],y+h+m)
        local_labels = labels[y0:y1,x0:x1]
        mask = local_labels == identifier
        dist = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
        weight = np.maximum(0,1-dist/m)
        weight[(local_labels>0)&~mask] = 0
        field[y0:y1,x0:x1] += shift*weight
        denominator[y0:y1,x0:x1] += weight
    # Preserve fade magnitude outside regions; only normalize overlapping weights.
    return field / np.maximum(denominator,1)


def local_gain(image, labels, regions, noise, component):
    gain = np.ones_like(image)
    anchor = np.zeros_like(image)
    if component not in {'J1','J4'}:
        return image.copy(),gain,anchor
    samples = []
    for identifier,x,y,w,h in regions:
        m = max(w,h)//2+4
        x0,y0 = max(0,x-m),max(0,y-m)
        x1,y1 = min(image.shape[1],x+w+m),min(image.shape[0],y+h+m)
        local_labels = labels[y0:y1,x0:x1]
        pad = local_labels == identifier
        background = cv2.dilate((local_labels>0).astype(np.uint8),np.ones((5,5),np.uint8))==0
        if pad.sum()<12 or background.sum()<20:
            continue
        patch = image[y0:y1,x0:x1]
        bg = float(np.median(patch[background]))
        contrast = float(np.percentile(patch[pad],80)-bg)
        if contrast > max(3.,noise*3):
            samples.append((x+w/2,y+h/2,bg,contrast))
    if len(samples)<20:
        return image.copy(),gain,anchor
    # Estimate slowly varying fields at coarse resolution separately for each panel.
    for panel in range(2):
        x0,x1 = panel*(image.shape[1]//2), image.shape[1] if panel else image.shape[1]//2
        selected = [p for p in samples if x0<=p[0]<x1]
        if len(selected)<10:
            continue
        target = np.percentile([p[3] for p in selected],60)
        num, bgnum, weights = [np.zeros((48,32),np.float32) for _ in range(3)]
        for x,y,bg,contrast in selected:
            ix = min(31,int((x-x0)/(x1-x0)*32))
            iy = min(47,int(y/image.shape[0]*48))
            num[iy,ix] += np.clip(target/contrast,1.,1.3)
            bgnum[iy,ix] += bg
            weights[iy,ix] += 1
        density = v3.blur(weights,3.)
        valid = density>.015
        g = np.where(valid,v3.blur(num,3.)/np.maximum(density,1e-6),1.)
        b = np.where(valid,v3.blur(bgnum,3.)/np.maximum(density,1e-6),np.median(image[:,x0:x1]))
        gain[:,x0:x1] = cv2.resize(g,(x1-x0,image.shape[0]),interpolation=cv2.INTER_LINEAR)
        anchor[:,x0:x1] = cv2.resize(b,(x1-x0,image.shape[0]),interpolation=cv2.INTER_LINEAR)
    gain = np.clip(v3.blur(gain,(0,3)),1,1.3)
    delta = np.clip((gain-1)*(image-anchor),-6,6)
    return image+delta,gain,anchor


def process_legacy(raw,component):
    flattened,_ = v3.broad_correction(raw)
    labels,regions,_,noise = v3.propose_regions(flattened,component)
    decisions=[]
    if component=='U29':
        labels,regions,grid=v4.square_grid(flattened,labels,regions,noise)
        if grid['used']:
            labels,regions,decisions=refine_squares(flattened,labels,regions,noise)
    balanced=flattened+v4.masked_bands(flattened,labels,component)
    if component=='U29':
        adjusted=balanced+outer_feather_correction(balanced,labels,regions)
    else:
        adjusted=balanced+v3.pad_correction(balanced,labels,regions)
    adjusted,gain,_=local_gain(adjusted,labels,regions,noise,component)
    cleaned,weight=v4.clean_background(adjusted,labels,noise)
    low,high=np.percentile(flattened,(5,99.5))
    return dict(reference_gray=v3.tone(cleaned,low,high),detail_gray=v3.tone(balanced,low,high),
                region_mask=(labels>0).astype(np.float32)*255,gain_field=(gain-1)/.3*255,
                background_smoothing=weight/.85*255),labels,regions,cleaned,dict(
                    boundary_decisions=decisions,max_gain=float(gain.max()),
                    estimated_healthy_pixels=False, note='70th percentile pad levels are provisional, not verified healthy tissue.')


def process(raw, component=None):
    return automatic_v5.process(raw, component)


def sheet(path,raw,previous,outputs,regions,previous_title='V4 baseline',mode='automatic'):
    overlay=cv2.cvtColor(v3.u8(raw),cv2.COLOR_GRAY2RGB)
    for _,x,y,w,h in regions:
        cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),(40,220,100),1)
    maximum = '1.6' if mode == 'automatic' else '1.3'
    panels=[('Input',raw),(previous_title,previous),(f'V5 {mode}',outputs['reference_gray']),
            ('Unreviewed region proposals',overlay),(f'Gain: black = 1, white = {maximum}',outputs['gain_field']),('Smoothing support',outputs['background_smoothing'])]
    width=560
    height=round(raw.shape[0]/raw.shape[1]*width)
    out=Image.new('RGB',(width*3,(height+28)*2),(24,24,24))
    draw=ImageDraw.Draw(out)
    for i,(title,a) in enumerate(panels):
        x,y=i%3*width,i//3*(height+28)
        out.paste(Image.fromarray(v3.u8(a)).convert('RGB').resize((width,height),Image.Resampling.LANCZOS),(x,y+28))
        draw.text((x+7,y+7),title,fill='white')
    out.save(path)


def input_output_sheet(path, raw, result, filename):
    width = min(1000, raw.shape[1])
    height = round(raw.shape[0] / raw.shape[1] * width)
    canvas = Image.new('RGB', (width * 2, height + 32), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (title, array) in enumerate(((f'Input: {filename}', raw), ('V5 automatic: reference_gray', result))):
        panel = Image.fromarray(v3.u8(array)).resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(panel, (i * width, 32))
        draw.text((i * width + 8, 9), title, fill='white')
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=v3.INPUT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mode',choices=['automatic','legacy'],default='automatic')
    parser.add_argument('--files',nargs='+')
    args=parser.parse_args()
    args.output = args.output or (AUTO_OUTPUT if args.mode == 'automatic' else OUTPUT)
    paths=sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'})
    if args.files:
        paths=[p for p in paths if p.name in args.files]
        if set(args.files)-{p.name for p in paths}: raise ValueError('Requested image missing')
    if not paths: raise ValueError('No input images')
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError('Duplicate image stems would overwrite outputs.')
    args.output.mkdir(parents=True,exist_ok=True)
    rows,manifest=[],[]
    for i,path in enumerate(paths,1):
        print(f'[{i}/{len(paths)}] {path.name}',flush=True)
        with Image.open(path) as source:
            if source.mode != 'L':
                raise ValueError(f'{path.name}: expected 8-bit grayscale, got {source.mode}; convert explicitly first.')
            raw=np.asarray(source,dtype=np.float32)
        component=path.stem.split('_')[0]
        processor = process if args.mode == 'automatic' else process_legacy
        outputs,labels,regions,numeric,diagnostic=processor(raw,component)
        prior=v4.OUTPUT/'reference_gray'/(path.stem+'.png')
        matched = prior.exists() and args.input.resolve() == v3.INPUT.resolve()
        previous=v3.read(prior) if matched else raw.copy()
        previous_title='V4 baseline (filename matched)' if matched else 'Input (no matched V4 baseline)'
        if previous.shape != raw.shape:
            previous=raw.copy()
            previous_title='Input (V4 shape mismatch)'
        for name,a in outputs.items():
            (args.output/name).mkdir(exist_ok=True)
            Image.fromarray(v3.u8(a)).save(args.output/name/(path.stem+'.png'))
        (args.output/'numeric').mkdir(exist_ok=True)
        np.savez_compressed(args.output/'numeric'/(path.stem+'.npz'),corrected=numeric,region_labels=labels)
        (args.output/'comparison_sheets').mkdir(exist_ok=True)
        sheet(args.output/'comparison_sheets'/(path.stem+'.jpg'),raw,previous,outputs,regions,previous_title,args.mode)
        if args.mode == 'automatic':
            (args.output/'input_output').mkdir(exist_ok=True)
            input_output_sheet(args.output/'input_output'/(path.stem+'.png'),raw,outputs['reference_gray'],path.name)
        records=v3.region_metrics(raw,dict(detail_gray=previous,reference_gray=outputs['reference_gray']),labels,regions)
        for r in records:
            r.update(filename=path.name,component=component)
            r['method']={'input':'input','detail_gray':'v4' if previous_title.startswith('V4') else 'input_fallback','reference_gray':'v5'}[r['method']]
        rows.extend(records)
        diagnostic.update(filename=path.name,regions=len(regions),shape=list(raw.shape),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest.append(diagnostic)
    fields=['region_id','x','y','width','height','method','interior_pixels','interior_mean','interior_std','interior_iqr','input_correlation','clipping_fraction','filename','component']
    v3.write_csv(args.output/'per_region_metrics.csv',rows,fields)
    source_paths = [Path(__file__), Path(automatic_v5.__file__), Path(v3.__file__), Path(v4.__file__)]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (args.output/'run.json').write_text(json.dumps(dict(mode=args.mode,input=str(args.input.resolve()),source_hashes=source_hashes,images=manifest,validation='unreviewed experimental outputs'),indent=2),encoding='utf-8')
    print(args.output.resolve(),flush=True)


if __name__=='__main__': main()

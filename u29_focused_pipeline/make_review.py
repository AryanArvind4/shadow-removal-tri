"""Verify a completed U29 run and prepare clearly labeled meeting comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import u29_focused as u


def panel_sheet(path, panels, width=760):
    height = round(672*width/760)
    sheet = Image.new('RGB',(width*len(panels),height+48),(24,24,24))
    draw = ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(panels):
        sheet.paste(im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS),(i*width,48))
        draw.text((i*width+8,10),name,fill='white')
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=u.OUTPUT)
    parser.add_argument('--reference',type=Path)
    args = parser.parse_args()
    folder = args.results
    manifest = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    source_dir = Path(manifest['input'])
    rows = list(csv.DictReader((folder/'per_region_metrics.csv').open(newline='',encoding='utf-8')))
    marks = list(csv.DictReader((folder/'source_mark_checks.csv').open(newline='',encoding='utf-8')))
    review = folder/'meeting_review'
    review.mkdir(exist_ok=True)
    per_image = []
    for entry in manifest['images']:
        name = entry['filename']
        stem = Path(name).stem
        source = source_dir/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['source_sha256'],name
        with np.load(folder/'numeric'/(stem+'.npz')) as data:
            for key in data.files:
                assert data[key].shape==tuple(entry['shape']) and np.isfinite(data[key]).all(),(name,key)
            result = data['enhanced']
            assert result.min()>=0 and result.max()<=255,name
            guard = data['edge_guard']
            labels = data['region_labels']
            assert guard.dtype==bool,name
            np.testing.assert_array_equal(data['background_delta'][guard],0.)
            np.testing.assert_array_equal(result[guard],data['detail'][guard])
            with Image.open(folder/'reference_gray'/(stem+'.png')) as image:
                np.testing.assert_array_equal(np.asarray(image),u.base.u8(result))
            with Image.open(folder/'edge_guard'/(stem+'.png')) as image:
                np.testing.assert_array_equal(np.asarray(image),guard.astype(np.uint8)*255)
        stats = dict(filename=name,pads=entry['layout']['accepted'])
        for method in ('input','v5a','u29_focused'):
            selected = [r for r in rows if r['filename']==name and r['method']==method]
            means = [float(r['interior_mean']) for r in selected]
            stats[method] = dict(across_pad_mean_std=float(np.std(means)),
                mean_within_pad_std=float(np.mean([float(r['interior_std']) for r in selected])),
                measured_pads=len(selected))
        selected = [r for r in marks if r['filename']==name]
        stats['source_marks'] = len(selected)
        stats['source_marks_positive_after'] = sum(float(r['focused'])>0 for r in selected)
        for method in ('input','v5a','focused'):
            stats['median_mark_contrast_'+method] = float(np.median([float(r[method]) for r in selected])) if selected else None
        per_image.append(stats)
    # Same native image geometry across all comparison panels.
    contact = Image.new('RGB',(4*380,((len(manifest['images'])+3)//4)*368),(24,24,24))
    draw = ImageDraw.Draw(contact)
    for i,entry in enumerate(manifest['images']):
        im = Image.open(folder/'reference_gray'/(Path(entry['filename']).stem+'.png'))
        x,y = (i%4)*380,(i//4)*368
        contact.paste(im.convert('RGB').resize((380,336),Image.Resampling.LANCZOS),(x,y+32))
        draw.text((x+8,y+8),entry['filename']+' | U29 V2',fill='white')
    contact.save(review/'all_u29_outputs.png')
    reference_record = None
    if args.reference:
        with Image.open(args.reference) as photo:
            photo = ImageOps.exif_transpose(photo).convert('RGB')
        # These coordinates crop only the photographed display. They are not
        # processing ROIs or a registration between source and competitor pads.
        corners = np.array([[47,438],[1197,378],[1281,1406],[94,1470]],np.float32)
        corners *= np.array([photo.width/1368.,photo.height/1824.],np.float32)
        transform = cv2.getPerspectiveTransform(corners,np.array([[0,0],[759,0],[759,671],[0,671]],np.float32))
        target = Image.fromarray(cv2.warpPerspective(np.asarray(photo),transform,(760,672)))
        target.save(review/'competitor_photo_rectified.png')
        for stem in ('U29_029','U29_034'):
            if (folder/'reference_gray'/(stem+'.png')).exists():
                panel_sheet(review/(stem+'_vs_competitor.png'),[
                    ('Our input: '+stem,Image.open(source_dir/(stem+'.jpg'))),
                    ('U29 V2: '+stem,Image.open(folder/'reference_gray'/(stem+'.png'))),
                    ('Competitor photo: unmatched acquisition\nDisplay crop rectified; visual target only',target)])
        reference_record = dict(path=str(args.reference.resolve()),sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                                role='Visual target only; no pixel or defect matching score.')
    summary = dict(files=len(per_image),pads=sum(s['pads'] for s in per_image),per_image=per_image,
                   reference=reference_record,validation='Source-selected dark responses are unreviewed, not ground-truth defects.')
    for method in ('input','v5a','u29_focused'):
        summary[method] = {key:float(np.mean([s[method][key] for s in per_image])) for key in ('across_pad_mean_std','mean_within_pad_std')}
    summary['source_mark_count'] = sum(s['source_marks'] for s in per_image)
    summary['source_marks_positive_after'] = sum(s['source_marks_positive_after'] for s in per_image)
    summary['median_mark_contrast'] = {method:float(np.median([float(r[method]) for r in marks])) if marks else None for method in ('input','v5a','focused')}
    edges = list(csv.DictReader((folder/'edge_profile_checks.csv').open(newline='',encoding='utf-8')))
    summary['edge_displacement_pixels'] = {}
    for method in sorted({r['method'] for r in edges}):
        selected = [r for r in edges if r['method']==method]
        valid = [float(r['displacement']) for r in selected if r['displacement']]
        summary['edge_displacement_pixels'][method] = dict(profiles=len(selected),missing=len(selected)-len(valid),
            median=float(np.median(valid)),p95=float(np.percentile(valid,95)),p99=float(np.percentile(valid,99)))
    for name in [r['filename'] for r in manifest['images']]:
        valid = [float(r['displacement']) for r in edges if r['filename']==name and r['method']=='v2' and r['displacement']]
        assert valid and float(np.percentile(valid,95))<.35, f'Edge displacement review required: {name}'
    (review/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    old,new = summary['v5a'],summary['u29_focused']
    contrast_old = f"{summary['median_mark_contrast']['v5a']:.2f}" if marks else 'Not available'
    contrast_new = f"{summary['median_mark_contrast']['focused']:.2f}" if marks else 'Not available'
    edge_old = summary['edge_displacement_pixels'].get('rejected_v1',{}).get('p95',float('nan'))
    edge_new = summary['edge_displacement_pixels']['v2']['p95']
    report = f'''# U29 focused V2: shape-correction review

Processed {summary['files']} U29 images with one parameter set; {summary['pads']} automatically supported pad regions.

The previous U29 V1 output was rejected because separate pad/background rendering introduced visible boundary distortions. V2 removes that compositing, the per-pad rendered planes, and the within-pad row correction. Smooth brightness and gain fields now act across the whole image, followed by one common tone curve. Background filtering is exactly zero within each pad and a seven-pixel surrounding guard.

## Where to look

- `../input_output/`: original and focused result, matching filename.
- `../comparison_sheets/`: original, V5a, focused result.
- `../shape_comparison/`: original, rejected U29 V1, corrected U29 V2.
- `../reference_gray/`: final native-resolution images for slides.
- `../pad_crops/`: enlarged upper, middle and lower comparisons.
- `../detail_gray/`: companion without background filtering; identical to reference within protected pads/edges.
- `all_u29_outputs.png`: overview of all eight results.
- `U29_034_vs_competitor.png`: photographed target beside our source and output, when a reference was provided.

## Measurements

The 95th-percentile edge displacement relative to the source changed from **{edge_old:.3f} pixels** in rejected V1 to **{edge_new:.3f} pixels** in V2. These are half-height crossings of source-selected edge profiles, not full contour ground truth. Weak edges and some corners are outside the measurement's coverage. The seven-pixel guard and numeric equality checks additionally verify that pad pixels receive no denoising or dot suppression before the photometric transform.

Measured on identical automatically selected pad interiors. Gray values use a 0-255 scale.

| Measurement | V5a | U29 focused |
|---|---:|---:|
| Across-pad SD of interior means, averaged over images | {old['across_pad_mean_std']:.2f} | {new['across_pad_mean_std']:.2f} |
| Mean within-pad SD, including dark features | {old['mean_within_pad_std']:.2f} | {new['mean_within_pad_std']:.2f} |
| Median local contrast at source-selected dark marks | {contrast_old} | {contrast_new} |

{summary['source_marks_positive_after']} of {summary['source_mark_count']} automatically selected source dark responses still have positive dark-to-surround contrast after enhancement. These may include shadow and texture; this is not defect recall.

Across-pad SD describes similarity of mean brightness. V2 gives priority to source geometry and does not force every pad to a common brightness. Within-pad SD includes defects, so lowering it is not automatically better. These are display measurements, not proof of accuracy or healthy-only uniformity.

## Suggested presentation wording

We found that separate pad rendering changed the visible pad outlines. We replaced it with smooth image-wide brightness and contrast correction, and checked edge locations against the source. U29 now uses this shape-corrected V2 result; J1, J4 and U30 still use V5a. The competitor photograph is our visual target; identical acquisition and confirmed-defect equivalence have not been established.

## Remaining gap

Upper-row shadows still cross pad outlines in the source. Those outlines have not been reconstructed as ideal squares. Background texture is more visible than in rejected V1 because suppressing it with a tightly fitted rendering mask caused the original defect. Fine marks in the competitor image remain more clearly separated. These outputs are a correction of the added deformation, not a completed competitor-quality shadow-removal result.
'''
    (review/'MEETING_NOTES.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_image','reference')},indent=2))


if __name__=='__main__':
    main()

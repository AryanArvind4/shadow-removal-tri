"""Run the U30 experiment and generate matched full-image and focus comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

import focused_pipeline as focused
from focused_pipeline import auto, base
from prepare_baseline import ROOT, SOURCE, digest


def sheet(path, panels, zoom=1):
    h, w = panels[0][1].shape
    canvas = Image.new('RGB', (len(panels)*w*zoom, h*zoom+32), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (title, array) in enumerate(panels):
        panel = Image.fromarray(base.u8(array)).convert('RGB')
        if zoom != 1:
            panel = panel.resize((w*zoom, h*zoom), Image.Resampling.NEAREST)
        canvas.paste(panel, (i*w*zoom, 32))
        draw.text((i*w*zoom+6, 10), title, fill='white')
    canvas.save(path)


def review_boxes(name):
    # Review coordinates only: these are never inputs to the enhancement.
    if int(Path(name).stem.split('_')[1]) >= 73:
        return dict(plate_top_left=(70, 200, 330, 320),
                    plate_top_right=(450, 200, 705, 320),
                    plate_bottom_left=(70, 330, 330, 525),
                    plate_bottom_right=(450, 330, 705, 525))
    return dict(row7_col3_neighborhood=(140, 455, 320, 620),
                row7_col3_candidate=(205, 500, 270, 565))


def measure(image):
    smooth = base.blur(image, 1.2)
    band = smooth - base.blur(image, 12.)
    fine = image - smooth
    p05, p95 = np.percentile(image, [5, 95])
    return dict(p05_p95_span=float(p95-p05), midscale_rms=float(np.sqrt(np.mean(band**2))),
                fine_residual_rms=float(np.sqrt(np.mean(fine**2))),
                clipping_fraction=float(np.mean((image <= 1) | (image >= 254))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'result/u30_focus_v1')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Refusing overwrite: ' + str(output))
    manifest = json.loads((ROOT / 'input_manifest.json').read_text())
    baseline = ROOT / 'result/v5a_baseline'
    for name, expected in manifest['baseline_source_hashes'].items():
        if digest(ROOT.parent / name) != expected:
            raise RuntimeError('Baseline source changed: ' + name)
    for entry in manifest['images']:
        name = entry['filename']
        if digest(ROOT/'input'/name) != entry['sha256'] or digest(SOURCE/name) != entry['sha256']:
            raise RuntimeError('Input changed: ' + name)
        if not (baseline/'reference_gray'/(Path(name).stem+'.png')).exists():
            raise FileNotFoundError('Generate V5a baseline first.')
    for folder in ('reference_gray', 'input_output', 'comparison_sheets', 'focus_crops',
                   'review_locations', 'numeric'):
        (output/folder).mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    records, metrics = [], []
    for entry in manifest['images']:
        name = entry['filename']
        stem = Path(name).stem
        raw = base.read(ROOT/'input'/name)
        if raw.shape != (700, 768):
            raise ValueError('Review crop coordinates require the selected native image size.')
        previous = base.read(baseline/'reference_gray'/(stem+'.png'))
        print('Processing ' + name, flush=True)
        result, fields, info = focused.process(raw)
        result = base.u8(result).astype(np.float32)
        if not np.isfinite(result).all() or np.ptp(result) == 0 or result.shape != raw.shape:
            raise RuntimeError('Invalid output: ' + name)
        Image.fromarray(base.u8(result)).save(output/'reference_gray'/(stem+'.png'))
        panels = [('Input', raw), ('V5a baseline', previous), ('U30 focus v1 (experimental)', result)]
        sheet(output/'comparison_sheets'/(stem+'.png'), panels)
        sheet(output/'input_output'/(stem+'.png'), [panels[0], panels[2]])
        overlay = Image.fromarray(base.u8(raw)).convert('RGB')
        draw = ImageDraw.Draw(overlay)
        boxes = review_boxes(name)
        for label, (x0, y0, x1, y1) in boxes.items():
            crops = [(title, image[y0:y1,x0:x1]) for title, image in panels]
            sheet(output/'focus_crops'/(stem+'_'+label+'.png'), crops, zoom=2)
            draw.rectangle((x0, y0, x1-1, y1-1), outline=(255, 200, 60), width=2)
            draw.text((x0, max(0,y0-13)), label, fill=(255,200,60))
            for method, patch in crops:
                metrics.append(dict(filename=name, review_region=label, method=method, **measure(patch)))
        overlay.save(output/'review_locations'/(stem+'.png'))
        np.savez_compressed(output/'numeric'/(stem+'.npz'), **fields)
        records.append(dict(filename=name, sha256=entry['sha256'], review_boxes=boxes, diagnostics=info))
        if digest(SOURCE/name) != entry['sha256']:
            raise RuntimeError('Original changed: ' + name)
    with (output/'review_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    sources = [Path(focused.__file__), Path(__file__), Path(auto.__file__), Path(base.__file__)]
    run = dict(pipeline='u30_focus_v1', status='complete', seconds=perf_counter()-start,
               source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               coordinate_convention='Provisional row 7 from top, column 3 from left, one-based.',
               metrics_note='Fixed review boxes, not confirmed healthy/defect ROIs. '
                            'Fine residual includes detail, not just noise. No accuracy or CNR claims.',
               images=records)
    (output/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()

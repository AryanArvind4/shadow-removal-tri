"""Reproducible, non-overwriting V5c review export against saved V5b results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

import v5c


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(study, manifest):
    for name, expected in manifest['frozen_code'].items():
        if digest(study / name) != expected:
            raise RuntimeError('Frozen baseline changed: ' + name)
    for name, expected in manifest['retained_files'].items():
        if digest(study.parent / name) != expected:
            raise RuntimeError('Retained V5b file changed: ' + name)


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    with Image.open(path) as saved:
        np.testing.assert_array_equal(np.array(saved), array)


def save_review_crops(folder, stem, raw, old, new, details):
    folder.mkdir(parents=True, exist_ok=True)
    crops = []
    records = details['selection']['regions']
    if details['selection']['method'].endswith('rectangular_proposals'):
        # Export every detected pad, not just the visually favorable examples.
        for index, record in enumerate(records, 1):
            x, y, w, h = record['box']
            crops.append((f'pad_{index:02d}', (max(0,x-5), max(0,y-5), min(raw.shape[1],x+w+5), min(raw.shape[0],y+h+5))))
    else:
        h, w = raw.shape
        crops = [('center', (w//4,h//4,3*w//4,3*h//4)),
                 ('lower_background', (0,4*h//5,w,h))]
    metadata = []
    for name, (x0,y0,x1,y1) in crops:
        sl = (slice(y0,y1), slice(x0,x1))
        filename = stem + '_' + name + '.png'
        v5c.v5b.v5a.save_sheet(folder / filename,
            [('Input',raw[sl]),('Saved V5b',old[sl]),('V5c candidate',new[sl])],
            220 if name.startswith('pad') else min(760,x1-x0))
        metadata.append(dict(file=filename, source_xyxy=[x0,y0,x1,y1],
                             note='Display crop resized equally using Lanczos; coordinates are native pixels.'))
    (folder / (stem + '_locations.json')).write_text(json.dumps(metadata, indent=2))


def main():
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=study/'results_v5c')
    parser.add_argument('--dataset', nargs='+')
    args = parser.parse_args()
    manifest = json.loads((study/'study_manifest.json').read_text())
    known = {e['dataset'] for e in manifest['inputs']}
    if args.dataset and set(args.dataset)-known:
        raise ValueError('Unknown dataset')
    entries = [e for e in manifest['inputs'] if not args.dataset or e['dataset'] in args.dataset]
    verify_snapshot(study, manifest)
    for entry in entries:
        if digest(study/entry['path']) != entry['sha256']:
            raise RuntimeError('Input changed: ' + entry['path'])
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(args.output))
    args.output.mkdir(parents=True)
    report = dict(pipeline='v5c', meeting='2026-11-19', status='running', images=[],
                  code_sha256={p.name:digest(p) for p in (Path(__file__),Path(v5c.__file__))},
                  settings=v5c.asdict(v5c.Settings()),
                  note='Exploratory display comparison. No labeled defect accuracy or competitor equivalence established.')
    run_path = args.output/'run.json'
    run_path.write_text(json.dumps(report, indent=2))
    rows = []
    began = perf_counter()
    try:
        for entry in entries:
            path = study/entry['path']
            with Image.open(path) as image:
                if image.mode not in ('L','RGB'):
                    raise ValueError('Expected 8-bit L or RGB: '+str(path))
                source = np.array(image)
                raw = np.asarray(image.convert('L'), np.float32)
            protected = None
            if source.ndim == 3:
                chroma = np.ptp(source.astype(np.int16), axis=2) > 18
                protected = cv2.dilate(chroma.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
            start = perf_counter()
            baseline = v5c.v5b.v5a.process(raw)
            baseline_seconds = perf_counter()-start
            start = perf_counter()
            result = v5c.process(raw, baseline=baseline, protected=protected)
            refinement_seconds = perf_counter()-start
            gray = v5c.v5b.v5a.base.u8(result['output'])
            final = v5c.render_original_mode(source, result)
            with Image.open(study.parent/entry['v5b_reference']) as image:
                old = np.array(image.convert('L'))
            recomputed = v5c.v5b.v5a.base.u8(result['v5b'])
            np.testing.assert_array_equal(old, recomputed, err_msg='V5b baseline mismatch')
            outside = ~result['support']
            np.testing.assert_array_equal(final[outside], source[outside])
            np.testing.assert_array_equal(gray[outside], raw.astype(np.uint8)[outside])
            assert np.all(result['output'] >= raw), 'Added darkening'
            folder = args.output/entry['dataset']
            arrays = dict(enhanced=final,reference_gray=gray,
                          component_mask=result['support'].astype(np.uint8)*255,
                          blend_weight=v5c.v5b.v5a.base.u8(result['alpha']*255))
            overlay = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            contours, _ = cv2.findContours(result['support'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(40,220,100),1)
            arrays['selection_overlay'] = overlay
            for name, array in arrays.items():
                save_png(folder/name/(path.stem+'.png'), array)
            for name in ('comparison_sheets','input_output','numeric','diagnostics','review_crops'):
                (folder/name).mkdir(parents=True,exist_ok=True)
            width = min(760,raw.shape[1])
            v5c.v5b.v5a.save_sheet(folder/'comparison_sheets'/(path.stem+'.png'),
                [('Input',raw),('Saved automatic V5b',old),('V5c candidate',gray)],width)
            v5c.v5b.v5a.save_sheet(folder/'input_output'/(path.stem+'.png'),
                [('Input',raw),('V5c candidate',gray)],width)
            if 'targeted_reference' in entry:
                with Image.open(study.parent/entry['targeted_reference']) as image:
                    targeted = np.array(image.convert('L'))
                (folder/'targeted_comparisons').mkdir(exist_ok=True)
                v5c.v5b.v5a.save_sheet(folder/'targeted_comparisons'/(path.stem+'.png'),
                    [('Input',raw),('V5b manual targeted fix',targeted),('Automatic V5c candidate',gray)],width)
            np.savez_compressed(folder/'numeric'/(path.stem+'.npz'),support=result['support'],alpha=result['alpha'])
            save_review_crops(folder/'review_crops',path.stem,raw,old,gray,result['details'])
            delta = gray.astype(float)-raw
            record = dict(dataset=entry['dataset'],filename=path.name,sha256=entry['sha256'],
                baseline_seconds=baseline_seconds,refinement_with_v5b_comparison_seconds=refinement_seconds,
                processing_seconds=baseline_seconds+refinement_seconds,
                saved_v5b_exact=True,background_max_abs_change=0,
                changed_fraction=float(np.mean(delta != 0)),
                mean_abs_change=float(np.mean(np.abs(delta))),
                output_clipped_fraction=float(np.mean((gray==0)|(gray==255))),
                input_clipped_fraction=float(np.mean((raw==0)|(raw==255))),**result['details'])
            report['images'].append(record)
            (folder/'diagnostics'/(path.stem+'.json')).write_text(json.dumps(record,indent=2))
            rows.append({k:record[k] for k in ('dataset','filename','processing_seconds','saved_v5b_exact',
                'background_max_abs_change','support_fraction','changed_fraction','mean_abs_change',
                'input_clipped_fraction','output_clipped_fraction')})
            run_path.write_text(json.dumps(report,indent=2))
            print(f"{entry['dataset']}/{path.name}: {record['selection']['accepted']}/{record['selection']['proposed']} accepted; outside mask exact; baseline reproduced",flush=True)
        verify_snapshot(study,manifest)
        for entry in entries:
            if digest(study/entry['path']) != entry['sha256']:
                raise RuntimeError('Input modified')
        with (args.output/'metrics.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report.update(status='complete',elapsed_including_exports_seconds=perf_counter()-began,
                      original_v5b_snapshot_unchanged=True,inputs_unchanged=True)
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        run_path.write_text(json.dumps(report,indent=2))
    print(f'Complete: {len(entries)} images -> {args.output.resolve()}',flush=True)


if __name__ == '__main__':
    main()

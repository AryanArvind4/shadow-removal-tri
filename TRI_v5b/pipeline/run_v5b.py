"""Run V5b on explicit original inputs; export comparisons and numeric diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

import v5b


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', help='Dataset keys recorded in study_manifest.json')
    parser.add_argument('--files', nargs='+', help='Optional original basenames to process')
    parser.add_argument('--output', type=Path, default=study/'results')
    args = parser.parse_args()
    manifest = json.loads((study/'study_manifest.json').read_text())
    frozen = Path(__file__).parent/'frozen_v5a'
    for name, expected in manifest['frozen_v5a']['source_hashes'].items():
        if digest(frozen/name) != expected:
            raise RuntimeError('Frozen V5a code changed: '+name)
    datasets = args.dataset or list(manifest['datasets'])
    if set(datasets)-set(manifest['datasets']):
        raise ValueError('Unknown dataset selection')
    selected = [(dataset, study/name) for dataset in datasets for name in manifest['datasets'][dataset]
                if not args.files or Path(name).name in args.files]
    if not selected or (args.files and set(args.files)-{p.name for _,p in selected}):
        raise ValueError('Missing selected input')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite results: '+str(args.output))
    args.output.mkdir(parents=True)
    started = perf_counter()
    report = dict(pipeline='v5b', status='running', images=[], source_hashes={
        p.name: digest(p) for p in (Path(__file__), Path(v5b.__file__))},
        frozen_v5a=manifest['frozen_v5a'],
        note='Component-only experimental display. Background outside supports equals source; no defect validation.')
    run_path = args.output/'run.json'
    run_path.write_text(json.dumps(report, indent=2))
    for dataset, path in selected:
        start = perf_counter()
        expected = manifest['inputs'][str(path.relative_to(study))] if str(path.relative_to(study)) in manifest['inputs'] else manifest['inputs'][path.relative_to(study).as_posix()]
        if digest(path) != expected:
            raise RuntimeError('Input changed: '+str(path))
        with Image.open(path) as image:
            if image.mode not in ('L', 'RGB'):
                raise ValueError('Expected 8-bit L or RGB source: '+path.name)
            source = np.array(image)
            raw = np.asarray(image.convert('L'), np.float32)
        protected = None
        if source.ndim == 3:
            chroma = np.ptp(source.astype(np.int16), axis=2) > 18
            protected = cv2.dilate(chroma.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
        result = v5b.process(raw, protected=protected)
        gray = v5b.v5a.base.u8(result['output'])
        final = v5b.render_original_mode(source, result)
        background = ~result['support']
        if not np.array_equal(final[background], source[background]):
            raise RuntimeError('Background changed')
        output = args.output/dataset
        arrays = dict(enhanced=final, reference_gray=gray, v5a_reference=v5b.v5a.base.u8(result['v5a']),
                      component_mask=result['support'].astype(np.uint8)*255,
                      blend_weight=v5b.v5a.base.u8(result['alpha']*255))
        overlay = cv2.cvtColor(v5b.v5a.base.u8(raw), cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(result['support'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (40,220,100), 1)
        arrays['selection_overlay'] = overlay
        for folder, array in arrays.items():
            (output/folder).mkdir(parents=True, exist_ok=True)
            destination = output/folder/(path.stem+'.png')
            Image.fromarray(array).save(destination)
            with Image.open(destination) as saved:
                if not np.array_equal(np.array(saved), array):
                    raise RuntimeError('Output verification failed')
        for folder in ('input_output', 'comparison_sheets', 'numeric'):
            (output/folder).mkdir(parents=True, exist_ok=True)
        v5b.v5a.save_sheet(output/'input_output'/(path.stem+'.png'),
            [('Input (grayscale)',raw),('V5b: component enhancement',gray)], min(1000,raw.shape[1]))
        v5b.v5a.save_sheet(output/'comparison_sheets'/(path.stem+'.png'),
            [('Input (grayscale)',raw),('V5a',result['v5a']),('V5b',gray)], min(768,raw.shape[1]))
        np.savez_compressed(output/'numeric'/(path.stem+'.npz'), support=result['support'],
                            alpha=result['alpha'], boundary_restore=result['halo'])
        if digest(path) != expected:
            raise RuntimeError('Original was modified')
        record = dict(filename=path.name, dataset=dataset, source_sha256=expected, shape=list(raw.shape),
                      output_mode='RGB' if source.ndim == 3 else 'L', seconds=perf_counter()-start,
                      original_unchanged=True, background_exact=True, **result['details'])
        report['images'].append(record)
        run_path.write_text(json.dumps(report, indent=2))
        print(f"{dataset}/{path.name}: {record['selection']['count']} regions, coverage={record['support_fraction']:.3f}, background exact", flush=True)
    report.update(status='complete', total_seconds=perf_counter()-started)
    run_path.write_text(json.dumps(report, indent=2))
    print(f"Complete: {len(selected)} images, {report['total_seconds']:.2f}s, {args.output.resolve()}", flush=True)


if __name__ == '__main__':
    main()

"""Run the frozen V5a pipeline on the two new RGB inspection screenshots."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'TRI_New_Image'
    work = source / 'v5a_run'
    gray_dir = work / 'input_grayscale'
    results = work / 'results'
    names = ['image001.png', 'image002.png']
    baseline = json.loads((root / 'competitor_target_pipeline' /
                           'v5a_source_manifest.json').read_text(encoding='utf-8'))
    for name, expected in baseline['source_hashes'].items():
        if sha256(root / 'competitor_target_pipeline' / name) != expected:
            raise RuntimeError('Selected V5a code has changed: ' + name)
    targets = [source / (Path(name).stem + suffix) for name in names
               for suffix in ('_v5a.png', '_input_output_v5a.png')]
    if work.exists() or any(path.exists() for path in targets):
        raise FileExistsError('Outputs already exist; refusing to overwrite them.')
    records = []
    for name in names:
        path = source / name
        with Image.open(path) as image:
            if image.mode not in ('L', 'RGB'):
                raise ValueError('Unsupported source mode: ' + image.mode)
            records.append(dict(filename=name, original_sha256=sha256(path),
                                original_mode=image.mode, width=image.width, height=image.height))
    gray_dir.mkdir(parents=True)
    for name in names:
        with Image.open(source / name) as image:
            image.convert('L').save(gray_dir / name)
    start = perf_counter()
    subprocess.run([sys.executable, str(root / 'competitor_target_pipeline' /
                    'competitor_target_v5a.py'), '--input', str(gray_dir),
                    '--output', str(results), '--files', *names], check=True)
    elapsed = perf_counter() - start
    run = json.loads((results / 'run.json').read_text())
    for record in records:
        name = record['filename']
        stem = Path(name).stem
        record['original_unchanged'] = sha256(source / name) == record['original_sha256']
        if not record['original_unchanged']:
            raise RuntimeError('Source changed during execution: ' + name)
        for folder, suffix in [('reference_gray', '_v5a.png'),
                               ('input_output', '_input_output_v5a.png')]:
            origin = results / folder / name
            destination = source / (stem + suffix)
            destination.write_bytes(origin.read_bytes())
        with Image.open(source / (stem + '_v5a.png')) as image:
            pixels = np.array(image)
            if image.mode != 'L' or image.size != (record['width'], record['height']):
                raise RuntimeError('Invalid result dimensions or mode')
            record['output_min_max'] = [int(pixels.min()), int(pixels.max())]
        diagnostic = next(item for item in run['images'] if item['filename'] == name)
        record['layout'] = diagnostic['layout']
        record['attenuated_spots'] = len(diagnostic['attenuated_spots'])
    report = dict(pipeline='Unchanged selected V5a',
                  preprocessing='Pillow RGB-to-L conversion; native dimensions preserved. No cropping, overlay removal, or tuning.',
                  limitation='Low-resolution screenshots with crosshairs; annotations also enter processing. Not defect-validated.',
                  execution_seconds_including_subprocess_startup=elapsed, images=records)
    (work / 'source_preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

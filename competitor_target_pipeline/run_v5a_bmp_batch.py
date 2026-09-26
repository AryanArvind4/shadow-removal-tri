"""Lossless BMP adapter for the frozen V5a batch runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'G2G3')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = (args.output or source / 'v5a_results').resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing result folder: ' + str(output))
    files = sorted(p for p in source.iterdir() if p.suffix.lower() == '.bmp')
    if not files or len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError('No BMP images or duplicate stems')
    baseline = json.loads((root / 'v5a_source_manifest.json').read_text(encoding='utf-8'))
    for name, expected in baseline['source_hashes'].items():
        if digest(root / name) != expected:
            raise RuntimeError('Selected V5a code has changed: ' + name)
    records = []
    for path in files:
        with Image.open(path) as image:
            if image.mode != 'L':
                raise ValueError('Expected 8-bit grayscale BMP: ' + path.name)
            records.append(dict(source=path.name, sha256=digest(path),
                                input_png=path.stem + '.png', dimensions=list(image.size)))
    prepared = output / 'prepared_inputs'
    prepared.mkdir(parents=True)
    for path, record in zip(files, records):
        with Image.open(path) as image:
            image.save(prepared / record['input_png'])
            with Image.open(prepared / record['input_png']) as check:
                if not np.array_equal(np.asarray(image), np.asarray(check)):
                    raise RuntimeError('Lossless input conversion failed: ' + path.name)
        record['prepared_sha256'] = digest(prepared / record['input_png'])
    # Persist provenance even if a later processing step fails.
    preparation = dict(status='prepared', input=str(source), images=records,
                       conversion='8-bit grayscale BMP to PNG; identical pixels, no resizing or intensity changes.')
    provenance = output / 'source_preparation.json'
    provenance.write_text(json.dumps(preparation, indent=2), encoding='utf-8')
    start = perf_counter()
    subprocess.run([sys.executable, str(root / 'competitor_target_v5a.py'),
                    '--input', str(prepared), '--output', str(output),
                    '--files', *[r['input_png'] for r in records]], check=True)
    elapsed = perf_counter() - start
    run = json.loads((output / 'run.json').read_text())
    if len(run['images']) != len(records):
        raise RuntimeError('Incomplete processing manifest')
    for path, record in zip(files, records):
        if digest(path) != record['sha256']:
            raise RuntimeError('Source changed: ' + path.name)
        with Image.open(output / 'reference_gray' / record['input_png']) as result:
            pixels = np.asarray(result)
            if result.mode != 'L' or list(result.size) != record['dimensions'] or np.ptp(pixels) == 0:
                raise RuntimeError('Unexpected output format, dimensions, or blank image: ' + path.name)
            record['output_range'] = [int(pixels.min()), int(pixels.max())]
        for folder in ('input_output', 'pad_only_gray', 'selection_overlay', 'comparison_sheets'):
            with Image.open(output / folder / record['input_png']) as result:
                result.verify()
        diagnostic = next(x for x in run['images'] if x['filename'] == record['input_png'])
        record['rectangular_branch'] = diagnostic['layout']['used']
        record['accepted_supports'] = diagnostic['layout']['accepted']
        record['repeated_spots'] = len(diagnostic['attenuated_spots'])
    preparation.update(status='complete', execution_seconds_including_subprocess_startup=elapsed,
                       originals_unchanged=True, native_output_dimensions_verified=True)
    provenance.write_text(json.dumps(preparation, indent=2), encoding='utf-8')
    (output / 'README.txt').write_text(
        'V5a results for G2/G3\n\n'
        'reference_gray: final native-resolution output PNGs.\n'
        'input_output: original grayscale input beside final V5a output.\n'
        'pad_only_gray: output without repeated-dot attenuation.\n'
        'comparison_sheets: input, automatic V5 baseline, and V5a.\n'
        'Other folders contain diagnostic masks, overlays, and numeric arrays.\n'
        'prepared_inputs: lossless PNG copies of the original BMP inputs.\n\n'
        f'{len(records)} images processed using unchanged selected V5a code and defaults.\n'
        'Original BMP files were not modified. No resizing or retuning was applied.\n'
        'Code and input provenance are in run.json and source_preparation.json.\n'
        'These are experimental inspection views, not verified defect classifications.\n',
        encoding='utf-8')
    print(json.dumps(preparation, indent=2), flush=True)


if __name__ == '__main__':
    main()

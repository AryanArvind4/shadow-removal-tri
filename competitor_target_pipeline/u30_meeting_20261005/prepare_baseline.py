"""Copy the seven requested originals and run the unchanged selected V5a."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from PIL import Image


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT.parent
SOURCE = PIPELINE.parent / 'TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset'
NAMES = [f'U30_{i:03d}.jpg' for i in (32, 33, 34, 35, 73, 74, 75)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = ROOT / 'input'
    output = ROOT / 'result/v5a_baseline'
    if inputs.exists() or output.exists():
        raise FileExistsError('Study inputs/results already exist; refusing overwrite.')
    frozen = json.loads((PIPELINE / 'v5a_source_manifest.json').read_text())
    for name, expected in frozen['source_hashes'].items():
        if digest(PIPELINE / name) != expected:
            raise RuntimeError('Selected V5a source changed: ' + name)
    records = []
    for name in NAMES:
        with Image.open(SOURCE / name) as image:
            if image.mode != 'L':
                raise ValueError('Expected grayscale: ' + name)
            records.append(dict(filename=name, sha256=digest(SOURCE / name),
                                dimensions=list(image.size)))
    inputs.mkdir(parents=True)
    for record in records:
        name = record['filename']
        shutil.copy2(SOURCE / name, inputs / name)
        if digest(inputs / name) != record['sha256']:
            raise RuntimeError('Copy verification failed: ' + name)
    manifest = dict(status='copied', source=str(SOURCE), images=records,
                    baseline_source_hashes=frozen['source_hashes'])
    provenance = ROOT / 'input_manifest.json'
    provenance.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    start = perf_counter()
    subprocess.run([sys.executable, str(PIPELINE / 'competitor_target_v5a.py'),
                    '--input', str(inputs), '--output', str(output)], check=True)
    manifest.update(status='baseline_complete', baseline_seconds=perf_counter()-start)
    for record in records:
        name = record['filename']
        if digest(SOURCE / name) != record['sha256']:
            raise RuntimeError('Original changed: ' + name)
        with Image.open(output / 'reference_gray' / (Path(name).stem+'.png')) as image:
            if image.mode != 'L' or list(image.size) != record['dimensions']:
                raise RuntimeError('Invalid baseline output: ' + name)
    provenance.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()

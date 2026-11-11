"""Check local manuscript links and saved V5b invariants; write only thesis evidence."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / 'thesis'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, grayscale=False):
    with Image.open(path) as image:
        return np.array(image.convert('L') if grayscale else image)


def main():
    links, figures, words = [], [], {}
    for document in sorted(THESIS.glob('*.md')):
        content = document.read_text(encoding='utf-8')
        words[document.name] = len(content.split())
        for match in re.finditer(r'(!?)\[[^\]]*\]\(([^)]+)\)', content):
            image, target = match.groups()
            if target.startswith(('https://', 'http://', '#')):
                continue
            path = (document.parent / target.split('#')[0]).resolve()
            if not path.exists():
                raise FileNotFoundError(f'{document.name}: {target}')
            links.append(dict(document=document.name, target=target))
            if image:
                with Image.open(path) as img:
                    img.verify()
                figures.append(dict(document=document.name, path=path.relative_to(ROOT).as_posix(),
                                    sha256=digest(path)))

    study = ROOT / 'TRI_v5b'
    manifest = json.loads((study / 'study_manifest.json').read_text())
    run = json.loads((study / 'results/run.json').read_text())
    checks = []
    for record in run['images']:
        dataset, name = record['dataset'], record['filename']
        source_path = next(study / p for p in manifest['datasets'][dataset] if Path(p).name == name)
        if digest(source_path) != record['source_sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        folder = study / 'results' / dataset
        filename = Path(name).stem + '.png'
        source, output = read(source_path), read(folder / 'enhanced' / filename)
        support = read(folder / 'component_mask' / filename) > 0
        gray = read(folder / 'reference_gray' / filename)
        baseline = read(folder / 'v5a_reference' / filename)
        with np.load(folder / 'numeric' / (Path(name).stem + '.npz')) as arrays:
            core = (arrays['alpha'] == 1) & (arrays['boundary_restore'] == 0)
        if source.shape != output.shape or not np.array_equal(source[~support], output[~support]):
            raise ValueError(f'Background or shape mismatch: {name}')
        if not np.array_equal(gray[core], baseline[core]):
            raise ValueError(f'Core mismatch: {name}')
        checks.append(dict(dataset=dataset, filename=name, original_sha256=record['source_sha256'],
                           output_sha256=digest(folder / 'enhanced' / filename),
                           source_shape_and_background_exact=True, core_equals_v5a=True))
    source_checks = {}
    for name, expected in manifest['frozen_v5a']['source_hashes'].items():
        for folder in [ROOT / 'competitor_target_pipeline', study / 'pipeline/frozen_v5a']:
            path = folder / name
            source_checks[path.relative_to(ROOT).as_posix()] = digest(path) == expected
    for name, expected in run['source_hashes'].items():
        path = study / 'pipeline' / name
        source_checks[path.relative_to(ROOT).as_posix()] = digest(path) == expected
    if not all(source_checks.values()):
        raise ValueError('Selected source code changed')
    logs = [p.read_text(encoding='utf-8') for p in (THESIS / 'evidence').glob('tests_*.txt')]
    tests = sum(int(match.group(1)) for log in logs
                if (match := re.search(r'Ran (\d+) tests? in ', log)))
    report = dict(status='pass', local_links_checked=len(links), figure_count=len(figures),
                  figures=figures, document_words=words, saved_v5b_checks=checks,
                  source_hash_checks=source_checks, recorded_test_count=tests,
                  limitations='No human defect validation or true-background labeling. Local-link checks do not verify external URLs.')
    (THESIS / 'evidence/draft_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'local_links_checked', 'figure_count',
                                           'document_words', 'recorded_test_count']}, indent=2))
    print(f'Saved V5b invariant checks: {len(checks)} passed; source hashes match.')


if __name__ == '__main__':
    main()

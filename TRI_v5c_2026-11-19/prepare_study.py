"""Copy the explicit V5b study inputs and baseline code without moving originals."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study = Path(__file__).resolve().parent
    old = study.parent / 'TRI_v5b'
    destination = study / 'study_manifest.json'
    if destination.exists() or (study / 'inputs').exists():
        raise FileExistsError('Study already prepared; do not overwrite source copies')
    source_manifest = json.loads((old / 'study_manifest.json').read_text())
    frozen = study / 'pipeline' / 'frozen_v5b'
    frozen.mkdir(parents=True, exist_ok=True)
    code = [old / 'pipeline' / 'v5b.py', *sorted((old / 'pipeline' / 'frozen_v5a').glob('*.py')),
            old / 'pipeline' / 'frozen_v5a' / 'v5a_source_manifest.json']
    report = dict(meeting='2026-11-19', pipeline='v5c', inputs=[], frozen_code={}, retained_files={})
    for source in code:
        relative = source.relative_to(old / 'pipeline')
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError('Copy mismatch')
        report['frozen_code'][target.relative_to(study).as_posix()] = digest(source)
    for dataset, names in source_manifest['datasets'].items():
        for name in names:
            source = old / name
            expected = source_manifest['inputs'][name]
            if digest(source) != expected:
                raise RuntimeError('V5b input changed: ' + name)
            target = study / 'inputs' / dataset / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != expected:
                raise RuntimeError('Input copy mismatch')
            entry = dict(dataset=dataset, path=target.relative_to(study).as_posix(), sha256=expected)
            original = old / 'results' / dataset / 'reference_gray' / (source.stem + '.png')
            entry['v5b_reference'] = original.relative_to(study.parent).as_posix()
            targeted = old / 'results_targeted' / dataset / 'reference_gray' / (source.stem + '.png')
            if targeted.exists():
                entry['targeted_reference'] = targeted.relative_to(study.parent).as_posix()
            report['inputs'].append(entry)
    for root in (old / 'pipeline', old / 'results', old / 'results_targeted'):
        for source in sorted(root.rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                report['retained_files'][source.relative_to(study.parent).as_posix()] = digest(source)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Copied {len(report["inputs"])} inputs; V5b files retained in place')


if __name__ == '__main__':
    main()

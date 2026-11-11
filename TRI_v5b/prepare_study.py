"""Organize authorized V5b inputs and preserve the selected V5a source."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study = Path(__file__).resolve().parent
    root = study.parent.resolve()
    manifest_path = study / 'study_manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Study is already prepared')
    source = root / 'competitor_target_pipeline'
    frozen = study / 'pipeline' / 'frozen_v5a'
    selected = json.loads((source / 'v5a_source_manifest.json').read_text())
    for name, expected in selected['source_hashes'].items():
        if digest(source / name) != expected:
            raise RuntimeError('V5a source differs from selected manifest: ' + name)
    transfers = []
    for name in ('TRI Images 1008', 'TRI_New_Image'):
        original = (root / name).resolve(strict=True)
        destination = (study / name).resolve()
        if not original.is_relative_to(root) or not destination.is_relative_to(study):
            raise ValueError('Folder transfer outside the intended workspace')
        if destination.exists():
            raise FileExistsError(destination)
        paths = list(original.rglob('*'))
        if any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()) for p in paths):
            raise ValueError('Unexpected link in source directory')
        files = {str(p.relative_to(original)): digest(p) for p in paths if p.is_file()}
        transfers.append(dict(original=str(original), destination=str(destination), files=files))
    u29_manifest = json.loads((root / 'u29_focused_pipeline/result/u29_focused_v2/run.json').read_text())
    originals = Path(u29_manifest['input'])
    u29 = [originals / record['filename'] for record in u29_manifest['images']]
    if any(not p.is_file() for p in u29):
        raise FileNotFoundError('Missing U29 input')
    frozen.mkdir(parents=True, exist_ok=False)
    for name in (*selected['source_hashes'], 'v5a_source_manifest.json'):
        shutil.copy2(source / name, frozen / name)
    # Record all source hashes before the explicitly requested directory moves.
    plan = dict(status='prepared_for_move', transfers=transfers, frozen_v5a=selected)
    manifest_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for transfer in transfers:
        original, destination = Path(transfer['original']), Path(transfer['destination'])
        shutil.move(str(original), str(destination))
        for name, expected in transfer['files'].items():
            if digest(destination / name) != expected:
                raise RuntimeError('Moved file hash mismatch: ' + name)
    target = study / 'U29_inputs'
    target.mkdir()
    for path in u29:
        shutil.copy2(path, target / path.name)
        if digest(path) != digest(target / path.name):
            raise RuntimeError('U29 copy mismatch')
    datasets = {
        'U29': [str((target / p.name).relative_to(study)) for p in u29],
        'TRI_1008': [str(p.relative_to(study)) for p in sorted((study / 'TRI Images 1008').glob('*.bmp'))],
        'TRI_new': ['TRI_New_Image/image001.png', 'TRI_New_Image/image002.png'],
    }
    plan.update(status='complete', datasets=datasets,
                inputs={name: digest(study / name) for names in datasets.values() for name in names})
    manifest_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='complete', datasets=datasets), indent=2))


if __name__ == '__main__':
    main()

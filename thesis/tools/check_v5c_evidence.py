"""Read saved V5c evidence and run scoped tests; write only thesis evidence files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'TRI_v5c_2026-11-19'
EVIDENCE = ROOT / 'thesis/evidence'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, grayscale=False):
    with Image.open(path) as image:
        return np.array(image.convert('L') if grayscale else image)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_saved():
    """Do not import enhancement modules or rewrite saved results during audit."""
    manifest_path = STUDY / 'study_manifest.json'
    run_path = STUDY / 'results_v5c/run.json'
    manifest = json.loads(manifest_path.read_text())
    run = json.loads(run_path.read_text())
    require(run['status'] == 'complete', 'V5c run is incomplete')
    snapshots = {}
    for mapping, base in [(manifest['frozen_code'], STUDY),
                          (manifest['retained_files'], ROOT),
                          (run['code_sha256'], STUDY / 'pipeline')]:
        for name, expected in mapping.items():
            path = base / name
            require(digest(path) == expected, f'Snapshot changed: {path}')
            snapshots[path.relative_to(ROOT).as_posix()] = expected
    entries = {(r['dataset'], Path(r['path']).name): r for r in manifest['inputs']}
    records = {(r['dataset'], r['filename']): r for r in run['images']}
    require(len(entries) == len(manifest['inputs']) == 14, 'Unexpected input cohort')
    require(len(records) == len(run['images']) and entries.keys() == records.keys(),
            'Run/input cohort mismatch')
    checks = []
    for key, entry in entries.items():
        record = records[key]
        source_path = STUDY / entry['path']
        require(digest(source_path) == entry['sha256'] == record['sha256'],
                f'Source changed: {key}')
        source, raw = read(source_path), read(source_path, True)
        folder = STUDY / 'results_v5c' / entry['dataset']
        filename = source_path.stem + '.png'
        final_path = folder / 'enhanced' / filename
        gray_path = folder / 'reference_gray' / filename
        mask_path = folder / 'component_mask' / filename
        final, gray, support = read(final_path), read(gray_path), read(mask_path) > 0
        require(final.shape == source.shape and gray.shape == raw.shape
                and support.shape == raw.shape, f'Dimensions changed: {key}')
        require(np.array_equal(final[~support], source[~support])
                and np.array_equal(gray[~support], raw[~support]), f'Background changed: {key}')
        require(np.all(gray >= raw), f'Added grayscale darkening: {key}')
        require(record['saved_v5b_exact'], f'Saved baseline verification failed: {key}')
        checks.append(dict(dataset=key[0], filename=key[1], source_sha256=entry['sha256'],
                           output_sha256=digest(final_path), gray_sha256=digest(gray_path),
                           mask_sha256=digest(mask_path), native_dimensions=True,
                           outside_support_exact=True, no_added_grayscale_darkening=True,
                           baseline_equality_recorded_at_export=True,
                           accepted_proposals=record['selection']['accepted'],
                           proposed_regions=record['selection']['proposed'],
                           changed_fraction=record['changed_fraction'],
                           mean_absolute_display_change=record['mean_abs_change']))
    # Detailed retained-file hashes stay in the private study manifest. Commit
    # aggregate checks and compact input/output identity evidence, not image data.
    return dict(status='pass', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                run_sha256=digest(run_path), manifest_sha256=digest(manifest_path),
                image_count=len(checks), checks=checks, code_sha256=run['code_sha256'],
                settings=run['settings'], snapshot_files_verified=len(snapshots),
                snapshot_digest=hashlib.sha256(json.dumps(snapshots, sort_keys=True).encode()).hexdigest(),
                recorded_export_seconds=run['elapsed_including_exports_seconds'],
                limitations='Development images; no semantic defect/background validation. '
                'Baseline equality was recorded during export, not recomputed by this read-only audit. '
                'Runtime includes comparisons/crops/exports, not deployment latency.')


def main():
    report = verify_saved()
    command = [sys.executable, '-m', 'unittest', 'discover', '-s',
               'TRI_v5c_2026-11-19/pipeline', '-p', 'test_v5c.py', '-v']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log = result.stdout + result.stderr
    (EVIDENCE / 'tests_v5c.txt').write_text(log, encoding='utf-8')
    require(result.returncode == 0, 'V5c tests failed; inspect thesis/evidence/tests_v5c.txt')
    report['test_command'] = command[1:]
    report['test_log_sha256'] = digest(EVIDENCE / 'tests_v5c.txt')
    (EVIDENCE / 'v5c_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(log)
    print(f'Verified {len(report["checks"])} saved V5c images; snapshots unchanged.')


if __name__ == '__main__':
    main()

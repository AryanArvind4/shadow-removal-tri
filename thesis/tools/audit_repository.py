"""Inventory thesis evidence without importing pipelines or modifying results."""
import argparse
import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'thesis' / 'evidence'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SOURCE_DIRS = ['competitor_target_pipeline', 'component_restoration',
               'inspection_view_pipeline', 'masked_background_research',
               'u29_focused_pipeline', 'TRI_v5b']
SKIP = {'result', 'results', 'results_targeted', 'development_previews',
        '__pycache__', 'numeric', 'frozen_v5a'}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory():
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.md'))
    for folder in SOURCE_DIRS:
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.suffix in {'.py', '.md'} and not SKIP.intersection(p.parts))
    records = []
    for path in sorted(set(paths)):
        content = path.read_text(encoding='utf-8-sig')
        item = dict(path=relative(path), sha256=sha(path), lines=len(content.splitlines()))
        if path.suffix == '.py':
            try:
                tree = ast.parse(content)
                item['functions'] = [dict(name=n.name, line=n.lineno,
                                          end_line=n.end_lineno)
                                     for n in ast.walk(tree)
                                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                item['classes'] = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                item['imports'] = sorted({n.module for n in ast.walk(tree)
                                          if isinstance(n, ast.ImportFrom) and n.module}
                                         | {a.name for n in ast.walk(tree)
                                            if isinstance(n, ast.Import) for a in n.names})
            except SyntaxError as exc:
                item['syntax_error'] = str(exc)
        records.append(item)
    return records


def datasets():
    folders = {
        'TRI_full': 'TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset',
        'TRI_concise': 'TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset_Concised',
        'G2G3': 'competitor_target_pipeline/G2G3',
        'TRI_1008': 'TRI_v5b/TRI Images 1008',
        'TRI_new': 'TRI_v5b/TRI_New_Image',
        'U29_copy': 'TRI_v5b/U29_inputs',
        'round_ball_copy': 'TRI_v5b/round_ball_review',
        'U30_study_copy': 'competitor_target_pipeline/u30_meeting_20261005/input',
    }
    result = {}
    for name, folder in folders.items():
        records = []
        excluded = []
        for path in sorted((ROOT / folder).iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                if name == 'TRI_new' and path.name not in {'image001.png', 'image002.png'}:
                    excluded.append(path.name)
                    continue
                with Image.open(path) as img:
                    identity = f'{img.mode}:{img.size}'.encode() + img.tobytes()
                    records.append(dict(path=relative(path), filename=path.name,
                                        width=img.width, height=img.height, mode=img.mode,
                                        sha256=sha(path), pixel_sha256=hashlib.sha256(identity).hexdigest()))
        result[name] = dict(folder=folder, count=len(records),
                            excluded_derived_files=excluded,
                            dimensions=dict(Counter(f"{r['width']}x{r['height']} {r['mode']}" for r in records)),
                            records=records)
    return result


def artifacts():
    records = []
    for folder in ['results'] + SOURCE_DIRS:
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or 'development_previews' in path.parts:
                continue
            if path.name in {'run.json', 'benchmark.json', 'verification.json',
                             'source_preparation.json', 'annotations.json'}:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                record = dict(path=relative(path), sha256=sha(path),
                              keys=list(data) if isinstance(data, dict) else ['list'])
                if isinstance(data, dict):
                    record['list_lengths'] = {k: len(v) for k, v in data.items() if isinstance(v, list)}
                    record['scalars'] = {k: v for k, v in data.items()
                                         if isinstance(v, (str, int, float, bool)) or v is None}
                records.append(record)
    return records


def run_tests():
    suites = [
        ('v3_to_v5a', ['discover', '-s', 'competitor_target_pipeline', '-p', 'test_competitor_target_v*.py']),
        ('v5b', ['discover', '-s', 'TRI_v5b/pipeline', '-p', 'test_*.py']),
        ('u29_experiments', ['discover', '-s', 'u29_focused_pipeline', '-p', 'test_*.py']),
        ('u30_experiment', ['discover', '-s', 'competitor_target_pipeline/u30_meeting_20261005', '-p', 'test_*.py']),
    ]
    records = []
    for name, args in suites:
        command = [sys.executable, '-m', 'unittest'] + args + ['-v']
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=600)
        log = OUT / f'tests_{name}.txt'
        log.write_text(completed.stdout + completed.stderr, encoding='utf-8')
        records.append(dict(suite=name, returncode=completed.returncode,
                            command=command, log=relative(log)))
        print(f'{name}: exit {completed.returncode}', flush=True)
    return records


def measured_evidence():
    def read(path):
        return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))
    v5a = read('competitor_target_pipeline/result/competitor_target_v5a/run.json')
    v5b = read('TRI_v5b/results/run.json')
    verification = read('TRI_v5b/verification.json')
    annotations = read('competitor_target_pipeline/review/annotations.json')
    metrics_path = ROOT / 'competitor_target_pipeline/result/competitor_target_v5a/per_region_metrics.csv'
    with metrics_path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    summaries = {}
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method'] == method]
        summaries[method] = {
            k: float(np.mean([float(r[k]) for r in subset]))
            for k in ['interior_std', 'interior_iqr', 'input_correlation', 'clipping_fraction']}
        summaries[method]['mean_across_pad_mean_sd_per_image'] = float(np.mean([
            np.std([float(r['interior_mean']) for r in subset if r['filename'] == name])
            for name in sorted({r['filename'] for r in subset})]))
    frozen = read('competitor_target_pipeline/v5a_source_manifest.json')
    # The manifest has a named source_hashes mapping in the saved release.
    hashes = frozen.get('source_hashes', frozen)
    verification_hashes = {name: sha(ROOT / 'competitor_target_pipeline' / name) == value
                           for name, value in hashes.items() if name.endswith('.py')}
    return dict(v5a_images=len(v5a['images']),
                v5a_layout_counts={component: dict(
                    images=sum(r['filename'].startswith(component+'_') for r in v5a['images']),
                    layout_used=sum(r['filename'].startswith(component+'_') and r['layout']['used']
                                    for r in v5a['images'])) for component in ['J1','J4','U29','U30']},
                v5a_region_count=len({(r['filename'],r['region_id']) for r in rows}),
                v5a_metrics=summaries, metrics_source=relative(metrics_path),
                v5a_selected_sources_match_manifest=verification_hashes,
                v5b_images=len(v5b['images']), v5b_total_seconds=v5b['total_seconds'],
                v5b_exact_background_records=sum(r['exact_background'] for r in verification['records']),
                v5b_verification_source='TRI_v5b/verification.json',
                review_candidates=len(annotations['images']),
                reviewed_images=sum(bool(r['reviewed']) for r in annotations['images']),
                annotations=sum(len(r['annotations']) for r in annotations['images']),
                benchmark=read('competitor_target_pipeline/result/benchmarks/v5a_20260918_163907/benchmark.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Authored Python/Markdown inventory and selected local evidence; no model training, checkpoint loading or full image-quality adjudication.',
                  sources=source_inventory(), datasets=datasets(), artifacts=artifacts(),
                  measurements=measured_evidence())
    by_pixel = defaultdict(list)
    for dataset, info in report['datasets'].items():
        for record in info['records']:
            by_pixel[record['pixel_sha256']].append(f"{dataset}:{record['filename']}")
    report['cross_folder_pixel_duplicates'] = [v for v in by_pixel.values() if len(v) > 1]
    report['unique_decoded_images_across_listed_folders'] = len(by_pixel)
    report['git_history'] = subprocess.run(
        ['git', 'log', '--format=%h %ad %s', '--date=short'], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    report['checkpoints'] = [dict(path=relative(p), bytes=p.stat().st_size)
                             for folder in [ROOT, ROOT / 'checkpoints']
                             for p in folder.iterdir() if p.suffix in {'.pt', '.pth'}]
    if args.tests:
        report['tests'] = run_tests()
    (OUT / 'repository_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(source_files=len(report['sources']),
                          datasets={k: {f: v[f] for f in ('count', 'dimensions')}
                                    for k, v in report['datasets'].items()},
                          unique_images=report['unique_decoded_images_across_listed_folders']), indent=2))


if __name__ == '__main__':
    main()

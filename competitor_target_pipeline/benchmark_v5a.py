"""Time the unchanged V5a CLI on the exact inputs recorded in its manifest."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np
import PIL
from PIL import Image
import scipy

import competitor_target_v5a as pipeline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    baseline = root / 'result' / 'competitor_target_v5a'
    manifest = json.loads((baseline / 'run.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_hashes'].items():
        if digest(root / name) != expected:
            raise RuntimeError('Code differs from the selected run: ' + name)
    source = Path(manifest['input'])
    files = sorted(manifest['images'], key=lambda entry: entry['filename'])
    for entry in files:
        if digest(source / entry['filename']) != entry['sha256']:
            raise RuntimeError('Input differs from the selected run: ' + entry['filename'])

    output = root / 'result' / 'benchmarks' / datetime.now().strftime('v5a_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    original_process = pipeline.process

    def timed_process(raw, *args, **kwargs):
        entry = files[len(records)]
        start = perf_counter()
        result = original_process(raw, *args, **kwargs)
        seconds = perf_counter() - start
        record = dict(filename=entry['filename'], component=entry['filename'].split('_')[0],
                      height=raw.shape[0], width=raw.shape[1], process_seconds=seconds)
        records.append(record)
        print(f"  PROCESS TIME: {seconds:.3f} s", flush=True)
        return result

    old_argv = sys.argv
    pipeline.process = timed_process
    sys.argv = ['competitor_target_v5a.py', '--input', str(source),
                '--output', str(output), '--files', *[entry['filename'] for entry in files]]
    start = perf_counter()
    try:
        pipeline.main()
    finally:
        elapsed = perf_counter() - start
        pipeline.process = original_process
        sys.argv = old_argv

    # Verification is outside the measured batch interval.
    identical = []
    for entry in files:
        filename = Path(entry['filename']).stem + '.png'
        with Image.open(baseline / 'reference_gray' / filename) as old:
            old_pixels = np.array(old)
        with Image.open(output / 'reference_gray' / filename) as new:
            identical.append(bool(np.array_equal(old_pixels, np.array(new))))
    grouped = defaultdict(list)
    for record in records:
        grouped[record['component']].append(record['process_seconds'])
    summaries = {name: dict(images=len(values), mean_seconds=statistics.mean(values),
                            median_seconds=statistics.median(values),
                            min_seconds=min(values), max_seconds=max(values),
                            total_process_seconds=sum(values))
                 for name, values in grouped.items()}
    report = dict(
        measured_at=datetime.now().astimezone().isoformat(),
        scope='One sequential pass over the 32 original inputs; no dedicated warm-up.',
        timer='time.perf_counter wall-clock seconds',
        process_scope='pipeline.process on an already-loaded float32 image; includes all algorithm branches and diagnostic computation returned by process.',
        batch_scope='Original pipeline.main: image reads, processing, image/mask/overlay writes, numeric compression, comparison sheets, region metrics, and run manifest. Excludes Python startup/imports, preflight hash checks, and post-run output verification.',
        hardware=dict(cpu='Intel Core 7 240H', physical_cores=10, logical_processors=16,
                      installed_ram_gib=16, gpu='NVIDIA GeForce RTX 5050 Laptop GPU',
                      nvidia_reported_memory_mib=8151, gpu_driver='596.36',
                      laptop='MSI Cyborg 15 B2RWEKG', os='Windows 11 Home 64-bit, build 26200'),
        environment=dict(python=platform.python_version(), opencv=cv2.__version__,
                         numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__,
                         opencv_threads=cv2.getNumThreads(), opencl_active=cv2.ocl.useOpenCL(),
                         acceleration='CPU implementation; no CUDA GPU processing'),
        image_count=len(records), batch_seconds=elapsed,
        processing_total_seconds=sum(x['process_seconds'] for x in records),
        batch_average_seconds=elapsed / len(records),
        component_processing=summaries, per_image=records,
        source_hashes=manifest['source_hashes'],
        reference_outputs_identical_count=sum(identical),
        note='Observed runtime on this PC under current load, not a guaranteed latency or repeated benchmark. Input hash checks read files before timing, so this is not a cold-cache disk benchmark.')
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# V5a hardware and measured runtime', '',
             '- CPU: Intel Core 7 240H, 10 cores / 16 logical processors.',
             '- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, approximately 8 GB VRAM (8151 MiB reported).',
             '- RAM: 16 GB installed.', '- OS: Windows 11 Home, 64-bit.',
             '- V5a processing: CPU-based; the installed NVIDIA GPU is not used for acceleration.', '',
             '## Processing-only time', '',
             '| Component | Images | Dimensions (W x H) | Mean seconds/image | Range (s) |',
             '| --- | --- | --- | --- | --- |']
    for name, summary in summaries.items():
        sample = next(x for x in records if x['component'] == name)
        lines.append(f"| {name} | {summary['images']} | {sample['width']} x {sample['height']} | {summary['mean_seconds']:.2f} | {summary['min_seconds']:.2f}-{summary['max_seconds']:.2f} |")
    lines.extend(['', f"- Total processing-only time: {report['processing_total_seconds']:.2f} seconds.",
                  f"- Full batch runtime: **{elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)** for {len(records)} images.",
                  f"- Full batch average: {elapsed / len(records):.2f} seconds/image, including exports and diagnostics.",
                  f"- Output verification: {sum(identical)}/{len(identical)} reference PNGs have identical pixels to the selected V5a results.",
                  '', '## Measurement details', '', report['scope'], report['process_scope'],
                  report['batch_scope'], report['note'], '',
                  'Software: ' + json.dumps(report['environment'])])
    (output / 'SLIDE_TIMING_NOTES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('batch_seconds', 'processing_total_seconds',
        'batch_average_seconds', 'component_processing', 'reference_outputs_identical_count', 'environment')}, indent=2), flush=True)
    print('REPORT: ' + str(output / 'SLIDE_TIMING_NOTES.md'), flush=True)


if __name__ == '__main__':
    main()

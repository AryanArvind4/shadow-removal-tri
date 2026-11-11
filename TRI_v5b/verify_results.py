"""Independently verify saved outputs and build matched source-pixel review crops."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, grayscale=False):
    with Image.open(path) as image:
        return np.array(image.convert('L') if grayscale else image)


def main():
    study = Path(__file__).resolve().parent
    manifest = json.loads((study/'study_manifest.json').read_text())
    run = json.loads((study/'results/run.json').read_text())
    if run['status'] != 'complete':
        raise RuntimeError('Run is incomplete')
    checks = []
    for record in run['images']:
        dataset, name = record['dataset'], record['filename']
        source_path = next(study/p for p in manifest['datasets'][dataset] if Path(p).name == name)
        source = read(source_path)
        gray_source = read(source_path, True)
        folder = study/'results'/dataset
        output_name = Path(name).stem+'.png'
        output = read(folder/'enhanced'/output_name)
        gray = read(folder/'reference_gray'/output_name)
        baseline = read(folder/'v5a_reference'/output_name)
        mask = read(folder/'component_mask'/output_name) > 0
        with np.load(folder/'numeric'/(Path(name).stem+'.npz')) as arrays:
            alpha, restore = arrays['alpha'], arrays['boundary_restore']
        assert digest(source_path) == record['source_sha256'], name
        assert source.shape == output.shape, name
        assert np.array_equal(output[~mask], source[~mask]), name
        assert np.array_equal(gray[~mask], gray_source[~mask]), name
        assert np.ptp(output) > 0, name
        core = (alpha == 1) & (restore == 0)
        assert np.array_equal(gray[core], baseline[core]), name
        historical = None
        if dataset == 'TRI_1008':
            historical = study/'TRI Images 1008/v5a_results/reference_gray'/output_name
        elif dataset == 'TRI_new':
            historical = study/'TRI_New_Image/v5a_run/results/reference_gray'/output_name
        elif dataset == 'U29':
            historical = study.parent/'competitor_target_pipeline/result/competitor_target_v5a/reference_gray'/output_name
        elif dataset == 'round_ball_review':
            historical = study.parent/'competitor_target_pipeline/G2G3/v5a_results/reference_gray'/output_name
        baseline_match = None
        if historical is not None and historical.exists():
            baseline_match = bool(np.array_equal(read(historical), baseline))
            assert baseline_match, 'V5a reproduction mismatch: '+name
        delta = baseline.astype(np.float32)-gray_source
        checks.append(dict(dataset=dataset, filename=name, source_hash_valid=True,
                           native_shape_valid=True, exact_background=True,
                           v5a_retained_in_fully_weighted_core=True,
                           matches_historical_v5a=baseline_match,
                           outside_support_pixels=int((~mask).sum()),
                           outside_support_v5a_mean_abs_change=float(np.abs(delta[~mask]).mean()),
                           outside_support_v5b_max_abs_change=0))
    for name, expected in manifest['frozen_v5a']['source_hashes'].items():
        assert digest(study/'pipeline/frozen_v5a'/name) == expected
        assert digest(study.parent/'competitor_target_pipeline'/name) == expected
    for name, expected in run['source_hashes'].items():
        assert digest(study/'pipeline'/name) == expected
    moved_count = 0
    for transfer in manifest['transfers']:
        assert not Path(transfer['original']).exists()
        for name, expected in transfer['files'].items():
            assert digest(Path(transfer['destination'])/name) == expected
            moved_count += 1
    crop_dir = study/'results/review_crops'
    crop_dir.mkdir(exist_ok=True)
    crop_specs = [
        ('U29','U29_034.jpg',(110,195,420,485),'U29_middle_pads'),
        ('TRI_1008','Component_U76_1_2048_708_636_201_z141.bmp',(115,170,400,460),'U76_middle_pads'),
        ('round_ball_review','Component_G2_1_1152_2472_2140_165_z0060.bmp',(1800,1500,2300,2000),'round_ball_rims'),
    ]
    crop_records = []
    for dataset, name, box, label in crop_specs:
        source_path = next(study/p for p in manifest['datasets'][dataset] if Path(p).name == name)
        result_dir = study/'results'/dataset
        files = [source_path, result_dir/'v5a_reference'/(Path(name).stem+'.png'),
                 result_dir/'reference_gray'/(Path(name).stem+'.png')]
        width, height = box[2]-box[0], box[3]-box[1]
        canvas = Image.new('RGB',(width*3,height+30),(25,25,25))
        draw = ImageDraw.Draw(canvas)
        for i,(title,path) in enumerate(zip(('Input','V5a','V5b'),files)):
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB').crop(box),(i*width,30))
            draw.text((i*width+8,8),title,fill='white')
        canvas.save(crop_dir/(label+'.png'))
        crop_records.append(dict(filename=name, box=list(box), interpolation='none; native pixels', output=label+'.png'))
    (crop_dir/'locations.json').write_text(json.dumps(crop_records, indent=2))
    report = dict(status='pass', verified_images=len(checks), unchanged_moved_files=moved_count,
                  original_v5a_unchanged=True, frozen_v5a_verified=True, current_v5b_sources_verified=True,
                  records=checks,
                  limitation='Equality applies outside estimated masks; it is not segmentation or defect-accuracy validation.')
    (study/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()

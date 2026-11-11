"""Explicit, hash-locked V5b corrections for three reviewed inputs only."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import v5b


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(path, profile):
    if digest(path) != profile['sha256']:
        raise ValueError('Profile is only valid for its exact source hash: '+str(path))


def boxes_for(profile):
    width, height = profile['size']
    boxes = list(profile['boxes'])
    for name in ('grid', 'small_columns'):
        if name in profile:
            grid = profile[name]
            boxes.extend([max(0,x-grid['half_width']),max(0,y-grid['half_height']),
                          min(width,x+grid['half_width']),min(height,y+grid['half_height'])]
                         for y in grid['y'] for x in grid['x'])
    for x0,y0,x1,y1 in boxes:
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError('Invalid review envelope')
    return boxes


def envelope_mask(shape, boxes):
    mask = np.zeros(shape, np.uint8)
    for x0,y0,x1,y1 in boxes:
        mask[y0:y1,x0:x1] = 1
    return mask.astype(bool)


def inside_feather(mask, width):
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(distance/width, 0, 1)
    return t*t*(3-2*t)


def screenshot_support(raw, boxes):
    support = np.zeros(raw.shape, np.uint8)
    records = []
    for box in boxes:
        x0,y0,x1,y1 = box
        patch = raw[y0:y1,x0:x1]
        large = min(patch.shape) > 35
        smooth = v5b.v5a.base.blur(patch, 1.6 if large else .85)
        low, high = np.percentile(smooth, [10,80])
        if high-low < 12:
            records.append(dict(box=box, accepted=False, reason='weak source contrast'))
            continue
        threshold = low+.22*(high-low)
        binary = (smooth > threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7) if large else (3,3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        local = np.zeros(patch.shape,np.uint8)
        accepted = 0
        for contour in contours:
            if cv2.contourArea(contour) < max(5,.035*patch.size):
                continue
            cv2.drawContours(local,[contour],-1,1,-1)
            accepted += 1
        support[y0:y1,x0:x1] |= local
        records.append(dict(box=box,accepted=accepted > 0,source_threshold=float(threshold),
                            support_pixels=int(local.sum())))
    return support.astype(bool), records


def correct(source, previous, baseline, previous_mask, profile, previous_color=None):
    if list(source.shape[1::-1]) != profile['size']:
        raise ValueError('Profile dimensions do not match source')
    raw = np.asarray(Image.fromarray(source).convert('L'),np.float32)
    boxes = boxes_for(profile)
    envelope = envelope_mask(raw.shape,boxes)
    if profile['mode'] == 'restrict_existing':
        # Restrict an already accepted display. Fully allowed pixels remain the
        # exact prior V5b values; excluded regions return to the source.
        support = previous_mask & envelope
        alpha = inside_feather(envelope,profile['feather'])*previous_mask
        target = previous.astype(np.float32)
        records = []
    elif profile['mode'] == 'complete_screenshot_components':
        support, records = screenshot_support(raw,boxes)
        alpha = inside_feather(support,profile['feather'])
        target = baseline.astype(np.float32)
        # Do not deepen the outer edge while transitioning to the input.
        target = target+(1-alpha)*np.maximum(raw-target,0)
    else:
        raise ValueError('Unknown correction profile')
    if source.ndim == 3:
        chroma = np.ptp(source.astype(np.int16),axis=2) > 18
        protected = cv2.dilate(chroma.astype(np.uint8),np.ones((3,3),np.uint8)) > 0
        support[protected] = False
        alpha[protected] = 0
    output = raw+alpha*(target-raw)
    result = dict(output=output,alpha=alpha,target=target,support=support)
    final = v5b.render_original_mode(source,result)
    if profile.get('retain_previous_outside_boxes',False):
        outside = ~envelope
        output[outside] = previous[outside]
        final[outside] = (previous_color if previous_color is not None else previous)[outside]
        support[outside] = previous_mask[outside]
        envelope |= previous_mask
    assert np.array_equal(final[~support],source[~support])
    assert np.array_equal(final[~envelope],source[~envelope])
    return final,v5b.v5a.base.u8(output),support,alpha,envelope,records


def main():
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=study/'results_targeted')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace earlier results')
    profile_path = Path(__file__).with_name('targeted_profiles.json')
    profiles = json.loads(profile_path.read_text())['profiles']
    manifest = json.loads((study/'study_manifest.json').read_text())
    sources = {}
    # Validate all inputs before any output is created.
    for profile in profiles:
        path = next(study/p for p in manifest['datasets'][profile['dataset']] if Path(p).name == profile['filename'])
        validate_source(path,profile)
        sources[profile['filename']] = path
    args.output.mkdir(parents=True)
    unchanged_paths = [p for p in (study/'results').rglob('*') if p.is_file()]
    unchanged_paths += list((study/'pipeline').glob('v5b.py'))+list((study/'pipeline/frozen_v5a').glob('*.py'))
    before = {str(p):digest(p) for p in unchanged_paths}
    report = dict(pipeline='V5b targeted image profiles',status='running',images=[],
                  source_hashes={str(p.relative_to(study)):digest(p) for p in (Path(__file__),profile_path,Path(v5b.__file__))},
                  scope='Three exact inputs only. Analyst-selected envelopes, not a general pipeline change.')
    for profile in profiles:
        name,dataset = profile['filename'],profile['dataset']
        source_path = sources[name]
        with Image.open(source_path) as image:
            source = np.array(image)
            raw = np.array(image.convert('L'))
        prior_dir = study/'results'/dataset
        output_name = Path(name).stem+'.png'
        with Image.open(prior_dir/'reference_gray'/output_name) as image:
            previous = np.array(image)
        with Image.open(prior_dir/'v5a_reference'/output_name) as image:
            baseline = np.array(image)
        with Image.open(prior_dir/'component_mask'/output_name) as image:
            previous_mask = np.array(image) > 0
        with Image.open(prior_dir/'enhanced'/output_name) as image:
            previous_color = np.array(image)
        final,gray,support,alpha,envelope,regions = correct(source,previous,baseline,previous_mask,profile,previous_color)
        folder = args.output/dataset
        overlay = cv2.cvtColor(raw,cv2.COLOR_GRAY2RGB)
        contours,_ = cv2.findContours(support.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(30,220,90),1)
        for x0,y0,x1,y1 in boxes_for(profile):
            cv2.rectangle(overlay,(x0,y0),(x1-1,y1-1),(230,170,30),1)
        arrays = dict(enhanced=final,reference_gray=gray,component_mask=support.astype(np.uint8)*255,
                      envelope_mask=envelope.astype(np.uint8)*255,selection_overlay=overlay,
                      blend_weight=v5b.v5a.base.u8(alpha*255))
        for key,array in arrays.items():
            (folder/key).mkdir(parents=True,exist_ok=True)
            path = folder/key/output_name
            Image.fromarray(array).save(path)
            with Image.open(path) as saved:
                assert np.array_equal(np.array(saved),array)
        for key in ('input_output','comparison_sheets','numeric'):
            (folder/key).mkdir(parents=True,exist_ok=True)
        v5b.v5a.save_sheet(folder/'input_output'/output_name,[('Input (grayscale)',raw),('V5b: targeted correction',gray)],raw.shape[1])
        v5b.v5a.save_sheet(folder/'comparison_sheets'/output_name,[('Input',raw),('Previous V5b',previous),('Targeted V5b',gray)],raw.shape[1])
        np.savez_compressed(folder/'numeric'/(Path(name).stem+'.npz'),refinement_alpha=alpha,support=support,envelope=envelope)
        excluded_checks = []
        for box in profile.get('unchanged_check_boxes',[]):
            x0,y0,x1,y1 = box
            assert np.array_equal(final[y0:y1,x0:x1],source[y0:y1,x0:x1]), box
            excluded_checks.append(dict(box=box,exact_source_pixels=True))
        if profile['mode'] == 'restrict_existing':
            full = alpha == 1
            assert np.array_equal(gray[full],previous[full])
        assert digest(source_path) == profile['sha256']
        report['images'].append(dict(filename=name,dataset=dataset,source_sha256=profile['sha256'],
            profile_mode=profile['mode'],source_shape=list(source.shape),source_unchanged=True,
            previous_support_fraction=float(previous_mask.mean()),support_fraction=float(support.mean()),
            previous_support_pixels=int(np.count_nonzero(previous_mask)),refinement_fully_blended_pixels=int(np.count_nonzero(alpha == 1)),
            changed_pixels_vs_previous=int(np.count_nonzero(gray != previous)),
            background_exact_outside_support=True,background_exact_outside_envelopes=True,
            unchanged_check_boxes=excluded_checks,regions=regions))
        print(f'{dataset}/{name}: support {previous_mask.mean():.3f} -> {support.mean():.3f}; background checks passed',flush=True)
    for path,expected in before.items():
        assert digest(Path(path)) == expected, 'Existing artifact changed: '+path
    report.update(status='complete',existing_artifacts_unchanged=True,verified_existing_files=len(before))
    (args.output/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(args.output.resolve(),flush=True)


if __name__ == '__main__':
    main()

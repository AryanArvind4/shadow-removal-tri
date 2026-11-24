"""Audit saved V5c files and export pixel-replicated U29 edge-detail comparisons."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run_v5c import digest, verify_snapshot


def main():
    study = Path(__file__).resolve().parents[1]
    output = study/'results_v5c'
    manifest = json.loads((study/'study_manifest.json').read_text())
    report = json.loads((output/'run.json').read_text())
    if report['status'] != 'complete':
        raise RuntimeError('Run not complete')
    verify_snapshot(study,manifest)
    for name, expected in report['code_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise RuntimeError('Pipeline changed after export: '+name)
    records = []
    for entry in manifest['inputs']:
        path = study/entry['path']
        if digest(path) != entry['sha256']:
            raise RuntimeError('Copied source changed')
        with Image.open(path) as image:
            source = np.array(image)
            raw = np.array(image.convert('L'))
        folder = output/entry['dataset']
        with Image.open(folder/'enhanced'/(path.stem+'.png')) as image:
            final = np.array(image)
        with Image.open(folder/'reference_gray'/(path.stem+'.png')) as image:
            gray = np.array(image)
        with Image.open(folder/'component_mask'/(path.stem+'.png')) as image:
            support = np.array(image) > 0
        np.testing.assert_equal(final.shape,source.shape)
        np.testing.assert_equal(gray.shape,raw.shape)
        np.testing.assert_array_equal(final[~support],source[~support])
        np.testing.assert_array_equal(gray[~support],raw[~support])
        if np.any(gray < raw):
            raise RuntimeError('Grayscale darkening found')
        records.append(dict(dataset=entry['dataset'],filename=path.name,
                            background_exact=True,native_dimensions=True,no_added_grayscale_darkening=True))
        if entry['dataset'] != 'U29':
            continue
        with Image.open(study.parent/entry['v5b_reference']) as image:
            old = np.array(image.convert('L'))
        locations = json.loads((folder/'review_crops'/(path.stem+'_locations.json')).read_text())
        destination = folder/'pixel_crops'
        destination.mkdir(exist_ok=True)
        for location in locations:
            if not any(location['file'].endswith(f'_pad_{index:02d}.png') for index in range(10,19)):
                continue
            x0,y0,x1,y1 = location['source_xyxy']
            width,height = (x1-x0)*5,(y1-y0)*5
            sheet = Image.new('RGB',(width*3,height+28),'#181818')
            draw = ImageDraw.Draw(sheet)
            for i,(label,array) in enumerate((('Input',raw),('Saved V5b',old),('V5c candidate',gray))):
                crop = Image.fromarray(array[y0:y1,x0:x1]).resize((width,height),Image.Resampling.NEAREST)
                sheet.paste(crop,(i*width,28))
                draw.text((i*width+6,7),label+' (5x pixels)',fill='white')
            sheet.save(destination/location['file'])
    summary = dict(status='passed',images=len(records),code_matches_export=True,
                   retained_v5b_snapshot_unchanged=True,checks=records,
                   note='Mechanical invariants only; semantic void preservation requires review.')
    (output/'audit.json').write_text(json.dumps(summary,indent=2))
    print(f'Audit passed for {len(records)} images. U29 row-2 pixel crops exported.')


if __name__ == '__main__':
    main()

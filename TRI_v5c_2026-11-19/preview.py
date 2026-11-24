"""Small representative run before exporting the full review batch."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

study = Path(__file__).resolve().parent
sys.path.insert(0, str(study / 'pipeline'))
import v5c

manifest = json.loads((study / 'study_manifest.json').read_text())
output = study / 'development' / 'v5c_preview'
output.mkdir(parents=True, exist_ok=True)
for entry in manifest['inputs']:
    name = Path(entry['path']).stem
    if not (name == 'U29_032' or name.endswith('z1') or name.endswith('z201')):
        continue
    raw = np.asarray(Image.open(study / entry['path']).convert('L'), np.float32)
    result = v5c.process(raw)
    v5c.v5b.v5a.save_sheet(output / (name + '.png'),
        [('Input', raw), ('V5b',result['v5b']), ('V5c',result['output'])], min(760,raw.shape[1]))
    (output / (name + '.json')).write_text(json.dumps(result['details'], indent=2))
    if name == 'U29_032':
        for i,(x,y,w,h) in enumerate([r['box'] for r in result['details']['selection']['regions'] if 110 < r['box'][1] < 185]):
            sl = (slice(y-5,y+h+5), slice(x-5,x+w+5))
            v5c.v5b.v5a.save_sheet(output / f'pad_{i+10}.png',
                [('Input',raw[sl]),('V5b',result['v5b'][sl]),('V5c',result['output'][sl]),
                 ('V5c weight',result['alpha'][sl]*255)],220)
    print(name, result['details']['selection']['accepted'], result['details']['support_fraction'], flush=True)

"""Inspect source, V5a and V5b on the reported U29 edge-feature region."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / 'pipeline' / 'frozen_v5b'))
import v5b

raw = np.asarray(Image.open(STUDY / 'inputs/U29/U29_032.jpg'), np.float32)
baseline = v5b.v5a.process(raw)
result = v5b.process(raw, baseline=baseline)
out = STUDY / 'development'
out.mkdir(exist_ok=True)
images, labels, regions, corrected, diag, _ = baseline
np.savez_compressed(out / 'probe32.npz', raw=raw, v5a=result['v5a'], v5b=result['output'],
                    support=result['support'], alpha=result['alpha'], regions=np.asarray(regions))
(out / 'probe32.json').write_text(json.dumps(diag, indent=2))
v5b.v5a.save_sheet(out / 'probe32.png', [(k, a[125:185, 60:650]) for k, a in [
    ('Input',raw), ('V5a',result['v5a']), ('V5b',result['output']),
    ('V5b support',result['support']*255)]], 1180)
for i, x, y, w, h in regions:
    if y < 185 and y > 110:
        p = (slice(y-5,y+h+5), slice(x-5,x+w+5))
        v5b.v5a.save_sheet(out / f'pad_{i}.png', [('Source',raw[p]), ('V5a',result['v5a'][p]),
             ('V5b',result['output'][p]), ('Mask',result['support'][p]*255)], 220)
print('Noise/diameter:', diag['baseline_diagnostics']['noise_sigma'], diag['baseline_diagnostics']['feature_diameter_native'])
print('Second row:', [r for r in regions if 110 < r[2] < 185])

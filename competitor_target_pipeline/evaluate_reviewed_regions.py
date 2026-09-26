"""Compare reviewed healthy/defect polygons at identical pixels across outputs."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from competitor_target_v3 import INPUT, OUTPUT, read, write_csv


def polygon_mask(points, shape):
    points = np.asarray(points, dtype=float)
    h, w = shape
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('A polygon needs at least three finite [x, y] coordinates.')
    if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError('Polygon lies outside native image bounds.')
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if mask.sum() < 4:
        raise ValueError('Polygon must contain at least four pixels.')
    return mask > 0


def measure(image, healthy, defect):
    if np.any(healthy & defect):
        raise ValueError('Healthy and defect polygons must not overlap.')
    h, d = image[healthy], image[defect]
    difference = float(h.mean() - d.mean())
    noise = float(np.sqrt((h.var() + d.var()) / 2))
    return dict(healthy_mean=float(h.mean()), healthy_std=float(h.std()),
                defect_mean=float(d.mean()), defect_std=float(d.std()),
                signed_healthy_minus_defect=difference,
                cnr=abs(difference) / noise if noise > 1e-6 else None,
                healthy_pixels=int(healthy.sum()), defect_pixels=int(defect.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--results', type=Path, default=OUTPUT)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    records = []
    for entry in annotations['images']:
        filename = entry['filename']
        if Path(filename).name != filename:
            raise ValueError('Use image filenames without directory paths.')
        source = read(args.input / filename)
        images = {'input': source}
        for method in ('detail_gray', 'reference_gray'):
            images[method] = read(args.results / method / (Path(filename).stem + '.png'))
            if images[method].shape != source.shape:
                raise ValueError(f'{filename}: output and source dimensions differ.')
        for region in entry['regions']:
            healthy = polygon_mask(region['healthy_polygon'], source.shape)
            defect = polygon_mask(region['defect_polygon'], source.shape)
            for method, image in images.items():
                records.append(dict(filename=filename, region_id=region['id'], method=method,
                                    **measure(image, healthy, defect)))
    if not records:
        raise ValueError('No reviewed regions supplied.')
    destination = args.results / 'reviewed_roi_metrics.csv'
    write_csv(destination, records)
    print(destination)
    print('Fixed-ROI contrast measurements only; defect boundary/area accuracy is not measured.')


if __name__ == '__main__':
    main()

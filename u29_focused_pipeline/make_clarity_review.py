"""Build review sheets for the U29 clarity pipeline (V3).

Produces, under ``result/u29_clarity_v3/review``:

* ``compare_<name>.png``   source, V2 if present, V3, side by side
* ``pads_<name>.png``      a magnified three by three pad block
* ``overview.png``         all eight V3 results on one sheet
* ``reference_side.png``   V3 next to the competitor photograph, labelled
* ``metrics.csv``          per image numbers used in the write up

The competitor image is a phone photograph of a screen running a different
acquisition. It is used only for a labelled visual comparison and for coarse
tonal statistics. It never takes part in generating any output.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u29_clarity as U


V2_DIR = r"E:\Shadow_Removal_SDE\u29_focused_pipeline\shape_preview\reference_gray"
ZOOM_BOX = (118, 340, 124, 346)      # y0, y1, x0, x1 of the 3x3 pad block


def _label(img, text, height=26):
    h, w = img.shape[:2]
    bar = np.zeros((height, w), np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, 255, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _stretch(src, lo=40.0, hi=110.0):
    return np.clip((src.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _hstack(tiles, gap=8):
    out = []
    for i, t in enumerate(tiles):
        if i:
            out.append(np.full((t.shape[0], gap), 128, np.uint8))
        out.append(t)
    return np.hstack(out)


def pad_metrics(img, centers):
    """Pad core, rim and gap statistics for one image."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dmin = np.full((h, w), 1e9, np.float32)
    for row in centers:
        for cx, cy in row:
            dmin = np.minimum(dmin, np.maximum(np.abs(xx - cx), np.abs(yy - cy)))
    core = img[dmin < 13].astype(np.float64)
    gap = img[dmin > 30].astype(np.float64)
    if core.size == 0 or gap.size == 0:
        return {}
    return {
        "pad_core_median": round(float(np.median(core)), 1),
        "pad_core_p5": round(float(np.percentile(core, 5)), 1),
        "void_contrast": round(float(np.median(core) - np.percentile(core, 5)), 1),
        "gap_median": round(float(np.median(gap)), 1),
        "gap_iqr": round(float(np.percentile(gap, 75) - np.percentile(gap, 25)), 1),
        "pad_to_gap": round(float(np.median(core) - np.median(gap)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=U.DEFAULT_SRC)
    ap.add_argument("--pattern", default="U29_*.jpg")
    ap.add_argument("--out", default=U.DEFAULT_OUT)
    ap.add_argument("--reference", default="")
    args = ap.parse_args()

    review = os.path.join(args.out, "review")
    os.makedirs(review, exist_ok=True)
    p = U.Params()

    files = sorted(glob.glob(os.path.join(args.src, args.pattern)))
    if not files:
        raise SystemExit("no inputs matched " + args.pattern)

    rows = []
    finals = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        out, diag, st = U.process(f, p)
        src = st["source"]
        centers = st["centers"]

        tiles = [_label(_stretch(src), "source (contrast stretched)")]
        v2_path = os.path.join(V2_DIR, name + ".png")
        if os.path.exists(v2_path):
            v2 = cv2.imread(v2_path, cv2.IMREAD_GRAYSCALE)
            if v2 is not None and v2.shape == out.shape:
                tiles.append(_label(v2, "V2"))
        tiles.append(_label(out, "V3 clarity"))
        cv2.imwrite(os.path.join(review, "compare_" + name + ".png"), _hstack(tiles))

        y0, y1, x0, x1 = ZOOM_BOX
        zs = cv2.resize(_stretch(src)[y0:y1, x0:x1], None, fx=3, fy=3,
                        interpolation=cv2.INTER_NEAREST)
        zo = cv2.resize(out[y0:y1, x0:x1], None, fx=3, fy=3,
                        interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(review, "pads_" + name + ".png"),
                    _hstack([_label(zs, "source x3"), _label(zo, "V3 x3")]))

        m_src = pad_metrics(src.astype(np.uint8), centers)
        m_out = pad_metrics(out, centers)
        rows.append(dict(image=name,
                         **{"src_" + k: v for k, v in m_src.items()},
                         **{"v3_" + k: v for k, v in m_out.items()},
                         shadow_span_dn=diag["shadow_span_dn"]))
        finals.append(_label(out, name))

    # overview sheet, two rows of four
    per_row = 4
    bands = []
    for i in range(0, len(finals), per_row):
        chunk = finals[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.zeros_like(finals[0]))
        bands.append(_hstack(chunk))
    cv2.imwrite(os.path.join(review, "overview.png"), np.vstack(bands))

    if args.reference and os.path.exists(args.reference):
        ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            best = cv2.imread(os.path.join(args.out, "U29_032.png"),
                              cv2.IMREAD_GRAYSCALE)
            ref_r = cv2.resize(ref, (best.shape[1], best.shape[0]))
            sheet = _hstack([
                _label(best, "ours V3, 2D projection"),
                _label(ref_r, "competitor photo, 3D CT slice, different part"),
            ])
            cv2.imwrite(os.path.join(review, "reference_side.png"), sheet)

    with open(os.path.join(review, "metrics.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    print("wrote " + review)
    for r in rows:
        print("  {0}: void contrast {1} -> {2} DN, gap IQR {3} -> {4} DN".format(
            r["image"], r.get("src_void_contrast"), r.get("v3_void_contrast"),
            r.get("src_gap_iqr"), r.get("v3_gap_iqr")))


if __name__ == "__main__":
    main()

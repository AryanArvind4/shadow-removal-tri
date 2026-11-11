"""Source-matched ablations; no competitor image enters processing."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

import u29_clarity as clarity


def tone_control(src):
    # One increasing affine display transform: no masks or spatial processing.
    return np.rint(np.clip(3.0 * (src.astype(np.float32) - 40.) + 24.,
                           0, 255)).astype(np.uint8)


def variants():
    conservative = replace(
        clarity.Params(), denoise_h=0, deconv_sigma=0, deconv_iters=0,
        dome_strength=0, void_amount=0, unsharp_amount=0,
        pad_norm=0.5, fine_strength=0.5, bg_gain=0.5,
        tone_x=(-1.5, -0.6, 0., 0.3, 0.7, 1.1, 1.6, 2.4),
        tone_y=(4., 16., 42., 87., 147., 207., 240., 251.),
    )
    return {
        "clarity_v3": clarity.Params(),
        "no_detail_boost": replace(clarity.Params(), denoise_h=0,
            deconv_sigma=0, deconv_iters=0, dome_strength=0,
            void_amount=0, unsharp_amount=0),
        "conservative": conservative,
    }


def labeled(image, label):
    out = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=20)
    cv2.putText(out, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, 240, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="U29_034.jpg")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent /
                        "result" / "u29_clarity_audit")
    args = parser.parse_args()
    files = sorted(Path(clarity.DEFAULT_SRC).glob(args.pattern))
    if not files:
        raise ValueError("No source images")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "experimental, not defect validated",
              "acquisition_match": "not confirmed by user",
              "parameters": {k: asdict(v) for k, v in variants().items()},
              "images": {}}
    for path in files:
        src = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        panels = [labeled(src, "Original source")]
        crops = [labeled(src[105:270, 115:425], "Source: rows 2-3")]
        record = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        control = tone_control(src)
        cv2.imwrite(str(args.out / (path.stem + "_tone_only.png")), control)
        crops.append(labeled(control[105:270, 115:425], "Tone only: no spatial edits"))
        for name, params in variants().items():
            out, diag, stages = clarity.process(str(path), params)
            target = args.out / name
            target.mkdir(exist_ok=True)
            if not cv2.imwrite(str(target / (path.stem + ".png")), out):
                raise OSError("Failed to save output")
            panels.append(labeled(out, name))
            crops.append(labeled(out[105:270, 115:425], name))
            record[name] = {"grid": [diag["n_rows"], diag["n_cols"]],
                "work_source_max_difference": float(np.max(np.abs(stages["work"] - src))),
                "endpoint_fraction": float(np.mean((out == 0) | (out == 255)))}
        cv2.imwrite(str(args.out / (path.stem + "_comparison.png")),
                    np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
        cv2.imwrite(str(args.out / (path.stem + "_boundary_crops.png")),
                    np.hstack(crops))
        report["images"][path.name] = record
        print(path.name, record, flush=True)
    (args.out / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

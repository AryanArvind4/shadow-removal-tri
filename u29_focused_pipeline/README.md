# U29 focused V2: shape correction

Use `result/u29_focused_v2/`. **U29 focused V1 was rejected for visible pad
deformation.** Its source is retained as `rejected_v1.py` solely for diagnosis;
the old outputs remain available for the before/after comparison.

## What was wrong

V1 blended a bright, locally normalized pad rendering with a separately darkened
background. The blending mask depended on the pad rectangle and local brightness.
Weak/shadowed corners received different treatment than bright parts of the same
edge. This created cut-in edges and made pads look reshaped. Keeping pixel
coordinates unchanged was not sufficient to preserve their visible shapes.

The old synthetic checks covered dark marks but missed boundary deformation.
They are replaced by explicit edge, notch, and invertibility regression tests.

## What V2 does

1. Detect candidate pad positions from the source. Use them to sample pad and
   surrounding-background brightness, not to paint a pad into the output.
2. Interpolate broad, smooth background and gain fields across the whole image.
   These fields vary on the scale of pad spacing, with no jump at a pad boundary.
3. Apply limited smoothing/dot attenuation only outside each pad and a seven-pixel
   surrounding guard. Save the exact source-change array and assert zero change
   throughout the protected regions.
4. Apply the smooth brightness/contrast correction to all pixels and use one
   common tone curve. The curve is linear across its central range and has soft
   tails. There are no separate foreground/background display curves, foreground
   alpha mask, per-pad plane rendering, or within-pad row correction.
5. Compare source-selected edge profiles against V2, V5a, and rejected V1. Save
   native PNGs, enlarged crops, numeric intermediates, and reports.

This preserves the source much more closely, but source shadows can still make
some pad outlines appear irregular. It does not reconstruct ideal squares or
copy defect patterns from the competitor photograph. Background suppression is
less aggressive than V1. Exact competitor quality remains unachieved.

## Run

From `E:\Shadow_Removal_SDE`:

```powershell
.\.venv312\Scripts\python.exe u29_focused_pipeline\u29_focused.py
.\.venv312\Scripts\python.exe -m unittest discover -s u29_focused_pipeline -p test_u29_focused.py -v
.\.venv312\Scripts\python.exe u29_focused_pipeline\make_review.py --reference "C:\Users\User\OneDrive\Documents\TRI_NTU Docs\TRI Target Results\res1.jpg"
```

All eight U29 inputs use one parameter set. Existing V5a outputs are unchanged.

## Review order

1. `shape_comparison/U29_034.png`: original, rejected V1, corrected V2.
2. `pad_crops/U29_034_upper.png`: enlarged troublesome upper rows.
3. `input_output/`: matching originals and V2 results for all images.
4. `reference_gray/`: native-resolution final V2 images.
5. `meeting_review/`: all-image overview, target comparison and measurement notes.

`detail_gray` omits background filtering and is exactly equal to `reference_gray`
within the protected pad/edge regions. `edge_guard`, `background_change`, and
`gain_field` are diagnostics. `numeric` retains all source/correction arrays.

## Validation limits

Edge displacement measures half-height crossings along selected source edges.
It is a source-relative proxy with incomplete coverage of weak boundaries and
corners. It is not a full contour-accuracy guarantee. Photometric invertibility
is tested on floating-point intermediates; saved 8-bit PNGs have quantization.

Source-selected dark responses include possible texture and shadows. Their
contrast is not defect recall. Within-pad SD includes dark marks and cannot
be described as healthy-only uniformity. The competitor phone photo is unmatched
acquisition and is used only in a labeled comparison, never in image generation.

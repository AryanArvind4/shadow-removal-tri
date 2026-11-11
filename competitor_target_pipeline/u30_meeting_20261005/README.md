# U30 Meeting Study: 032-035 and 073-075

## What Was Run

Seven original JPEG files were copied byte-for-byte from the full TRI dataset
into `input/`. `input_manifest.json` records original hashes and image dimensions.
Originals and the selected V5a processing modules were not changed.

1. `result/v5a_baseline/`: unchanged selected V5a, all seven images.
2. `result/u30_focus_v1/`: first experimental U30-specific contrast view, all seven.

The experiment imports the frozen automatic core used by V5a. It does not fork
or overwrite those modules. It omits V5a's rectangular-pad correction and
repeated-dot attenuation, then applies bounded smooth local adjustments over the
whole image. No review boxes, pad masks, defect labels or filenames determine
the enhancement. The crop coordinates are used only AFTER processing.

## Review These First

- `result/u30_focus_v1/comparison_sheets/`: input | unchanged V5a | experiment.
- `result/u30_focus_v1/focus_crops/`: matched 2x nearest-neighbor review crops of
  all four plate regions for 073-075 and the target neighborhood for 032-035.
- `result/u30_focus_v1/review_locations/`: rectangles showing crop locations.
- Both runs' `reference_gray/`: native 768 x 700 lossless PNG outputs.
- Both runs' `input_output/`: two-panel input/output comparisons.

Row 7 / column 3 is provisionally interpreted as one-based counting from the
top-left of the pad grid, around native pixel (237, 533). The candidate crop is
`[205,500,270,565)` and the wider neighborhood is `[140,455,320,620)`.
These are NOT defect annotations. Confirm the intended site with the professor.

## How the Experiment Differs

The automatic V5 core supplies a denoised, broadly corrected float image before
display mapping. A wide Gaussian context estimate (sigma 45.5 at this image size)
describes slowly changing brightness. A lightly smoothed image minus that context
describes signed local detail: both darker and brighter features are retained.

Local gain is limited to 1.75 and reduced where contrast evidence is weak relative
to the noise estimate. Only the added detail term is smoothed; the core image's
fine detail is not replaced by that smoothed image. Added detail is bounded to
10 source-intensity levels, and a gradual midtone lift is bounded to 8 levels.
The same automatic-core tone parameters render the result. No pad is filled,
reshaped, forced to a template, or moved. This does not guarantee unchanged
perceived boundaries or faint-defect preservation: contrast can create halos and
the inherited denoising can affect weak features.

## Initial Assessment

V5a improves visibility over the input but also makes grain more prominent.
Its rectangular branch activates for 033-035 (69, 70 and 71 proposed pads), but
not for 032 or 073-075. This is an important difference across this small set.

The experiment raises visibility of some darker details and midtones, including
the provisional row-7/column-3 neighborhood. It does not cleanly separate every
plate-interior feature from texture. Omitting per-pad correction also means some
pads are less uniformly bright than in V5a. It is an exploratory alternative,
NOT a validated replacement for V5a or a completed solution to the requirement.

We cannot conclude that every dark shape is a defect. Nor can enhancement reliably
reconstruct content genuinely absent or obscured in the acquisition. The source
images contain some internal contrast worth studying; whether the four plate
regions physically obscure another layer requires acquisition information.

## Diagnostics, Not Defect Scores

`review_metrics.csv` measures the SAME crop locations in all three images:

- `p05_p95_span`: local intensity spread. Higher can mean stronger useful contrast
  or stronger unwanted texture.
- `midscale_rms`: strength of variations between two smoothing scales; includes
  structures, boundaries, noise and possible artifacts.
- `fine_residual_rms`: fine variation relative to a lightly smoothed image. It is
  not a calibrated noise estimate, and lower can also mean lost small details.
- `clipping_fraction`: near-black/near-white pixels; low values alone prove little.

For example, the provisional 032 target crop's p05-p95 span is 17 (input),
81 (V5a), and 112 (experiment). Its fine residual RMS also increases from
1.14 to 3.97 to 4.81. This is a contrast/texture tradeoff, not evidence that all
defects became easier to identify. No CNR, defect accuracy, or healthy-pad
uniformity claim is made without reviewed labels.

`numeric/` saves corrected values, context, gain, added detail and midtone lift.
`run.json` records processing source hashes, settings and crop coordinates.

## Reproduction

From the repository root with the existing Python environment:

```powershell
.\.venv312\Scripts\python.exe competitor_target_pipeline\u30_meeting_20261005\prepare_baseline.py
.\.venv312\Scripts\python.exe competitor_target_pipeline\u30_meeting_20261005\run_focused.py
.\.venv312\Scripts\python.exe -m unittest discover -s competitor_target_pipeline\u30_meeting_20261005 -p "test_*.py"
```

Preparation refuses existing study input/results. The experiment refuses an
existing output folder; provide `--output` with a new path for another run.
Keep input images and generated results local, as with the rest of the project.

## Next Review Decisions

1. Confirm the intended row-7/column-3 site using `review_locations`.
2. Compare the matched crops at equal scale. Identify which source-supported
   details are clearer, and which apparent dots are simply amplified texture.
3. Before stronger filtering, select a few small real features that must remain
   visible. Synthetic signed-feature tests cannot establish clinical/industrial
   defect preservation on these images.
4. Ask TRI whether adjacent files are reconstruction depths, repeated acquisitions
   or different boards before attempting alignment or averaging. No cross-image
   averaging is used in this study.

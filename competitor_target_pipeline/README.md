# Competitor-Target Pipeline

## Current improvement: V5a

V5a is the selected workflow. See the [repository quick start](../README.md)
for installation, custom input paths, BMP batches, output folders and limitations.
Datasets and generated results described below are local artifacts, not included
in the GitHub checkout. Historical sections describe earlier experiments, not
validated competitor-equivalence claims.

For new 8-bit grayscale BMPs (including G2/G3):

```powershell
.\.venv312\Scripts\python.exe competitor_target_pipeline\run_v5a_bmp_batch.py --input "path\to\bmp_images" --output "path\to\new_results"
```

The output folder must not already exist. This adapter checks the selected code
against `v5a_source_manifest.json`, converts BMP to lossless PNG without changing
pixels, and validates native output dimensions and unchanged originals. It does
not require historical image outputs to be present.

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v5a.py
```

Results: `result/competitor_target_v5a/`. The complete `competitor_target_v5_auto`
run is kept unchanged as the comparison baseline. No manual labels are required.

V5a estimates repeated rectangular layouts from four opposing edge responses,
regular spacing, and boundary support across rows and columns. It does not use
component filenames or a prescribed number of rows/columns. If the layout is
not supported, the V5 automatic result is retained exactly, rather than applying
rectangular-pad processing to a different kind of component.

Within supported pads it applies bounded brightness offsets and a mild robust
plane correction. Dark residual outliers have reduced influence on that fit;
this does NOT mean that the remaining pixels are verified healthy solder.
Correction fades outside each pad support. Rounded/irregular pad boundaries,
large defects and overlapped structures remain potential failure cases.

Small bright spots are attenuated only when away from guarded pad boundaries,
within the layout, and repeated at similar positions in at least four cells.
Attenuation is partial and bounded to eight pre-tone intensity levels. Repetition
is not proof of harmlessness: this is a display choice, not defect classification.
Inside guarded pads, the dot attenuation is exactly zero. Other corrections can
still affect pad pixels and may affect faint defects.

Review in this order:

1. `comparison_sheets`: input / V5 automatic / V5a, matching filenames.
2. `reference_gray`: native-resolution V5a output including dot attenuation.
3. `pad_only_gray`: same pad correction but NO dot attenuation, for comparison.
4. `input_output`: simple before/after sheets for presentation.
5. `selection_overlay`: green proposed pad boundaries, orange attenuated spots.

`pad_mask`, `dot_attenuation_mask`, and `pad_correction` are diagnostics.
`numeric` retains pre-tone arrays. `per_region_metrics.csv` measures the same
proposed interiors for input, V5 auto and V5a, including any defects: these are
not healthy-only uniformity or defect-accuracy scores. `run.json` records input
and code hashes, layout decisions, pad adjustments and attenuated-spot locations.

## Current experiment: automatic V5 (2026-09-15)

No manual labels are required to run this path. It does not inspect component
names, use a fixed grid, or fit against the competitor photographs. It is an
image-processing algorithm, not a trained neural network. Its assumptions
include 8-bit grayscale input and locally brighter solder features. Generalization
to other acquisitions, polarities and component types is not validated.

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v5.py
```

Current V5 results: `result/competitor_target_v5_auto/`. This is the only retained
V5 result run. The old `competitor_target_v5`, `competitor_target_v5_sample`, and
`v5_auto_preview` result folders were removed at the user's request. V4 remains.

The next improvement, **V5a**, is implemented above. Keep automatic V5 as its
comparison baseline. The new correction is selective; unsupported layouts
continue to use this automatic V5 result.

1. Estimate fine and correlated texture levels from the source image.
2. Use OpenCV non-local means (similar-patch denoising), blended with the source.
   Large signed differences receive less smoothing. Faint defects remain at risk.
3. Estimate feature size from compact bright contours at reduced resolution.
   These are unreviewed proposals, not confirmed solder or defect labels.
4. In dense small-feature scenes only, test row-profile agreement across six
   strips and apply bounded band correction when sufficient evidence exists.
   Genuine horizontal structure can also be affected; this is not artifact proof.
5. Estimate a smooth lower background envelope with morphological opening;
   subtract a bounded part of its variation from the native-resolution image.
6. Estimate slow local gain from tile brightness ranges. Low-evidence tiles
   receive no boost, gain stays below 1.6, and the extra change is bounded at 14
   source levels. No individual pad is painted or forced to a constant value.
7. Apply one continuous increasing grayscale curve, not binary thresholding.

Use `reference_gray` for the main native-resolution lossless PNGs, `input_output`
for paired presentation sheets, and `comparison_sheets` for diagnostic views.
`detail_gray` omits denoising and local gain but uses the same correction fields
and tone parameters; it is not an independent raw reference. `numeric` retains
pre-tone float results. `run.json` records settings, decisions and source hashes.
Masks and correction visualizations are diagnostics, not inspection outputs.

`per_region_metrics.csv` compares input/V4/V5 within the same automatic proposed
regions. These may contain defects, missed boundaries or non-solder structure.
Lower interior standard deviation alone does NOT establish better defect
visibility or healthy-pad uniformity. The automatic branch deliberately does
not generate confirmed-defect accuracy or claim parity with the competitor.

The optional annotation workbench below now targets `competitor_target_v5_auto`.
Annotations are for future evaluation, not a runtime prerequisite. Existing
review crop sheets from the earlier run are historical; run `prepare` again to
refresh them before using the workbench with the current automatic results.

## Earlier component-specific V5 (historical)

Its result folder has been deleted. Legacy source is retained for reproducibility
and existing tests, not as the active results workflow. The old algorithm used:

- J1/J4 use slowly varying local gain estimated from sufficiently strong regional
  contrast, capped at 1.3 with the additional source-intensity change capped at 6.
  This can increase visibility AND noise. Weak evidence leads to no gain estimate.
- U29 grid supports are refined using evidence on all four outer edges. Uncertain
  boundaries retain their prior ROI. Constant pad shifts use a provisional 70th
  percentile brightness estimate. Their fades occur outside pads so the shift
  does not introduce an interior gradient. This is not healthy-only estimation:
  healthy ground truth still requires human labels.
- U30 retains V4 processing pending source acquisition/depth clarification.
- Denoising still excludes selected solder regions. Misidentified or missed
  regions remain a risk; true defect preservation is not certified.

The review workflow prepares eight candidates and supports polygon annotation:

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\review_workbench.py prepare
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\review_workbench.py annotate
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\review_workbench.py evaluate
```

See `review/REVIEW_GUIDE.md` for instructions and `review/U30_SOURCE_FINDINGS.md`
for source findings and questions for TRI. No reviewed scores are generated from
pending annotations. The eight candidates were previously viewed and are not
an unseen holdout. Review labels are used for evaluation, not fed into V5 fitting.

## Baseline: V4 component-aware cleanup

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v4.py
```

Results: `competitor_target_pipeline/result/competitor_target_v4/`.
Start with `comparison_sheets`, then `reference_gray`. Each sheet compares the
same input filename with V3 and V4, and shows region and correction diagnostics.
The previous V3 run remains available as a baseline.

V4 addresses visible grain, remaining horizontal bands, and missed U29 pads:

- J1/J4 band estimation excludes proposed solder regions and their margins.
  It estimates row profiles in three strips per panel, interpolates within each
  panel, and limits correction to 8 source intensity levels. This does not prove
  that all row variation is an artifact; genuine horizontal structure can remain
  or be affected. U29/U30 do not receive row correction.
- U29 uses a 9-column, 8-row grid prior specific to this dataset. Grid origin and
  spacing are fitted to the current image response. A cell needs local brightness
  evidence before it becomes an inspection ROI. If fewer than 40 cells qualify,
  processing falls back to V3's contour proposals. These rectangles are inspection
  supports, not exact solder segmentation or defect boundaries. Review overlays.
- Background denoising uses a bounded bilateral correction, faded out around
  selected solder regions. The denoising delta is exactly zero inside selected
  regions and a four-pixel margin. Brightness and band corrections can still
  change those pixels. Unclassified edges receive less smoothing; missed low-
  contrast structures may still be smoothed. No universal preservation claim.
- Source detail stays at native resolution. V4 retains V3's gradual per-region
  brightness correction and global grayscale tone curve. It does not paint pads
  or darken a presumed void map.

Outputs are lossless PNG. `numeric/*.npz` contains the actual cleaned float
intensities before tone mapping and the region labels. `detail_gray` omits local
pad correction and background denoising, retaining broad and row correction.
`run.json` records input hashes, source hashes, grid decisions, and diagnostics.

`metrics.csv` compares input, V3 reference_gray, and V4 reference_gray using the
same V4 region proposals for all three. `background_high_frequency_rms` measures
fine variations outside guarded regions; lower can mean less grain OR lost fine
structure. `background_row_variation` measures variation in the detrended row
median outside these regions; lower can mean fewer bands OR suppressed real
structure. Neither is an independently validated noise/artifact measurement.
Per-region interior spread still includes defects and requires visual review.
V3 and V4 use the same tone function but can estimate different percentile ranges.

To run a small review batch in its own output folder:

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v4.py --files J1_007.jpg J4_049.jpg U29_034.jpg U30_085.jpg --output .\competitor_target_pipeline\result\competitor_target_v4_sample
```

Run all restoration tests:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s competitor_target_pipeline -p "test_competitor_target_v*.py"
```

The reviewed-ROI evaluator below also supports V4 by passing
`--results .\competitor_target_pipeline\result\competitor_target_v4`.

## Previous: V3 native-resolution grayscale correction

Run from the repository root:

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v3.py
```

Results are saved in `competitor_target_pipeline/result/competitor_target_v3/`.
Review `comparison_sheets` first, then `reference_gray`. `detail_gray` is an
ablation: the same display contrast and broad/row correction, without per-region
normalization or background smoothing. Final images use lossless PNG at the
input dimensions. `numeric` contains floating-point corrected intensities and
integer region labels before display mapping.

For a separate four-image smoke run:

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v3.py --sample --output .\competitor_target_pipeline\result\competitor_target_v3_sample
```

### What changed

- Estimate only the broad correction field at reduced resolution. Apply it to
  original pixels. J1/J4 are no longer processed at 700 pixels and enlarged.
- Bound broad correction to 15 intensity levels. J1/J4 use separate panel row
  profiles, with row corrections bounded to 5 levels. U29/U30 skip that step.
- Propose complete regions from several contour thresholds and shape checks.
  Fill interiors in the SUPPORT MASK only, retaining source intensities and holes
  in the actual image. These proposals are not verified solder segmentation.
- Fit a robust plane inside each region to estimate gradual brightness changes.
  Correct part of its slope and shift regional brightness by at most 10 levels.
  Regional corrections taper toward the boundary and total at most 15.2 levels.
- Smooth weak-detail background conservatively; protect proposed pads and strong
  unclassified structure. This can leave substantial unwanted texture visible.
- Use one increasing grayscale display curve. No thresholded defect map is
  subtracted, no cavity is painted, and no pad is filled with a constant intensity.

### Reading the diagnostics

Each comparison sheet shows input, V2, V3 detail_gray, V3 reference_gray,
region outlines, and background smoothing support. White in the smoothing map
means stronger smoothing, black means none. Green outlines need visual review:
missed regions and background false positives remain possible. The correction
visualization uses gray 128 for zero, scaled by 3; inspect NPZ for numeric values.

`per_region_metrics.csv` contains separate measurements for every proposed region:

- `interior_std`: standard deviation inside ONE eroded region, in 0-255 units.
- `interior_iqr`: middle-50-percent intensity spread in that same region.
- `input_correlation`: association with source intensities at the same pixels.
  High correlation is a useful change check, not proof of defect preservation.
- `clipping_fraction`: fraction at 0-1 or 254-255. The display curve deliberately
  reserves headroom, so low clipping alone does not establish image quality.

`metrics.csv` averages the individual interior standard deviations, unlike V2's
pooled statistic. Interior samples INCLUDE defects and normal structural detail;
they are not confirmed healthy solder pixels. Lower spread is not automatically
better. Compare detail_gray versus reference_gray at the same display mapping;
raw-versus-output spread is affected by the contrast increase itself.

No confirmed defect annotations or healthy-pad annotations were available for
this run. These statistics cannot establish defect accuracy, defect area retention,
or healthy-solder uniformity. Review matched native-resolution crops, annotate
healthy areas and defects, and confirm acquisition/slice compatibility with TRI
before claiming parity with the competitor. Large U30 interior cavities absent
from the source cannot be reconstructed by this enhancement.

Tests:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s competitor_target_pipeline -p test_competitor_target_v3.py
```

### Evaluating reviewed defects

`evaluate_reviewed_regions.py` accepts human-reviewed healthy and defect polygons
in native input pixel coordinates. Its JSON structure is an `images` list, each
entry containing `filename` and `regions`. Each region contains `id`,
`healthy_polygon` and `defect_polygon`; a polygon is a list of `[x, y]` points.
Choose healthy and defect areas within the same solder region. Do not label
every dark area as a defect. Polygons must not overlap.

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\evaluate_reviewed_regions.py --annotations .\reviewed_regions.json
```

This writes `reviewed_roi_metrics.csv` into the V3 result directory, measuring
healthy standard deviation, signed healthy-minus-defect contrast, and CNR
(absolute contrast divided by pooled within-ROI standard deviation). CNR is
blank when both ROIs have effectively zero variance. These are fixed-area
measurements, not automatic defect boundary or area validation. No reviewed
metrics are produced until actual annotations are supplied.

## Historical V2

The descriptions below apply only to V2; its defect maps are heuristic.

This folder contains the rebuilt competitor-target pipeline after the V1 failure.

The main lesson from V1 was:

```text
high contrast is not the same as solder/pad
```

V2 therefore uses a different strategy:

1. Estimate and flatten broad shadow/background.
2. Build candidate bright-feature maps.
3. Detect pad/solder regions using geometry rules.
4. Reject long seams, traces, and large background structures.
5. Preserve dark void/defect candidates inside detected pads.
6. Suppress the background outside pads.
7. Render inspection-grade grayscale contrast.

Run:

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\competitor_target_v2.py
```

Default input:

```text
TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset_Concised
```

Default output:

```text
competitor_target_pipeline/result/competitor_target_v2
```

Priority result folders:

```text
reference_gray
defect_preserving
pad_uniform_gray
inspection_grade
comparison_sheets
```

`reference_gray` is the main result to check first. It is designed to look like
the TRI competitor reference: dark suppressed background, bright solder regions,
and preserved dark defects/voids inside the solder.

Debug folders:

```text
pad_mask
void_mask
rejected_features
flattened_gray
background_field
```

# V5c review study: 19 November 2026

## Purpose and status

Experimental successor to V5b for the U29 boundary-void issue and more conservative
component selection. **V5b code, inputs, automatic results and targeted corrections
are retained unchanged.** This is an inspection-display experiment, not a validated
defect detector or a reconstruction of missing solder.

The study copies the 14 previously reviewed original inputs: U29_029 through
U29_036 (8), TRI_1008 U76 (3), TRI_new (2), and the G2 round-ball example (1).
There are no filename-specific enhancement profiles or hand-drawn component masks.

## What changed

1. Keep frozen V5a/V5b as the baseline and as a source of component proposals.
2. For grid pads, replace V5b's center-seeded watershed selection with a
   source-contrast selection. Fit a bounded local background plane from a ring
   outside the proposed pad. Strong source pixels seed the selection; connected
   weaker pixels can retain thin, visible rims.
3. Do not close dark gaps with mask morphology. Filling *enclosed mask holes*
   does not fill image pixels. Eight-connected background handling and the
   source-intensity mapping leave open, dark channels dark.
4. Apply a bounded positive intensity lift to source-supported component pixels.
   V5a supplies a brightness anchor, not the final per-pixel texture. The lift is
   zero at/below the estimated background and for pixels already above the white
   level. The scalar curve is monotone for a fixed background; the complete
   spatially weighted image is **not** guaranteed globally rank-preserving.
5. Blend continuously according to source contrast, rather than feathering only
   from a hard pad silhouette. All pixels outside the effective support remain
   byte-identical to the input. No pixels are moved, resized, or synthesized in
   the exported full-resolution result.
6. For non-grid candidates, require adequate surrounding background, compactness
   and solidity before local contrast validation. Feather these boundaries.
   Ambiguous or fragmented regions are left as input. This reduces unwanted
   background selection but also reduces coverage of irregular plates.

The observed V5b problem is consistent with selection and blending excluding a
thin edge rim. It is not evidence that every notch is a closed void. Some U29
features already have weak or open-looking boundaries in the original image.

For background on marker-based watershed, see the
[OpenCV tutorial](https://docs.opencv.org/4.13.0/d3/db4/tutorial_py_watershed.html).

## Review order

Start in `results_v5c/`:

1. `U29/comparison_sheets/U29_032.png`: input, saved automatic V5b, V5c.
2. `U29/review_crops/U29_032_pad_12.png`: row 2, column 3, the lower-edge dark
   feature that is cut into the boundary in V5b. Inspect at native scale too.
3. `U29/review_crops/U29_032_pad_14.png`: an intentionally retained difficult
   example; V5c does not establish a closed rim where the source is unclear.
4. All eight `U29/comparison_sheets/` images. Every proposed pad has a crop,
   not only favorable examples. Crop coordinates are recorded alongside them.
5. `TRI_1008/comparison_sheets/`, especially U76 z201 and its lower-background
   crop. Assess both reduced background alteration and missed plate enhancement.
6. `round_ball_review/comparison_sheets/` and `review_crops/` for regressions.
7. `TRI_new/comparison_sheets/` and `targeted_comparisons/`. These low-resolution,
   already strongly contrasted images may improve little. Do not claim equivalence
   to the separately curated V5b targeted outputs.

Per-dataset folders:

| Folder | Contents |
| --- | --- |
| `enhanced` | Full-resolution final PNGs, retaining original L/RGB mode |
| `reference_gray` | Full-resolution grayscale output |
| `input_output` | Two-panel input/V5c comparison |
| `comparison_sheets` | Input/saved automatic V5b/V5c comparison |
| `targeted_comparisons` | Separate comparison with existing manually targeted V5b, when available |
| `review_crops` | Equal-scale crops and source-coordinate records |
| `pixel_crops` | U29 row-2 crops enlarged 5x with nearest-neighbor pixel replication |
| `component_mask`, `blend_weight` | Effective enhancement support and blend strength |
| `selection_overlay` | Source with selected region outlines |
| `numeric`, `diagnostics` | Float weights, masks, per-region decisions and diagnostics |

Comparison sheets and enlarged crops use equal Lanczos resizing for display.
Only `enhanced` and `reference_gray` are native-resolution outputs. A component
mask is an enhancement-support estimate, not a defect or solder segmentation label.

## Verification and metrics

`run.json` records progress, source/code hashes, timings and decisions. A successful
run ends with `status: complete`. `metrics.csv` contains:

- Pixel agreement between the recomputed frozen V5b baseline and saved V5b output.
- Exact input preservation outside the V5c support, checked in exported PNGs.
- Selected and actually changed pixel fractions, mean absolute display change,
  and endpoint clipping fractions. These describe behavior, not inspection accuracy.
- Processing time split into baseline and refinement. Refinement includes the
  V5b comparison calculation; total export time includes image writing and crops.
  These are review-run timings, not an optimized deployment benchmark.

The runner verifies all retained V5b files and copied inputs by SHA-256 before and
after processing. Seventeen synthetic regression tests cover a near-edge void with
a visible rim, an open notch, a diagonal dark channel, translation, unchanged
background/protected RGB, no added grayscale darkening, weak candidates, generic
selection, constant inputs, scalar mapping and invalid input handling. Additional
cases check a blurred rim, sloped background, lower brightness anchor and full
protection. `audit.json` independently checks saved images, native dimensions,
unchanged outside-mask pixels, no added grayscale darkening and code hashes.

These checks do **not** prove that every real void is preserved or every background
pixel is correctly classified. Background equality is conditional on the selected
support: a wrongly selected background pixel can still be enhanced.

## Reproduce

From the repository root, using the existing environment:

```powershell
# Initial preparation only; refuses to overwrite the prepared study.
.\.venv312\Scripts\python.exe TRI_v5c_2026-11-19\prepare_study.py

# Focused tests only; do not use unrestricted root-level test discovery.
.\.venv312\Scripts\python.exe -m unittest discover -s TRI_v5c_2026-11-19/pipeline -p test_v5c.py -v

# Full review export; refuses to overwrite an existing output folder.
.\.venv312\Scripts\python.exe TRI_v5c_2026-11-19\pipeline\run_v5c.py

# Independently audit the completed default run and export pixel-replicated crops.
.\.venv312\Scripts\python.exe TRI_v5c_2026-11-19\pipeline\audit_results.py

# Optional new U29-only run.
.\.venv312\Scripts\python.exe TRI_v5c_2026-11-19\pipeline\run_v5c.py --dataset U29 --output TRI_v5c_2026-11-19/results_u29_repeat
```

Development-only `probe.py` and `preview.py` write to `development/` and may
overwrite their own previews. They are not the final export workflow. The new
pipeline depends on NumPy, OpenCV, SciPy and Pillow plus the frozen V5a dependencies
already installed in the repository's environment.

## Remaining limitations and next steps

- Review matching native input/output crops with the professor before replacing
  the established meeting results. Keep V5b and targeted outputs available.
- No closure is invented for unresolved rim/void boundaries. Matched acquisition
  depth or higher-quality source data may be needed to resolve those cases.
- Grid boxes are still inferred proposals. Conservative generic filtering can
  reject real irregular plates or components near image borders.
- Noise and source texture can still be amplified inside accepted components.
  Lower variance alone is not a valid goal if it removes void evidence.
- Next validation should label a small set of confirmed boundary voids, true
  notches and background regions, then measure void/rim visibility and false
  background enhancement on held-out images. Compare acquired data at matching
  depth before making competitor-equivalence claims.

Do not describe this candidate as recovering physical geometry, diagnosing voids,
or matching the competitor's acquisition. Competitor/source acquisition equivalence
is still unconfirmed.

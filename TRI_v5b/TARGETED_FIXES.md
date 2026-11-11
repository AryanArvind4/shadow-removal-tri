# V5b Targeted Corrections: 2026-10-18

Only three exact inputs are revised. The general V5b algorithm, original inputs,
original result folder, U29 outputs, and all other component results are unchanged.

## Review

Open `results_targeted/TRI_1008/comparison_sheets` for the U76 z201 correction,
and `results_targeted/TRI_new/comparison_sheets` for image001 and image002.
Panels are input / previous V5b / targeted V5b. Final full-resolution files are
in each dataset's `enhanced` folder. RGB input backgrounds and colored annotations
are retained there. `reference_gray` and comparison sheets are grayscale views.

Use these three revised files alongside the other accepted images in `results/`.

## What changed

- **U76 z201:** explicit component search envelopes restrict the previous support.
  False background selections below and beside the plates, and between the top
  components, return to their exact original pixels. Fully retained component
  pixels remain the previous V5b values. Four excluded background regions,
  including the whole image below row 565, are checked for exact equality.
- **image001:** source-based masks are completed within the visible pad grid's
  search windows. Mask closing and enclosed-hole filling operate on the support
  only. Pixel values still come from the frozen V5a result and source blending.
- **image002:** fuller supports are estimated within four visible textured plate
  envelopes, with a wider inner transition. The previous V5b RGB/grayscale results
  outside those four windows are retained exactly, including the smaller pads.

The screenshots already contain strong contrast, texture, and overlays. The
change is fuller coverage of existing V5a processing, not new reconstruction or
invented cavities. image001 has a modest visual improvement; image002 has stronger
internal contrast but retains substantial texture and uncertain detail.

## Scope and provenance

The manually chosen coordinates in `pipeline/targeted_profiles.json` are search
envelopes, not defect annotations. Each profile is locked to the exact source
SHA-256 and dimensions and is deliberately unsuitable for arbitrary new images.
This is an image-specific correction for the reviewed batch, not evidence of
improved generalization. Source contours still need visual review.

`selection_overlay` shows green support outlines and orange search envelopes.
The effective `envelope_mask` also includes retained previous supports for
image002. `numeric/refinement_alpha` describes this correction only; previous
V5b is passed through outside image002's four search windows.
No synthetic image generation, repainting, geometric warping, or image-pixel
hole filling is used. No additional intensity curve or denoising is introduced.

`results_targeted/run.json` records hashes, support coverage, background checks,
and verification that all previously generated V5b files remain unchanged.
Coverage measures are not defect accuracy or image-quality scores.

## Reproduce

```powershell
.\.venv312\Scripts\python.exe TRI_v5b\pipeline\targeted_fixes.py --output TRI_v5b\results_targeted_new
.\.venv312\Scripts\python.exe -m unittest discover -s TRI_v5b\pipeline -p "test_*.py" -v
```

An existing output folder is never overwritten. The script reads the previous
V5b and V5a outputs from `results/`; it does not reprocess other images.

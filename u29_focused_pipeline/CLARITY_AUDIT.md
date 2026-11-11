# U29 contrast and shape audit

## Status

Experimental comparison, not an accepted replacement pipeline. The user has
not confirmed that our inputs and the competitor reference have matching
board, acquisition, or reconstruction depth. Prior notes describing this as
a confirmed projection-versus-CT or different-specimen comparison should be
treated as hypotheses, not established facts. No competitor pixels are used
to create outputs.

## Findings

The clarity V3 output increases dark texture and bright rims markedly.
Disabling deconvolution, dome correction, band-pass enhancement, and unsharp
masking makes this less harsh, but the same source-linked bands and uneven
pad interiors remain. Brightness-only crops show dark interruptions already
present in the source. This does not establish whether they represent defects,
overlapping structures, or acquisition artifacts.

We must not paint ideal squares, fill dark regions, or copy voids from the
reference to force a cosmetic match. None of these comparisons establishes
defect preservation or competitor parity.

## Implemented

- Fixed `denoise_h=0` to bypass NLM completely, preserving floating-point input.
- Added three source-matched variants and a spatially independent tone control.
- Saved native-resolution boundary crops, parameters, and source hashes.
- Existing production results and V5a are not overwritten.

Run from the workspace:

```powershell
.venv312\Scripts\python.exe u29_focused_pipeline\audit_clarity.py --pattern 'U29_03[246].jpg'
.venv312\Scripts\python.exe -m unittest discover -s u29_focused_pipeline -p test_clarity_audit.py
```

Inspect `result/u29_clarity_audit/*_boundary_crops.png` first, then
`*_comparison.png`. The source panel is unmodified; the tone-only control is
explicitly labeled and uses a single affine brightness transform with clipping.
The conservative variant is a diagnostic candidate, not an approved final.

The next acceptance step needs matched source/target acquisition information
or reviewer confirmation of which dark pad structures are defects. Without
that information, aggressive suppression risks deleting the very evidence
the inspection view must retain.

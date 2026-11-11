# TRI V5b: Component Enhancement With Original Background

V5b reuses a byte-for-byte snapshot of the selected V5a processing code and
adds component-region selection and compositing. The original V5a modules in
`../competitor_target_pipeline` have not been edited.

## Results to review

For the three images revised after the October 18 feedback, see
[TARGETED_FIXES.md](TARGETED_FIXES.md) and `results_targeted/`. The original
results below remain unchanged; all other accepted images still use them.

Open `results/`, then select a dataset:

- `U29`: eight original U29_029 through U29_036 images from the earlier focused study.
- `TRI_1008`: three U76 BMP originals from TRI Images 1008.
- `TRI_new`: the two original image001/image002 screenshots from TRI_New_Image.
- `round_ball_review`: one additional earlier G2/G3 input for a round-ball halo check.
  The exact source of the user's photographed close-up is not confirmed.

Within each dataset:

1. `comparison_sheets`: input / unchanged V5a / V5b at matching scale.
2. `enhanced`: final PNGs at native resolution and original L/RGB mode.
3. `input_output`: simple grayscale before/after sheets.
4. `selection_overlay`: green outlines of estimated component supports.
5. `reference_gray`: grayscale V5b view; RGB inputs have a grayscale background here.

`component_mask`, `blend_weight`, `v5a_reference`, and `numeric` are diagnostics.
Use `enhanced` when retaining the exact source background, including RGB color,
matters. Comparison sheets use grayscale input for direct comparison with V5a.
`review_crops` contains enlarged, matching native-pixel crops for review.

## Processing changes

1. Run the frozen V5a algorithm with its existing defaults.
2. For supported pad grids, use V5a's proposals as markers for OpenCV watershed
   on source-image gradients. These are estimated supports, not exact solder labels.
3. Otherwise, propose compact source contours at two context scales. Smoothing
   and small mask closing help establish supports; they do not replace image pixels.
4. Blend V5a into these supports, fading within their inner edges. Pixels outside
   the supports are copied exactly from the input; no global output contrast curve
   or background blur is applied after compositing.
5. Near support edges, reduce only V5a's additional negative intensity change.
   Fully interior pixels retain V5a values. Existing source rims remain visible;
   they are not painted away. Dark interior cavities are not filled with brightness.
6. Protect conspicuously colored annotations in the two RGB screenshots and retain
   the source's original RGB pixels outside the enhancement support.

Background equality is verified outside the **estimated** mask. It does not prove
the mask has no false positives or omissions. Broad textured plates in the small
screenshots remain difficult; some features may retain their input appearance.
V5a's inherited denoising and all display changes can affect faint features. No
claim of defect preservation, exact segmentation, or competitor equivalence is made.

## Files and provenance

The user-requested folder moves placed `TRI Images 1008` and `TRI_New_Image` here,
including all their prior V5a results. Every moved file was hash-verified. Old run
logs retain their historical absolute paths. `U29_inputs` contains verified copies;
the original U29 dataset and previous U29 pipelines remain where they were.
`study_manifest.json` records source hashes, transfers, and explicit input selection,
so prior generated images are never accidentally processed as new inputs.

`pipeline/frozen_v5a/` is the unchanged local copy of the three selected V5a modules
and their source manifest. `pipeline/v5b.py` implements the new stage.
`results/run.json` records settings, source hashes, timings, masks and checks.
`verification.json` records independent checks of saved outputs and retained sources.
`development_previews/` contains superseded development runs; do not present them.

## Run from the repository root

```powershell
.\.venv312\Scripts\python.exe TRI_v5b\pipeline\run_v5b.py --output TRI_v5b\results_new
.\.venv312\Scripts\python.exe -m unittest discover -s TRI_v5b\pipeline -p "test_*.py" -v
```

The output directory must be new. Select a subset with `--dataset U29 TRI_1008
TRI_new`, or use `--files` with original basenames. Preparing/moving the inputs is
a one-time step already completed; do not rerun `prepare_study.py`.

## Review limits

The test suite checks exact outside-mask background retention, native source-rim
retention, signed interior feature behavior, translated mask selection, RGB handling,
and constant/invalid inputs. Synthetic tests do not certify real defect visibility.
The new version addresses added background grain and added boundary darkening;
removing a ring already present in the original would require a separately reviewed
change to source structure, which this version deliberately does not perform.

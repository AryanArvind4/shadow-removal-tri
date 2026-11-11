# U29 clarity pipeline (V3)

Use `result/u29_clarity_v3/`. V1 was rejected for pad deformation and V2 was
too timid to remove the shadow. V3 replaces both for U29. Their code and
outputs are untouched and still available for comparison.

## Run

From `E:\Shadow_Removal_SDE`:

```powershell
.\.venv312\Scripts\python.exe u29_focused_pipeline\u29_clarity.py
.\.venv312\Scripts\python.exe u29_focused_pipeline\u29_clarity.py --stack U29
.\.venv312\Scripts\python.exe u29_focused_pipeline\make_clarity_review.py --reference "C:\Users\User\OneDrive\Documents\TRI_NTU Docs\TRI Target Results\res1.jpg"
cd u29_focused_pipeline
..\.venv312\Scripts\python.exe -m unittest test_u29_clarity -v
```

All eight U29 inputs run on one parameter set.

## The full delivery is a height stack

`TRI_Dataset` and `TRI_Dataset_Concised` together hold 68 U29 frames, indices
0 to 67. They are not eight views of one thing plus spares. They are a
laminography height sweep, and different layers come into focus at different
indices:

| frames | what is in focus |
| --- | --- |
| 0 to 16 | package and die, bond pads, lead frame |
| 29 to 37 | the BGA solder balls |
| 48 to 67 | board side, plated vias with their annular rings |

`--stack U29` loads all 68, picks the solder plane automatically and averages
it. **Focus must be measured as gradient magnitude around the pad boundary,
not as pad-versus-gap brightness.** Brightness contrast peaks near index 25,
about eight slices below the sharpest plane, because slices nearer the die
carry more attenuation over the pads without resolving them any better.
Selecting on brightness picks frames 23 to 28 and the pads come out visibly
fragmented. Boundary sharpness picks 30 to 36, peak 33.

The eight concised frames, 29 to 36, are exactly the sharpest slices at the
solder plane, so the existing frame selection was already correct. Averaging
the plane buys a small gain, void contrast 98 to 100, and nothing more.

## What changed, and why

The detector image is an attenuation map. Solder is bright, a void inside a
pad is dark, and the board behind the array adds a slowly varying offset. The
signal is therefore additive, which means the right correction is a
subtraction and a rescale, not a blend of two renderings. Four stages:

1. **Pad lattice detection.** A centre-surround response is projected onto
   each axis and a periodic comb is fitted for spacing and phase, then each
   pad centre is refined locally. Detection returns 8 by 9 pads at a pitch
   near 73.7 by 73.3 px on every input.

2. **Two-field affine normalisation.** The board level `B` and the healthy
   solder level `P` are sampled on the lattice and interpolated into smooth
   fields, then `N = (I - B) / (P - B)`. Board level goes to about 0 and
   solder to about 1 everywhere, so the shadow disappears and every pad
   reads the same regardless of where it sits. Both fields vary on the scale
   of the pad pitch, so no pad boundary sits on a discontinuity. This is what
   V1 got wrong: it used a brightness-dependent mask, which cut pad edges.

3. **Board structure removal.** Pads are masked out and the remaining pixels
   are smoothed by normalised convolution, which interpolates the board level
   straight across each pad. Subtracting that removes traces, thermal vias
   and the die edge. What is left between pads is flattened further by a gain
   field derived only from detected pad geometry, never from pixel values.

4. **Deblurring.** Richardson-Lucy against a Gaussian point spread function,
   sigma 1.6, twenty iterations, applied before anything measures a level or
   an edge. The Gaussian model and the sigma are set by a direct measurement of
   the imaging blur; see **Measured PSF** and the sharpness section below.

5. **Display mapping.** A modest dome correction, a void-scale band pass
   confined to the pad body, and one monotone tone curve.

## Sharpness

The source is **blur limited, not noise limited**. Pad contrast is about 34
grey levels against 0.3 grey levels of noise, a signal to noise ratio near
113, so inverting the imaging blur is well conditioned. That is what makes
deconvolution the right tool here rather than more unsharp masking.

Measured on the averaged pad edge profile across all 72 pads:

| | 10 to 90 per cent rise |
| --- | --- |
| raw | 4.39 px |
| deblurred, sigma 1.1 | 3.95 px |
| deblurred, sigma 1.7 | 3.66 px |

The operating point is **sigma 1.6**, set to sit just below the measured 1.8 px
core (see **Measured PSF**). End to end it raises the pad boundary gradient by
23 per cent over no deblur, holds the pad edges in place (edge shift 0.25 px
median, 1.10 px at the 95th percentile), and lifts mark contrast to 1.66 times
median, 0.98 minimum. Void contrast on the plane average is unchanged; on a
single frame it eases by about one per cent, which is the brightening of the
real rim rather than a shallower void. Sigma 1.4 was the previous value and
remains a safe, slightly softer choice.

Two notes on reading the profile. The 20 per cent bright shoulder just inside
the pad edge is **the solder ball rim, present in the source**, not ringing;
deconvolution leaves it essentially unchanged, at 22.6 per cent. And per-pad
rise measurements are too noisy to judge this, they disagree with the
averaged profile; use the average.

### Measured PSF

The imaging blur was measured directly, from the pad edge-spread averaged over
all 72 pads (the only reliable edge measurement here; per-pad and single-row
rises are too noisy and disagree with it). Three findings, each with a
consequence for processing:

* The line-spread **core is a clean Gaussian, sigma about 1.8 px** (X 1.96,
  Y 1.78, a mild ten per cent anisotropy not worth correcting, like the warp).
  So a Gaussian PSF is the right deconvolution model, and the source really is
  blur limited. The pipeline deconvolves at 1.6, just under the measured core,
  because RL also sharpens the real rim and matching the full 1.8 rings it.
* The system is **not scatter limited**: only five to eight per cent of the
  line-spread energy lies beyond three pixels of the edge. There is no
  veiling-glare tail to invert, so a blind or heavy-tailed PSF buys nothing.
  That avenue is closed, not untried.
* The edge carries two real, non-Gaussian features a Gaussian PSF must not and
  cannot touch: the bright rim overshoot just inside the pad (the solder ball,
  in the source), and a broad dark moat just outside it, which is a
  limited-angle, thirty degree, laminography reconstruction halo and is already
  damped by the background gain. A Gaussian has no negative lobe, so RL neither
  creates nor removes the moat.

A sigma sweep confirms the trade and the trap. Raising sigma monotonically
raises a pad-edge gradient metric (182 at 1.4, 190 at 1.6, 205 at 2.0) with no
halo, but past 1.6 the rim visibly rings into a hard bright frame while the
gradient number keeps climbing. This is the same measurement trap that rewarded
out-of-plane subtraction: **look at the image, not only the number.** Sigma 1.6
is the ceiling that still sharpens without ringing.

This measurement also bounds what processing can achieve. The core blur is only
about 1.8 px and is now largely inverted, so the remaining gap to a clean
CT-slice interior is **not** a sharpness deficit that more deconvolution can
close. It is the projection integrating through the whole ball, an acquisition
limit. The one lever that moves it is the 128 raw projections from TRI, from
which the solder plane could be reconstructed with proper depth deblurring.

### Two sharpening routes that did not pay off

* **Per-pad focus stacking.** The package is genuinely warped: fitting a
  quadratic surface to each pad's best-focus slice gives corner heights from
  30.0 to 35.4, about five slices of tilt, with a 1.6 slice residual.
  Sampling the stack along that warped surface improves pad boundary
  sharpness by only 1.1 per cent, because the depth response is broad enough
  that five slices of tilt barely defocuses anything. Not worth the
  complexity. The focus map needs an orientation-selective measure to come
  out at all: plain gradient magnitude is captured by whichever layer happens
  to be strong above a given pad, and returns garbage.
* **More unsharp masking.** Deconvolution does the same job with a physical
  model behind it. A small unsharp pass is kept at 0.20 on top and adds about
  4 per cent more edge gradient.

## Measured results

Across all eight inputs, comparing V3 against the source:

| Quantity | Source | V3 |
| --- | --- | --- |
| Void contrast inside pads, grey levels | 20 | 98 to 107 |
| Background spread between pads, interquartile range | 29 to 32 | 7 |
| Shadow span removed, grey levels | 27 to 29 | 0 |

### Do not quantify against the competitor image

An earlier version of this file claimed V3 beat the reference on void
contrast, 124 against 101. **That claim is withdrawn.** The metric measures
how dark the darkest pixels in a pad get, which our broad mottled dark
regions inflate, and it says nothing about whether voids are resolved as
discrete features. Their pads show smooth interiors with distinct round
voids; ours show irregular mottling.

More generally, their reference is a phone photograph of a screen and cannot
support any quantitative claim about spatial detail. Measuring edge sharpness
against it gave contradictory answers, theirs 3.4 times sharper after
downsampling and 0.56 times at native resolution. Only comparisons against
our own source are reliable.

What does hold up, from their own capture dialog rather than from the photo:
they are viewing a CT slice, 128 projections at 30 degrees, 10 um. In a slice
the only things present are solder and air, so a void is a full contrast hole.
Our projection integrates through the whole ball, so a void is diluted by the
solder above and below it. Their sample is also a different physical part, so
void patterns must not be compared pad for pad.

## Regression checks

`test_u29_clarity.py`, twelve checks, all passing:

* Pad edges do not move. Half-height crossings are compared per side using
  each image's own local levels, because the pipeline deliberately changes
  both the plateau and the background, so a fixed grey level or a gradient
  extremum would appear to move even when the boundary does not. Median
  shift 0.25 px, 95th percentile 1.10 px. 47 per cent of edges are excluded
  because the source presents no clean step there, usually under heavy
  shadow or against a trace; that exclusion rate is a limit of the check.
* Background attenuation is exactly 1.0 over every pad out to the measured
  18 px half width.
* The void band pass weight reaches zero well inside the pad edge, so the
  boundary is never band-pass amplified.
* The tone curve is monotone across its whole domain, and inputs are clamped
  into the knot range rather than extrapolated, since a PCHIP end cubic can
  turn over outside its knots and invert contrast.
* Dark marks present in the source gain contrast. Median gain 1.66 times,
  minimum 0.98 times, measured relative to each image's own pad amplitude.
* Deblurring sharpens pad edges by at least 8 per cent, and does not brighten
  the gap beside a pad relative to the gap between pads, which is what a
  deconvolution halo would do.
* The height stack is complete at 68 contiguous frames, focus selection lands
  on the solder plane, and averaging it does not cost void contrast.
* The lattice resolves 8 by 9 on every input.
* Background uniformity improves on every input.

## What this does not do, and what to watch

* **Residual out-of-plane haze cannot be removed from reconstructed slices.**
  Our frames are already reconstructed by TRI, not raw projections, so each
  slice has the other layers baked into it as blur. Subtracting far slices
  was tried at several strengths. It raises measured in-pad detail from 0.24
  to 0.34 but the image gets visibly worse, because a far slice contains its
  own layer sharply in focus and subtracting it stamps that structure onto
  the pads as a negative imprint. What is needed is the contribution of each
  layer blurred by its depth offset, which the reconstructed stack does not
  give you. **Ask TRI for the 128 raw projections.** With those the solder
  plane can be reconstructed with proper depth deblurring, which is the only
  route to their pad interiors.
* **Depth resolution here is broad.** Pad boundary sharpness stays within 15
  per cent of its peak across about nine slices, so the reconstruction cannot
  separate nearby depths. That is consistent with a 30 degree angular range
  and is why haze from neighbouring layers survives into the solder plane.
* **There is no parallax between frames.** Regional phase correlation across
  the stack returns near zero shifts. These are reconstructed height slices,
  not angular views, so nothing can be recovered by registering them.
* **Per-pad equalisation hides pad-to-pad volume differences.** `pad_norm`
  defaults to 0.85, so most of the per-pad level difference is removed. That
  is what makes the array read evenly, and it also means a uniformly thin
  pad looks normal. The measured per-pad solder and board levels are written
  to `run.json` so that information is recoverable.
* **Background attenuation is a deliberate loss of information.** Contrast
  between pads is reduced to `bg_gain`, 0.30 by default. Strong excursions
  are exempted so a bridge or a stray ball keeps full contrast, but weak
  board detail in the gaps is suppressed on purpose. On these eight inputs
  no gap pixel reaches the exemption threshold, so the exemption is untested
  on real bridges and should be validated on a part that has one before the
  view is trusted for bridge detection.
* **Marks are measured relative to pad amplitude.** Because normalisation
  divides by the local pad amplitude, an equally deep mark on an unusually
  thick pad is rendered as a smaller fractional defect. That is intended,
  but it means absolute grey depth in the output is not a void volume.
* **Geometry is validated as a source-relative proxy.** The edge check
  covers resolvable edges only and is not a full contour-accuracy guarantee.
* **The competitor image is a different physical part.** Its void pattern
  differs from ours, so do not read pad-for-pad differences as a processing
  gap. It is used only for a labelled comparison and never in generating any
  output.

## Review order

1. `review/reference_side.png`, V3 next to the competitor photograph.
2. `review/overview.png`, all eight results.
3. `review/pads_U29_032.png`, magnified pad block, source against V3.
4. `review/compare_U29_032.png`, source, V2 and V3 side by side.
5. `review/metrics.csv` and `run.json` for the numbers.

## Tuning

`--pad-norm` per-pad equalisation, `--bg-gain` background suppression,
`--unsharp` edge crispness. Every other knob is in `Params` in
`u29_clarity.py`, with the constraint it has to satisfy written next to it.

# Component-Aware Enhancement of Industrial PCB X-ray Images for Visual Inspection

## Working Master's Thesis - Draft 0.2

Evidence update: 2026-11-24. Original repository audit: 2026-10-22.

Author: [TO CONFIRM: full official name]

University, department and degree: [TO CONFIRM]

Supervisor: [TO CONFIRM: official name/title]

Industrial collaboration: Test Research, Inc. (TRI), subject to confirmation of
acknowledgment wording and permission to publish the included images.

**Draft status.** This manuscript describes the retained V5a/V5b implementations
and the experimental V5c boundary-support extension
and available preliminary evidence. It is not a completed validation study.
Bracketed items identify missing information. Proposed experiments are separated
from completed work. The methods and result chapters are intended to evolve during
the remaining research period. University formatting and a complete bibliography
will be added after the required template is confirmed.

## Abstract

Industrial printed circuit board X-ray images can contain spatially varying
brightness, weak component contrast, overlapping structure and texture that make
visual inspection difficult. Increasing contrast throughout an image can improve
component visibility while also amplifying unwanted background variation. More
aggressive local processing can alter the apparent boundaries of solder regions.
These competing requirements motivate an inspection-oriented enhancement workflow
that controls where and how image intensities change.

This work develops a non-generative, component-aware enhancement framework using
industrial images supplied through collaboration with TRI. The current V5a method
combines guarded denoising, bounded broad-background and conditional band
correction, noise-limited local gain, and a continuous grayscale display mapping.
An evidence-gated branch estimates repeated rectangular layouts and applies
bounded regional correction. Unsupported layouts retain the automatic core result.
The V5b extension estimates component supports and blends the V5a result into the
source only within those supports, retaining original pixels outside them and
reducing additional darkening near support boundaries.

A subsequent V5c candidate replaces grid-support selection and edge blending with
source-contrast connectivity and a bounded positive lift of original intensities.
It addresses apparent merging of near-edge dark features with pad boundaries.
On a 14-image development cohort, selected U29 crops show more visible source rims,
while conservative non-grid selection reduces coverage of irregular plates and
weak balls. V5c is therefore reported as a focused candidate, not a universal
replacement for V5b. Synthetic tests and saved-file checks support narrow software
properties; they do not establish true-void geometry or diagnostic improvement.

Preliminary evidence includes a reproducible 32-image V5a development run across
four component families and saved verification of outside-support source retention
on 14 V5b images. Qualitative comparisons show stronger component visibility in
selected cases, alongside unresolved grain, support-selection errors and ambiguous
internal detail. Three manually guided corrections are reported separately from
automatic processing. The current evidence establishes implementation behavior and
motivates further investigation; it does not establish defect-detection accuracy,
universal defect preservation or competitor equivalence. The planned evaluation
uses independently reviewed regions, grouped unseen acquisitions, boundary analysis,
ablations and task-oriented assessment.

Keywords: PCB X-ray inspection; image enhancement; component-aware processing;
background retention; contrast; reproducibility.

## Chapter 1. Introduction

### 1.1 Industrial Motivation

X-ray inspection provides access to internal PCB assembly information that cannot
be assessed from an ordinary surface image. TRI develops automated X-ray inspection
equipment for PCB assemblies. This provides the industrial setting for the project,
but does not identify the scanner model or acquisition mode of the supplied data.
[Official TRI product context](https://www.tri.com.tw/en/product/product_detail-12-2-1462-21.html).

The project uses TRI-supplied images and iterative feedback communicated through
academic meetings. The practical requirements evolved from reducing shadow-like
variation to making solder regions and their internal structure easier to inspect.
Feedback emphasized clearer components, less distracting background texture,
more consistent presentation of pads, and retention of small dark features and
boundaries. Later feedback specifically requested keeping the background as it
appears in the input while enhancing components.

The collaboration informs the problem definition and selection of challenging
cases. The precise institutional arrangement, data ownership, acknowledgment and
publication permissions remain to be confirmed. Positive professor feedback is
treated as informal development feedback, not formal industrial acceptance or a
controlled usability study.

### 1.2 Problem Statement

Let `I` denote an exported grayscale image. The aim is to produce an inspection
view `Y` in which relevant components are easier to examine without introducing
misleading apparent structure or suppressing relevant evidence. This is not the
same objective as producing the most visually uniform or highest-contrast image.

A low standard deviation inside a pad may indicate reduced noise, but it may also
indicate that a real dark feature was removed. A sharper edge may be an actual
visibility improvement or an artificial bright rim. Background changes may make
normal texture resemble additional components. Therefore, quality must be assessed
using several complementary measurements and domain review.

The method operates on supplied image exports. It does not reconstruct a physical
volume, estimate calibrated solder thickness, or classify defects. The physical
origin of every dark or bright structure is not known. This distinction is
especially important when viewing overlapping layers or heavily processed exports.

### 1.3 Objectives and Research Questions

The current objectives are to improve component visibility with bounded changes,
retain native output dimensions, adapt processing to available geometric evidence,
and provide diagnostics that make failures inspectable. V5b adds the explicit
requirement of original-pixel retention outside its estimated enhancement support.

Three research questions organize the thesis:

1. Does bounded enhancement improve useful component/defect visibility compared
   with the input and simple baselines, without unacceptable changes to boundaries
   and faint features?
2. Can component-selective processing reduce unwanted background modification
   while retaining useful enhancement within components?
3. Under which component, noise and acquisition conditions does the method work,
   and how should unsupported or uncertain cases be handled?

These questions are not yet fully answered by the preliminary results.

### 1.4 Current Contributions and Scope

The implemented contribution is an inspection workflow integrating bounded
correction, evidence-gated layout handling, continuous display mapping and
source-supported selective compositing. The workflow includes source hashes,
diagnostic maps, native-resolution outputs and regression checks.

A second contribution is the documented analysis of failure modes, including
over-enhanced texture, apparent pad deformation and incorrect foreground supports.
These observations motivate evaluation criteria beyond global contrast or
uniformity. Their broader significance requires systematic testing.

NLM, morphology, robust fitting, watershed and blending are established techniques.
This thesis does not claim their invention. The research contribution must be
established through the application-specific formulation, constraints, integration
and comparative evidence, after a complete literature review.

## Chapter 2. Background and Related Work

### 2.1 Enhancement Versus Physical Restoration

For an ideal narrow monoenergetic beam, transmitted X-ray intensity follows an
exponential attenuation relationship. Exported industrial images may additionally
involve reconstruction, inversion, windowing and other processing. Consequently,
an additive brightness field or multiplicative illumination model used by an
enhancer should not automatically be interpreted as a physical material model.
[NIST attenuation reference](https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html).

The earlier repository uses the term "shadow removal" broadly. In this thesis,
"shadow-like variation" denotes observed spatial intensity variation without
assigning a verified cause. "Inspection enhancement" denotes modifying the display
to aid examination, not recovering a known clean image or restoring physically
occluded information.

### 2.2 Classical Correction and Denoising

Classical approaches in the repository include profile balancing, smooth background
subtraction/division, log-domain correction and local contrast normalization.
These can reduce broad variation, but their background estimates may absorb real
structure and their contrast operations can magnify noise.

Non-local means uses similarity between image neighborhoods to obtain a denoised
estimate. It is an established method used in the automatic core, not a new
algorithm proposed here. The project's guarded blend reduces reliance on that
estimate where the signed change is large; small-feature retention still requires
evaluation. [Buades, Coll and Morel, 2011](https://doi.org/10.5201/ipol.2011.bcm_nlm).

CLAHE provides a useful local-contrast baseline. Its inclusion in future comparisons
will test whether the additional components of the proposed workflow provide a
benefit beyond a standard contrast adjustment. The selected V5a implementation
does not use CLAHE as its main display curve.
[OpenCV histogram equalization documentation](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html).

### 2.3 Stack-Based and Learned Alternatives

The repository explores stack-derived fields and optional BaSiCPy processing.
BaSiC estimates background/shading using a sequence-based decomposition developed
for optical microscopy. Its use on TRI exports is exploratory; assumptions about
stable fields and changing image content require checking. Optional-library
fallbacks must be identified in comparisons.
[Peng et al., 2017](https://doi.org/10.1038/ncomms14836).

Historical learned experiments include intrinsic decomposition and SDEdit-style
processing. SDEdit uses a generative prior to edit an input by adding noise and
iteratively denoising. The ability to synthesize plausible images is not itself
evidence that an inspection feature has been preserved. The repository's learned
experiments are historical alternatives, not stages of V5a/V5b, and are not ranked
as inferior without a controlled reproduction.
[Meng et al.](https://arxiv.org/abs/2108.01073).

### 2.4 Quality Assessment

SSIM is an established reference-based image-quality measure, but similarity to
an input image is not the same as improved inspection performance. Without an
aligned clean target, SSIM/PSNR cannot establish recovery of a true clean image.
[Wang et al., 2004](https://eceweb.uwaterloo.ca/~z70wang/publications/ssim.html).

This project therefore separates image-domain integrity, source-relative change
checks, reviewed visibility measures and eventual task performance. Competitor
screen photographs are qualitative examples only. Their scene identity,
acquisition and reconstruction depth have not been confirmed to match the inputs.

[TO COMPLETE: systematic literature comparison of industrial X-ray enhancement,
task-based quality assessment, boundary fidelity and uncertainty-aware processing.]

## Chapter 3. Data and Development Context

### 3.1 Available Data

The local audit found the following primary collections. File counts do not
represent numbers of independent boards, specimens or defects.

| Collection | Count | Native dimensions / format |
| --- | ---: | --- |
| Full-named TRI folder | 487 | J1/J4: 2304 x 2148; U29: 760 x 672; U30: 768 x 700; grayscale exports |
| Concise development cohort | 32 | Eight images per J1, J4, U29 and U30, at the dimensions above |
| G2/G3 batch | 13 | Grayscale BMPs, approximately 4.4-5.5 megapixels, varying dimensions |
| TRI Images 1008 | 3 | U76 grayscale BMPs, 708 x 636 |
| TRI_New_Image originals | 2 | RGB screenshots, 373 x 282 and 405 x 320 |

The full-named and concise folders together contain 519 unique decoded exports.
Including G2/G3, the three U76 images and two screenshot originals gives 537 unique
decoded images across the audited TRI source collections. Study copies are
deduplicated. Four derived images stored beside the two screenshots are excluded.
These are inventory counts, not evidence that all files were evaluated with V5a/b.

Separate legacy material contains 85 files in clean-named training folders, 473
in noisy-named inference folders, and 112 solder-ball TIFFs. Their labels and
pairing require provenance review before use as ground truth.

### 3.2 Current Experimental Cohorts

The selected V5a run uses the 32-image concise cohort. The saved V5b study contains
eight U29 images, three U76 images, two screenshots and one copied G2 image for
rim review. The U29 and G2 copies are not new specimens. A separate U30 study
uses seven source images, indices 032-035 and 073-075, for focused review.

All these collections have contributed to development, visual assessment or
feedback. They are not claimed as a pristine held-out test set. Related frames,
copies and stack-fitting exposure must be accounted for in future split design.

### 3.3 Acquisition and Reference Limitations

The U30 source audit did not establish acquisition mode, reconstruction depth,
original bit depth or calibrated pixel size from the inspected JPEG metadata.
Historical notes make conflicting interpretations of projections and slices;
this manuscript treats those interpretations as unconfirmed.

Competitor photographs show desirable visual characteristics, including clearer
pad interiors and separated dark features, but cannot support pixel-level target
matching. An exact visual match is neither a reliable scientific target nor a
license to invent missing detail. Confirmation of matching specimen and acquisition
is required before a quantitative competitor comparison.

### 3.4 Development Lessons

The development history motivates three requirements. First, global correction
must not equate all low-frequency variation with unwanted shading. Second,
region processing must not make pad boundaries appear newly cut, squared or
displaced. Third, the ability to retain source background depends on selecting
the correct regions, not merely on implementing a mathematically exact blend.

The U29 focused V1 failure is particularly informative: visible shape changed
without geometric resampling. The later U76 support failure demonstrates that
copying pixels outside a mask does not help background mistakenly included inside
the mask. Both cases are retained as negative evidence.

## Chapter 4. Methodology

### 4.1 Input, Output and Design Principles

The selected processing core accepts finite grayscale intensities in `[0,255]`
with both dimensions at least 32 pixels. The standard export workflow uses
8-bit grayscale inputs. BMP adapters preserve decoded input pixels; the screenshot
adapter explicitly converts RGB to grayscale for enhancement. V5b can retain the
source RGB background in its final native-mode output.

The methods do not use a trained checkpoint. They do not move image coordinates
or render ideal pad templates. However, intensity corrections and denoising can
still change apparent shape and faint detail. Native dimensions are retained,
while some slowly varying fields are estimated at reduced resolution.

### 4.2 Automatic Core

**Noise estimate and guarded denoising.** The core estimates fine and correlated
variation using robust differences and a smoothed residual. These are operational
noise estimates, not calibrated detector-noise measurements. An OpenCV NLM image
`N` is blended with the input using a weight that decreases for large differences:

```text
D(x) = I(x) - w(x) [I(x) - N(x)]
w(x) = s / {1 + [|I(x)-N(x)| / max(1, 2.5*n)]^4}
```

Here `n` is the estimated noise scale and default `s = 0.85`. Large signed
departures are less smoothed, but faint defects may still resemble noise.

**Feature scale.** Compact bright contours at several thresholds estimate a
characteristic feature diameter. The estimate selects scales for later processing.
These contours are proposals, not confirmed solder labels.

**Conditional band correction.** For dense small-feature scenes, the method
examines row fluctuations in six vertical strips while excluding proposed bright
regions. A correction is used only when strip evidence meets the implemented
agreement threshold. Its magnitude is bounded at approximately six input levels.
True repeated horizontal structure can also satisfy the condition.

**Broad correction.** A morphological opening on a reduced-resolution analysis
image estimates a lower background envelope. After smoothing and resizing the
estimated field, a bounded adjustment reduces its deviation from a median anchor:

```text
C(x) = L * tanh[s_f * (b - B(x)) / L]
F(x) = D(x) + C(x) + C_band(x)
```

Defaults are `L = 28` input intensity levels and `s_f = 0.85`. This is a bounded
heuristic estimate, not a physical decomposition of solder and background.

**Limited local gain.** Tile statistics estimate positive signal relative to the
background and noise. Weak-evidence tiles receive no extra gain. Gain is capped
at 1.6 and reduced further by the noise estimate. The additional signed intensity
term is smoothly bounded to 14 levels. This favors moderate visibility correction
over unconstrained local equalization.

**Display mapping.** One continuous increasing sigmoid maps the corrected values
to grayscale, with headroom at both ends:

```text
T(z) = 8 + 239 / [1 + exp(-4.2*(z-b-0.40*S)/S)]
S = max(24, estimated signal span)
```

A monotone display curve avoids inversions introduced by that curve itself.
It does not prove that all spatially varying preceding corrections preserve
feature ordering or geometry. Uniform constant inputs have an identity special
case in the automatic core.

### 4.3 V5a: Evidence-Gated Pad Correction

V5a searches for repeated rectangular layouts using opposing edge responses and
approximately regular row/column spacing. It does not select a pipeline from
the filename or require a fixed 9-by-8 grid. Candidate pads require support on
four sides and repetition across the layout. When the criteria fail, V5a retains
the automatic core's reference image exactly.

Within supported pads, a robust plane estimates gradual intensity slope. Large
negative residuals receive less influence, reducing their effect on the fit.
The remaining samples are not thereby certified as healthy solder. A provisional
brightness level guides a bounded regional offset: the implemented shift is
limited to -6 through +18 levels, and the slope term to four levels. The correction
fades outside the proposed support.

Small bright inter-pad responses are partially attenuated only when they are
outside guarded pads and recur at similar positions in at least four layout cells.
The attenuation is bounded to eight pre-tone levels. Repetition is not proof that
a feature is harmless. A `pad_only_gray` output omits this step so its effect can
be inspected separately.

The final reference uses the automatic core's tone parameters. Masks, correction
fields, proposed regions and floating-point intermediates are retained for review.

### 4.4 V5b: Component-Selective Enhancement

V5b begins with the unchanged V5a output `E`. For supported grids, it uses source
intensity evidence and marker-based watershed to refine supports around proposals.
The watershed implementation follows established OpenCV functionality; it is not
a new segmentation primitive.
[Official watershed documentation](https://docs.opencv.org/4.12.0/d3/db4/tutorial_py_watershed.html).

For other layouts, compact contours are proposed at two context scales and filtered
using size, aspect, area and shape criteria. The method may miss weak components
or include real background. Closing/filling operations apply to the support mask,
not to image intensities; dark cavities are not filled with bright image pixels.

Let `M` denote the estimated binary support. Distance from its boundary defines
an inward feather weight `a` between zero and one. A boundary weight `e` reduces
V5a's extra negative intensity change near the support edge:

```text
R(x) = max(I(x) - E(x), 0) * e(x)
Y(x) = I(x) + a(x) * [E(x) + R(x) - I(x)]
Y(x) = I(x) explicitly wherever M(x) = 0
```

The default inner feather width is feature-scale dependent and clipped to
1.5-4 pixels. Where feathering is complete and the boundary safeguard is inactive,
the grayscale output equals V5a. Outside support, direct source copying provides
exact equality, including the final native-mode export.

This last statement is a conditional image-domain invariant, not proof of
segmentation correctness. A false-positive region can still enhance true background.
An omitted component remains unchanged even if it requires enhancement. An existing
dark source rim is not painted away; only added processing darkening is reduced.

For RGB screenshots, conspicuously colored annotations are excluded from support
and source RGB values are retained outside it. This is a heuristic for these inputs,
not general overlay detection or color restoration.

### 4.5 Separate Targeted Corrections

After review of automatic V5b, three exact inputs received manually selected
component search envelopes. U76 z201 had false selections restricted; the two
screenshots received more complete component supports in specified windows.
Profiles are locked to source hashes and dimensions.

These are assisted, image-specific repairs. They do not improve the automatic
algorithm for arbitrary new images and are excluded from claims of generalization.
They provide a useful diagnostic demonstration of the importance of support quality.

### 4.6 V5c: Source-Supported Boundary Enhancement

The November study investigates a specific V5b failure: an edge-adjacent dark
feature may appear joined to the exterior when support selection or inward
feathering excludes its thin surrounding rim. This is a hypothesis about display
behavior, not a classification of every notch as a closed void.

V5c retains frozen V5a/V5b for proposals and comparison. For grid candidates, a
robust, slope-bounded plane fitted to a surrounding ring estimates local background.
Strong positive source residuals seed a support; connected weaker residuals can
retain faint rims. There is no morphological closing across dark gaps.
Eight-connected background handling distinguishes enclosed support holes from
open channels. Filling a mask hole does not replace the dark image pixels inside it.

Within accepted regions, V5a supplies a brightness anchor, while a bounded positive
lift is applied to source intensities. Pixels at or below estimated background,
and pixels already above the white level, receive no lift. Continuous source-contrast
weights and an outer proposal taper limit transitions. For fixed background, the
scalar mapping is monotone; the final spatially varying composite is not guaranteed
to preserve global intensity ordering. Native exports neither warp nor synthesize
geometry. Apparent shapes can nevertheless change with intensity and must be reviewed.

For non-grid proposals, sufficient surrounding context, solidity and occupancy
are required before contrast validation. This more conservative acceptance can
reduce false background enhancement but reject true weak or irregular components.
Defaults are weak/strong contrast fractions 0.10/0.40, minimum contrast 3 levels,
noise multiplier 2.5, maximum plane slope parameter 6 and white level 245.
There are no per-filename profiles in V5c. The output explicitly equals the source
outside effective support and adds no grayscale darkening. Neither invariant proves
that semantic background is excluded or that all real defects retain visibility.

## Chapter 5. Evaluation Design and Available Evidence

### 5.1 Evidence Levels

Current evidence comprises saved runs, implementation invariants, regression tests,
source-relative statistics and qualitative review. The existing annotation file
contains eight candidates but zero completed reviews or annotations. Consequently,
this draft reports no confirmed-defect accuracy or reviewed healthy/defect CNR.

The next evaluation will separate healthy solder, confirmed defects, boundaries,
true background and ambiguous regions. Automatic processing will not require
these labels at runtime. Labels are needed to assess whether the processing is
beneficial and to prevent a metric from rewarding removal of real structure.

### 5.2 Metrics

Current proposed-region statistics include interior standard deviation/IQR, input
correlation and near-endpoint clipping fraction. These describe intensity behavior
within the same estimated interiors. They do not distinguish healthy solder from
defects or shadows and are influenced by display scaling.

The planned reviewed-ROI measures include signed healthy-minus-defect contrast
and a pooled-variability CNR proxy:

```text
contrast = mean(H) - mean(D)
CNR = |mean(H)-mean(D)| / sqrt[(var(H)+var(D))/2]
```

The denominator is within-ROI variability, not necessarily detector noise; a zero
denominator is undefined. Independent background labels will evaluate background
changes and false selections. Reviewed boundaries and source-relative edge profiles
will evaluate apparent deformation, with ambiguous-edge exclusions reported.

### 5.3 Comparison Protocol

[PLANNED] Compare input, simple window/level contrast, CLAHE, a classical field
correction, the automatic V5 core, V5a, V5b and V5c on the same images. Ablate denoising,
gain, layout correction, dot attenuation and V5b boundary handling. Fix ROIs across
methods and disclose whether tone parameters are held constant.

Split future data by independent board/acquisition groups. Do not place adjacent
frames or study copies in separate splits. Report paired per-group results,
uncertainty and failure rates, not significance based on pixel count. Where feasible,
include blinded expert localization/confidence or a controlled downstream detector
task. Manual targeted corrections remain a separately labeled condition.

## Chapter 6. Preliminary Results

### 6.1 V5a Processing Coverage

| Component family | Images in selected run | Rectangular branch used |
| --- | ---: | ---: |
| J1 | 8 | 0 |
| J4 | 8 | 0 |
| U29 | 8 | 8 |
| U30 | 8 | 0 |

Source: [selected V5a run manifest](../competitor_target_pipeline/result/competitor_target_v5a/run.json).
This behavior shows conditional execution, not proof that the gate always makes
the correct decision. A separate U30 study contains inputs on which the branch
does activate, so the table should not be generalized to every U30 image.

### 6.2 Proposed-Region Statistics

The following table aggregates 576 proposed regions across the eight U29 development
images. Mean interior statistics average the region measurements. Across-pad level
spread is the mean, over images, of the SD of their proposed-region means.

| Diagnostic, 0-255 intensity units unless noted | Input | Automatic V5 | V5a |
| --- | ---: | ---: | ---: |
| Mean within-region SD | 7.55 | 19.69 | 18.97 |
| Mean within-region IQR | 10.23 | 26.97 | 25.29 |
| Across-pad level spread | 10.53 | 35.20 | 18.82 |
| Mean input correlation | 1.000 | 0.954 | 0.940 |
| Mean near-endpoint clipping fraction | 0.000 | 0.000 | 0.000 |

Source: [saved region CSV](../competitor_target_pipeline/result/competitor_target_v5a/per_region_metrics.csv);
aggregation is reproducible in `thesis/tools/audit_repository.py`.

V5a reduces the across-pad level spread relative to the automatic baseline on
these supports. It also changes source correlation and preserves considerably
more intensity spread than the low-contrast input. This table cannot establish
healthy-only uniformity or defect preservation. In particular, the low input SD
does not imply superior inspection quality, and the low clipping fraction is
partly a consequence of display headroom.

### 6.3 Qualitative V5a Results

The following are illustrative development cases, not an unbiased selection or
independent proof of superiority. Panels use matching filenames. Presentation
sheets are resized for viewing; native output PNGs are retained separately.

![Input and V5a J1 comparison](../competitor_target_pipeline/result/competitor_target_v5a/input_output/J1_007.png)

**Figure 6.1. J1_007.** Ring-like structures become more conspicuous after
enhancement, but middle-region visibility remains uneven and background texture
is amplified. This motivates separating useful contrast from grain amplification.

![Input and V5a J4 comparison](../competitor_target_pipeline/result/competitor_target_v5a/input_output/J4_048.png)

**Figure 6.2. J4_048.** Stronger feature contrast is visible in the enhanced view.
Repeated horizontal variation and regional differences remain. The display is
not evidence that each small feature has been identified as healthy or defective.

![Input and V5a U29 comparison](../competitor_target_pipeline/result/competitor_target_v5a/input_output/U29_034.png)

**Figure 6.3. U29_034.** Pad visibility improves, but irregular apparent boundaries,
upper-row interference and uncertain interior texture remain. The method does
not reconstruct ideal squares or prove that all observed dark regions are voids.

![Input and V5a U30 comparison](../competitor_target_pipeline/result/competitor_target_v5a/input_output/U30_085.png)

**Figure 6.4. U30_085.** Bright small components become clearer; plate interiors
remain ambiguous and textured. This is not competitor-equivalent cavity separation.

### 6.4 V5b Background Retention

The saved verification report records successful source/dimension checks and exact
outside-support background equality for all 14 V5b images. Fully weighted interior
pixels outside the boundary safeguard retain their corresponding V5a values.
These checks assess implementation behavior, not correctness of the selected
support or preservation of every real defect.

![Input, V5a and V5b U29 comparison](../TRI_v5b/results/U29/comparison_sheets/U29_034.png)

**Figure 6.5. U29_034, input / V5a / V5b.** Selective blending retains a background
appearance closer to the input while applying stronger display contrast to selected
pads. Apparent boundary differences still need independent review. No pixel-coordinate
warp is applied, but this alone is not a geometric fidelity guarantee.

### 6.5 False Support and Assisted Repair

![Input, automatic V5b and targeted V5b U76 comparison](../TRI_v5b/results_targeted/TRI_1008/comparison_sheets/Component_U76_1_2048_708_636_201_z201.png)

**Figure 6.6. U76 z201, input / automatic V5b / targeted V5b.** Incorrect automatic
supports enhanced bottom and other background regions. Manually specified component
search envelopes restore excluded locations to source pixels while retaining allowed
component enhancement. The assisted output is a failure-analysis case, not an
automatic held-out result.

The two small screenshots also exposed support omissions: automatic outputs could
remain close to their inputs. Fuller manually guided supports improve coverage,
but the first screenshot changes only modestly and both retain substantial texture.
This is a limitation of current source quality and support selection, not a reason
to synthesize competitor-like details.

### 6.6 Runtime and Reproducibility

The saved September 18 V5a benchmark reports the following processing times for
an already-loaded float32 image, including returned diagnostics but excluding exports.

| Family | Images | Mean processing time per image |
| --- | ---: | ---: |
| J1 | 8 | 16.80 s |
| J4 | 8 | 18.15 s |
| U29 | 8 | 7.96 s |
| U30 | 8 | 3.14 s |

Total recorded processing time is 368.40 s; the full batch including reading,
processing, images, diagnostics and exports is 612.47 s, or 19.14 s/image. The
batch timer excludes Python startup/imports and pre/post verification. The saved
report records pixel equality for all 32 selected reference outputs.

The report lists an Intel Core 7 240H, 16 GiB RAM and an NVIDIA GeForce RTX 5050
Laptop GPU. V5a uses the CPU implementation, not CUDA acceleration. These are
historical recorded hardware fields, not a new hardware measurement by this audit.
The run used Python 3.12.10, OpenCV 4.10.0, NumPy 1.26.4 and SciPy 1.12.0. It was
a single sequential pass without dedicated warm-up, not a latency guarantee.

The separate V5b run records 67.45 s for 14 different images, including its exports.
It cannot be compared directly with the V5a total to claim a speedup. A common-cohort,
repeated timing experiment is planned.

The October 22 audit passes 33 V3-V5a, 13 V5b, 22 U29-experiment and five U30-study
tests. These 73 checks establish selected software properties, not industrial
inspection accuracy. The evidence register identifies the saved logs and manifests.

### 6.7 V5c Development Study and Boundary Trade-Off

The November study reuses 14 previously reviewed inputs, not a new held-out set.
All eight U29 images are U29_029 through U29_036. Results for the complete cohort
are summarized below; proposal acceptance is not measured component accuracy.

| Inputs | Count | Accepted / proposed regions per image | Interpretation |
| --- | ---: | --- | --- |
| U29_029-036 | 8 | 72 / 72 each | Selected boundary crops improve; ambiguous examples remain |
| U76 z1 | 1 | 23 / 91 | Conservative selection omits many candidates |
| U76 z141 | 1 | 69 / 69 | Grid branch; some internal detail is more conspicuous |
| U76 z201 | 1 | 7 / 26 | Less lower-background alteration; irregular plates under-enhanced |
| TRI_new image001 | 1 | 5 / 121 | Little change; retain separate targeted V5b result |
| TRI_new image002 | 1 | 1 / 145 | Little change; retain separate targeted V5b result |
| G2 round-ball case | 1 | 371 / 590 | Weak/irregular ball coverage regresses relative to V5b |

![U29 boundary comparison](../TRI_v5c_2026-11-19/results_v5c/U29/pixel_crops/U29_032_pad_12.png)

**Figure 6.7. U29_032, row 2, column 3: input / saved automatic V5b / V5c.**
Each source pixel is replicated 5x, without smoothing. The lower-edge dark feature
is more clearly separated from the outline in this inspected crop. This does not
establish its physical classification or a measured defect-detection improvement.

![Unresolved U29 boundary](../TRI_v5c_2026-11-19/results_v5c/U29/pixel_crops/U29_032_pad_14.png)

**Figure 6.8. U29_032, row 2, column 5, at the same display convention.**
The difficult feature remains ambiguous. V5c does not establish a closed rim where
the input evidence is insufficient; reporting this case avoids claiming every
apparent notch has been repaired. Full-image and all-pad comparisons are retained.

In U76 z201, the inspected lower background remains closer to source than automatic
V5b, but plate coverage is reduced. The earlier manually targeted V5b already
addressed that background issue and is a distinct comparator. Round-ball and
TRI_new examples likewise prevent a claim of universal improvement. The current
recommendation is U29-focused review of V5c while retaining V5b alternatives.

The saved export reports exact recomputed frozen-V5b agreement on all 14 inputs.
The November 24 read-only recheck verifies input and retained-baseline snapshots,
native dimensions, exact source pixels outside effective support, and no added
grayscale darkening in all 14 saved outputs. Seventeen scoped synthetic V5c tests
pass, including thin/blurred rims, open notches and diagonal dark channels. These
are additional software checks, not real-defect sensitivity measurements. The
October 73-test logs remain historical and are not represented as newly rerun.
The complete recorded V5c review export took 358.0 seconds, including comparison
sheets, crops and diagnostics; it is not an isolated processing benchmark.

## Chapter 7. Discussion

### 7.1 What the Current Results Support

The project demonstrates an implemented, inspectable enhancement workflow with
native-resolution exports, conditional regional correction and a component-selective
extension. Available outputs support a qualitative finding that relevant structures
can become more conspicuous. Saved verification supports a narrower exact claim:
V5b retains source pixels outside its estimated support.

The methods respond to the changing inspection objective. Global flattening was
not enough when important regions remained faint. Increasing whole-image contrast
made background texture more distracting. Selective enhancement addresses that
tradeoff but introduces dependence on region selection. This progression provides
a coherent research narrative rather than a collection of unrelated filters.

### 7.2 What Remains Unproven

No current reviewed study establishes improved defect detection, complete retention
of small flaws, or stable perceived boundaries for all components. Output correlation,
low clipping and exact outside-mask pixels are insufficient for those claims.
Likewise, positive informal feedback does not replace blinded task assessment.

The dataset may contain related frames and repeated development exposure. Without
board/acquisition grouping, uncertainty estimates would risk treating correlated
observations as independent. The automatically inferred masks are not ground truth;
evaluating background only outside those masks would ignore precisely the false
positives that produced the U76 artifacts.

### 7.3 Risks of Uniformity and Suppression

Increasing pad uniformity can hide real differences in thickness, shape or solder
volume. Dark/bright features cannot be classified from sign alone. Repeated-dot
attenuation is a display decision, not evidence that every attenuated object is
unimportant. Denoising can suppress weak details even when large signed residuals
are protected. Boundary safeguards can change perceived edge appearance and do
not justify removing every native dark rim.

The enhanced view should therefore remain linked to its source and diagnostics.
It should not silently replace the original for quantitative material measurement.
Future integration should make it easy to inspect the input, result and changed
regions together and to flag cases where support evidence is uncertain.

### 7.4 Industrial and Scientific Implications

The collaboration gives access to practical requirements and difficult data, but
the thesis must distinguish feedback-driven engineering from validated performance.
Formalizing constraints, reporting negative results and testing independent cases
can turn the current system into a defensible applied imaging contribution.

The next method change should address a measured failure, such as unreliable
support for textured plates or low-confidence boundaries. Adding a neural network
is not required merely to make the project a thesis. Any learned extension would
need appropriate training data, independent evaluation and safeguards against
introducing unsupported inspection content.

## Chapter 8. Conclusion and Planned Work

The current project has progressed from broad shadow-removal experiments to a
controlled inspection-enhancement framework. V5a is the current automatic reference
system; V5b adds component-selective application with explicit source retention
outside estimated supports. Preliminary results and verification provide a sound
basis for drafting the thesis, while documented failures reveal the limits of
contrast-based optimization and uncertain segmentation.

V5c extends this investigation to source-supported boundary handling. It improves
some inspected U29 edge presentations without establishing every void's geometry,
and its conservative selection reduces coverage elsewhere. Its mixed results are
part of the thesis evidence, not grounds to replace all retained V5b outputs.

The next stage will prioritize dataset provenance, a small reviewed evaluation
set, matched baselines and ablations, boundary/halo analysis, and independent
TRI acquisitions. Task-oriented assessment and repeated runtime measurements will
strengthen the final conclusions. Manually guided corrections will remain separate
from automatic results. Future validated improvements will update Chapters 4-7
without changing the core research questions or rewriting the thesis from scratch.

## References and Reproducibility Appendices

The [starter bibliography](REFERENCES.md) contains verified primary sources.
The [evidence register](EVIDENCE_REGISTER.md) maps claims to files and limitations.
The [evaluation plan](EVALUATION_PLAN.md) specifies proposed studies and the one-year
roadmap. The [repository review](REPOSITORY_REVIEW.md) documents historical branches,
data counts and contradictory older claims. Full per-image data and test logs are
in `thesis/evidence/`; industrial data should remain local until publication
permissions are confirmed.

[TO COMPLETE BEFORE SUBMISSION: official formatting, complete literature review,
reviewed experiments, independent test results, publication permissions,
authorship/contribution statements, acknowledgments and final abstract.]

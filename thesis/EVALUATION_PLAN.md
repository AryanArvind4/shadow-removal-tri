# Evaluation and One-Year Research Plan

All experiments in this document are proposed unless linked to an existing run.
The exact schedule and acceptance thresholds require supervisor agreement.

## 1. Central Questions

**RQ1: Useful visibility.** Does V5a improve reviewed component/defect visibility
over the input and simple contrast/flat-field baselines, without unacceptable
boundary displacement or loss of faint features?

**RQ2: Selective enhancement.** Does V5b reduce changes to independently identified
background while retaining V5a's useful component contrast? What errors arise from
missed components and false background selections?

**RQ3: Robustness.** Which geometry, noise, resolution and acquisition conditions
support reliable operation, and can unsupported cases be recognized and reported?

The focus is a measurable tradeoff, not maximizing a single visual-quality score.

## 2. Establish Data Identity Before Splitting

Ask TRI for anonymized board/specimen IDs, acquisition-session IDs, stack or view
relationships, pixel size, bit depth, polarity and prior processing. Confirm whether
each export is a projection, slice or processed display. These are currently unknown
for important collections.

Record filename, file hash, decoded-pixel hash, dimensions, component geometry,
export type and parent group. Put duplicates, crops, recompressed copies and related
frames from the same board/stack in the same split. Pixel hashing catches exact
copies, not near-duplicate frames; add near-duplicate review where needed.

Treat the concise set, current focused studies, G2/G3 and the feedback-driven new
batches as development/retrospective evidence because they have already been
inspected or used for changes. Do not retroactively relabel them as pristine tests.
The full archive may have been used for stack fitting, so its unused-looking files
are not automatically unseen either.

Obtain a genuinely new test batch where possible. Split by board/acquisition group,
not by random pixels or neighboring frame numbers. Tune only on development data,
freeze parameters and selection rules, then evaluate once on the held-out groups.
Report per-component and per-acquisition results, with sample counts at each level.

## 3. A Small, Practical Review Set

Propose an initial 12-20-image pilot spanning at least the four established
component families plus the new geometries. This is a workload proposal, not a
power-calculated minimum. A few regions per image can test whether the evaluation
procedure is practical before larger annotation work.

Ask a domain reviewer to mark:

- Healthy solder interior, excluding edges and ambiguous shadows.
- Confirmed defects, with their matching local healthy reference and confidence.
- Relevant boundaries, real notches and small features that must remain visible.
- True background where enhancement is undesirable.
- Ambiguous structures, kept separate rather than forced into healthy/defect labels.

Annotation is needed for evaluation, not for normal V5a/V5b operation. Reuse the
existing review workbench, but validate its schemas against the chosen evaluator.
The standalone paired-polygon evaluator and workbench annotation format are not
identical. Record reviewer identity/date and agreement; have a second reviewer
adjudicate difficult cases when feasible.

## 4. Baselines and Ablations

Compare on identical input pixels and fixed reviewed ROIs:

| Method | Purpose |
| --- | --- |
| Unmodified input | Essential reference; detects enhancement-induced harm |
| Global linear/window-level contrast | Tests whether simple display adjustment is sufficient |
| CLAHE with declared settings | Standard local-contrast control; assess grain and halos |
| Conservative additive/divisive background correction | Tests the original shadow-flattening idea |
| One reproducible historical classical method | Places development history on a common benchmark |
| Automatic V5 core | Separates layout correction from general enhancement |
| V5a without dot attenuation (`pad_only_gray`) | Existing diagnostic isolating that display choice |
| Full V5a | Main reference method |
| V5b with the same V5a intermediate | Isolates selective compositing |
| V5b without edge-darkening safeguard | Tests that safeguard, with all other settings fixed |

Further planned ablations: turn off NLM, band correction, local gain or pad-plane
correction one at a time while explicitly deciding whether display parameters
are fixed. Report both fixed-tone mechanistic comparisons and end-to-end method
comparisons if necessary. V5's `detail_gray` changes more than one stage; it is
not a pure denoising-only ablation.

Do not include learned or library-based historical outputs in a ranking until
their configuration, actual backend, training exposure and preprocessing are
reproducible. Label the wavelet/Fourier branch as an approximation if that is what
the code implements. Manual targeted V5b is a separate assisted mode, never pooled
with automatic V5b.

## 5. Metrics and Their Limits

| Measure | Definition or protocol | Interpretation and limitation |
| --- | --- | --- |
| Signed local contrast | `mean(healthy) - mean(defect)` on fixed reviewed ROIs | Useful for the assumed bright-solder/dark-defect polarity; does not measure shape or detectability alone |
| ROI CNR proxy | `abs(mu_H-mu_D) / sqrt((var_H+var_D)/2)` | Matches the existing paired-ROI evaluator. ROI variability includes texture; denominator zero is undefined, not infinite quality |
| Healthy interior spread | SD/IQR in reviewed healthy areas | Use alongside contrast and feature retention; making all pixels constant is not a valid improvement |
| Across-pad level spread | Dispersion of reviewed healthy pad levels within an image | Equalization may hide real solder-thickness/volume differences; retain original measurements |
| True-background change | MAE, maximum change and changed-pixel fraction on independent background labels | Evaluates real background, unlike the method's own support complement |
| Outside-support equality | Exact pixel comparison where V5b support is zero | Software invariant only; does not validate the support mask |
| Support quality | IoU/coverage, component misses, false-positive background area | Requires reviewed component masks; report small components separately |
| Boundary fidelity | Reviewed contour distance and source-relative edge-profile displacement | Report missing/ambiguous edge rate and notches; no resampling does not guarantee unchanged apparent geometry |
| Added dark rim | Compare source/output intensity profiles across reviewed edges and nearby background | Distinguish a real source rim from additional processing darkening |
| Fine-detail/clipping checks | Matched small-feature review, residual maps, near-endpoint fractions | Residual energy includes genuine structure; low clipping is not sufficient |
| Observer task | Blinded paired review, defect localization or confidence with expert reference | Directly addresses usefulness; preference alone is weaker than task performance |
| Detector task (optional) | Frozen detector, consistent train/test exposure and operating point | Report false positives, sensitivity/precision and threshold protocol; not just enhanced-image quality |

PSNR/SSIM are appropriate only when an aligned, meaningful clean reference exists
or for a carefully labeled synthetic experiment. Against the degraded input they
measure similarity, not restoration quality. Competitor screen photographs are
not reference targets. Do not optimize global entropy or edge energy as substitutes
for inspection performance.

## 6. Statistical and Visual Reporting

Use paired comparisons on the same images/ROIs. Report per-image and per-board
outcomes, effect sizes, median/IQR or mean/SD as appropriate, and confidence
intervals using the independent board/acquisition group as the resampling unit.
Thousands of pixels or 576 pads do not automatically create 576 independent trials.
Choose the final statistical test after learning the group count and distribution.

Predefine primary endpoints and acceptable boundary/defect-loss tolerances with
the professor/TRI. The pilot can inform sample-size planning; do not select a
tolerance after seeing the test outputs. Report exclusions and failed processing,
not only completed images. If group identity cannot be recovered, label the study
exploratory and limit inferential claims.

Show typical, difficult and failed cases at equal scale. Retain native PNGs and
float intermediates. Do not auto-normalize each crop independently without labeling
that choice. Include original, enhanced, support overlay and change-map views.

## 7. Runtime Protocol

Measure the same input cohort on the same hardware/environment for V5a and V5b.
Record image dimensions, pixel count, CPU, RAM, library versions, threads and power
mode. Separate image decoding, enhancement and diagnostic exports. Use a declared
warm-up and multiple repeated passes; report median and spread, peak memory and
failure count. Verify output equality after performance changes. Existing logs
are historical measurements, not evidence that V5b accelerates V5a.

## 8. One-Year Roadmap

| Period relative to this draft | Main deliverable | Thesis update |
| --- | --- | --- |
| Months 1-2 | Supervisor-approved scope; metadata/permissions; reproducible release; reviewed pilot and dataset exposure map | Finalize introduction, data chapter and evaluation protocol |
| Months 3-4 | Fixed baselines, ablations, blinded pilot review and common runtime benchmark | Replace preliminary metrics with reviewed comparative tables |
| Months 5-7 | One focused method contribution: more reliable support/confidence and conservative fallback; test boundaries and low-SNR features | Update method and failure-analysis chapters |
| Months 8-9 | Frozen-system evaluation on independent TRI groups, with observer/task assessment if feasible | Write main results and robustness chapter |
| Months 10-11 | Resolve limitations, complete literature comparison, publication permissions, repeatability package | Complete discussion and full draft review |
| Month 12 | Formatting, consistency checks, supervisor revisions and defense preparation | Final thesis and defense material |

Do not add reconstruction, diffusion and defect classification simultaneously.
Select a major extension only if it addresses a measured failure and fits the time
available. If TRI provides calibrated projections or depth stacks, treat that as
a separately scoped study with its own forward model and evaluation.

## 9. Highest-Priority Next Action

Agree on the title and evaluation endpoints with the professor, then review a
small balanced pilot using the tools already present. This provides a stronger
foundation for the next implementation than another round of visual tuning alone.

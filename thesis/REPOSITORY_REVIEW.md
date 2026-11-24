# Repository-to-Thesis Review

**24 November 2026 addendum:** The original inventory and assessment below remain
the October snapshot. Draft 0.2 adds the [V5c study](../TRI_v5c_2026-11-19/README.md)
and [fresh saved-file evidence](evidence/v5c_checks.json). V5a/V5b remain baselines;
V5c is a U29 boundary-review candidate with documented cross-component coverage
regressions. The current [handoff](../PROJECT_HANDOFF.md) supersedes older root
context notes for operational status. No new whole-repository inventory is claimed.

Audit date: 2026-10-22. This document distinguishes inspected implementation,
saved experimental evidence, user-reported feedback, and future work.

## 1. Assessment

There is enough material to start a substantive master's thesis now: an industrial
problem, several implemented approaches, documented failures, a selected automatic
method, an extension responding to industrial feedback, reproducible outputs,
and regression checks. There is not yet enough evidence to conclude that the
method improves defect detection, is equivalent to a competitor, or generalizes
reliably to unseen boards and acquisitions.

The main gap is experimental validation, not a shortage of pipeline versions.
Additional filters alone will not establish novelty. A credible contribution
would be the formulation and evaluation of an evidence-gated inspection workflow
that explicitly balances useful contrast, boundary fidelity and background change.
Whether that contribution meets the department's expectations requires supervisor
agreement and a broader literature review.

Do not organize the thesis as a diary of every version. Organize it around the
problem, hypotheses, design decisions, experiments, failures and conclusions.

## 2. What Was Inspected

- Inventoried 75 authored Python/Markdown files across root scripts and all six
  pipeline/research directories; recorded hashes, imports and function locations.
- Read the research history and method documentation; inspected processing,
  evaluation and testing code across each method family, with detailed review of
  the selected V5a/V5b core and batch runners.
- Parsed selected saved runs, timing records, review annotations and verification
  reports; recounted original-image folders and decoded-image duplicates.
- Viewed representative saved J1/U30 V5a and U29/U76 V5b comparisons. This was not
  an exhaustive visual or defect-by-defect review of the dataset.
- Re-ran 73 tests in four explicitly selected suites, all passing. These are
  software/synthetic/source-proxy checks, not 73 clinical or industrial trials.
- Listed checkpoints without deserializing or retraining them. Historical model
  quality and training provenance were not independently reproduced.
- Checked primary references for imaging context and method families. This is a
  starter bibliography, not a systematic novelty search.

Presentations, archives, environment dependencies and large checkpoint internals
were not exhaustively reviewed. Older work that exists only outside this repo
must be supplied before it can be included as documented research history.

## 3. Research Lineage

| Family and source | What it contains | Thesis treatment |
| --- | --- | --- |
| `stage1_dataset.py`, `stage1_model.py`, `stage1_train.py`, `stage2_inference.py`, `run_pipeline.py` | Grayscale U-Net, VP-SDE training, stochastic editing and optional mask branch; checkpoints and output folders exist | Historical learned exploration. Do not present as a fair failed baseline without reproducing checkpoint/configuration/data provenance |
| `decomposition_model_v3.py`, `decomposition_model.pth` | Shared encoder, structure/illumination decoders; reconstruction, smoothness, anti-collapse and synthetic supervision; tiled inference | Motivates identifiability and scale ambiguity. Reflectance/illumination are computational labels, not verified physical X-ray quantities |
| `classical_shadow_removal.py`, `classical_shadow_removal_v2.py`, `weight_sweep.py` | Profile balancing, additive/divisive fields, masked log correction, local mean/variance and Retinex-like mixtures | Useful classical baselines. Historical component-name parameter routing and tuning must be disclosed |
| `component_restoration/` | Component-aware correction, stack-derived fields, BaSiCPy, directional denoising and fallback implementations | Exploratory alternatives. Verify which optional backend actually ran; do not label a fallback as a BaSiC result |
| `masked_background_research/` | Hard-mask inpainting and later soft weighted background estimates | Negative evidence: an erroneous structure mask can contaminate the estimated field |
| `inspection_view_pipeline/` | Aggressive feature rendering, less saturated grayscale variants and heuristic dark-response maps | Explains why cosmetic contrast and automatic feature scores were insufficient |
| `competitor_target_pipeline/` V2-V4 | Successive region proposals, native-resolution correction, component-specific guards and metrics | Development history and selected ablations, not separate claimed inventions |
| `automatic_v5.py`, `competitor_target_v5a.py` | Image-driven automatic core plus gated repeated-layout correction | Main reference method for draft 0.1; no trained weights or filename-based processing selection |
| `u29_focused_pipeline/` | Rejected pad-local rendering; smooth-field correction; later clarity/deconvolution/stack experiments and audit | Important negative case: unchanged pixel coordinates do not imply unchanged visible boundaries |
| `competitor_target_pipeline/u30_meeting_20261005/` | Seven-image baseline/focused comparison with signed-detail enhancement | Focused exploratory study; crop coordinates are review-only, not enhancement inputs |
| `TRI_v5b/pipeline/v5b.py` | Frozen V5a plus source-supported region blending and edge-darkening control | Main extension; exact source outside estimated supports, not verified true-background segmentation |
| `TRI_v5b/pipeline/targeted_fixes.py` | Three source-hash-locked manual search-envelope profiles | Separate manually guided troubleshooting study; exclude from automatic generalization scores |

Git contains four commits from May 14 to September 26, 2026. The student's longer
project history may be correct, but this Git history alone does not establish its
start date. Recover earlier reports, meeting records and training logs rather than
inventing a year-long chronology from file names or modification dates.

## 4. Data Inventory and Independence

| Collection | Original images found | Interpretation |
| --- | ---: | --- |
| `TRI_Dataset` | 487: J1 172, J4 167, U29 60, U30 88 | Available files, not 487 independent boards |
| `TRI_Dataset_Concised` | 32: eight per family | Repeatedly used development cohort; not a fresh holdout |
| `G2G3` | 13 | Additional TRI batch; one is copied into the V5b rim review |
| `TRI Images 1008` | 3 U76 BMPs | Additional batch; targeted edits were tuned on z201 |
| `TRI_New_Image` | 2 original RGB screenshots | Four adjacent files are derived images, not extra originals |
| U29 and U30 study copies | 8 and 7 | Copies of existing data; must not inflate sample size |
| Transferred `clean_auto` / `clean_manual_fixed` | 62 / 23 PNG files | Legacy prior-training material; folder names do not certify clean ground truth |
| Transferred `noisy_auto` / `noisy_manual_fixed` | 217 / 256 PNG files | Legacy inference material; no one-to-one pairing assumed |
| `solder_ball_data` | 112 TIFF files | Separate historical material; calibration/provenance to confirm |

The selected TRI-related source folders contain 537 unique decoded images after
excluding derived screenshot outputs and collapsing explicit copies. This is a
file-content count, not an independent-sample count or a count of tested images.
The full and concise TRI folders together contain 519 unique source exports;
do not assume the concise folder is duplicated inside the folder named full.

Some historical stack methods used the full collection to estimate templates.
Therefore the remainder of that collection cannot automatically be described as
unseen. Exposure to fitting, manual review and parameter selection needs recording.
Adjacent index values and filename depth suffixes are not reliable independent
board identities or acquisition metadata.

## 5. Current Evidence

- V5a: 32 saved native-resolution results. The rectangular branch activates for
  all eight concise U29 images, but not the J1/J4/U30 images in that run.
- Its saved benchmark reproduces all 32 selected output images pixel-for-pixel.
  It records 368.40 seconds processing and 612.47 seconds including batch exports.
- V5b: 14 saved results and a saved independent verifier reporting native shape,
  source hashes, exact outside-support pixels and retained fully weighted cores.
- Targeted V5b: three revised inputs, explicitly manually guided and hash-locked.
- Reviewed evaluation file: eight candidates, zero reviewed images and zero
  annotations. There are no confirmed-defect results in that evaluation set.
- Regression suites rerun here: V3-V5a 33; V5b 13; U29 experiments 22; U30 study 5.

Observed repeatability and positive meeting feedback are useful, but neither is
statistical evidence of generalization or defect-detection reliability.

## 6. Documentation and Reproducibility Risks

1. **Partner naming:** the historical context file sometimes expands TRI as
   Taiwan Research Institute. The industrial partner name to verify for the
   thesis is Test Research, Inc. Public product pages establish the company and
   its inspection business, not the terms or scope of this collaboration.
2. **Acquisition claims conflict:** `README_CLARITY_V3.md` makes definite statements
   about slices, projections and different specimens. The later `CLARITY_AUDIT.md`
   explicitly says these are unconfirmed. Use the cautious audit wording until
   TRI supplies matching metadata. Do not use screen-dialog values as proof of
   our source acquisition.
3. **Unsupported physical explanation:** old notes imply that added diffusion
   noise selectively destroys smooth shading while preserving fine structure.
   This is not established. SDEdit is a synthesis/editing method, not a guarantee
   of industrial evidence preservation.
4. **Metric naming:** older metrics named shadow reduction, void contrast or
   healthy-pad uniformity can measure ordinary content, unconfirmed dark responses
   or display scaling. Re-label them as proxies unless validated ground truth exists.
5. **Legacy mask claim:** `stage2_inference.py` describes exact preservation in its
   masked branch, but its final preserved sample still contains noise at the last
   positive timestep. Do not transfer that claim to the thesis; V5b's explicit
   source-copy invariant is a different, tested mechanism.
6. **Historic weight search:** a 0.10-step grid cannot include the stated
   `(0.45, 0.35, 0.20)` baseline. The script can therefore print an unavailable
   baseline comparison. Recompute a valid comparison before citing improvement.
7. **Environment and timing:** root requirements are unpinned and cover the
   classical workflow, not all historical neural/library experiments. Hardware
   fields in `benchmark_v5a.py` are literal recorded values, not fresh hardware
   discovery. The V5a/V5b runtime logs have different scopes and datasets; do not
   interpret their totals as a speedup comparison.
8. **Version control:** much relevant code is currently untracked. A thesis
   release needs an explicit code snapshot, environment lock and result manifest.
9. **Import side effects:** `test_hybrid_postprocess.py` is a top-level batch
   experiment, not a unit-test module. The audit accidentally invoked it before
   discovering this and stopped it. Fourteen historical JPEG outputs may have
   been replaced; their original pixels were not snapshotted. Exclude that folder
   as immutable historical evidence. V5a/V5b code and result folders were not
   regenerated. Details are in the change log.

## 7. Thesis Contribution Candidates

**Implemented system contribution:** a reproducible non-generative inspection
pipeline with bounded corrections, conditional layout processing, retained
diagnostics, and a component-only extension with an explicit source-retention rule.

**Research contribution to establish:** demonstrate, through fixed-area expert
review, shape measurements, unseen grouped data and ablations, when selective
enhancement improves inspection visibility and when it fails. Quantify the tradeoff
between missed component support and false background enhancement.

**Useful negative contribution:** show why reducing intensity variance or increasing
contrast/edge energy can reward erased structure, amplified noise or apparent pad
deformation. The rejected U29 and U76 false-support cases already motivate this.

The basic ingredients (NLM, morphology, robust fitting, watershed, tone curves,
blending) are established techniques. Do not claim their invention. The novelty
argument must concern the formulation, constraints, integration and evidence in
this application, subject to literature and supervisor review.

## 8. Recommendation

Start drafting now using V5a/V5b as the **current reference system**. Keep results
explicitly preliminary. Prioritize reviewed evaluation and independently acquired
data before adding another complex model. A neural model is optional, not a
requirement for the thesis to be research-worthy.

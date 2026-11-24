# Evidence and Claim Register

Updated: 2026-11-24; October audit evidence retains its original date. Relative paths resolve from this folder. Evidence levels:
**implementation**, **saved measurement**, **qualitative observation**,
**user-reported feedback**, **hypothesis**, **planned evaluation**.

| ID | Defensible statement | Evidence and limitation |
| --- | --- | --- |
| E01 | V5a combines an automatic core with conditional repeated-layout correction | [V5a source](../competitor_target_pipeline/competitor_target_v5a.py), [automatic core](../competitor_target_pipeline/automatic_v5.py); implementation, not a learned network |
| E02 | The concise V5a run contains 32 images, with layout correction on eight U29 images | [Saved run](../competitor_target_pipeline/result/competitor_target_v5a/run.json); development data, not independent held-out boards |
| E03 | A benchmark reproduced all 32 selected V5a outputs | [Benchmark](../competitor_target_pipeline/result/benchmarks/v5a_20260918_163907/benchmark.json); saved single-machine, single-pass measurement |
| E04 | V5a run has 576 proposed U29 regions with input/automatic-V5/V5a statistics | [Region metrics](../competitor_target_pipeline/result/competitor_target_v5a/per_region_metrics.csv); proposals include unconfirmed dark structure, not healthy-only or true-defect labels |
| E05 | V5b explicitly copies source pixels outside its estimated support | [V5b source](../TRI_v5b/pipeline/v5b.py); exact image-domain property, not proof that every true background pixel was excluded |
| E06 | Saved V5b verification passes on 14 images; outside-support equality and core retention rechecked on October 22 | [Verifier report](../TRI_v5b/verification.json), [fresh checks](evidence/draft_checks.json), [run](../TRI_v5b/results/run.json); processing integrity, not defect validation |
| E07 | Three targeted revisions depend on manually selected search envelopes | [Profiles](../TRI_v5b/pipeline/targeted_profiles.json), [targeted run](../TRI_v5b/results_targeted/run.json); report separately from automatic results |
| E08 | The provided review file has eight candidates and no completed annotations | [Review file](../competitor_target_pipeline/review/annotations.json); no reviewed accuracy/CNR table can currently be derived from it |
| E09 | U29 focused V1 was rejected for visible pad deformation | [Focused-method README](../u29_focused_pipeline/README.md), [later audit](../u29_focused_pipeline/CLARITY_AUDIT.md); negative case, not final method |
| E10 | Seven U30 images have matched baseline/experimental crops | [Study README](../competitor_target_pipeline/u30_meeting_20261005/README.md), [run](../competitor_target_pipeline/u30_meeting_20261005/result/u30_focus_v1/run.json); no confirmed defect annotations |
| E11 | Source/competitor acquisition compatibility remains unconfirmed | [U30 source audit](../competitor_target_pipeline/review/U30_SOURCE_FINDINGS.md), [U29 audit](../u29_focused_pipeline/CLARITY_AUDIT.md), user reply "Not confirmed"; cannot claim a particular source modality or exact reference matching |
| E12 | Selected source folders contain 537 unique decoded images | [Machine-readable audit](evidence/repository_audit.json); original exports and distinct screenshots, not independent boards or all processed images |
| E13 | 73 allowlisted tests pass in the audit | [V3-V5a](evidence/tests_v3_to_v5a.txt), [V5b](evidence/tests_v5b.txt), [U29](evidence/tests_u29_experiments.txt), [U30](evidence/tests_u30_experiment.txt); regression checks, not accuracy experiments |
| E14 | Professor feedback was positive, with requests for component-only enhancement | User-reported meeting feedback in project conversation; informal feedback, not a blinded observer study or formal industrial acceptance |
| E15 | TRI supplies images and practical feedback in this collaboration | User/project records; exact institutional wording, contract scope, acknowledgments and publication rights require confirmation |

## November V5c Evidence

| ID | Defensible statement | Evidence and limitation |
| --- | --- | --- |
| E16 | V5c uses source-contrast connectivity, bounded positive lifting and conservative non-grid acceptance | [Implementation](../TRI_v5c_2026-11-19/pipeline/v5c.py), [method documentation](../TRI_v5c_2026-11-19/README.md); no claim of semantic defect segmentation |
| E17 | All 14 saved V5c outputs pass dimensions, outside-support equality and no-added-grayscale-darkening checks | [Fresh saved-file audit](evidence/v5c_checks.json), [checker](tools/check_v5c_evidence.py); source/baseline hashes verified without rerunning enhancement; masks are estimates |
| E18 | Seventeen scoped synthetic V5c regression tests pass | [Test log](evidence/tests_v5c.txt), [tests](../TRI_v5c_2026-11-19/pipeline/test_v5c.py); not real-defect validation; separate from the 73 historical October tests |
| E19 | Selected U29 rims appear clearer but some boundaries remain ambiguous | [Review](../TRI_v5c_2026-11-19/REVIEW_NOTES.md), Figures 6.7-6.8; qualitative observations on development data |
| E20 | V5c has reduced coverage on some irregular plates, weak balls and TRI_new images | [Complete-cohort counts](evidence/v5c_checks.json), [review](../TRI_v5c_2026-11-19/REVIEW_NOTES.md); accepted/proposed counts are not precision or recall |
| E21 | V5c export recorded 358.0 seconds and exact recomputed V5b agreement for 14 inputs | [Saved run](../TRI_v5c_2026-11-19/results_v5c/run.json); includes comparisons, crops and diagnostics; baseline equality recorded at export, not reprocessed by the November thesis audit |

## Claims Not Supported Yet

- "All defects are preserved" or "no structural change": requires reviewed
  real defects, full-boundary assessment and explicit sensitivity limits.
- "Matches/exceeds the competitor": requires matched data and a fair benchmark;
  photographs of displays are insufficient.
- "Generalizes to all TRI images": requires independent acquisition/board groups
  and a frozen parameter set, including failures and unsupported image types.
- "Improves detection accuracy": requires an observer or detector task with
  labels; contrast and standard deviation are not detection endpoints.
- "Removes the physical cause of shadows": current processing modifies exported
  intensities; acquisition/reconstruction causes are not identified.
- "Exact original background everywhere": supported only outside estimated masks,
  which can include false-positive true background, as U76 demonstrates.
- "V5b is faster than V5a": current logs differ in input resolution, workload,
  environment state and timing scope.

## Figure selection rules

Use original file identity, not appearance, to match input and output. Include
same-scale crops, masks and a difference view where appropriate. Caption any
manual intervention, display resampling and uncertainty. Keep original reference
and float intermediates alongside the enhanced view. Do not call an unconfirmed
dark region a void merely because the enhancement makes it conspicuous.

## Preservation of evidence

The frozen V5a module hashes match the selected release. The audit did not rerun
the accepted V5a/V5b batches. The separate classical-batch audit incident is
documented in [CHANGELOG.md](CHANGELOG.md); affected historical outputs are not
used for conclusions or tables in this draft.

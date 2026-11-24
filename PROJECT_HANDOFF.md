# Project Handoff

Context tracking updated: 2026-11-24T16:42:13.484765+08:00 (Asia/Taipei).

Verified working-state update: **24 November 2026 (Asia/Taipei)**.
Repository: `E:\Shadow_Removal_SDE`; remote:
`https://github.com/AryanArvind4/shadow-removal-tri.git`, branch `master`.
Use `git log -1` and `git status --short` for the current commit and pending work;
this document intentionally does not embed its own commit ID.

## Immediate Task State

Context maintenance now follows `CONTEXT_TRACKING.md` after every change task.
`PROJECT_HISTORY.jsonl` is the append-only timestamped history; its recorder stamps
this handoff automatically. `AGENTS.md` requires semantic updates, while
`.githooks/pre-commit` rejects missing or stale records using staged-file hashes.
Enable `core.hooksPath=.githooks` once per clone. This is task/commit tracking, not
a watcher on every filesystem save. Include ignored artifact paths explicitly.
Pipeline methods and image outputs are unchanged by this tooling addition.

The user requested durable context, a thesis update and a GitHub push. This update
documents the completed V5c study without changing its algorithm or outputs.
**Next: wait for the user's latest professor feedback**, then scope the next
experiment. No new professor feedback or next pipeline change has been supplied.
Do not interpret the 19 November presentation as formal validation or acceptance.

## Goal and Research Position

Enhance TRI-supplied PCB X-ray inspection views: clearer solder/component interiors
and dark features, less distracting background change, and faithful apparent
boundaries. Keep originals accessible. This is currently classical image
processing, not a trained defect detector or physical reconstruction method.
The user is developing this as a living master's thesis, with future revisions
expected. Official institution, supervisor, collaboration wording and publication
permissions still require confirmation.

The competitor screen photographs are qualitative visual references only. Same
board and reconstruction depth were explicitly **not confirmed**. Do not fabricate
missing cavities or use those photographs as pixel-aligned ground truth.

## Current Methods and Decisions

| Method | Status and role |
| --- | --- |
| V5a | Retained baseline: noise-limited contrast enhancement, bounded background/band correction, conditional repeated-pad correction |
| Automatic V5b | Retained component-selective extension; source pixels copied outside estimated supports; limits added darkening near edges |
| Targeted V5b | Three manually guided repairs, locked to source hashes: U76 z201 and two TRI_new screenshots; never label as automatic |
| V5c | Latest candidate for U29 edge-feature presentation; mixed cross-component results; not a blanket V5b replacement |
| Earlier U29 focused branches | Historical experiments; V1 rejected for apparent pad deformation; later variants are not current final results |

User feedback drove the progression: global correction amplified background grain;
V5b limited processing to components but some near-edge dark features appeared as
notches. V5c tests whether source-supported thin rims can be retained without
inventing closure. Boundary selection/blending is a working explanation, not proof
that every dark opening is an imaging artifact.

V5c uses frozen V5a/V5b proposals. Around a grid pad, a bounded robust background
plane estimates local context. Strong source contrast seeds selection; connected
weaker pixels can preserve a visible thin rim. It does not close open dark gaps.
Enclosed holes are filled in the support mask, not with bright image pixels.
A bounded positive lift acts on original intensities, with V5a providing a brightness
anchor rather than final texture. Contrast-based weights blend the correction.
The scalar curve is monotone for fixed background; the spatial composite is not
guaranteed globally rank-preserving. Non-grid regions require more conservative
shape/context checks, which can miss irregular plates and weak balls.

Default settings: weak fraction 0.10; strong fraction 0.40; minimum contrast 3;
noise multiplier 2.5; maximum slope 6; white level 245. No filename-specific
enhancement profiles in V5c. Full-resolution exports do not warp or resize pixels.
These properties do not guarantee preservation of every real defect.

## Files and Results

All paths below are relative to the repository root. Private images and generated
outputs remain local and will not appear in a fresh GitHub clone.

| Artifact | Location |
| --- | --- |
| V5a implementation | `competitor_target_pipeline/competitor_target_v5a.py` |
| V5a retained outputs | `competitor_target_pipeline/result/competitor_target_v5a/` |
| V5b implementation / frozen baseline | `TRI_v5b/pipeline/` |
| V5b automatic results | `TRI_v5b/results/` |
| V5b three targeted repairs | `TRI_v5b/results_targeted/` |
| Latest V5c study | `TRI_v5c_2026-11-19/` |
| V5c implementation / tests | `TRI_v5c_2026-11-19/pipeline/` |
| V5c copied originals / identities | `TRI_v5c_2026-11-19/inputs/`, `study_manifest.json` |
| V5c complete results | `TRI_v5c_2026-11-19/results_v5c/` |
| V5c interpretation / review order | `TRI_v5c_2026-11-19/REVIEW_NOTES.md` |
| Current thesis | `thesis/THESIS_DRAFT.md` (draft 0.2) |
| Thesis evidence and update history | `thesis/EVIDENCE_REGISTER.md`, `thesis/CHANGELOG.md`, `thesis/evidence/` |
| Latest presentation | `presentations/V5c_Progress_2026-11-19/output/TRI_V5c_Meeting_2026-11-19.pptx` |
| Latest slide scripts | `presentations/V5c_Progress_2026-11-19/.build/speaker_notes.json` |
| Previous V5b presentation | `presentations/V5b_Progress_2026-10-27/output/TRI_V5b_Meeting_2026-10-27.pptx` |

V5c processed 14 previously reviewed images: U29_029-036 (8); TRI_1008 U76 z1,
z141, z201 (3); TRI_new image001/image002 (2); one G2 round-ball image (1).
These are development/review images, not independent held-out cases.

Within each result dataset: `enhanced/` contains native-mode final PNGs;
`reference_gray/` native grayscale; `input_output/` two panels;
`comparison_sheets/` input/automatic V5b/V5c; `targeted_comparisons/` the separate
manual baseline where available; `review_crops/` all-pad or generic review crops;
`pixel_crops/` U29 row-2 nearest-neighbor 5x crops; masks, weights, diagnostics and
numeric arrays support inspection. Ordinary sheets use equal Lanczos resizing.
`run.json`, `audit.json` and `metrics.csv` record the completed study.

## Observations and Known Failures

1. Start with `results_v5c/U29/comparison_sheets/U29_032.png`, then
   `U29/pixel_crops/U29_032_pad_12.png` (row 2, column 3): the inspected lower-edge
   feature is better separated from the outline than in V5b.
2. `U29/pixel_crops/U29_032_pad_14.png` (row 2, column 5) remains ambiguous.
   Preserve this negative/uncertain example alongside favorable ones.
3. Review all eight U29 comparisons, not just U29_032. Each accepted 72/72 proposed
   pads; this is algorithmic coverage, not detection accuracy.
4. U76 z141 accepted 69/69 proposals; some internal detail is more visible.
5. U76 z201 accepted 7/26: less unwanted lower-background enhancement but missed
   irregular plate enhancement. Prior targeted V5b already addressed its background.
6. U76 z1 accepted 23/91; round-ball case 371/590. Weak/irregular components can be
   left unchanged; retain V5b round-ball output rather than claiming universal gain.
7. TRI_new accepted only 5/121 and 1/145; little change. Retain targeted V5b here.

## Verification and Limits

The saved study reports exact recomputed-versus-saved automatic V5b agreement for
all 14 cases; native dimensions, exact source retention outside V5c support and no
added grayscale darkening. Input and retained V5b hashes match their snapshots.
The new thesis evidence checker re-reads saved files without rerunning enhancement.
Seventeen focused synthetic V5c tests cover thin/blurred rims, genuine open gaps,
diagonal channels, background/protected pixels, translation, mappings and rejection.
The October thesis audit separately recorded 73 earlier tests; do not portray all
90 as one newly rerun test suite.

V5c full export took 358.0 seconds including comparisons, crops and diagnostics.
This is not deployment latency and is not directly comparable with other cohorts.
Hardware previously recorded: Intel Core 7 240H, 16 GiB RAM, NVIDIA RTX 5050 Laptop
GPU; processing uses CPU, not CUDA. Treat these as historical benchmark fields,
not a fresh hardware measurement. Existing environment: `.venv312`, Python 3.12.

Exact outside-mask equality does not prove that the mask excludes every real
background pixel. Proposal counts, variance, correlation and clipping are not
defect accuracy. There are no completed reviewed defect labels establishing
universal preservation, competitor parity, or generalization to all TRI images.

## Safe Commands

```powershell
# Synthetic tests: no private data needed.
.\.venv312\Scripts\python.exe -m unittest discover -s TRI_v5c_2026-11-19/pipeline -p test_v5c.py -v
# Fresh saved-file checks + focused test log; writes thesis evidence only.
.\.venv312\Scripts\python.exe thesis/tools/check_v5c_evidence.py
.\.venv312\Scripts\python.exe thesis/tools/check_draft.py
# Only when a new experiment is requested; choose an unused output folder.
.\.venv312\Scripts\python.exe TRI_v5c_2026-11-19/pipeline/run_v5c.py --dataset U29 --output TRI_v5c_2026-11-19/results_u29_repeat
```

Do not run unrestricted test discovery. `test_hybrid_postprocess.py` runs processing
at import; an October audit accidentally rewrote 14 historical classical outputs.
The incident and affected scope are preserved in `thesis/CHANGELOG.md`. Those
mixed-provenance outputs are excluded from thesis conclusions. Do not undo them
using guessed originals. Do not rerun `prepare_study.py` on prepared studies;
V5b preparation originally moved data folders. Use new result directories.

## Git and Context Hygiene

Publish reviewed code, tests, documentation and compact numerical evidence only.
Do not push industrial images, results, numeric arrays, presentations, checkpoints,
virtual environments or private study manifests. A fresh clone cannot run full
saved-result checks without the separately retained local data.

`sites/arvind-academic/` is unrelated website work and is left local in this update.
`PROJECT_CONTEXT.md` and `HANDOFF_PROMPT_U29.md` are older untracked handoffs, left
unchanged. Their old "current" methods and unconfirmed acquisition/institutional
claims must not override this verified state. Historical README statements may
likewise describe superseded experiments rather than validated final conclusions.

## Resume Prompt

"Read AGENTS.md, PROJECT_HANDOFF.md, TRI_v5c_2026-11-19/README.md and REVIEW_NOTES.md,
then thesis/CHANGELOG.md and EVIDENCE_REGISTER.md. Check git status before editing.
Preserve V5a/V5b and saved studies. V5c is a U29-focused candidate with documented
coverage regressions, not a universal replacement. We are waiting for the latest
professor feedback before choosing the next method change. Update the handoff and
thesis evidence when a new experiment is completed."

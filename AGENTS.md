# Project Onboarding

Read `PROJECT_HANDOFF.md` before continuing this project, then the relevant study
README and `thesis/EVIDENCE_REGISTER.md`. The handoff records the verified state,
local artifact locations, open questions and the next requested action.

- V5a and V5b are retained baselines. V5c is a U29-focused review candidate, not
  a universal replacement. Preserve prior inputs, pipelines and result folders.
- Do not synthesize defect geometry or claim that every notch is a closed void.
  Competitor/source acquisition equivalence and defect ground truth are unconfirmed.
- Source equality outside an estimated mask does not prove correct segmentation.
  Keep automatic and manually targeted V5b results explicitly separate.
- Use explicit, scoped tests. Never run root-wide test discovery or import
  `test_hybrid_postprocess.py`: it executes a historical batch at import time.
- Do not rerun study preparation or overwrite accepted output directories.
- Keep industrial images, generated results, models and presentations out of new
  Git commits. Local data/manifests are needed for full evidence verification.
- Root `PROJECT_CONTEXT.md` and `HANDOFF_PROMPT_U29.md`, when present, are older
  unreviewed historical notes. Their "current" labels, global constraints and
  acquisition assertions are not authoritative for the present project.
- After major work, update the handoff and thesis change log. Distinguish measured
  results, qualitative observations, hypotheses and planned evaluation.

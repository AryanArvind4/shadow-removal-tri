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
- After EVERY completed change task, update the relevant sections of
  `PROJECT_HANDOFF.md`, then record a timestamped entry in `PROJECT_HISTORY.jsonl`
  using `tools/project_context.py record`. Follow `CONTEXT_TRACKING.md` for the
  stage/record/stage sequence. Record summary, validation (including not run),
  next action and produced artifact paths, including ignored results/slides.
- Also record new feedback/decisions that change the project direction. Read-only
  questions need no entry unless they establish a new decision. Before a pause or
  handoff, record partial progress and what is unfinished; never mark it complete.
- Update the thesis change log for substantive research changes, not every small
  tooling edit. Distinguish measured results, observations, hypotheses and plans.
- The context hook must remain enabled (`git config core.hooksPath .githooks`).
  Never use `--no-verify` or disable it to bypass a stale-context error. New clones
  need this local setup once. Semantic context must be written by the agent;
  timestamps/file hashes are automated, not an automatic scientific interpretation.

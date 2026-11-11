# Living Master's Thesis

Working snapshot: 2026-10-22, draft 0.1. This is a research manuscript in progress,
not a submission-ready thesis or a claim that the research is finished.

## Recommended framing

**Component-Aware Enhancement of Industrial PCB X-ray Images for Visual Inspection**

Working research question: Can bounded, evidence-gated enhancement improve
component visibility while limiting unwanted background changes and preserving
inspection-relevant source features?

V5a and automatic V5b are the current reference implementations. Their names are
software versions, not the thesis's scientific contribution. Three image-specific
V5b repairs are an explicitly separate case study. Future methods can replace the
main results without requiring a new thesis structure.

## Read in this order

1. [Repository review](REPOSITORY_REVIEW.md): what exists, what it demonstrates,
   and what should not yet be claimed.
2. [Working thesis draft](THESIS_DRAFT.md): initial prose, methods, measured tables,
   result figures, limitations and conclusions.
3. [Evaluation and research plan](EVALUATION_PLAN.md): the work needed to turn
   encouraging outputs into defensible research results over the coming year.
4. [Evidence register](EVIDENCE_REGISTER.md): traceability for important claims.
5. [References](REFERENCES.md): verified starter sources, not a completed survey.
6. [Change log](CHANGELOG.md): updates to the manuscript and evidence.

Machine-readable evidence is in [evidence/repository_audit.json](evidence/repository_audit.json).
It inventories 75 Python/Markdown files, source hashes and function locations,
selected result manifests, dataset dimensions and decoded-image duplicates.
It is an inventory and targeted technical review, not a claim that every image,
checkpoint, slide or proprietary acquisition has been exhaustively validated.

## Update after each major experiment

1. Keep the original inputs and old outputs. Use a new run directory.
2. Record the question, dataset identities/groups, code hash, settings and baseline.
3. Add every attempted variant to the experiment log, including rejected variants.
4. Record complete-cohort results, failures and representative figures, not just
   the best examples. Keep automatic and manually guided outputs separate.
5. Update the evidence register and only then update numerical manuscript claims.
6. Add a dated change-log entry. Archive a PDF only at agreed milestones.

Use [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for each new study. Keep
unresolved material marked `[TO CONFIRM]`, `[PLANNED]` or `[NOT YET MEASURED]`.
Never replace these markers with estimates presented as measured results.

## Reproduce the audit

```powershell
.\.venv312\Scripts\python.exe thesis\tools\audit_repository.py --tests
.\.venv312\Scripts\python.exe thesis\tools\check_draft.py
```

This writes inventory/test logs only under `thesis/evidence`; it does not train
models, load checkpoints, or rerun enhancement batches. Use the explicit test
allowlist in that script, not repository-wide discovery: several legacy files
execute experiments at import. See the audit incident in the change log.

## Decisions needed from the student and supervisor

- Confirm the title, primary contribution and minimum evaluation requirements.
- Supply official university/department template, degree name, author name,
  supervisor details and required thesis language.
- Confirm the permitted wording for the TRI collaboration and which images,
  product information and code can be published. Do not assume public GitHub
  availability authorizes industrial image publication.
- Confirm acquisition metadata and whether a new untouched board-level test set
  and a small expert-reviewed evaluation set can be obtained.

Markdown keeps this draft easy to revise and version. University-specific LaTeX
or Word formatting should follow once the required template is confirmed.

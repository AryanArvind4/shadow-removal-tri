# Thesis Change Log

## 2026-10-22 - Draft 0.1 and repository evidence review

- Created initial thesis framing, working manuscript, evidence register, evaluation
  plan and starter references without changing the selected pipelines.
- Inventoried authored code/documentation and selected original image collections.
- Parsed saved V5a/V5b results; reran four allowlisted suites, 73 tests passing.
- Identified unsupported modality, metric and generalization claims in older notes.
- Distinguished automatic V5b from three manually guided targeted corrections.

### Audit incident: historical classical outputs

An initial audit command mistakenly included `test_hybrid_postprocess` in a
unittest invocation. That legacy module runs a batch at import, writing to
`results/aggressive_local_hybrid/`. It was stopped, and process inspection
confirmed no Python process remained from that run. The allowlist was corrected.

Files with refreshed timestamps: J1_000.jpg through J1_007.jpg and J4_044.jpg
through J4_049.jpg, 14 images total. Pre-audit hashes of these outputs were not
available, so unchanged pixel content cannot be asserted. The batch did not
finish; its old metrics CSV must not be assumed to match those refreshed images.
No guessed rollback was performed. Treat the historical folder as mixed-provenance
until it is reconstructed in a new directory from a documented code revision.

Original input data and selected V5a, V5b and targeted-V5b output folders were not
rerun or edited by this incident. The manuscript does not use the affected folder
as quantitative evidence. This limitation is kept visible rather than hidden.

## Next update

- Date and experiment ID:
- What changed:
- New dataset groups and review status:
- Evidence/result locations and source hashes:
- Chapters/tables/figures updated:
- Claims added, narrowed or withdrawn:
- Failures and unresolved questions:

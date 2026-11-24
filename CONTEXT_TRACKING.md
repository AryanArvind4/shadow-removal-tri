# Timestamped Project Context

`PROJECT_HANDOFF.md` is the current readable state. `PROJECT_HISTORY.jsonl` is an
append-only chronological record: one JSON object per task/change set, with an
ISO 8601 Asia/Taipei timestamp (`+08:00`), summary, validation, next action, local
artifacts, parent commit and SHA-256 hashes of staged file contents. Git records
the subsequent commit identity and full text diffs. Do not rewrite old entries.

## Required Workflow

1. Finish the change and its checks, or explicitly document incomplete work.
2. Update the relevant handoff sections with actual progress, decisions and paths.
3. Stage only the intended code/docs. Never stage large/private images or results.
4. Run the recorder below. It captures the staged changes, automatically updates
   the handoff's tracking timestamp, and appends a history entry. It does not commit,
   push or stage anything itself. Review the generated record.
5. Stage `PROJECT_HANDOFF.md` and `PROJECT_HISTORY.jsonl`. Commit normally.
   If staged code changes again, record a new entry before committing.

```powershell
.\.venv312\Scripts\python.exe tools/project_context.py record --summary "What changed and why" --validation "Checks run and results, or explicitly not run" --next-step "Next action or waiting for feedback" --artifact "relative/path/to/local/result"
git add -- PROJECT_HANDOFF.md PROJECT_HISTORY.jsonl
.\.venv312\Scripts\python.exe tools/project_context.py check
```

`--artifact` is optional and repeatable. It records locations, not image content.
The handoff should also describe significant ignored artifacts: a Git hook cannot
detect changes to ignored results or presentation images. Context-only decisions
can be recorded after staging the updated handoff. Reading a file alone does not
require a history entry.

## Automatic Commit Guard

The tracked `.githooks/pre-commit` runs `check`. It checks that the staged handoff
matches the latest history record, the staged file set/hashes match, the record
belongs to the current parent commit, and history was only appended. A mismatch
blocks the commit with instructions to refresh context. It never edits the index
or files during a commit. Existing non-sample hooks must be reviewed before setup.

Enable once in each clone (Git does not transfer local hook configuration):

```powershell
git config --local core.hooksPath .githooks
```

The hook uses the repo's `.venv312` Python when present, otherwise Python 3 from
PATH. The tracker itself uses only the Python standard library. Run its isolated
tests without private data:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tools/tests -p test_project_context.py -v
```

## Limits

This is task-level tracking with a commit-time safeguard, not a background watcher
on every keystroke. Agents follow `AGENTS.md`; external editors or models need to
follow the same workflow. No hook can validate the scientific truth of a summary,
force another model to read instructions, or run when deliberately bypassed.
Uncommitted or ignored changes still need an explicit task record. Pipeline outputs
remain untouched by the tracker. No claim of continuously monitored filesystem
history is made.

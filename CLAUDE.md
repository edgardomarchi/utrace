# utrace

Backend-agnostic uncertainty quantification (conformal prediction) for black-box models.

## Build & test
- Run tests: `uv run --extra=torch --extra=viz pytest tests/ -q --no-cov`
- Core tests (`tests/core/`) must NOT import torch.
- Integration tests live in `tests/integration/torch/`.

## Active refactor — READ FIRST
This repo is mid-refactor (PyTorch → backend-agnostic core). Before touching
any core code, tests, or example scripts, read `MIGRATION.md`. It defines the
phase state, the canonical migration recipe, and the dependency rule (core
never imports torch). Do not deviate from it without flagging the conflict.

## Working style
- One script / one concern per change. Do not migrate multiple scripts at once.
- Preserve behavior; the golden tests are the safety net.
- Some things that look like bugs are intentional (see MIGRATION.md "Decisions
  to respect"). Flag, don't "fix" silently.
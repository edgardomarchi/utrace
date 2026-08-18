# utrace

Backend-agnostic uncertainty quantification (conformal prediction) for black-box models.

## Build & test
- Run tests: `uv run --extra=viz pytest tests/ -q --no-cov`
  `--extra=torch` is gone: torch now comes from the default `dev` dependency group (see
  MIGRATION.md, "Phase 6 step 3c — packaging cleanup" and the reshape entry above it), so a plain
  `uv sync`/`uv run` already has it. `--extra=viz` is still required on the `uv run` that executes
  pytest, not on a preceding `uv sync` — a bare `uv run pytest` re-syncs the environment down to
  base plus `dev` and silently drops `matplotlib`/`pandas`, which are NOT part of the `dev` group.
  This was observed directly during the redesign: a strict `uv sync` with no `--extra=viz`
  actively uninstalls matplotlib and pandas from the venv if they were present. Both packages are
  needed: `tests/integration/torch/` needs torch (from `dev`), and one test in `tests/core/`
  exercises the plotting helpers and needs matplotlib genuinely installed to pass (it skips
  cleanly via `pytest.importorskip` when matplotlib is absent).
- Core tests (`tests/core/`) must NOT import torch.
- Integration tests live in `tests/integration/torch/`.

## Contributor environment setup
The `dev`, `dev-cuda13` and `dev-rocm7` dependency groups are mutually conflicting (one torch
build per hardware target) and `dev` is the default group, so pick exactly one:

```bash
uv sync --extra=viz                                              # CPU (default group, no flags needed)
uv sync --extra=viz --no-default-groups --group dev-cuda13       # NVIDIA GPU
uv sync --extra=viz --no-default-groups --group dev-rocm7        # AMD GPU
```

`--extra=viz` is needed in all three cases for the same reason as the test command above.
Forgetting `--no-default-groups` on the GPU cases does not silently double-install two torch
builds — `uv` refuses outright, naming the conflicting groups:
```
error: Groups `dev` (enabled by default) and `dev-cuda13` are incompatible with the conflicts:
{`utrace:dev`, `utrace:dev-cuda13`, `utrace:dev-rocm7`}
```

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
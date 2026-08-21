# utrace

Backend-agnostic uncertainty quantification (conformal prediction) for black-box models.

## Build & test
- Run tests: `uv run --extra=viz pytest tests/ -q --no-cov`
  `--extra=torch` is gone: torch now comes from the default `dev` dependency group (see
  FINDINGS.md, "Phase 6 step 3c — packaging cleanup" and the reshape entry above it), so a plain
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
build per hardware target) and `dev` is the default group, so pick exactly one. The dependency
GROUP only routes `torch`/`torchvision` to the right wheel index — it does NOT give `jax` GPU
support. That comes from a separate EXTRA (`cuda13` / `rocm7-local`), which must be requested
too, or `jax.default_backend()` silently falls back to `'cpu'` (with an easy-to-miss warning)
while torch sees the GPU fine. Confirmed by resolution/execution on an RTX 3070 — see
`.reports/2026-08-20_gpu_verification_3070.md` and `.reports/2026-08-20_gpu_packaging_fixes.md`;
the ROCm line below carries the identical defect by construction (same group/extra split) but
was only checked by resolution, no ROCm hardware:

```bash
uv sync --extra=viz                                                             # CPU (default group, no flags needed)
uv sync --extra=viz --extra=cuda13      --no-default-groups --group dev-cuda13  # NVIDIA GPU (jax GPU + torch GPU)
uv sync --extra=viz --extra=rocm7-local --no-default-groups --group dev-rocm7   # AMD GPU (jax GPU + torch GPU)
```

`--extra=viz` is needed in all three cases for the same reason as the test command above.
Forgetting `--no-default-groups` on the GPU cases does not silently double-install two torch
builds — `uv` refuses outright, naming the conflicting groups:
```
error: Groups `dev` (enabled by default) and `dev-cuda13` are incompatible with the conflicts:
{`utrace:dev`, `utrace:dev-cuda13`, `utrace:dev-rocm7`}
```

**Every flag above must be repeated on every subsequent `uv run`, not just the first `uv sync`.**
This is the same trap as `--extra=viz` on the test command, but worse on the GPU paths: a bare
`uv run <anything>` (missing `--no-default-groups --group dev-cuda13 --extra=cuda13`, or the
ROCm equivalent) silently re-syncs down to the default `dev` group and replaces an
already-installed GPU torch with the CPU build — same version number, only the `+cu130`/`+cpu`
local segment changes, and `jax` reverts to CPU the same way. There is no error and no obvious
warning beyond a routine "Uninstalled 2 packages / Installed 2 packages" line. Verified directly:
this happened on the first bare `uv run` issued right after a correct GPU install. Always pass
the full flag set — `--no-default-groups --group dev-cuda13 --extra=viz --extra=cuda13` (or the
ROCm equivalent) — on every `uv run`, not only `uv sync`.

## Active refactor — READ FIRST
This repo is mid-refactor (PyTorch → backend-agnostic core). Before touching
any core code, tests, or example scripts, read `MIGRATION.md` and `CONTRIBUTING.md`.
MIGRATION.md defines the phase state; CONTRIBUTING.md defines the canonical migration
recipe and the dependency rule (core never imports torch). Do not deviate from either
without flagging the conflict.

## Working style
- One script / one concern per change. Do not migrate multiple scripts at once.
- Preserve behavior; the golden tests are the safety net.
- Some things that look like bugs are intentional (see CONTRIBUTING.md "Decisions
  to respect"). Flag, don't "fix" silently.
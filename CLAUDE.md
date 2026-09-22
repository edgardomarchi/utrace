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
local segment changes. There is no error and no obvious warning beyond a routine "Uninstalled 2
packages / Installed 2 packages" line. Verified directly: this happened on the first bare `uv run`
issued right after a correct GPU install. Always pass the full flag set — `--no-default-groups
--group dev-cuda13 --extra=viz --extra=cuda13` (or the ROCm equivalent) — on every `uv run`, not
only `uv sync`.

Torch and jax do not necessarily revert together: torch always reverts on the group swap described
above, but whether `jax` reverts to CPU depends on whether the specific bare invocation included
`--extra=cuda13`/`--extra=rocm7-local` — a bare `uv run --extra=viz` (matching the canonical test
command, not a GPU sync command) has been observed to revert torch to `+cpu` while leaving the
`jax-cuda13-*` packages installed and `jax.default_backend()` still reporting `'gpu'`.

## Project status
`utrace` reached `v0.1.0`: Phases 0-6 of the backend-agnostic migration are complete and tagged
(`MIGRATION.md`'s `## Phase status`). What remains open — step-ladder items D and E, ruff rungs
2-3, and the items in `BACKLOG.md` — is scoped explicitly by `MIGRATION.md` itself as work beyond
Phase 6, not a continuation of an ongoing refactor.

Read `CONTRIBUTING.md` before touching any code, tests, or example scripts — it holds the
canonical conventions and the dependency rule (core never imports torch), and applies regardless
of what you're working on. Read `MIGRATION.md` too, but only when the work touches step-ladder
item D (jit/vmap over classes) or E2 (device coherence in the script/wrapper layer — E1, the
core-level device coherence, already shipped) specifically — it is the design-in-progress record
for that work, not a general orientation document.

## Working style
- One script / one concern per change. Do not migrate multiple scripts at once.
- Preserve behavior; the golden tests are the safety net.
- Some things that look like bugs are intentional (see CONTRIBUTING.md "Decisions
  to respect"). Flag, don't "fix" silently.
# U-TraCE — Contributing

How to set up an environment, run the tests, and work with this codebase.

## Environment setup

Pick exactly one command, matching your hardware. The `dev`, `dev-cuda13` and `dev-rocm7`
dependency groups are mutually conflicting (each routes to a different `torch` build), so only
one can be installed at a time.

```bash
uv sync --extra=viz                                                             # CPU (default group, no flags needed)
uv sync --extra=viz --extra=cuda13      --no-default-groups --group dev-cuda13  # NVIDIA GPU (jax GPU + torch GPU)
uv sync --extra=viz --extra=rocm7-local --no-default-groups --group dev-rocm7   # AMD GPU (jax GPU + torch GPU)
```

`--extra=viz` is required in all three cases — it installs `matplotlib`/`pandas`, needed for one
test in `tests/core/` and for the plotting helpers.

**Repeat the full flag set on every subsequent `uv run`, not only on `uv sync`.** This is easy to
get wrong in a way that fails silently on the GPU paths: a bare `uv run <anything>` (missing
`--no-default-groups --group dev-cuda13 --extra=cuda13`, or the ROCm equivalent) re-syncs down to
the default `dev` group and silently replaces an already-installed GPU `torch` with the CPU
build — same version number, only the `+cu130`/`+cpu` local segment changes, and `jax` reverts to
CPU the same way. There is no error, just a routine-looking "Uninstalled 2 packages / Installed 2
packages" line. Forgetting `--no-default-groups` itself, by contrast, fails loudly — `uv` refuses
outright:
```
error: Groups `dev` (enabled by default) and `dev-cuda13` are incompatible with the conflicts:
{`utrace:dev`, `utrace:dev-cuda13`, `utrace:dev-rocm7`}
```
Always pass the full flag set — `--no-default-groups --group dev-cuda13 --extra=viz
--extra=cuda13` (or the ROCm equivalent) — on every `uv run`.

## Running the tests

```bash
uv run --extra=viz pytest tests/ -q --no-cov
```

Expect `120 passed, 4 skipped` on a machine with no GPU-backed JAX device; on a GPU-capable machine those four tests run and pass instead (124 passed). The four skips are the GPU-only device-reconciliation tests for calibrate and get_uncertainty — inert without GPU hardware, not a failure.

`--extra=viz` must be on the `uv run` invocation itself, not only on a preceding `uv sync` — a
bare `uv run pytest` re-syncs the environment down to whatever the active group provides and can
silently drop `matplotlib`/`pandas` if they're not part of it.

- `tests/core/` does not import torch (or any framework) — inputs are synthetic numpy/jnp.
  Run it alone with `uv run --extra=viz pytest tests/core/ -q --no-cov`.
- `tests/integration/torch/` imports torch and needs one of the three environments above (any of
  them; the `dev` default is enough). Run it alone with `uv run pytest tests/integration/torch/ -q
  --no-cov`.
- Regenerate the integration baselines with `uv run python
  tests/integration/torch/regenerate_baselines.py --api new` if you deliberately change behavior
  they cover — `--api` currently accepts only `new`.

`scripts/` has no test coverage. If you change a script, verify it by actually running it
(before/after, comparing output) — a passing test suite says nothing about whether a script
still works.

## Linting

```bash
uvx ruff@0.16.3 check src/ tests/ scripts/
```

This is the exact command and pinned version CI runs; a plain `ruff check src/ tests/ scripts/`
with whatever `ruff` you have installed will usually agree with it. CI enforces this on every
push and pull request — a PR that doesn't pass it won't pass CI.

## What CI checks

`.github/workflows/tests.yml` runs five jobs on every push and pull request against `main`:

- **lock-check** — `uv lock --check`; fails if `uv.lock` has drifted from `pyproject.toml`.
- **lint** — `ruff check src/ tests/ scripts/` (default ruleset), as above.
- **base-import** — installs only the base package (no extras, no dependency group) across Python
  3.11–3.14, and asserts `import utrace` succeeds with torch absent from `sys.modules`.
- **core** — runs `tests/core/` across the same four Python versions, plus a `compileall` pass
  over `scripts/` (catches syntax errors on the oldest supported Python even though nothing in
  `tests/` imports `scripts/`).
- **integration** — runs `tests/integration/torch/` across the same four Python versions.

## Test conventions

- `tests/core/`: does NOT import torch (or any framework). Inputs are synthetic numpy/jnp.
- `tests/integration/torch/`: may import torch.
- Baselines live in `tests/integration/torch/baselines/`. Regenerate with
  `regenerate_baselines.py --api new` (paths in that script are relative to the file itself).
- `tests/core/test_import_properties.py` asserts properties of the package as an importable
  artifact — global JAX x64 config, torch/matplotlib/pandas absent from `sys.modules` after
  `import utrace` — rather than API behavior. The torch-absence check runs in a subprocess,
  because the rest of the integration suite imports torch into the main test process and would
  otherwise contaminate the check.

## Working with `UncertaintyQuantifier`

The canonical alpha-search method is the **binary** one, `get_uncertainty`, which accepts
`max_iters` to adjust precision. Recipe:

1. **Construction**: `UncertaintyQuantifier(N=..., classes=[C], max_batch_size=...)`. There is no
   `model` parameter — `UncertaintyQuantifier(model=...)` raises `TypeError`. For the per-class
   case, construct one `UncertaintyQuantifier` per class.

   **Buffer sizing for high-volume cases (segmentation).** The default `N=1000` is correct for
   moderate sample counts (classification, MNIST). In segmentation each image contributes ~65k
   pixel-samples, so the fixed-size padded buffer (`_max_N`) overflows if you leave `N` at its
   default. Size `N` per class from the data instead: one pass over the full dataset counts
   per-class samples (this is invariant to noise, since the ground truth is not transformed),
   then `N_class_C = ceil(count_C * cal_fraction * margin)` (e.g. `cal_fraction=0.2`,
   `margin~1.5`). Background gets a large buffer, the foreground structures small ones.

2. **Calibration (can stream)**: iterate the calibration loader ONCE, batch outside / class
   inside, accumulating with `batched=True`:
   ```python
   for X_cal, y_cal in calDataLoader:
       smx_cal = model.predict_proba(X_cal)      # tensor; to_jax handles DLPack conversion
       y_cal_arr = flatten_batch(y_cal).ravel()  # flatten batch/spatial dims (labels)
       for C in classes:
           uqs[C].calibrate(smx_cal, y_cal_arr, batched=True)
   ```
   This keeps only one batch of logits in memory at a time — important when "samples" are pixels,
   as in segmentation.

3. **Tuning (NOT batched-and-averaged)**: materialize the tune set (it is small by CP design) and
   make ONE call per class over the full set:
   ```python
   tune_smx, tune_y = precompute_softmax(tuneDataLoader, model)
   U, alpha = uqs[C].get_uncertainty(tune_smx, tune_y, max_iters=30)
   ```
   **FORBIDDEN: `alpha = np.nanmean([alpha_per_batch...])`.** This is statistically incorrect —
   alpha is a non-linear function of the data, so averaging per-batch alphas does not converge to
   the correct value. Averaging over `L` distinct splits (a full experimental repetition, each
   split producing its own complete alpha) IS valid; that is a different thing.

4. **Apply alpha (explicit, non-mutating)**: `get_uncertainty` does not mutate `self.alpha` or
   `self.q_hat`; the caller sets it:
   ```python
   uqs[C].alpha = alpha
   ```

5. **Test**: predict and compute coverage as a GLOBAL proportion over the full set, not an average
   of per-batch proportions:
   ```python
   y_p, y_s = uqs[C].predict(test_smx)
   # coverage over the full set for class C
   ```

Precompute the model's softmax output once per (noise/split) wherever possible, to eliminate
redundant forward passes — this is where most of the speedup in the analysis scripts comes from.
`scripts/_common.py`'s `precompute_softmax(loader, classifier)` does this and returns raw torch
tensors (no numpy conversion), so the result can take the zero-copy DLPack path into `to_jax`.

### Passing tensors to the core

Pass backend tensors directly to `calibrate`/`predict`/`get_uncertainty`. `to_jax()` (in
`utils/tensors.py`) handles conversion via DLPack: a CPU PyTorch tensor is consumed zero-copy. Do
NOT call `.cpu().numpy()` by hand on values that feed `calibrate`/`predict`/`get_uncertainty` —
that conversion is the library's job, and writing it yourself defeats the zero-copy path and
clutters the example.

`.cpu().numpy()` is still legitimate for values that do NOT go into the core — e.g. computing
accuracy, building arrays for matplotlib. Only avoid it on the path into
`calibrate`/`predict`/`get_uncertainty`.

## Decisions to respect (do NOT "fix" without confirming)

- In the coverage test scripts, `uq.alpha = U` (setting alpha to the U value, not the tuned alpha)
  is INTENTIONAL: it is part of the alignment tests between U and `(1-Cov)`. It may be changed to
  the tuned `alpha` in the future, but it is not a bug.

- `classes=[labels]` calibrates on the subpopulation whose label is in `labels`
  (group-conditional coverage); `classes=None` calibrates marginally. Multiple classes are fully
  supported under these semantics.

## Architecture: agnostic core, backend-specific integrations

"Backend-agnostic" applies to the CORE, not to the whole package. The package legitimately
contains backend-specific code; what matters is where it lives and the direction of dependencies.

### Dependency rule

- **Core** (`uncertaintyQuantifier`, alpha-search functions, masked quantile): imports ONLY numpy
  and jax. NEVER imports torch, onnx, or any backend, and NEVER imports from the backend
  subpackages below. Data flow is always: user code → backend wrapper → softmax output → core
  (`calibrate`/`predict`/`get_uncertainty`).
- **`utrace.utils.pytorch.*`**: everything that touches torch — `Pytorch_wrapper`, example
  models, dataset loaders, transforms, and any helper that needs torch (e.g. `flatten_batch` /
  `unflatten_batch` when operating on torch tensors).
- **`utrace.utils.onnx.*`**: analogous, for the ONNX backend.
- **`utrace.utils`** (root): only truly backend-agnostic helpers (pure numpy).
- The JAX backend's GPU acceleration is available via the optional extras `cuda13` and
  `rocm7-local` (neither declares torch). The PyTorch backend (`utrace.utils.pytorch.*`) is
  reached differently: `utrace` never installs torch for a library user, it only requires torch to
  already be present (the `examples` extra is the one exception, for running `scripts/`;
  contributors get torch from the `dev`/`dev-cuda13`/`dev-rocm7` groups — see "Environment
  setup" above). Either way, the core must be installable and importable WITHOUT torch.

### Placement test for each symbol

Does the function import or assume a backend?
- Yes → it belongs in that backend's subpackage (`utils/pytorch/`, `utils/onnx/`).
- No (pure numpy/JAX) → it may stay in `utils/` root.

For example: `flatten_batch`, `unflatten_batch`, `unflatten_pixels`, `unflatten_set_sizes`,
`view_classify` all touch torch tensors, so they live in `utils/pytorch/`. `get_coverage`,
`relabel`, `check_row_sums` are pure numpy, so they stay in `utils/` root, which is itself
torch-free — guarded directly by
`tests/core/test_import_properties.py::test_core_does_not_import_torch`.

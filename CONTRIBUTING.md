# U-TraCE — Contributing

How work is done in this repo: the canonical migration recipe, test conventions, decisions to respect, and the agnostic-core architecture rule. A contributor should not have to read a refactor-status document to find these. Split out of MIGRATION.md on 2026-08-19 (see `.reports/2026-08-19_docs_restructure.md`).

### Criterion change: renames and collected test IDs

[ESTABLISHED] Every Phase 6 removal step (steps 4-6 in MIGRATION.md's Phase status) was held to a zero-test-edit
property: the collected test-ID diff before/after had to be empty. The forwarding-accessor
commit retired that property deliberately — its declared purpose was exactly to edit tests, the
one case where the rule and the task are incompatible — not by accident or erosion. What
replaced it, since a rename necessarily moves the test IDs that name the renamed thing:

- Declare the old→new mapping BEFORE editing (both the derived mapping and, where a diagnostic
  had proposed one already, an independent re-derivation reported before touching anything).
- After editing, walk the collected-ID diff line by line against the declared mapping: every
  disappeared ID must be explained by an entry in the mapping, every appeared ID likewise, and
  the COUNT of disappeared IDs must equal the count of appeared ones. An unexplained change, or
  an unequal count, is a stop condition, not something to reconcile after the fact.
- Where a commit renames bindings but no test file or test name — the forwarding-accessor commit
  (test bodies changed, but no test was added, removed, or renamed) and the dead-code commit (no
  test touched at all) — the empty-diff property still applies, and both in fact produced
  byte-identical `--collect-only` output before and after.
- For `scripts/`, which no test covers, a rename that misses a use site fails silently at
  runtime, not at collection time — nothing about a passing test suite would catch it. The
  batch's answer: execute at least two scripts per script-touching commit (one exercising every
  renamed method, one as a cheap sanity check) with output redirected outside the repo, import
  every other touched script (catches `ImportError`/syntax errors, not runtime misses), and diff
  a `pyflakes` run against the previous commit's saved output — an `undefined name` finding is
  exactly the signature a missed rename site leaves, and nothing else in that workflow would
  surface it.

### Verification method for unvalidated scripts

[ESTABLISHED] The scripts are covered by no test, so step C developed a repeatable equivalence
procedure worth reusing for any future change to them:

- Check out HEAD into a git worktree for the "before" side.
- Assert environment parity between the two trees before running anything — interpreter path,
  python version, jax and torch versions — because a worktree is a fresh checkout and does not
  carry untracked files, including any tool-version-manager config that decides which
  interpreter is active.
- Seed from an external driver, since none of the scripts seeds itself and several have multiple
  randomness sources, including per-image noise transforms.
- Where a full run is too slow, apply an IDENTICAL scope reduction to both copies and diff them,
  asserting the diff contains exactly the intended edit and nothing else.
- Compare exactly rather than approximately, and capture accumulators out of the running frame
  where they do not reach disk.
- Run sequentially and, for timing, interleave the variants.

## Canonical migration recipe (new API)

Apply to each script. The canonical alpha-search method is the **binary** one (`get_uncertainty`,
renamed from `get_uncertainty_from_proba` by the rename batch — see "Rename batch" in FINDINGS.md), which
accepts `max_iters` to adjust precision.

1. **Construction**: `UncertaintyQuantifier(N=..., classes=[C], max_batch_size=...)`.
   There is no `model` parameter to pass — it was removed in Phase 6 step 6;
   `UncertaintyQuantifier(model=...)` raises `TypeError`. For the per-class case, one UQ per class.
   **Buffer sizing for high-volume cases (segmentation).** The default `N=1000` is correct for moderate sample counts (classification, MNIST). In segmentation each image   contributes ~65k pixel-samples, so the fixed-size padded buffer (`_max_N`) overflows.
   Size `N` per class from the data: one pass over the full dataset counts per-class samples (invariant to noise — the GT is not transformed), then `N_class_C = ceil(count_C * cal_fraction * margin)` (e.g. cal_fraction=0.2, margin~1.5).
   Background gets a large buffer, the structures small ones. The fixed-size padding from Phase 2 assumes moderate volumes; a buffer design that scales to high-volume regimes is a possible future task (backlog).

2. **Calibration (can stream)**: iterate the calibration loader ONCE, batch outside /
   class inside, accumulating with `batched=True`:
```python
   for X_cal, y_cal in calDataLoader:
       smx_cal = model.predict_proba(X_cal)      # tensor; to_jax handles DLPack conversion
       y_cal_arr = flatten_batch(y_cal).ravel()  # flatten batch/spatial dims (labels)
       for C in classes:
           uqs[C].calibrate(smx_cal, y_cal_arr, batched=True)
```
   This keeps only one batch of logits in memory at a time (important when "samples" are pixels, e.g. segmentation).

3. **Tuning (NOT batched-and-averaged)**: materialize the tune set (it is small by CP design) and make ONE call per class over the full set:
```python
   tune_smx, tune_y = precompute_logits(tuneDataLoader, model)
   U, alpha = uqs[C].get_uncertainty(tune_smx, tune_y, max_iters=30)
```
   FORBIDDEN: `alpha = np.nanmean([alpha_per_batch...])`. This is statistically incorrect:
   alpha is a non-linear function of the data. Averaging over L distinct splits (full experimental repetition) IS valid and is a different thing.

4. **Apply alpha (explicit, non-mutating)**: `get_uncertainty` does not mutate `self.alpha` or
   `self.q_hat`; the caller sets:
```python
   uqs[C].alpha = alpha
```

5. **Test**: predict and compute coverage as a GLOBAL proportion, not an average of per-batch proportions:
```python
   y_p, y_s = uqs[C].predict(test_smx)
   # coverage over the full set for class C
```

6. **Precompute logits** once per (noise/split) wherever possible, to eliminate redundant model forward passes. This is where the bulk of the speedup in the analysis scripts comes from.

### Passing tensors to the core

Pass backend tensors directly to the `calibrate`/`predict`/`get_uncertainty` methods (renamed
from `calibrate_from_proba`/`predict_from_proba`/`get_uncertainty_from_proba` by the rename
batch). `to_jax()` (in `utils/tensors.py`) handles conversion via DLPack: a CPU PyTorch tensor is
consumed zero-copy. Do NOT call `.cpu().numpy()` manually on values that feed `calibrate` /
`predict` / `get_uncertainty` — that conversion is the library's job, and writing it by hand
defeats the zero-copy path and clutters the example.

Note: `.cpu().numpy()` is still legitimate for values that do NOT go into the core (e.g. computing accuracy, building arrays for matplotlib). Only remove the conversions on the path to `calibrate`/`predict`/`get_uncertainty`.

## Test conventions

- `tests/core/`: does NOT import torch (or any framework). Inputs are synthetic numpy/jnp.
- `tests/integration/torch/`: may import torch (legitimate).
- Baselines: `tests/integration/torch/baselines/` (new API only — `baselines/legacy/` was removed in Phase 6 step 4). Regenerate with `regenerate_baselines.py --api new` (the flag now has a single remaining choice; paths relative to the file).
- Core property tests that survive Phase 6 (cache, performance smoke) live alongside the new-API golden.
- `tests/core/test_import_properties.py` (which includes `test_x64_is_enabled`, consolidated into this file — see "Forwarding accessors: temporary scaffolding (removed)" in FINDINGS.md) asserts properties of the package as an importable artifact (global JAX x64 config, torch/matplotlib/pandas absent from `sys.modules` after `import utrace`) rather than API behavior; these are grouped by that subject, not by mechanism — the torch-absence check runs in a subprocess because the rest of the integration suite imports torch into the main test process.

## Decisions to respect (do NOT "fix" without confirming)

- In the coverage test scripts, `uq.alpha = U` (setting alpha to the U value, not the tuned alpha) is INTENTIONAL: it is part of the alignment tests between U and (1-Cov). It may be changed to `alpha` in the future, but it is not a bug.

- **SUPERSEDED**: The per-class branch with a *full* class list was equivalent to `classes=None` (commit 1a2c8a); with a *partial* list it already calibrated group-conditionally on the listed classes only. We have formalized this as group semantics: `classes=[labels]` calibrates on the subpopulation whose label ∈ labels (group-conditional coverage); `classes=None` is marginal. Multiple classes are fully supported under these semantics, and the internal per-class buffers (`_class_scores`, `_class_N`, `_class_alphas`, `_class_q_hats`) have been removed.

## Architecture: agnostic core, backend-specific integrations

"Backend-agnostic" applies to the CORE, not to the whole package. The package legitimately contains backend-specific code; what matters is where it lives and the direction of dependencies.

### Dependency rule

- **Core** (`uncertaintyQuantifier`, alpha-search functions, masked quantile): imports ONLY numpy + jax. NEVER imports torch, onnx, or any backend, and NEVER imports from the backend subpackages below. Data flow is always: user code → backend wrapper → softmax output → core (`calibrate`/`predict`/`get_uncertainty`).
- **`utrace.utils.pytorch.*`**: everything that touches torch — `Pytorch_wrapper`, example models, dataset loaders, transforms, and any helper that needs torch (e.g. `flatten_batch` / `unflatten` if they operate on torch tensors).
- **`utrace.utils.onnx.*`**: analogous, for the ONNX backend.
- **`utrace.utils`** (root): only truly backend-agnostic helpers (pure numpy).
- The JAX backend's GPU acceleration is available via the **optional extras** `cuda13` and
  `rocm7-local` (neither declares torch — see "Phase 6 step 3c, second pass" in FINDINGS.md). The PyTorch
  backend (`utrace.utils.pytorch.*`) is reached differently: `utrace` never installs torch for a
  library user, it only requires torch to already be present (the `examples` extra is the one
  exception, for running `scripts/`; contributors get torch from the `dev`/`dev-cuda13`/
  `dev-rocm7` groups). Either way, the core must be installable and importable WITHOUT torch.

### Placement test for each symbol

Does the function import or assume a backend?
- Yes → it belongs in that backend's subpackage (`utils/pytorch/`, `utils/onnx/`).
- No (pure numpy/Jax) → it may stay in `utils/` root.

### Current state (rule enforced)

The dependency rule above is already enforced in the code:
- Torch-dependent helpers — `flatten_batch`, `unflatten_batch` (typo `unflatten_bath` fixed), `unflatten_pixels`, `unflatten_set_sizes`, `view_classify` — live in `utils/pytorch/`. `utils.py` is torch-free.
- Pure-numpy helpers (`get_coverage`, `relabel`, `check_row_sums`, etc.) remain in `utils/` root.
- [RESOLVED, step 5] The core's one residual torch dependency — `uncertaintyQuantifier.py` importing `flatten_batch` (from `utils/pytorch/`) for the deprecated `calibrate` path — was removed together with `calibrate` in Phase 6 step 5. `flatten_batch` itself is untouched and still lives in `utils/pytorch/helpers.py` with its other callers; only the core's import of it is gone. `tests/core/test_import_properties.py::test_core_does_not_import_torch` now guards this directly.

This rule guides both the script migration (Phase 4) and the packaging cleanup tracked in the backlog and Phase 6 — the goal recorded here as "make torch an optional extra" is what the first packaging pass did; the second pass (see "Phase 6 step 3c, second pass" in FINDINGS.md) replaced the `torch` extra with the `examples` extra plus contributor dependency groups, for the reasons given there. The underlying rule — core installable and importable without torch — is unchanged by which mechanism supplies torch to whom.

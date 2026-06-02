# U-TraCE — Migration and Refactor Guide

Context document for the ongoing refactor of the `utrace` package. Captures the phase
status, the canonical migration pattern, and agreed conventions. This is the source of
truth for the refactor: when in doubt about "how something is done here," this document
and the tests are authoritative.

## Refactor goal

Make the `UncertaintyQuantifier` core independent of the tensor backend (PyTorch, JAX,
etc.). Core methods operate on precomputed probability arrays (via `to_jax`, zero-copy
through DLPack), not on an embedded model. The model is removed from the class: the user
computes probabilities externally and passes them to the `*_from_proba` API.

## Phase status

- [x] Phase 0 — Safety net: golden tests with fixed seeds and `.npy` baselines.
- [x] Phase 1 — Pure function extraction (`_calibrate_impl`, `_get_uncertainty_jit_impl`).
- [x] Phase 2 — Conformity scores with fixed-size padding (`_max_N`, sentinel +inf),
      masked quantile, masked samples instead of filtered. Eliminates JAX recompilations
      (the root cause of the original performance issue).
- [x] Phase 3 — New `*_from_proba` API (`calibrate`/`predict`/`get_uncertainty`),
      legacy `*(X)` deprecated and delegating to shared impls. Three goldens: legacy,
      new-API, and synthetic equivalence.
- [ ] Phase 4 — Migrate example scripts to the new API (this document lists them).
- [ ] Phase 5 — Migrate remaining state (`_class_scores`, etc.) to `jnp` storage.
- [ ] Phase 6 — Remove legacy API, `model` parameter, `*_opt`/`get_uncertainty`/`_trn`
      methods, `USE_JAX` flag, legacy golden, and `baselines/legacy/`.

## Backlog (does not block the phases)

- `get_uncertainty_grid_from_proba`: alpha search by grid, as a method separate from the
  binary search (kept to investigate differences). Pending.
- `tuning_stability(probs, y, n_splits)`: diagnostic for tuning-set size adequacy (runs
  the search on disjoint subsets and reports spread). This is the formalization of the
  "L random splits" scheme from the paper.
- Golden test with a trained model (current ones use an untrained model: reproducible but
  in a degenerate regime, unstable alphas).
- Packaging: remove torch from the main dependencies.
- Performance benchmark per phase.

## Canonical migration recipe (new API)

Apply to each script. The canonical alpha-search method is the **binary** one
(`get_uncertainty_from_proba`), which accepts `max_iters` to adjust precision.

1. **Construction**: `UncertaintyQuantifier(N=..., classes=[C], max_batch_size=...)`.
   Do NOT pass `model` (it is deprecated). For the per-class case, one UQ per class.

2. **Calibration (can stream)**: iterate the calibration loader ONCE, batch outside /
   class inside, accumulating with `batched=True`:
```python
   for X_cal, y_cal in calDataLoader:
       p_cal = model.predict_proba(X_cal).cpu().numpy()   # or backend equivalent
       y_cal_arr = flatten_batch(y_cal).ravel().astype(int)
       for C in classes:
           uqs[C].calibrate_from_proba(p_cal, y_cal_arr, batched=True)
```
   This keeps only one batch of logits in memory at a time (important when "samples" are
   pixels, e.g. segmentation).

3. **Tuning (NOT batched-and-averaged)**: materialize the tune set (it is small by CP
   design) and make ONE call per class over the full set:
```python
   tune_probs, tune_y = precompute_logits(tuneDataLoader, model)
   U, alpha = uqs[C].get_uncertainty_from_proba(tune_probs, tune_y, max_iters=30)
```
   FORBIDDEN: `alpha = np.nanmean([alpha_per_batch...])`. This is statistically incorrect:
   alpha is a non-linear function of the data. Averaging over L distinct splits (full
   experimental repetition) IS valid and is a different thing.

4. **Apply alpha (explicit, non-mutating)**: `get_uncertainty_from_proba` is pure and
   does not touch state. The caller sets:
```python
   uqs[C].alpha = alpha
```

5. **Test**: predict and compute coverage as a GLOBAL proportion, not an average of
   per-batch proportions:
```python
   y_p, y_s = uqs[C].predict_from_proba(test_probs)
   # coverage over the full set for class C
```

6. **Precompute logits** once per (noise/split) wherever possible, to eliminate redundant
   model forward passes. This is where the bulk of the speedup in the analysis scripts
   comes from.

## Test conventions

- `tests/core/`: does NOT import torch (or any framework). Inputs are synthetic numpy/jnp.
- `tests/integration/torch/`: may import torch (legitimate).
- Baselines: `tests/integration/torch/baselines/legacy/` (legacy) and
  `tests/integration/torch/baselines/` (new API). Regenerate with
  `regenerate_baselines.py --api {legacy,new}` (paths relative to the file).
- Core property tests that survive Phase 6 (cache, performance smoke) live alongside the
  new-API golden.

## Decisions to respect (do NOT "fix" without confirming)

- In the coverage test scripts, `uq.alpha = U` (setting alpha to the U value, not the
  tuned alpha) is INTENTIONAL: it is part of the alignment tests between U and (1-Cov).
  It may be changed to `alpha` in the future, but it is not a bug.

## Example script inventory

Mapping to paper figures (Marchi & Liebl 2026, Mach. Learn.: Sci. Technol. 7 015017) and
status. The scripts are reproducibility artifacts for the paper (ref [30] of the paper
itself).

| Script | Paper figures | Legacy methods used | Status |
|---|---|---|---|
| `MNIST_class_conditional_example.py` | 11, 12 | calibrate, get_uncertainty_jit, predict | Directly migratable (Phase 4) |
| `ACDC_example.py` | 13–16, tables B1/B2 | calibrate, get_uncertainty, predict | Migratable (per-class, pixels) |
| `convergence_analysis.py` | 7(b) | fit, get_uncertainty | Rewrite |
| `data_size_analysis.py` | 7(a,c) | fit, get_uncertainty_opt | Rewrite |
| `setsize_analysis.py` | 4, 5 | fit, get_uncertainty, predict | Rewrite |
| `MNIST_test_coverage.py` | Appendix A | fit_opt, get_uncertainty_opt, predict_opt | Rewrite |
| `MNIST_test_convergence.py` | Appendix A | fit, get_uncertainty_opt, predict | Rewrite |
| `btorch_MNIST_test.py` | Appendix C | (none — bayesian-torch) | DO NOT TOUCH |

Notes:
- `fit` was the former name of `calibrate`. `*_opt` were "optimized" variants; part of
  their logic was folded into the main methods. These scripts are written against a
  previous API and do NOT run as-is against the current package: migrating them means
  rewriting them against the new API. Note: `fit`, `fit_opt`, and `predict_opt` are
  fully absent from the current `UncertaintyQuantifier` (not merely deprecated).
- `ACDC_example.py`: cardiac model loaded via MONAI bundle (do not touch the loading);
  CP "samples" are pixels (high volume → calibration streaming matters); generates LaTeX
  tables (preserve).

## Architecture: agnostic core, backend-specific integrations

"Backend-agnostic" applies to the CORE, not to the whole package. The package
legitimately contains backend-specific code; what matters is where it lives and the
direction of dependencies.

### Dependency rule

- **Core** (`uncertaintyQuantifier`, alpha-search functions, masked quantile): imports
  ONLY numpy + jax. NEVER imports torch, onnx, or any backend, and NEVER imports from
  the backend subpackages below. Data flow is always:
  user code → backend wrapper → probabilities → core (`*_from_proba`).
- **`utrace.utils.pytorch.*`**: everything that touches torch — `Pytorch_wrapper`,
  example models, dataset loaders, transforms, and any helper that needs torch
  (e.g. `flatten_batch` / `unflatten` if they operate on torch tensors).
- **`utrace.utils.onnx.*`**: analogous, for the ONNX backend.
- **`utrace.utils`** (root): only truly backend-agnostic helpers (pure numpy).
- Backends are **optional extras** in `pyproject.toml` (e.g. `[cpu]`, `[cuda]`).
  The core must be installable and importable WITHOUT torch.

### Placement test for each symbol

Does the function import or assume a backend?
- Yes → it belongs in that backend's subpackage (`utils/pytorch/`, `utils/onnx/`).
- No (pure numpy) → it may stay in `utils/` root.

Known violations in the current code (verified against source):
- `flatten_batch`: currently lives in `utrace.utils.utils` root, but its implementation
  calls torch Tensor methods (`.dim()`, `.permute()`, `.reshape()`), and `utils.py`
  imports `torch` at the top. This means importing `utrace.utils` already pulls in torch,
  breaking torch-free installation. Must move to `utrace.utils.pytorch`; update script
  imports accordingly.
- `get_coverage`: receives `np.ndarray` in scripts and is pure numpy — confirmed safe to
  stay in `utils/` root once `flatten_batch` and the `import torch` are removed from
  `utils.py`.

This rule guides both the script migration (Phase 4) and the packaging cleanup (make
torch an optional extra) tracked in the backlog and Phase 6.

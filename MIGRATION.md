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
- [x] Phase 3 — New `*_from_proba` API (calibrate_from_proba / predict_from_proba /
      get_uncertainty_from_proba); legacy `*(X)` methods deprecated and delegating to
      shared impls. Three goldens: legacy, new-API, and synthetic equivalence.
- [x] Phase 4 — Migrate example scripts to the new API (COMPLETE — all 7 example scripts migrated to *_from_proba; migration-complete does NOT mean numerically validated, see note below).
      Done: MNIST_class_conditional_example.py (merged), ACDC_example.py (migrated, pending numerical validation against the paper), setsize_analysis.py (migrated), convergence_analysis.py (migrated to *_from_proba; core fixes already in), MNIST_example.py (migrated; quick AWGN sweep, 2 iter, reproduces figs 9(a)/10(a): U_bar tracks U_E and (1-Cov), slightly conservative on the Linear model; full multi-degradation / higher-iteration reproduction pending), data_size_analysis.py (migrated; quick num_sizes=8 sweep reproduces fig 7 trends — U converges onto the U_E plane as calibration grows, stabilizing past cal ~1000; tuning size negligible; surface spiky at small sizes; full 40x40 reproduction pending), MNIST_test_coverage.py (migrated; Appendix-A coverage test validated at 10 iterations — KS and Cramér-von Mises pass for both CNN and FC; Linear KS p marginal at ~0.05, higher iteration count advisable for a stronger check), MNIST_test_convergence.py (migrated; aligned to MNIST_test_coverage's Appendix-A convention — predicts at cp.alpha = U and parameterizes the BetaBinom null with U, resolving the script's prior internal alpha/U inconsistency; numerical validation at full iteration count pending). Pending: none.
      Open paper-level validations (separate from migration): convergence_analysis (fig 7b full sweep), ACDC_example (numerical vs paper), MNIST_test_convergence (full 200-iteration run), MNIST_test_coverage (advisable to strengthen beyond 10 iterations).
- [ ] Phase 5 — Migrate remaining state (`_class_scores`, etc.) to `jnp` storage.
- [ ] Phase 6 — Remove legacy API, `model` parameter, `*_opt`/`get_uncertainty`/`_trn`
      methods, legacy golden, and `baselines/legacy/`, remove the `USE_JAX` flag entirely. The JAX-based `_masked_quantile_higher` is part of the core and should always be available; the core should not import a symbol that the utils `__init__` exports only conditionally, and package importability must not depend on an environment flag or on where Python is launched from. Removing the flag also likely removes the need for the in-package `.env`, which is itself unusual

## Backlog (does not block the phases)

- `get_uncertainty_grid_from_proba`: alpha search by grid, as a method separate from the binary search (kept to investigate differences). Pending.
- `tuning_stability(probs, y, n_splits)`: diagnostic for tuning-set size adequacy (runs the search on disjoint subsets and reports spread). This is the formalization of the "L random splits" scheme from the paper.
- Golden test with a trained model (current ones use an untrained model: reproducible but in a degenerate regime, unstable alphas).
- Packaging: remove torch from the main dependencies.
- Performance benchmark per phase.
- Buffer/padding design for high-volume regimes (segmentation): the fixed-size `_max_N` buffer must currently be sized per class by hand. Consider a design that scales without manual sizing (without reintroducing variable shapes / JAX recompilation).

- force_non_empty_sets is silently ignored in the new prediction path. The jit _predict_sets does not implement it, and predict_from_proba accepts the parameter but does not pass it through. The legacy _predict_sets (initial commit) honored it (y_sets[arange, y_pred] = True). This is behavior lost in the jit migration. Harmless for callers passing False, but a latent bug for any script relying on force_non_empty_sets=True.

- [RESOLVED] The global batched branch of _calibrate_impl concatenated conformity scores into the buffer without re-sorting (.at[_N:_N+num].set with no np.sort), while the non-batched and per-class batched branches do sort. _masked_quantile_higher assumes an ascending-sorted buffer, so the tuning quantile (q_hat) became non-monotonic in alpha when calibrating global+batched, breaking the binary search for U (it failed to converge; U  collapsed to 0 or oscillated). Fix: sort the concatenation, matching the per-class branch.
  - The _masked_quantile_higher unit test did not catch this because it is fed an already-sorted array: the bug was in the integration (calibration violating the sort precondition), not in the function itself.
  - Coverage gap: no test exercises the global+batched path. TODO: add a test that calibrates global+batched and asserts the buffer stays sorted.

- [RESOLVED] Per-class calibration double-counted _N: the trailing _N update ran unconditionally and overwrote the correct `_N = total` set inside the per-class branch, adding the last class's num_scores on top (e.g. N=66 for a 60-sample calibration). Fix: move the _N update into the global branch only. Also switched per-class accounting to a per-class count (_class_N) and fixed _class_scores initialization (was np.empty(_max_N), garbage). classes=[full list] now matches classes=None (commit 1a2c8a).

### TODO: make device handling in to_jax() explicit (deferred)

`to_jax()` (utils/tensors.py) currently handles a device mismatch silently.
For a CUDA tensor with a CPU JAX backend, the DLPack path raises, the exception is caught and logged at `debug` level, and execution falls through to a `.cpu().numpy()` copy to host. The result is correct, but the host transfer is invisible: a user who believes they are running zero-copy on GPU is silently paying a copy on every call, with no visible signal.

This contradicts the intended design goal (use the user's backend device by default, with an option to specify the device explicitly).

When addressing device handling (separate task, own branch / design discussion):
- Make a device mismatch explicit rather than silent — a visible warning, an error, or a `device=` parameter controlling the behavior.
- Narrow the `except` around the DLPack call: it currently catches all exceptions and routes them to `logger.debug`, which also hides non-device failures (unsupported dtype, version mismatch, malformed array). Catch only the expected device/DLPack exceptions and let the rest propagate.

Out of scope for the current script-migration work. Recorded here so the context is not lost.

- Two quantile implementations: np.nanquantile(scores[:_N], method='higher') in the alpha setter vs _masked_quantile_higher (jax, assumes sorted input, used by the tuning path via _q_hat_from_alpha) in the tuning path. Unify toward _masked_quantile_higher for consistency, BUT first: (a) verify numerical equivalence against np.nanquantile across several q_level values, edges, and ties; (b) the setter would inherit the  sort precondition, so ensure its buffer is sorted.
_masked_quantile_higher is called only from the tuning fori_loop, which is why it does not sort internally: re-sorting unchanging data on each of the ~30 iterations would be wasteful. The sort precondition is the price of that optimization; prefer a sortedness assertion in _calibrate_impl (cheap, once per alibration) over sorting inside the loop.

- Performance: _calibrate_impl sorts the full buffer on every batch (O(N log N) per batch). For large datasets (ACDC) it would be better to sort once when calibration is finalized, not per batch.

- Zero-copy in tuning: `get_uncertainty_from_proba` does `np.asarray(to_jax(...))`
  (`uncertaintyQuantifier.py:431`), forcing a host copy and negating DLPack zero-copy on the tuning path; `calibrate_from_proba` / `predict_from_proba` keep zero-copy. Make tuning consume the jnp array directly (see the TODO at :430).

- Disconnected `transform` parameter in MNIST_example.py: main() receives a `transform`  argument but the noise injection (~:176) uses a hardcoded `AddGaussianNoise`, ignoring it — so the __main__ transform_str dispatch (AWGN/RandomPerspective/ElasticTransform) currently has no effect on the experiment; AWGN is always applied. Likely a remnant of the lambda->class migration done to support num_workers>0 (a lambda transform is not picklable   and breaks multi-worker DataLoaders). To resolve: decide whether to reconnect the transform  sweep (as other scripts do) or whether fixed-AWGN is intentional for this script. If  reconnecting, note the three transforms have different signatures (AddGaussianNoise(0., n), RandomPerspective(n, 1), ElasticTransform(n)), so the swept parameter must be mapped per signature — this is a behavior change, warranting its own commit and revalidation. Separate from the I/O refactor.

## Canonical migration recipe (new API)

Apply to each script. The canonical alpha-search method is the **binary** one (`get_uncertainty_from_proba`), which accepts `max_iters` to adjust precision.

1. **Construction**: `UncertaintyQuantifier(N=..., classes=[C], max_batch_size=...)`.
   Do NOT pass `model` (it is deprecated). For the per-class case, one UQ per class.
   **Buffer sizing for high-volume cases (segmentation).** The default `N=1000` is correct for moderate sample counts (classification, MNIST). In segmentation each image   contributes ~65k pixel-samples, so the fixed-size padded buffer (`_max_N`) overflows.
   Size `N` per class from the data: one pass over the full dataset counts per-class samples (invariant to noise — the GT is not transformed), then `N_class_C = ceil(count_C * cal_fraction * margin)` (e.g. cal_fraction=0.2, margin~1.5).
   Background gets a large buffer, the structures small ones. The fixed-size padding from Phase 2 assumes moderate volumes; a buffer design that scales to high-volume regimes is a possible future task (backlog).

2. **Calibration (can stream)**: iterate the calibration loader ONCE, batch outside /
   class inside, accumulating with `batched=True`:
```python
   for X_cal, y_cal in calDataLoader:
       p_cal = model.predict_proba(X_cal)        # tensor; to_jax handles DLPack conversion
       y_cal_arr = flatten_batch(y_cal).ravel()  # flatten batch/spatial dims (labels)
       for C in classes:
           uqs[C].calibrate_from_proba(p_cal, y_cal_arr, batched=True)
```
   This keeps only one batch of logits in memory at a time (important when "samples" are pixels, e.g. segmentation).

3. **Tuning (NOT batched-and-averaged)**: materialize the tune set (it is small by CP design) and make ONE call per class over the full set:
```python
   tune_probs, tune_y = precompute_logits(tuneDataLoader, model)
   U, alpha = uqs[C].get_uncertainty_from_proba(tune_probs, tune_y, max_iters=30)
```
   FORBIDDEN: `alpha = np.nanmean([alpha_per_batch...])`. This is statistically incorrect:
   alpha is a non-linear function of the data. Averaging over L distinct splits (full experimental repetition) IS valid and is a different thing.

4. **Apply alpha (explicit, non-mutating)**: `get_uncertainty_from_proba` is pure and
   does not touch state. The caller sets:
```python
   uqs[C].alpha = alpha
```

5. **Test**: predict and compute coverage as a GLOBAL proportion, not an average of per-batch proportions:
```python
   y_p, y_s = uqs[C].predict_from_proba(test_probs)
   # coverage over the full set for class C
```

6. **Precompute logits** once per (noise/split) wherever possible, to eliminate redundant model forward passes. This is where the bulk of the speedup in the analysis scripts comes from.

### Passing tensors to the core

Pass backend tensors directly to the `*_from_proba` methods. `to_jax()` (in `utils/tensors.py`) handles conversion via DLPack: a CPU PyTorch tensor is consumed zero-copy. Do NOT call `.cpu().numpy()` manually on values that feed `calibrate_from_proba` / `predict_from_proba` / `get_uncertainty_from_proba` — that conversion is the library's job, and writing it by hand defeats the zero-copy path and clutters the example.

Note: `.cpu().numpy()` is still legitimate for values that do NOT go into the core (e.g. computing accuracy, building arrays for matplotlib). Only remove the conversions on the path to `*_from_proba`.

## Test conventions

- `tests/core/`: does NOT import torch (or any framework). Inputs are synthetic numpy/jnp.
- `tests/integration/torch/`: may import torch (legitimate).
- Baselines: `tests/integration/torch/baselines/legacy/` (legacy) and `tests/integration/torch/baselines/` (new API). Regenerate with `regenerate_baselines.py --api {legacy,new}` (paths relative to the file).
- Core property tests that survive Phase 6 (cache, performance smoke) live alongside the new-API golden.

## Decisions to respect (do NOT "fix" without confirming)

- In the coverage test scripts, `uq.alpha = U` (setting alpha to the U value, not the tuned alpha) is INTENTIONAL: it is part of the alignment tests between U and (1-Cov). It may be changed to `alpha` in the future, but it is not a bug.

- Per-class branch with classes=[multiple] is an UNFINISHED feature, not a supported mode: _predict uses a single global q_hat, so passing the full class list must behave identically to classes=None (verified by equivalence test). Do not rely on per-class quantiles until the meta-class semantics are designed. See 1a2c8a for the _N accounting fix that restored the global-equivalence.

## Post-migration analysis (open items)

1. U-vs-alpha for the two Appendix-A scripts (`MNIST_test_coverage.py`, `MNIST_test_convergence.py`): both now use `U` (not the tuned alpha) on BOTH the prediction threshold (`cp.alpha = U`) and the BetaBinom null parameter (`a_p = U_mean`), per the `uq.alpha = U` decision above. Whether the tuned alpha is preferable instead is an open question — revisit once, for both scripts together, not independently.
2. BetaBinom null fragility: in both Appendix-A scripts, `Nr` (= `Nv`) is taken from only the last loop iteration's test-set size, while per-iteration sizes vary by ~1 sample due to `random_split`'s remainder rounding. The null distribution's trial count is therefore a (very close) approximation, not exact, across all recorded iterations.

## Legacy method state (discovered during ACDC migration)

Only the legacy subset exercised by the legacy golden — `calibrate`, `get_uncertainty_jit`, `predict` — is maintained and tested. The rest of the legacy surface is broken against the
current core and has NO test coverage:
- `get_uncertainty` (no suffix) calls `_predict_sets`, which no longer exists → raises `AttributeError` at runtime.
- `get_uncertainty_opt` IS defined (`uncertaintyQuantifier.py:435-490`) but is model-bound
  (calls `self.model.predict_proba`) and has no `*_from_proba` counterpart; it raises
  `AttributeError` if the UQ is constructed without `model=`, so it is unusable under the
  no-model migration.
- `fit_opt`, `predict_opt`, `fit` are absent entirely.

Consequence: any script using these does NOT run against the current package and cannot produce a local reference. Such scripts (convergence_analysis, data_size_analysis, setsize_analysis, MNIST_test_coverage, MNIST_test_convergence) are full rewrites — validate them against the paper, not a prior local run. Tests passing does NOT imply these methods work; tests cover the core, not the scripts, and not the unexercised legacy paths. All of this is removed in Phase 6.

## Example script inventory

Mapping to paper figures (Marchi & Liebl 2026, Mach. Learn.: Sci. Technol. 7 015017) and status. The scripts are reproducibility artifacts for the paper (ref [30] of the paper itself).

| Script | Paper figures | Legacy methods used | Status |
|---|---|---|---|
| `MNIST_class_conditional_example.py` | 11, 12 | calibrate, get_uncertainty_jit, predict | Directly migratable (Phase 4) |
| `MNIST_example.py` | 9, 10 | calibrate, get_uncertainty_opt, predict | Migrated (Phase 4) |
| `ACDC_example.py` | 13–16, tables B1/B2 | calibrate, get_uncertainty, predict | Migratable (per-class, pixels) |
| `convergence_analysis.py` | 7(b) | fit, get_uncertainty | Rewrite |
| `data_size_analysis.py` | 7(a,c) | fit, get_uncertainty_opt | Migrated (Phase 4) |
| `setsize_analysis.py` | 4, 5 | fit, get_uncertainty, predict | Rewrite |
| `MNIST_test_coverage.py` | Appendix A | fit_opt, get_uncertainty_opt, predict_opt | Migrated (Phase 4) |
| `MNIST_test_convergence.py` | Appendix A | fit, get_uncertainty_opt, predict | Migrated (Phase 4) |
| `btorch_MNIST_test.py` | Appendix C | (none — bayesian-torch) | DO NOT TOUCH |

Notes:
- `fit` was the former name of `calibrate`. `*_opt` were "optimized" variants; part of their logic was folded into the main methods. These scripts are written against a previous API and do NOT run as-is against the current package: migrating them means rewriting them against the new API. Note: `fit`, `fit_opt`, and `predict_opt` are fully absent from the current `UncertaintyQuantifier` (not merely deprecated).
- `ACDC_example.py`: cardiac model loaded via MONAI bundle (do not touch the loading); CP "samples" are pixels (high volume → calibration streaming matters); generates LaTeX tables (preserve).

## Validation notes (analysis scripts)

Alpha-sweep / set-size scripts (setsize_analysis, and likely the others) validate a METHOD CLAIM, not a pixel-exact reproduction of the paper figures.
What the figure asserts is that the prediction-set size distribution follows the theoretical Beta — the histogram should track the Beta overlay. It is NOT expected to be bit-identical to the paper.

- The variability between runs comes from the MODEL (the trained network), not from the calibration split. Verified empirically on setsize_analysis: with a fixed model, changing the random_split seed leaves the result unchanged; with a fixed split, different model trainings change it. With ~12k calibration samples the split barely moves the quantile; the model's score distribution is what matters.
- For a simpler model (e.g. the FC) the figure reproduces the paper closely. For a complex one (CNN), it does not match bit-for-bit and should not be expected to — U-TraCE estimates the uncertainty of the given model, which is not the paper's exact model.
- The degree of training affects set-size concentration: an over-trained CNN saturates (scores pushed to the 0/1 extremes, U-shaped), producing more concentrated sets. setsize_analysis uses 10 epochs (not 20) for a closer match to the paper's Beta fit.
- For reproducible figures, fix BOTH seeds: the model (torch.manual_seed beforeinstantiation + train-loader generator) and the random_split. Same model seed
  -> identical weights hash.
- convergence_analysis (paper fig 7b): validates that U converges to the empirical error (1 - accuracy) as calibration size grows. Noisy at small calibration (expected), settles onto the flat 1-Cov / Ue lines at large calibration. Reference: post-fix run converges to U ~ 0.05 / 0.50 / 0.75 / 0.81 for sigma_n = 0 / 0.75 / 1.25 / 2.0. Requires both core fixes (sort + _N).

## Architecture: agnostic core, backend-specific integrations

"Backend-agnostic" applies to the CORE, not to the whole package. The package legitimately contains backend-specific code; what matters is where it lives and the direction of dependencies.

### Dependency rule

- **Core** (`uncertaintyQuantifier`, alpha-search functions, masked quantile): imports ONLY numpy + jax. NEVER imports torch, onnx, or any backend, and NEVER imports from the backend subpackages below. Data flow is always: user code → backend wrapper → probabilities → core (`*_from_proba`).
- **`utrace.utils.pytorch.*`**: everything that touches torch — `Pytorch_wrapper`, example models, dataset loaders, transforms, and any helper that needs torch (e.g. `flatten_batch` / `unflatten` if they operate on torch tensors).
- **`utrace.utils.onnx.*`**: analogous, for the ONNX backend.
- **`utrace.utils`** (root): only truly backend-agnostic helpers (pure numpy).
- Backends are **optional extras** in `pyproject.toml` (e.g. `[cpu]`, `[cuda]`). The core must be installable and importable WITHOUT torch.

### Placement test for each symbol

Does the function import or assume a backend?
- Yes → it belongs in that backend's subpackage (`utils/pytorch/`, `utils/onnx/`).
- No (pure numpy/Jax) → it may stay in `utils/` root.

### Current state (rule enforced)

The dependency rule above is already enforced in the code:
- Torch-dependent helpers — `flatten_batch`, `unflatten_batch` (typo `unflatten_bath` fixed), `unflatten_pixels`, `unflatten_set_sizes`, `view_classify` — live in `utils/pytorch/`. `utils.py` is torch-free.
- Pure-numpy helpers (`get_coverage`, `relabel`, `check_row_sums`, etc.) remain in `utils/` root.
- One residual torch dependency remains in the core: `uncertaintyQuantifier.py` imports `flatten_batch` (from `utils/pytorch/`) for the deprecated `calibrate` path only. This is removed in Phase 6 with `calibrate`. Until then it is expected, not a violation to "fix" now.

This rule guides both the script migration (Phase 4) and the packaging cleanup (make torch an optional extra) tracked in the backlog and Phase 6.

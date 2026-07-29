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
- [x] Phase 5 — Migrate remaining state to `jnp` storage. Closure audit: zero array-valued object state remains on numpy. `conformity_scores_` is the only persistent array state and is a jnp buffer. Achieved via (1) the quantile swap — host-side `np.nanquantile` replaced by the jit `_masked_quantile_higher` in the alpha setter (commit 0f8512f) — and (2) defer-sort — new scores are written into the fixed jnp buffer without per-batch sorting; a single `jnp.sort` runs lazily on first read via the `conformity_scores_` property, gated by a `_sorted` dirty flag (commit 4852c3b). Deliberate exclusion: `__alpha` and `__q_hat` REMAIN `np.float64` host scalars — they are Python-level static args (never traced inside `jit`), not array state, so they are out of scope for "state → jnp" by design. NOTE: "migrating state to jnp" is DONE; this does NOT mean the core is numpy-free. Residual scalar/host-prep numpy (label arrays, class masks, staging buffers on transient locals) is intentional and untouched by this phase. The remaining numpy round-trips are Phase-6 perf/cleanup items (see Backlog), not Phase-5 state.
- [ ] Phase 6 — Remove legacy API, `model` parameter, the `USE_JAX` flag, and the legacy
      golden/baselines. Revised step ordering below (supersedes the original single-item
      plan above; re-derived after a diagnostic pass found the original order had an
      avoidable dependency and after the first two steps were already executed):
      1. DONE — `_max_N` overflow guard.
      2. DONE — remove dead legacy: `get_uncertainty`, `get_uncertainty_opt`, `get_U`.
      3. Remove USE_JAX: `src/utrace/__init__.py`, `utils/__init__.py`, `scores/__init__.py`,
         the two call sites in `uncertaintyQuantifier.py`, `config.py`, `.env`, `.env.example`,
         `scores/numpy_impl.py`, and the `python-dotenv` dependency.
         Rationale: moved earlier (was step 5) because it was established that the USE_JAX=false
         branch is unreachable, which makes this change behaviour-preserving by construction.
      4. Remove the legacy tests, `baselines/legacy/`, the `--api legacy` branch of
         `regenerate_baselines.py`, and the `LEGACY_BASELINE_DIR` entry in `_baselines.py`.
         Note: no property tests need extracting first — the two that survive Phase 6
         (`test_jax_cache_does_not_grow`, `test_golden_run_under_threshold`) are commented-out dead
         code in `test_golden_mnist.py` and already live in `test_golden_mnist_new_api.py`.
      5. Remove `calibrate`, `predict`, `get_uncertainty_jit`, and the `flatten_batch` import.
         This is what makes the core importable without torch.
      6. Remove the `model=` constructor parameter. Note: this step DOES require deleting two
         tests (`tests/core/test_from_proba_api.py` and
         `tests/integration/torch/test_legacy_equivalence.py` both test the `model=` deprecation
         warning); it is the only Phase 6 step that requires a test edit.
      7. Labels host-copy: remove the `# COMPAT` numpy round-trip in the core and in scripts.
      8. `flatten_batch` -> `flatten_to_pixels` unification.

      (The original text of this item also mentioned removing "`_trn`" methods; no symbol
      matching `_trn` exists anywhere in `src/`, `tests/`, or `scripts/` as of this pass —
      dropped from the ordering above as unverifiable rather than carried forward.)

## Backlog (does not block the phases)

- `get_uncertainty_grid_from_proba`: alpha search by grid, as a method separate from the binary search (kept to investigate differences). Pending.
- `tuning_stability(probs, y, n_splits)`: diagnostic for tuning-set size adequacy (runs the search on disjoint subsets and reports spread). This is the formalization of the "L random splits" scheme from the paper.
- Golden test with a trained model (current ones use an untrained model: reproducible but in a degenerate regime, unstable alphas).
- Packaging cleanup (post-Phase 6): see the index/extras finding under "Phase 6 diagnostic findings". Note that torch is already absent from `[project].dependencies` - it only appears in the optional-dependency groups. What actually keeps torch mandatory is that the core imports `flatten_batch` at module level; Phase 6 step 5 removes that.
- Performance benchmark per phase.
- Buffer/padding design for high-volume regimes (segmentation): the fixed-size `_max_N` buffer must currently be sized per class by hand. Consider a design that scales without manual sizing (without reintroducing variable shapes / JAX recompilation).
- [DONE] `_max_N` overflow guard (Phase 6 step 1): `_calibrate_impl` now checks capacity before writing, in both branches, raising `ValueError` instead of allowing an out-of-bounds write. The pre-fix diagnostic found the failure was not one mode but three distinct behaviours, empirically reproduced:
  1. Batched write whose start index (`_N`) is already `>= _max_N`: JAX's `.at[].set()` silently dropped the update; `_N` still incremented regardless; neither reading `conformity_scores_` afterward nor setting `alpha` raised anything — the only fully silent failure of the three, and the one a caller relying on repeated `batched=True` streaming calibration would hit first.
  2. Batched write straddling the boundary (starts in-bounds, overruns past `_max_N`): raised `ValueError` from JAX broadcasting (the in-bounds slice clips shorter than the incoming values).
  3. Non-batched write with `num_scores > _max_N` in a single call: same `ValueError` as (2).
  Both branches of `_calibrate_impl` now check `_N + num_scores <= _max_N` (batched) / `num_scores <= _max_N` (non-batched) before any state mutation, so all three cases now raise consistently instead of only two of three doing so. Cross-references the manual buffer-sizing item above — a design that scales sizing automatically would still need this guard as a backstop.

- force_non_empty_sets is silently ignored in the new prediction path. The jit _predict_sets does not implement it, and predict_from_proba accepts the parameter but does not pass it through. The legacy _predict_sets (initial commit) honored it (y_sets[arange, y_pred] = True). This is behavior lost in the jit migration. Harmless for callers passing False, but a latent bug for any script relying on force_non_empty_sets=True.

- [RESOLVED] The global batched branch of _calibrate_impl concatenated conformity scores into the buffer without re-sorting (.at[_N:_N+num].set with no np.sort), while the non-batched and per-class batched branches do sort. _masked_quantile_higher assumes an ascending-sorted buffer, so the tuning quantile (q_hat) became non-monotonic in alpha when calibrating global+batched, breaking the binary search for U (it failed to converge; U  collapsed to 0 or oscillated). Fix: sort the concatenation, matching the per-class branch.
  - The _masked_quantile_higher unit test did not catch this because it is fed an already-sorted array: the bug was in the integration (calibration violating the sort precondition), not in the function itself.
  - [RESOLVED] Coverage gap: no test exercises the global+batched path. `tests/core/test_deferred_sort_buffer.py::test_property_sorts_on_read` calibrates `classes=None` with `batched=True` across 3 separate calls and, after reading `conformity_scores_`, asserts the valid prefix is ascending-sorted; `test_no_sort_between_batches` covers the same global+batched shape (4 calls) and asserts no sort happens between batches. Together they cover the previously-uncovered global (`classes=None`) batched path (commit 4852c3b).

- [RESOLVED] Per-class calibration double-counted _N: the trailing _N update ran unconditionally and overwrote the correct `_N = total` set inside the per-class branch, adding the last class's num_scores on top (e.g. N=66 for a 60-sample calibration). Fix: move the _N update into the global branch only. Also switched per-class accounting to a per-class count (_class_N) and fixed _class_scores initialization (was np.empty(_max_N), garbage). classes=[full list] now matches classes=None (commit 1a2c8a).

- [RESOLVED] to_jax device mismatch on GPU backends. to_jax routed any object with __dlpack__ through jax.dlpack.from_dlpack; numpy arrays implement __dlpack__, so numpy label arrays landed on CPU (DLPack preserves host origin) while CUDA torch probability tensors landed on GPU. The jitted score (lac_cal) then received its two arguments on different devices and raised "Received incompatible devices for jitted computation". The numpy DLPack path also emitted a "buffer is not aligned, creating a copy" warning (neither zero-copy nor correct-device). Invisible on the CPU backend; only reproduces with a GPU JAX backend. Fix: check isinstance(np.ndarray) BEFORE the __dlpack__ branch, route numpy via jnp.asarray (lands on JAX default compute device); DLPack kept only for genuine framework tensors. Device contract documented in the to_jax docstring: preserve device for tensors, normalize host arrays to the default compute device, do not reconcile mismatches between two genuine tensors. Validated on CUDA (to_jax(numpy)->GPU, to_jax(cuda tensor)->GPU, MNIST_example --extra=cuda matches CPU results). NOTE: the test suite runs on CPU and does NOT exercise this path; it only guards against regression.

### TODO: make device handling in to_jax() explicit (deferred)

`to_jax()` (utils/tensors.py) currently handles a device mismatch silently.
For a CUDA tensor with a CPU JAX backend, the DLPack path raises, the exception is caught and logged at `debug` level, and execution falls through to a `.cpu().numpy()` copy to host. The result is correct, but the host transfer is invisible: a user who believes they are running zero-copy on GPU is silently paying a copy on every call, with no visible signal.

This contradicts the intended design goal (use the user's backend device by default, with an option to specify the device explicitly).

When addressing device handling (separate task, own branch / design discussion):
- Make a device mismatch explicit rather than silent — a visible warning, an error, or a `device=` parameter controlling the behavior.
- Narrow the `except` around the DLPack call: it currently catches all exceptions and routes them to `logger.debug`, which also hides non-device failures (unsupported dtype, version mismatch, malformed array). Catch only the expected device/DLPack exceptions and let the rest propagate.

Out of scope for the current script-migration work. Recorded here so the context is not lost.

- [RESOLVED] Two quantile implementations: np.nanquantile(scores[:_N], method='higher') in the alpha setter vs _masked_quantile_higher (jax, assumes sorted input, used by the tuning path via _q_hat_from_alpha) in the tuning path. Unified toward `_masked_quantile_higher` in the alpha setter (commit 0f8512f). Prerequisite (a) numerical equivalence: verified by the committed test `tests/core/test_alpha_setter_quantile_equiv.py`, which pins exact float64 equality between the two paths across N in {1,2,5,10,100,600,1200,5000,50000} and 502 alpha values each, including tied buffers and the cap boundary. Prerequisite (b) sort precondition: the setter now reads `conformity_scores_` through the lazy-sort property (commit 4852c3b), so the sortedness precondition is satisfied by the defer-sort mechanism itself rather than a per-call assertion.
_masked_quantile_higher is called only from the tuning fori_loop, which is why it does not sort internally: re-sorting unchanging data on each of the ~30 iterations would be wasteful. The sort precondition is the price of that optimization; prefer a sortedness assertion in _calibrate_impl (cheap, once per alibration) over sorting inside the loop.

- [RESOLVED]/[DONE] Performance: _calibrate_impl sorts the full buffer on every batch (O(N log N) per batch). For large datasets (ACDC) it would be better to sort once when calibration is finalized, not per batch. Implemented as defer-sort (commit 4852c3b): new scores are written into the fixed buffer at `_conformity_scores_[_N:_N+num_scores]` without per-batch sorting; a single `jnp.sort` runs lazily on first read of `conformity_scores_`, gated by the `_sorted` flag. MEASURED on RTX 3070 (CPU-unmeasured): streaming calibration ~357x faster than the prior sort-per-batch design at ~2M-score ACDC scale (7.2s → 20ms).

- [Phase 6] Zero-copy in tuning: `get_uncertainty_from_proba`'s body does `np.asarray(to_jax(...))`, forcing a host copy and negating DLPack zero-copy on the tuning path; `calibrate_from_proba` / `predict_from_proba` keep zero-copy. Make tuning consume the jnp array directly (see the adjacent bare `# TODO: ... espera numpy` comment — no symbol name to anchor to; that comment and the call sit at `uncertaintyQuantifier.py:417-418` as of HEAD `ebc5ddb`, but re-verify by symbol/grep rather than trusting that number after further commits). Re-confirmed still present as of this pass; perf impact is UNMEASURED (the RTX 3070 GPU benchmark above measured the calibration path, not the tuning/uncertainty path).

- Disconnected `transform` parameter in MNIST_example.py: main() receives a `transform`  argument but the noise injection (~:176) uses a hardcoded `AddGaussianNoise`, ignoring it — so the __main__ transform_str dispatch (AWGN/RandomPerspective/ElasticTransform) currently has no effect on the experiment; AWGN is always applied. Likely a remnant of the lambda->class migration done to support num_workers>0 (a lambda transform is not picklable   and breaks multi-worker DataLoaders). To resolve: decide whether to reconnect the transform  sweep (as other scripts do) or whether fixed-AWGN is intentional for this script. If  reconnecting, note the three transforms have different signatures (AddGaussianNoise(0., n), RandomPerspective(n, 1), ElasticTransform(n)), so the swept parameter must be mapped per signature — this is a behavior change, warranting its own commit and revalidation. Separate from the I/O refactor.

- [Phase 6] Labels passed to the *_from_proba API still go through .numpy() in the canonical recipe and in all example scripts (e.g. flatten_batch(y).ravel().numpy().astype(int)). This violates the "do not call .cpu().numpy() on values feeding the core" rule and, after the to_jax fix, forces a host->device hop (correct device now, but not zero-copy). Task: drop the .numpy() so labels stay backend tensors (DLPack zero-copy, same device as probas), moving the int-dtype guarantee elsewhere (torch .long() before the call, or the core ensuring int). Touches the recipe section AND the scripts — its own design + commit. The six call sites are now marked `# COMPAT` (grep COMPAT scripts/) — the cleanup is to drop those lines and keep labels as zero-copy tensors, verifying downstream label indexing (coverage counts, masks, get_coverage) still works with tensor labels.

- to_jax DLPack unaligned-copy: even for genuine tensors the DLPack path can emit "buffer is not aligned ... Creating a copy", so zero-copy is not guaranteed. Decide whether to make such copies VISIBLE (warn/error) rather than silent. Connects to the existing to_jax device-handling backlog item. Perf/observability task.

- User-configurable target device for to_jax (like torch's device=): host arrays currently go to JAX's default compute device; a future API should let the user choose. The current fix is written so the default-device path is the single point a future device= would generalize.

- Noise-sweep scripts rebuild the dataset (and DataLoader) inside the iteration loop, partly to reshuffle the split per iteration and partly to change the noise level. Reconstructing the full dataset per iteration is wasteful — only the noise (and the split) need to change, not the 60000-sample base. Optimization: instantiate the base dataset (and loader) ONCE outside the loop, and inside the loop either mutate the transform's sigma (transform.std sigma — valid because AddGaussianNoise reads self.std in __call__, not __init__) or reassign it (dataset.transform = AddGaussianNoise(0., sigma)). IMPORTANT: the random_split must STAY inside the loop (with a varying generator) to preserve per-iteration reshuffling — only the dataset/loader construction moves out. Caveat: mutating transform.std from the main process only propagates with num_workers=0; with spawn/fork workers, each worker holds its own copy and the loader would need rebuilding (ties into the num_workers decision). Applies to several sweep scripts (MNIST_class_conditional, and others with a noise sweep). Behavior-adjacent — revalidate numbers after the change. Its own diagnostic + commit.

### GPU / scalability (example scripts)

- [DONE] precompute_proba helper consolidated. scripts/_common.py now provides precompute_proba(loader, classifier) returning raw torch tensors (torch.cat of probs and labels, no conversion) so probabilities take the zero-copy DLPack path into to_jax. Adopted in the six MNIST-like scripts (MNIST_example, MNIST_class_conditional, MNIST_test_coverage, MNIST_test_convergence, convergence_analysis, data_size_analysis). ACDC_example (pixel-scale segmentation) and setsize_analysis (non-batched calibration use) intentionally keep their own logic. Faithful dedup: in the five scripts whose labels were numpy, a temporary compatibility line `flatten_batch(y).ravel().numpy().astype(int)` tagged `# COMPAT` preserves current behavior; convergence_analysis already consumed tensor labels and has no COMPAT line.
- Forward-pass batch vs jit padding are SEPARATE knobs; do not tie them. The DataLoader batch size only chunks the model forward (no effect on results — the tune set is re-concatenated and passed whole to get_uncertainty_from_proba). max_batch_size is the jit padding and must be >= the materialized tune set. On an 8GB GPU the OOMs were ALWAYS in the model forward (predict_proba), never in the utrace core/tuning. Scripts pin max_batch_size to a hardcoded constant (e.g. 12000) tied to the 0.2 tune split of 60k MNIST; prefer deriving it (ceil(tune_split * len(dataset)) + margin) instead of a magic number.
- [DONE for now] DataLoader num_workers set to 0 in MNIST_class_conditional_example (the only script that used workers; ACDC already used 0, the rest default to 0). Reason: num_workers>0 forks, and forking after JAX has initialized its threads can deadlock (generic fork-with-multithreading hazard, not a JAX bug). Resolved at the root by disabling workers. Deferred alternatives if workers are wanted back (e.g. if data loading becomes the bottleneck rather than the GPU forward pass): (a) spawn start method — robust but pays interpreter startup cost, requires picklable transforms (already satisfied: lambda->AddGaussianNoise) and the __main__ guard (already present); (b) lazy JAX initialization so all worker forks happen before XLA threads start — fragile/non-deterministic, NOT recommended. Decide by measuring workers=4 vs 0 wall-time on GPU first (the per-class script is likely GPU-forward-bound, so workers may add little). A permanent user-facing note belongs in docs/ (future), since this affects anyone using torch DataLoaders alongside the package — MIGRATION.md is process log only.
- 8GB VRAM is a hard constraint, not a bug: the per-class script (10 CPs) runs but only just fits with small forward batches. Not something to "fix"; scripts should scale by config.

## Phase 6 diagnostic findings (empirically verified, not yet acted on)

- Running with `USE_JAX=false` raises `ImportError` at package-import time, reproduced directly (`USE_JAX=false uv run --extra=cpu python -c "import utrace"`):
  ```
  File "src/utrace/__init__.py", line 2, in <module>
      from .uncertaintyQuantifier import UncertaintyQuantifier
  File "src/utrace/uncertaintyQuantifier.py", line 16, in <module>
      from .utils import _masked_quantile_higher, _bucket_size
  ImportError: cannot import name '_masked_quantile_higher' from 'utrace.utils'
  ```
  `utils/__init__.py` only defines `_masked_quantile_higher` inside `if USE_JAX:`. Consequently, `scores/numpy_impl.py` is unreachable code today: no configuration can load it — the package fails to import before `scores/__init__.py`'s own `if USE_JAX:` branch would even matter.
- Consequence: `score='aps'` is non-functional in every currently-reachable configuration. `scores/jax_impl.py`'s `aps`/`aps_cal` both unconditionally `raise NotImplementedError()`; the numpy implementation that does implement them (`scores/numpy_impl.py`) can never load, per the point above. Open decision, not resolved here: drop `'aps'` from the constructor's accepted `score` values, or implement it under JAX. Flagging for Phase 6 step 3 (USE_JAX removal), since that step touches `scores/numpy_impl.py` directly and will have to decide this one way or the other.
- `src/utrace/.env.example` ships `USE_JAX=False`; the actual development `src/utrace/.env` in this checkout has `USE_JAX=True`. A fresh clone following the repo's own example file would get a package that raises `ImportError` on `import utrace`, per the first point above.
- `scores/numpy_impl.py`'s `lac_cal` is not actually numpy-typed despite the module name: `return 1 - smx[np.arange(len(y)), y].cpu().numpy().astype(np.float64)` calls `.cpu().numpy()` on the indexed result, which only works if `smx` is a torch tensor at that point — a plain `np.ndarray` has neither method. Currently unreachable per the first point above, so this doesn't bite today, but it means "numpy_impl" was already a misnomer even when the branch was reachable.
- x64 precision is enabled and verified working (checked directly: `jnp.zeros(1, dtype=jnp.float64).dtype` is `float64`) even though `src/utrace/__init__.py` imports `.uncertaintyQuantifier` (line 2) BEFORE calling `jax.config.update("jax_enable_x64", True)` (lines 13-15, itself gated behind `if USE_JAX:`). This works because nothing at import time of `uncertaintyQuantifier.py` actually computes a float64 array before the flag gets set — the module only *defines* jit functions and the class at import time; no array materializes until the caller uses the class. Preserve this ordering (or verify the replacement is equivalent) when the flag becomes unconditional in Phase 6 step 3.
- Packaging: `uv.lock` and `pyproject.toml` are knowingly out of sync as of Phase 6 step 3c, and this is deliberate — do not "fix" it by reverting the pyproject change. The `torch-rocm` index declared `eexplicit = true` (a typo). Unrecognised, the key left that index general-purpose, so it was serving ordinary packages: regenerating the lock reassigned `numpy`, `pillow` and `fsspec` from `pypi.org/simple` to `download.pytorch.org/whl/rocm6.4` - same versions, same content hashes, different attributed origin. The typo is now fixed, but the committed lock was resolved while  it was still in effect, so the lock is no longer derivable from the pyproject. Correcting the spelling makes `uv lock` FAIL for the `rocm5` extra: `pytorch-triton-rocm` is a transitive dependency of torch, is not mapped in `[tool.uv.sources]`, and is not on any explicit index; and `rocm5` pins torch 2.3.x, which has no cp314 wheels under `requires-python = ">=3.11,<3.15"`. Separately, `jax[rocm]` does not exist (`uv lock` warns: the package jax==0.9.2 does not have an extra named rocm). uv stops at the first unsatisfiable split, so there may be more behind it.
  Consequence: `python-dotenv` remains in `dependencies` even though its only consumer (`config.py`) was deleted in step 3b - removing it forces a re-resolve, which is what surfaced all of the above. It is inert (nothing imports it), so it is deferred rather than forced through.
  Deferred to a packaging cleanup after Phase 6, deliberately kept out of the phase so that a moving golden in step 5 has only one candidate cause. That cleanup should decide, per extra, whether it is actually supported — `rocm`/`rocm5` look aspirational - and its success criterion is that every declared extra resolves, not merely that `uv lock` exits zero.

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

- **SUPERSEDED**: The per-class branch with a *full* class list was equivalent to `classes=None` (commit 1a2c8a); with a *partial* list it already calibrated group-conditionally on the listed classes only. We have formalized this as group semantics: `classes=[labels]` calibrates on the subpopulation whose label ∈ labels (group-conditional coverage); `classes=None` is marginal. Multiple classes are fully supported under these semantics, and the internal per-class buffers (`_class_scores`, `_class_N`, `_class_alphas`, `_class_q_hats`) have been removed.

## Post-migration analysis (open items)

1. U-vs-alpha for the two Appendix-A scripts (`MNIST_test_coverage.py`, `MNIST_test_convergence.py`): both now use `U` (not the tuned alpha) on BOTH the prediction threshold (`cp.alpha = U`) and the BetaBinom null parameter (`a_p = U_mean`), per the `uq.alpha = U` decision above. Whether the tuned alpha is preferable instead is an open question — revisit once, for both scripts together, not independently.
2. BetaBinom null fragility: in both Appendix-A scripts, `Nr` (= `Nv`) is taken from only the last loop iteration's test-set size, while per-iteration sizes vary by ~1 sample due to `random_split`'s remainder rounding. The null distribution's trial count is therefore a (very close) approximation, not exact, across all recorded iterations.
3. GPU validation for per-class calibration: the per-class calibration path (classes=[...]) has not been fully validated on a GPU backend end-to-end; only the global path (classes=None, MNIST_example) has a clean GPU run. MNIST_class_conditional ran on GPU only with ad-hoc batch tuning (a probe, not the final structure). This remains an open GPU-validation item.

## Legacy method state (discovered during ACDC migration)

Only the legacy subset exercised by the legacy golden — `calibrate`, `get_uncertainty_jit`, `predict` — is maintained and tested. These three, plus the `model=` constructor parameter, are still present, pending Phase 6 steps 5-6.

Two further legacy methods were found broken/orphaned during the ACDC migration and have since been **removed** (Phase 6 step 2; confirmed absent from `src/utrace/uncertaintyQuantifier.py` as of current HEAD — grepping either name in `src/` returns nothing):
- `get_uncertainty` (no suffix) called `self._predict_sets`, which was never defined as a method anywhere in the class (only as a same-named module-level function taking no `self`) → raised `AttributeError` at runtime on every call. Removed.
- `get_uncertainty_opt` was model-bound (called `self.model.predict_proba`, had no `*_from_proba` counterpart) and raised `AttributeError` if the UQ was constructed without `model=`, making it unusable under the no-model migration. Removed, together with its sole helper `get_U` — which had exactly one call site in the entire repo (inside `get_uncertainty_opt` itself) and so had no other consumer to preserve.

`fit_opt`, `predict_opt`, `fit` remain absent entirely — unrelated to this pass; they were never part of the current `UncertaintyQuantifier` and are not something Phase 6 needs to remove.

Consequence: any script relying on the two now-removed methods, or on `fit`/`fit_opt`/`predict_opt`, does NOT run against the current package and cannot produce a local reference. Such scripts (convergence_analysis, data_size_analysis, setsize_analysis, MNIST_test_coverage, MNIST_test_convergence) are full rewrites — validate them against the paper, not a prior local run. Tests passing does NOT imply the remaining legacy methods (`calibrate`, `predict`, `get_uncertainty_jit`) are correct in every path either; tests cover the core, not the scripts, and not every legacy branch. The remaining legacy surface (`calibrate`, `predict`, `get_uncertainty_jit`, `model=`) is removed in Phase 6 steps 5-6.

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
- `fit` was the former name of `calibrate`. `*_opt` were "optimized" variants; part of their logic was folded into the main methods. These scripts are written against a previous API and do NOT run as-is against the current package: migrating them means rewriting them against the new API. Note: `fit`, `fit_opt`, and `predict_opt` are fully absent from the current `UncertaintyQuantifier` (not merely deprecated). `get_uncertainty_opt`, bare `get_uncertainty`, and their helper `get_U` (also named in this table's "Legacy methods used" column) are now ALSO fully absent (removed, Phase 6 step 2) — unlike `get_uncertainty_jit`, `calibrate`, and `predict`, which still exist as deprecated methods pending Phase 6 step 5. This table's "Legacy methods used" column is a historical record of what each script called before its Phase 4 rewrite; it is not a claim about what still exists in the current class.
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

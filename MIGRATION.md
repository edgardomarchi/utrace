# U-TraCE — Migration and Refactor Guide

> **Note (2026-08-19 split):** MIGRATION.md, FINDINGS.md, BACKLOG.md and CONTRIBUTING.md were split from one file on this date (see `.reports/2026-08-19_docs_restructure.md`). This file now holds only the refactor's phase plan and architecture/design direction — content that expires when the refactor ends. Verified findings live in FINDINGS.md; open work in BACKLOG.md; contributor conventions in CONTRIBUTING.md.


Context document for the ongoing refactor of the `utrace` package. Captures the phase
status, the canonical migration pattern, and agreed conventions. This is the source of
truth for the refactor: when in doubt about "how something is done here," this document
and the tests are authoritative.

Diagnostic and execution reports produced during this refactor are kept in a private
companion repository, checked out at `.reports/` (gitignored).
Filenames are cited here for internal traceability; the reports are not publicly
available. Every claim in this document is written to stand on its own — the figures and
findings are transcribed here, not merely referenced.

## Refactor goal

Make the `UncertaintyQuantifier` core independent of the tensor backend (PyTorch, JAX,
etc.). Core methods operate on precomputed softmax arrays (via `to_jax`, zero-copy
through DLPack), not on an embedded model. The model is removed from the class: the user
computes softmax output externally and passes it to the `calibrate`/`predict`/`get_uncertainty`
API (renamed from `*_from_proba` by the rename batch; see "Rename batch" in FINDINGS.md).

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
- [x] Phase 6 — Remove legacy API, `model` parameter, the `USE_JAX` flag, and the legacy
      golden/baselines. DONE: the original six-step plan (steps 1-6, including sub-steps
      3a-3c) delivered exactly what this phase's own goal states — the legacy `calibrate`/
      `predict`/`get_uncertainty_jit`/`get_uncertainty`/`get_uncertainty_opt`/`get_U` methods
      and the `model` constructor parameter are gone (confirmed absent from
      `src/utrace/uncertaintyQuantifier.py`; `UncertaintyQuantifier(model=x)` raises
      `TypeError`), `USE_JAX` no longer exists anywhere in `src/`, and the legacy tests and
      `.npy` baselines were deleted. Two follow-on items appended to the step list after the
      original plan — step 7 (labels host-copy) and step 8 (`flatten_batch` ->
      `flatten_to_pixels` unification) — remain open and are tracked in the step list below;
      they are cleanup discovered while doing Phase 6, not part of what this phase's stated
      goal requires. Revised step ordering below (supersedes the original single-item
      plan above; re-derived after a diagnostic pass found the original order had an
      avoidable dependency and after the first two steps were already executed):
      1. DONE — `_max_N` overflow guard.
      2. DONE — remove dead legacy: `get_uncertainty`, `get_uncertainty_opt`, `get_U`.
      3a. DONE — `score='aps'` now raises `ValueError` at construction, naming the numpy
          backend as the only (unreachable) implementation and `'lac'` as the supported
          value. The generic `case _` branch, which previously fell back silently to
          `'lac'` for any unrecognized `score` value, was also closed to raise `ValueError`:
          with `'lac'` and `'aps'` as explicit cases, the default branch had to do
          *something*, and silently substituting a different scoring function on a typo is
          a correctness failure, not a convenience worth preserving. Verified: no call site
          anywhere in the repo (`src/`, `tests/`, `scripts/`) passes `score=` explicitly —
          every existing construction relies on the default.
      3b. DONE — `USE_JAX` removed: unconditional now in `src/utrace/__init__.py`,
          `utils/__init__.py`, `scores/__init__.py`, and both call sites in
          `uncertaintyQuantifier.py` (in `calibrate_from_proba` and the since-removed
          legacy `calibrate`); `config.py`, `.env.example`, and `scores/numpy_impl.py`
          deleted, along with the `aps`/`aps_cal` stub functions in `scores/jax_impl.py`.
          All six golden `.npy` baseline files (three new-API, three legacy — the legacy
          set still existed at this point in the sequence) were confirmed byte-identical
          (SHA-256) before and after, which is what confirmed the USE_JAX=false branch
          really was unreachable: the change was behaviour-preserving by construction, not
          merely by luck.
      3c. DONE — packaging cleanup, across two passes. First pass: `jax` and `numpy` moved into
          base `[project].dependencies`; `matplotlib`, `pandas`, `torch` and `torchvision` moved
          to extras (`viz`, `torch`, `cuda13`, `rocm7-local`); `python-dotenv`, `flax`,
          `scikit-image` and `tqdm` removed outright; `monai`, `nibabel` and `scikit-learn`
          moved to the `dev` group; the CUDA/ROCm indexes corrected. Second pass, superseding
          the first pass's extras shape: the `torch` extra was deleted outright — the GPU
          extras (`cuda13`, `rocm7-local`) no longer declare torch at all, torch instead comes
          from an `examples` extra or from the `dev`/`dev-cuda13`/`dev-rocm7` contributor
          groups, and torch/torchvision are upgraded to 2.13.0/0.28.0 uniformly across all
          routes. Current extras: `cuda13`, `rocm7-local`, `viz`, `examples`. See "Phase 6 step
          3c — packaging cleanup [RESOLVED]" and "Phase 6 step 3c, second pass — reshaping
          extras around usage [RESOLVED]", both in FINDINGS.md.
      4. DONE — removed the legacy tests (`test_golden_mnist.py`,
         `test_legacy_equivalence.py`), `baselines/legacy/`, the `--api legacy` branch of
         `regenerate_baselines.py` (the flag now has a single choice, `'new'`; narrowing
         rather than collapsing the flag was a deliberate, separate decision — see Backlog),
         and the `LEGACY_BASELINE_DIR` entry in `_baselines.py`. Required zero test edits:
         the two property tests that survive Phase 6 (`test_jax_cache_does_not_grow`,
         `test_golden_run_under_threshold`) were already commented-out dead code in
         `test_golden_mnist.py` and already lived, active, in `test_golden_mnist_new_api.py`.
         Test count: 113 → 106 (minus exactly the 7 active tests in the two deleted files).
      5. DONE — removed `calibrate`, `predict`, `get_uncertainty_jit`, and the
         `flatten_batch` import from `uncertaintyQuantifier.py`. This is what makes the core
         importable without torch — confirmed by `tests/core/test_import_properties.py`'s
         `test_core_does_not_import_torch` (a subprocess check; see the honest limitation
         noted under "What steps 4-6 established" below). Required zero test edits. Test
         count stayed at 106 (a separate commit then added
         `tests/core/test_x64_is_enabled.py` and `tests/core/test_import_properties.py`,
         bringing the count to 107 ahead of step 6).
      6. DONE — removed the `model=` constructor parameter outright (not kept as an
         accepted-but-warning parameter): `UncertaintyQuantifier(model=x)` now raises
         `TypeError` from Python itself, with no compatibility shim. Required exactly one
         test deletion, predicted by name in advance:
         `tests/core/test_from_proba_api.py::test_constructor_warns_when_model_passed` (the
         only construction anywhere in the repo passing `model=`). Test count: 107 → 106.
      7. Labels host-copy: remove the `# COMPAT` numpy round-trip in the core and in scripts.
      8. `flatten_batch` -> `flatten_to_pixels` unification. Diagnostic finding: as of this
         pass, `flatten_to_pixels` (in `utils/tensors.py`) and the three `unflatten_*`
         functions (`utils/pytorch/helpers.py`) have ZERO call sites anywhere in `src/`,
         `tests/`, or `scripts/`. `flatten_to_pixels` also raises `TypeError` on a raw torch
         tensor input (`jnp.moveaxis` requires an ndarray/scalar, not a `torch.Tensor`)
         because it does not route through `to_jax` first. This step is real work, not a
         swap.

      (The original text of this item also mentioned removing "`_trn`" methods; no symbol
      matching `_trn` exists anywhere in `src/`, `tests/`, or `scripts/` as of this pass —
      dropped from the ordering above as unverifiable rather than carried forward.)

      **What steps 4-6 established (verified):**
      - The suite's entire warning budget (92) came from the two legacy test files deleted
        in step 4, and is now 0. This zero baseline makes any new warning introduced by a
        later change immediately visible, rather than lost among pre-existing ones.
      - The core no longer imports torch at package-import time (step 5). This is verified
        by `tests/core/test_import_properties.py::test_core_does_not_import_torch`, a
        subprocess check asserting `'torch' not in sys.modules` after `import utrace`. Stated
        honestly: this proves nothing on the import path reaches for torch in an environment
        where torch happens to be installed but is otherwise unused — it does NOT prove the
        package works in an environment where torch is not installed at all. That still needs
        a clean, torch-free environment to confirm, and remains open.
      - `flatten_batch` itself was NOT removed and was never at risk: it survives with its
        callers intact in `tests/integration/torch/test_golden_mnist.py` (renamed from
        `test_golden_mnist_new_api.py` by the rename batch's first commit; see "Rename batch"
        in FINDINGS.md) and in several `scripts/`. Only the core's own module-level import of it
        (`uncertaintyQuantifier.py`) was removed in step 5.
      - `regenerate_baselines.py`'s `--api` flag now has a single remaining choice (`'new'`).
        It was deliberately left in place with `choices=['new']` rather than collapsed to no
        flag at all — collapsing the CLI is a separate, deliberate decision, not a side
        effect of removing the legacy branch. Open, minor.

      **Remaining for Phase 6:** steps 7 (labels host-copy) and 8 (`flatten_batch` ->
      `flatten_to_pixels` unification, real work per the diagnostic under step 8 above — not
      a swap). The step 3c packaging block (see FINDINGS.md) is resolved, across two passes — it is
      no longer part of what remains. The `_new_api` suffixes and
      `NEW_API_BASELINE_DIR` naming, once flagged here as redundant and a rename candidate, were
      renamed by the rename batch's first commit (`NEW_API_BASELINE_DIR` -> `BASELINE_DIR`,
      `test_golden_mnist_new_api.py` -> `test_golden_mnist.py`); see "Rename batch" in FINDINGS.md.

### Status at a glance

Done: Phase 6 (the original six-step plan); step-ladder items A, B1, B.5, C, and the rename
batch; packaging (both passes, see "Phase 6 step 3c" and "Phase 6 step 3c, second pass" in
FINDINGS.md); and, outside the phase numbering entirely (see "CI and tooling" in FINDINGS.md),
the CI workflow, ruff
adoption rung 1, the pytest dependency-group split, and the `scripts/` lint cleanup.

Removed rather than done: step-ladder item B2, after a measured negative result (see "Measured
negative result: jnp-native padding" below) — its content was absorbed into D, not lost.

Partially done: step-ladder item D — the marginal (`classes=None`) slice is done (see "Step D,
marginal slice" below); the rest (the class filter under jit, vmap over classes, the `N`
semantics change) remains fully coupled and not started.

Not started: step-ladder item E; Phase 6 steps 7 and 8 (see above); ruff rungs 2 (pydocstyle) and
3 (`ruff format`); and the backlog (see BACKLOG.md).

[INTENT] This document is now over 1200 lines and does three jobs at once — phase plan, findings
record, and backlog. Splitting the findings into their own file (leaving this one as the phase
plan and backlog) is a pending structural pass, deliberately kept separate from this content sync
so that pass's diff is purely structural and reviewable as such.

## Architecture / design direction (not started — Phase 6 scope unchanged)

This section records INTENT for work that has not been started, plus a small number of
findings verified directly in the pass that wrote this section, and several explicitly
unverified hypotheses. It is not a phase and nothing in it is DONE. It does not add to or
change the Phase 6 step list above.

Every statement below is labeled: **[ESTABLISHED]** (verified by reading or running code in
this pass), **[INTENT]** (a design decision taken but not implemented), or **[UNVERIFIED]** (a
hypothesis nobody has tested).

### Principles

1. [INTENT] The package accepts any array-like input at its boundary. DLPack-capable inputs
   are preferred and are converted zero-copy where possible.
2. [INTENT] Conversion happens ONCE, at the boundary, via `to_jax`. Nothing inside the core
   touches numpy for data that arrives as a device array.
3. [INTENT] The core works exclusively with jax arrays.
4. [INTENT] The core should be jittable end to end.

### Endpoint

[INTENT] A jittable core requires the state to be explicit — a PyTree passed in and returned —
because attribute mutation inside a traced function does not survive tracing.

[INTENT] This does NOT require the public API to become functional. The stateful class remains
the user-facing interface and absorbs the state threading:

```python
self._state = _calibrate_pure(self._state, to_jax(y_proba), to_jax(y), batched)
```

The reassignment happens in Python, outside the jit, which is legal. This is the pattern optax
and flax use.

[INTENT] Consequence: this is an internal refactor, not a breaking API change. No deprecation
cycle, no major version bump.

### User-visible consequence

[INTENT] The ONE thing that changes for users is the meaning of `N` under class-conditional
calibration. Today the class filter compacts before writing (`_calibrate_impl`'s
`y = y[mask]; smx = smx[mask]`, see Step 0(f) call sites), so `N=1000` with
`classes=[3]` means 1000 scores of class 3. Under a constant-size-mask design the buffer
receives all B entries with invalid ones marked, so `N=1000` becomes "1000 samples seen, of
which some fraction is mine." The same `N` that suffices today would overflow sooner — via the
`ValueError` added in Phase 6 step 1 (see Backlog, `_max_N` overflow guard), which is the
correct failure mode but is a behaviour change.

[INTENT] This is precisely what vmap over classes resolves: with a shared buffer, entries
belonging to other classes are not wasted. Therefore the class-filter redesign and the
vectorisation over classes are COUPLED and must be done together. Done in isolation, the class
filter makes memory use worse (one buffer per class, each sized for the full stream); combined,
it improves.

### Two obstacles

**The lazy sort.** [ESTABLISHED] The buffer is +inf-padded beyond the valid prefix (`reset()`
and the non-batched branch of `_calibrate_impl`; quoted in full below), so sorting the FULL
buffer is bit-identical to sorting `[:self._N]` — the +inf entries sort to the tail either way.
This makes the sort shape-stable at essentially no cost and makes `_sorted` the easiest piece
of state to move. [ESTABLISHED] Verified by running, in this pass, on CPU with
`jax_enable_x64` on (matching the package's own config): built a `max_N=20` buffer with `N=7`
random float64 values +inf-padded exactly as `_calibrate_impl`'s non-batched branch does, then
compared `jnp.sort(buf)` (full buffer) against `buf.at[:N].set(jnp.sort(buf[:N]))` (prefix-only,
today's behaviour) — `jnp.array_equal` and a raw byte comparison both returned `True`. 
Implemented and wired into conformity_scores_ in 972ef7f; the equivalence was verified against 
the real class, byte-comparing results, including _N == 0, _N == 1 and _N == _max_N.

**The class filter.** [ESTABLISHED] `y[mask]` produces a data-dependent shape, which raises
`NonConcreteBooleanIndexError` under `jit`. Verified directly: jitting a function that does
`y[jnp.isin(y, classes)]` and calling it raises

```
jax.errors.NonConcreteBooleanIndexError: Array boolean indices must be concrete; got bool[5]
```

[INTENT] The fix is the padding-and-masking pattern already used by
`_get_uncertainty_jit_impl`, which requires splitting `_N` (occupancy) from a separate valid
count, and that touches the overflow guard, the `conformity_scores_` property, and the alpha
setter.

### Measured negative result: jnp-native padding (B2, reverted)

[ESTABLISHED] jnp-native padding in the tuning path was implemented and reverted. Outside a
jit boundary it is a measured performance regression, not a step toward the jit goal:

- Isolated uncertainty section of a real MNIST class-conditional workload at B=12002: 0.5043s
  median under numpy versus 0.8524s median under jnp, five cold-process runs each, clusters
  non-overlapping (numpy max 0.5197 < jnp min 0.8427). About a 69% increase.
- A controlled call-count sweep showed the absolute delta GROWING with call count rather than
  flattening: at B=12002, 0.227s at 10 calls, 0.308s at 40, 0.580s at 160. Fitting those gives
  roughly 2.4 ms per call of persistent overhead plus about 0.20s of fixed compilation cost. At
  B=200 the same pattern holds at roughly 1.0 ms per call. There is no break-even: more calls
  widen the gap.
- The golden integration run went from 6.13s to 6.28s median (+2.4%), measured independently
  before the production-scale work, consistent in sign.
- Numerically value-preserving: golden `.npy` baselines byte-identical, and all per-class (U,
  alpha, coverage, setsize) values identical across all ten runs of both variants.

[ESTABLISHED] The mechanism: jax arrays are immutable, so `arr.at[:B].set(x)` allocates a new
array and dispatches an op, where numpy's `arr[:B] = x` is an in-place memcpy. The padding
construction is memory-movement work, which eager jnp does worse; this is the opposite of the
class mask, where `jnp.isin`'s compiled kernel beats `np.isin` at the same sizes. An isolated
benchmark of the mask alone therefore pointed the wrong way about the change as a whole.

[ESTABLISHED] The expectation recorded here — that inside a single jit trace XLA should fuse the
`.at[].set()` chain and eliminate the intermediates, so the same construction that costs ~2.4 ms
per call eagerly may cost nothing — has now been tested for the marginal (`classes=None`) slice
of step D, and was directionally right and mechanically wrong. It is right that the jit boundary
makes the write cheaper, measured directly. It is wrong that the mechanism is "fusion eliminates
the intermediates": the mechanism is dispatch-overhead amortisation, and the shipped design does
not even fuse the score computation and the write into one trace — an earlier attempt at that
exact fusion was reverted for reasons unrelated to performance (see "Step D, marginal slice"
below). Full result, numbers, and the two design findings this pass produced are recorded in the
new section immediately following "Measurement conditions" below; this paragraph is left in place,
relabelled, so the historical prediction stays visible next to what actually happened rather than
being silently deleted.

**Measurement conditions** (recorded because these numbers are now in a document people will
rely on):

- The Task 1 timings were taken on a workstation (Ryzen 7 5700G) that was in interactive use
  during the runs.
  Two of the five jnp runs show elevated values in both total and isolated time, consistent
  with external contention. The effect survives regardless: the signal is roughly 5-15x the
  within-cluster spread, and the three uncontended jnp runs still sit far above every numpy
  run.
- Run ordering was not documented, so runs were not verifiably interleaved between variants.
  Interleaving would be the definitive control against drift. Record this as a known
  limitation of the measurement, not as a reason to distrust the conclusion.
- All measurements are CPU backend. GPU is unmeasured and the balance could differ, since
  dispatch overhead and memory-movement costs have different relative weights there.

### Step D, marginal slice — jitting the write for classes=None [RESOLVED for this slice only]

[ESTABLISHED] Shipped in commit `c7617d2` ("Jit the marginal calibration write"): a new
module-level `@jit` function, `_calibrate_write_jit`, replaces `_calibrate_impl`'s eager
`.at[].set()` write, but **only** when `self._classes_jax is None` (the marginal path).
`_calibrate_impl` now branches explicitly — `if self._classes_jax is None: <jitted write> else:
<eager write, byte-identical to before>` — rather than sharing one implementation between the two
cases. Confirmed directly in the source (`_calibrate_write_jit` at module level before `_UQState`,
alongside the file's other module-level jit functions `_predict_sets`/`_q_hat_from_alpha`/
`_search_uncertainty`; the class-conditional branch of `_calibrate_impl` still ends in `.at[
old_N:old_N + num_scores].set(...)` and `.at[:num_scores].set(...)`, unchanged).

**The fusion that was tried and reverted.** The diagnostic that motivated this work (and the
`[UNVERIFIED]` paragraph above) tested a prototype that fused the score computation
(`cal_score_`/`lac_cal`) and the buffer write into a single trace. That is **not** what shipped.
During implementation, the fused design broke an existing test —
`tests/core/test_label_dtype_canonicalisation.py` asserts `lac_cal._cache_size() == 1` directly,
and calling `lac_cal` from inside another jit trace inlines its body without going through its
own dispatch/cache path, so the assertion silently failed (cache stayed at 0 even though `lac_cal`
was, in effect, being traced). The shipped `_calibrate_write_jit` therefore takes
**already-computed** `scores` as an argument, not `y`/`smx` plus the score function;
`self.cal_score_(y, smx)` remains a separate, standalone, already-jitted top-level call, exactly
as it was before this change, in both the marginal and class-conditional branches.

[ESTABLISHED] **The mechanism, corrected.** The jit boundary does make the write cheaper,
measured — but not by "fusing a `.at[].set()` chain and eliminating intermediates," since there
is no chain and nothing is fused: the shipped design replaces one eager dispatch (the write) with
one jit dispatch, while the score computation was, and remains, its own separate jitted call on
both sides of the change. JAX's eager mode still compiles each primitive separately on first use,
so an eager `.at[].set()` pays a real first-use compile cost (isolated measurement: ~40 ms) close
in magnitude to what `lac_cal` alone pays (~30 ms) — replacing just the write's eager dispatch
with a cached jit dispatch removes that cost on every call after the first, and most of it even on
the first, without needing to fuse anything with the score computation.

[ESTABLISHED] **Break-even is zero calls.** The jitted write's cold first call was cheaper than
the eager write's cold first call at every tested size — the reverse of the naive "compilation is
pure overhead" framing, and the reverse of B2, where more calls widened the gap in favour of the
eager/numpy side. Here, more calls widen the gap in favour of jit (steady-state timings below
confirm the gap does not close with repetition).

[ESTABLISHED] **This does not contradict the B2 finding above.** B2 compared eager-jnp against
eager-numpy, both un-jitted. This compares eager-jnp against a jitted dispatch of the same
operation. Different pairs, same underlying fact — JAX eager pays real per-call/per-first-use
costs — read from the other side: B2 showed eager-jnp loses to eager-numpy; this shows eager-jnp
loses to jitted-jnp. Neither overturns the other.

**Numbers, each with its machine.** All figures in this subsection were measured on a **Ryzen AI 7
PRO 350 laptop, CPU backend** (`jax.default_backend() == 'cpu'`) — not comparable to the 5700G
figures elsewhere in this document.

- Diagnostic prototype (fused design, `.reports/2026-08-18_stepD_jit_marginal_diagnostic.md`),
  streaming (repeated batched calibration into one buffer, warm): **~13.5x at B=500**, **~2.3x at
  B=12000**, **~1.08x at ACDC pixel scale (B=65000)** — the win shrinks sharply as B grows, because
  at large B the operations' own compute time dominates over the dispatch overhead the jit
  boundary is amortising.
- Shipped implementation (write-only jit, `.reports/2026-08-18_stepD_jit_marginal_execution.md`),
  measured end-to-end through the public `calibrate()` — which includes `to_jax()` conversion and
  method-call overhead, identical on both sides of the comparison and therefore diluting the
  *relative* size of the win without changing its direction: **~2-3x at B=500 and at B=12000**
  (two independent run-sets: 2.2x/3.1x at B=500, 3.0x/3.2x at B=12000). The smaller multiplier
  against the diagnostic's figures is a different, more inclusive measurement target, not a
  failed reproduction — the diagnostic isolated the write; this measures the whole public call.
- A batch-size change (a dataloader remainder) forces a retrace: **~30 ms** for the jitted write.
  This is not a new cost `_calibrate_write_jit` introduces — `lac_cal` is already `@jit`-decorated
  and already pays a retrace on exactly the same event, measured at **~54 ms**, both before and
  after this change. The jitted write's retrace is smaller than the pre-existing one, not an
  addition to it.

[ESTABLISHED] **Design finding 1 — `N` stays a host Python int; no sync, no per-N retrace.** The
prediction under "What the state PyTree revealed for step D" below assumed `N` would need to
become a traced value living inside `_UQState`, forcing either a device-to-host sync per batch (to
check the overflow guard) or moving the guard somewhere untraceable. Neither was needed. JAX's jit
cache keys on the *abstract* shape and dtype of a non-static argument, not its *concrete* value —
so passing the buffer offset as a plain Python `int` at the call site (never read back from a
device array, never round-tripped through the state) is lifted to a traced scalar once and reused
for every subsequent value. Verified directly: 45 distinct offsets in a streaming sequence produced
a `_calibrate_write_jit._cache_size()` of 1. `_UQState.N` is untouched by this change — still a
plain Python `int`, confirmed in the source — and the overflow guard is untouched too: still a
Python `if` in `_calibrate_impl`, before the write, never traced.

[ESTABLISHED] **Design finding 2 — the literal slice syntax fails outright, not "needs
adjustment."** `buffer.at[start:start+size].set(...)` (the syntax used everywhere else in this
file, including the still-eager class-conditional branch) does not merely need adjusting for a
traced `start` — it raises `IndexError: Slice entries must be static integers` immediately.
`jax.lax.dynamic_update_slice_in_dim(buffer, scores, start, axis=0)` is the form that actually
traces, because the *update size* is static (shape-derived) even though the *offset* is not. This
is a sharper finding than this document's earlier hedge ("needs a dynamic-slice update with a
statically known size") — the existing syntax is not adjustable, it is a different function.

[ESTABLISHED] **Both named predictions under "What the state PyTree revealed for step D" held,
confirmed by direct reproduction, not by reading:** `if state.sorted` inside `_ensure_sorted`
raises `TracerBoolConversionError` when `sorted` is a traced value, and the overflow guard's
Python `if` on a traced offset raises the same error — both cannot be traced and must stay at the
untraced wrapper level. `_ensure_sorted` itself, the `sorted` dirty flag, and the whole read path
(`conformity_scores_`, the `alpha` setter, `_get_uncertainty_jit_impl`) are **unchanged** by this
commit — confirmed by diff: no line outside `_calibrate_impl` and the one new function changed.

[ESTABLISHED] **Why the branch is explicit, not shared.** Sharing `_calibrate_write_jit` between
the marginal and class-conditional paths was the natural single-implementation choice and was
deliberately rejected: the class-conditional path's filtered batch size is data-dependent per
batch and per class, so a shared jitted write would retrace on every distinct filtered size, a
cost the class-conditional path pays nothing for today (it is fully eager). A test,
`tests/core/test_calibrate_jit_marginal.py::test_class_conditional_never_hits_jit_write`, asserts
directly that class-conditional calibration never reaches `_calibrate_write_jit`
(`_cache_size() == 0` after class-conditional-only calibration, with a non-vacuousness check that
a marginal call *does* reach it). Whether sharing would in fact be a net win for the
class-conditional path is unmeasured — this pass did not attempt it, and answering it would need
its own diagnostic with realistic per-class batch-size distributions, not an assumption either way.

[UNVERIFIED] **What this slice does NOT resolve — the pixel-scale regime, which is the case that
motivated step D in the first place** (see "User-visible consequence" above: high-volume,
per-pixel calibration is exactly where the fixed-size buffer and the class-filter redesign matter
most). At ACDC pixel scale, the realistic per-batch jitted design — one jitted call per incoming
batch from an ordinary Python loop, the direct drop-in for today's streaming pattern — measured
**~1.08x on CPU**, barely distinguishable from noise. A substantially larger win (**~5.7x**) was
measured only for a design that pre-stacks an entire stream of batches into one
`lax.fori_loop` trace, dispatched once — but that conflicts with this package's explicit
bounded-memory streaming goal (see the canonical migration recipe: "This keeps only one batch of
logits in memory at a time"). That tension is not resolved by anything measurable on a CPU
backend. Whether the shrink-to-near-parity pattern holds on GPU is also unknown — the mechanism
identified (dispatch-overhead amortisation shrinking as compute time grows) is exactly the kind of
effect whose balance could differ where kernel-launch and host/device synchronisation costs have a
different shape than CPU dispatch overhead. This joins the existing list of open questions waiting
on the RTX 3070 (see "What is unverified" in FINDINGS.md). **The marginal slice is justified on the
moderate-N regime alone** (MNIST-family scripts, N in the low tens of thousands); the pixel-scale
case that motivates step D as a whole still needs GPU measurement before a performance case can be
made for it there.

### Step ladder

[ESTABLISHED] A, B1, B.5, C, and the rename batch are DONE (see "Rename batch" in FINDINGS.md). B2 was
removed as a standalone step after a measured negative result (see "Measured negative result:
jnp-native padding (B2, reverted)" above); its content was absorbed into D. The step 3c
packaging block is now resolved, across two passes (see "Phase 6 step 3c — packaging cleanup
[RESOLVED]" and "Phase 6 step 3c, second pass — reshaping extras around usage [RESOLVED]",
both in FINDINGS.md). What remains after this pass is D and E, plus the backlog (BACKLOG.md).

- [ESTABLISHED] **A. The boundary.** DONE. `to_jax(y)` once on entry, no numpy in the core,
  `jnp.isin` for the class mask, eager boolean indexing left as-is. Plus tests for the
  torch-tensor input path, which the golden baselines do not currently cover.
- [ESTABLISHED] **B1. Sort the full buffer.** DONE. Sort the full buffer instead of the
  variable slice (commit 972ef7f). This one WAS a win, and stands on its own merits
  independently of D: 40 compiled sort shapes collapsed to 1, and a 40-batch streaming loop
  went from 5.20s to 2.44s. Measured on a Ryzen 7 5700G workstation (see "Convention:
  performance figures carry their machine", in FINDINGS.md).
- **B2. REMOVED as a standalone step.** It is not a stepping stone to D, because the jnp
  padding only pays off inside a single jit trace — see "Measured negative result" above. Its
  content is absorbed into D.
- [ESTABLISHED] **B.5. Extract state to an explicit PyTree.** DONE (commit 53b1e8d). A plain
  `typing.NamedTuple` (`_UQState`, holding `N`, `conformity_scores`, `sorted`, `alpha`, `q_hat`)
  sufficed — no `flax.struct.dataclass`, no hand-registered pytree node, no flax import. The
  reason is the partition itself: with configuration (`_max_N`, `label_dtype_`, `cal_score_`,
  `score_`, `classes`, `_classes_jax`, `_max_batch_size`) excluded from the state, no field
  needed a static designation, so JAX's automatic NamedTuple registration was enough. The lazy
  sort kept its laziness: a pure module-level `_ensure_sorted(state) -> state` returns a new
  state, and the `conformity_scores_` property calls it and reassigns `self._state` in ordinary
  Python, outside any jit — sorting eagerly on write was rejected because it would revert the
  measured 357x defer-sort win recorded in FINDINGS.md (the Backlog-derived "defer-sort"
  entry there). Verified in two halves so the mechanical
  extraction and the sort-laziness rewrite stay separately attributable: the mechanical
  extraction alone passed with byte-identical goldens before the sort work started; the full
  change passed with zero test edits and an empty collected-test-ID diff. Also corrected
  `get_uncertainty_from_proba`'s docstring, which claimed the method was pure — a diagnostic had
  verified by execution that it flips `_sorted` and rebinds the buffer through the same
  lazy-sort path as `conformity_scores_`; the mutation is idempotent and value-preserving, so
  only the documented claim was wrong, not the behaviour.
- [ESTABLISHED] **C. The scripts.** DONE, across three commits (32a1309, 6a630e5, 0caad12).
  Removed the `.numpy().astype(int)` / `# COMPAT` conversions that fed the core, in all eight
  scripts touched by this refactor (five scripts in 32a1309, `MNIST_class_conditional_example.py`
  in 6a630e5, `ACDC_example.py` in 0caad12). This completes the boundary work started in step A:
  labels now reach the core as torch tensors and take the DLPack path in `to_jax`, rather than
  being forced through the host via `.numpy()`. Conversions that feed a script's own numpy math
  (accuracy tallies, coverage indexing, plotting) are untouched by design — see "Passing tensors
  to the core" in CONTRIBUTING.md and "Step C: the `# COMPAT` marker post-mortem" in FINDINGS.md.
- **D. jit + vmap over classes + the `N` semantics change**, all together because they are
  coupled (see "User-visible consequence" above). [ESTABLISHED] A first, narrow slice is DONE
  (commit `c7617d2`, see "Step D, marginal slice" above): the marginal (`classes=None`) buffer
  write is jitted via `_calibrate_write_jit`, behind an explicit branch, leaving the
  class-conditional path byte-identical. [INTENT] What remains of D, still fully coupled and not
  started, is unchanged by the marginal slice: the class filter under jit (see "Two obstacles"
  above), vmap over classes, and the `N` semantics change these two require together (see
  "User-visible consequence" above). The marginal slice deliberately did not touch any of the
  three, and does not commit the project to a particular resolution for them — it answered only
  whether jitting pays off at all, for the narrowest slice that could test it in isolation. D also
  still carries what was B2: the `_get_uncertainty_jit_impl` padding, the class mask leftover in
  `_get_uncertainty_jit_impl` (its own `np.asarray(self.classes)`), and the
  `np.asarray(to_jax(...))` on entry to `get_uncertainty` — because all of them must land inside
  the jit boundary or not at all.
- [INTENT] **E. Device coherence in the script layer.** Not started. Make probabilities and
  labels reach the core committed to the same device (see "Step C: device-commitment risk"
  in FINDINGS.md). Open questions to record, none of them decided:
  - Where the transfer belongs: inside the dataset/model wrappers, or in the scripts. Moving it
    into the dataset wrapper is not free — several consumers legitimately need host arrays
    (ACDC's accuracy tally over `gt_img`, the coverage indexing over `y_test_arr`), so a wrapper
    that unconditionally returns device tensors would force those back down.
  - It touches `src/utrace/utils/pytorch/`, which no step of this refactor has modified, and
    changes the contract of helpers all eight scripts consume.
  - It is only observable on CUDA hardware: on a CPU backend `.to(device)` is a no-op and a
    passing test suite would demonstrate nothing.
  Sequenced after D deliberately: D restructures the core with jit and vmap, and the wrapper
  contract should not move at the same time.

### What the state PyTree revealed for step D

[ESTABLISHED] `jax.tree_util.tree_flatten` of a populated `_UQState` yields five leaves: `N` as
a Python `int`, the conformity-score buffer as a jax array, `sorted` as a Python `bool`, and
`alpha`/`q_hat` as numpy `float64` scalars. Only one of the five is a jax array. No
configuration value appears among the leaves — `_max_N`, `label_dtype_`, etc. are not fields of
`_UQState` and never reach `tree_flatten`.

The consequences for D, recorded here before any of it was tested. The first two bullets have
since been tested, for the marginal write slice specifically (see "Step D, marginal slice" above)
— confirmed by direct reproduction where marked, and one of them turned out more favorable than
predicted. The third remains exactly as open as when this was written.

- [ESTABLISHED] Under jit every leaf becomes a traced value if it is part of the jitted function's
  arguments. `sorted` as a traced bool breaks the Python `if` inside `_ensure_sorted` (`if
  state.sorted: return state`) — a data-dependent Python conditional cannot be traced. **Confirmed
  by direct reproduction**, not just reasoned about: jitting a function with this exact body and a
  traced `sorted` argument raises `TracerBoolConversionError`. That flag has NOT left the state,
  become static, or been eliminated — `_ensure_sorted` and `_UQState.sorted` are byte-identical to
  before this pass, because the marginal slice touches only the write path, not the read path this
  bullet is about. The observation from "Two obstacles" above still stands, now doubly confirmed,
  and the fix it names is still not implemented.
- [ESTABLISHED, corrected] The prediction was: "`N` as a traced value means the buffer write needs
  a dynamic-slice update with a statically known size." The dynamic-slice-update part is confirmed
  exactly (`lax.dynamic_update_slice_in_dim`, not the `.at[start:start+size].set(...)` syntax used
  elsewhere in this file, which raises `IndexError: Slice entries must be static integers` under a
  traced start). The "`N` as a traced value" premise turned out to be avoidable, not necessary:
  the shipped design keeps `N` a plain Python `int` — never placed inside `_UQState` as a device
  value, never synced from device — and passes it directly as the jitted function's offset
  argument; JAX lifts a concrete Python int to a traced scalar automatically and caches by
  abstract shape/dtype, not by the int's value, so this costs nothing per distinct `N` (45 distinct
  offsets, one compiled trace). The overflow guard's Python comparison is confirmed untraceable
  (`TracerBoolConversionError`, reproduced directly) and stays exactly where this prediction said
  it would: the untraced wrapper level, in `_calibrate_impl`, unchanged in placement or behaviour.
- [UNVERIFIED] `alpha` and `q_hat` are host scalars sitting in the same structure as a device
  array. D has to decide whether they convert at the jit boundary or leave the state entirely
  (living instead as plain wrapper attributes, the way configuration already does). This interacts
  with the output-types rule recorded below: scalars are returned as host floats. Untouched by the
  marginal slice — still exactly as open as when this was written.

### Output types

[INTENT] Scalars are returned as host floats. They have no device to preserve, no zero-copy to
protect, and no pipeline to return to. Python's float is IEEE 754 binary64, identical in
precision to `np.float64`; the float64 guarantee comes from `jax_enable_x64` upstream, not from
the return type.

[INTENT] Arrays are returned on device, with a conversion helper for callers who want them
elsewhere. `predict` returns arrays and currently returns numpy; changing it is a
real API decision, not yet taken. Open questions to record: the helper's signature, whether it
lives in `utils.tensors` next to `to_jax`, and whether it supports only torch and numpy or
anything with `__dlpack__`.


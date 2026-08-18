# U-TraCE — Migration and Refactor Guide

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
API (renamed from `*_from_proba` by the rename batch; see "Rename batch" below).

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
          extras around usage [RESOLVED]" below.
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
        below) and in several `scripts/`. Only the core's own module-level import of it
        (`uncertaintyQuantifier.py`) was removed in step 5.
      - `regenerate_baselines.py`'s `--api` flag now has a single remaining choice (`'new'`).
        It was deliberately left in place with `choices=['new']` rather than collapsed to no
        flag at all — collapsing the CLI is a separate, deliberate decision, not a side
        effect of removing the legacy branch. Open, minor.

      **Remaining for Phase 6:** steps 7 (labels host-copy) and 8 (`flatten_batch` ->
      `flatten_to_pixels` unification, real work per the diagnostic under step 8 above — not
      a swap). The step 3c packaging block (see below) is resolved, across two passes — it is
      no longer part of what remains. The `_new_api` suffixes and
      `NEW_API_BASELINE_DIR` naming, once flagged here as redundant and a rename candidate, were
      renamed by the rename batch's first commit (`NEW_API_BASELINE_DIR` -> `BASELINE_DIR`,
      `test_golden_mnist_new_api.py` -> `test_golden_mnist.py`); see "Rename batch" below.

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
on the RTX 3070 (see "What is unverified" below). **The marginal slice is justified on the
moderate-N regime alone** (MNIST-family scripts, N in the low tens of thousands); the pixel-scale
case that motivates step D as a whole still needs GPU measurement before a performance case can be
made for it there.

### Step ladder

[ESTABLISHED] A, B1, B.5, C, and the rename batch are DONE (see "Rename batch" below). B2 was
removed as a standalone step after a measured negative result (see "Measured negative result:
jnp-native padding (B2, reverted)" above); its content was absorbed into D. The step 3c
packaging block is now resolved, across two passes (see "Phase 6 step 3c — packaging cleanup
[RESOLVED]" and "Phase 6 step 3c, second pass — reshaping extras around usage [RESOLVED]"
below). What remains after this pass is D and E, plus the Backlog.

- [ESTABLISHED] **A. The boundary.** DONE. `to_jax(y)` once on entry, no numpy in the core,
  `jnp.isin` for the class mask, eager boolean indexing left as-is. Plus tests for the
  torch-tensor input path, which the golden baselines do not currently cover.
- [ESTABLISHED] **B1. Sort the full buffer.** DONE. Sort the full buffer instead of the
  variable slice (commit 972ef7f). This one WAS a win, and stands on its own merits
  independently of D: 40 compiled sort shapes collapsed to 1, and a 40-batch streaming loop
  went from 5.20s to 2.44s. Measured on a Ryzen 7 5700G workstation (see "Convention:
  performance figures carry their machine" below).
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
  measured 357x defer-sort win recorded above. Verified in two halves so the mechanical
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
  to the core" above and "Step C: the `# COMPAT` marker post-mortem" below.
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
  below). Open questions to record, none of them decided:
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

## Rename batch

[ESTABLISHED] Five commits, landed in sequence: dropping the `_new_api` suffixes from the
golden test (`2574c41`); removing the forwarding accessors left by the B.5 state extraction
(`cfba206`); dropping `proba` from the library's public API (`70fa11e`); dropping `proba` from
the callers — scripts and tests (`b23cef2`); and removing two pieces of dead code the batch's
diagnostics surfaced (`372defd`). The rename mappings themselves live in the commit messages and
in `.reports/2026-08-06_rename2_forwarding_accessors.md`, `2026-08-07_rename3_drop_proba.md`,
`2026-08-07_rename4_caller_locals.md`, and `2026-08-07_rename5_dead_code.md`; this section is
the index, not a duplicate.

**Why the `proba` rename happened.** Not tidiness. The paper this package accompanies argues
that treating a model's scaled logits as an approximation to a probability distribution is a
conceptual error. A public API whose method names said `proba` — `calibrate_from_proba`,
`predict_from_proba`, `get_uncertainty_from_proba` — contradicted the package's own central
claim, at exactly the place (the API surface) where a reader who has just followed that argument
would look next. That is the reason for the rename. The fact that the `_from_proba` suffix had
also stopped distinguishing anything — the model-taking variant it originally contrasted with
was removed in Phase 6 steps 5-6 — is secondary; renaming it because it was redundant, alone,
would not have been worth a five-commit batch.

**Naming convention now in force.** Method names are task-agnostic — `calibrate`, `predict`,
`get_uncertainty` — so they generalise to a future regression class without another rename; the
task-specific meaning (that the input is a classifier's softmax output today) lives in the
parameter name, not the method name. At the parameter boundary: `softmax` where the name is read
as documentation (public method signatures, e.g. `calibrate(self, softmax, y, ...)`); `smx` for
short-lived internal locals in dense expressions, extending the convention `scores/jax_impl.py`
already used (`lac(smx)`, `lac_cal(y, smx)`). Prose describing the argument says "softmax
output," or "scaled logits" where a generic term is wanted — not "probabilities."

**`calibrate`, `predict`, and `get_uncertainty` are reused names.** These three names were
removed from `UncertaintyQuantifier` in Phase 6 — bare `get_uncertainty` in step 2 (dead/broken:
it called `self._predict_sets`, never defined as a method), `calibrate` and `predict` in steps
5-6 (legitimate legacy deprecation, alongside `get_uncertainty_jit`) — and the rename batch has
now reintroduced `calibrate`, `predict`, and `get_uncertainty` as the current public API, with
unrelated implementations that expect softmax-output input rather than a raw model/raw data.
`get_uncertainty_jit` was NOT reused; the new binary-search method is named `get_uncertainty`,
a distinct string. Consequence, accepted at version 0.0.1: old code written against the
pre-Phase-6 legacy API and calling `uq.calibrate(X, y)` or `uq.predict(X)` with raw inputs no
longer raises `AttributeError` — it now runs, silently, against the wrong implementation, and
passes raw data where softmax output is expected. This is a materially different failure mode
than "does not run," and any part of this document written before the rename batch that claims
`calibrate`/`predict`/`get_uncertainty` are "fully absent" from the current class is describing
the pre-rename-batch state, not the current one — see the corrections in "Legacy method state"
and "Example script inventory" below.

**`Pytorch_wrapper.predict_proba` was deliberately NOT renamed.** It is a scikit-learn
convention (`predict_proba` as a model-wrapper method name), lives outside the core in
`utils/pytorch/model_wrapper.py`, and was the single largest remaining source of `proba` hits
after the library and caller renames — left untouched on purpose, not missed.

### Criterion change: renames and collected test IDs

[ESTABLISHED] Every Phase 6 removal step (steps 4-6 above) was held to a zero-test-edit
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

### Accidental public surface

[ESTABLISHED] `src/utrace/scores/__init__.py` does `from .jax_impl import *`, and
`scores/jax_impl.py` defines no `__all__`, so every public (non-underscore) module-level name
bound in `jax_impl.py` is re-exported through `utrace.scores`. Before the dead-code commit this
included `jax_print` (`utrace.scores.jax_print`, reachable with zero consumers anywhere in the
repo); it still includes `jnp` and `jit` — `utrace.scores.jnp` and `utrace.scores.jit` are
reachable today, without anyone having decided that as an API surface. Verified by execution
during the batch (`from utrace import scores; 'jax_print' in dir(scores)` returned `True` before
the dead-code commit), not inferred from reading the wildcard import.

[INTENT] An explicit `__all__` on `scores/jax_impl.py` (or replacing the wildcard import in
`scores/__init__.py` with named imports) would close this. Not done — added to the Backlog
below.

### Discrepancies found against earlier counts

[ESTABLISHED] Two of the batch's execution passes disagreed with the rename diagnostic's counts
and were right to report the disagreement rather than silently reconcile it:

- The diagnostic counted 75 occurrences of the three method names (`calibrate_from_proba`,
  `predict_from_proba`, `get_uncertainty_from_proba`) across 18 files. The execution pass
  independently counted 84 across the same 18 files. Ten of the 84 were docstring/comment prose
  mentioning a method name rather than a `def`/call site; subtracting those gives 74 — one short
  of the diagnostic's 75, with no further split found to close that last occurrence. Scope was
  unaffected either way — all 84 were individually enumerated and either updated or explicitly
  deferred with a reason, so nothing depended on which count was authoritative.
- The diagnostic's signature count of 7 (functions/methods taking the old `y_pred_proba`
  parameter) was numerically correct but its composition was not fully verified until execution:
  it missed `_search_uncertainty`, whose `y_pred_proba` parameter sits on its own line inside a
  multi-line `def`, matching no single-line `grep 'def .*y_pred_proba'` pattern. It was found by
  reading the function, not by any grep tried.

Lesson: a grep-derived count is a lower bound, not an exact count, and multi-line signatures —
one parameter per line, common in this codebase's jit-decorated functions — are exactly where a
single-line pattern under-reports. Re-derive counts independently before trusting a prior
diagnostic's number, and report disagreement rather than silently adopting either figure.

### Step C: the `# COMPAT` marker post-mortem

[ESTABLISHED] The `# COMPAT` markers left by the Phase 4 helper consolidation (see "GPU /
scalability (example scripts)" below) were not a reliable inventory of the conversions step C
needed to remove, and failed in both directions:

- They UNDERCOUNTED. In the first group of five scripts, nine core-bound conversions existed
  where only four were tagged (commit 32a1309). The markers were placed during the Phase 4
  helper consolidation and only ever tagged the tune-set conversions that flow through
  `precompute_proba`; the per-batch calibration loops convert the DataLoader's labels directly
  and were never tagged.
- They MISFIRED. In `MNIST_class_conditional_example.py` a tag sat on `test_y_all`, which never
  reaches an `UncertaintyQuantifier` at all (commit 6a630e5).
- They were ABSENT. `ACDC_example.py`, the largest-scale consumer, carried none at all (commit
  0caad12).

The lesson: a marker placed for one batch of work is not an index of a category, and later
passes should not treat it as one. The criterion that actually worked was tracing each variable
to the end of its file and asking whether it feeds the core — not whether it would still work if
left unconverted, which in several cases it would have, by `__array__` coercion.

### Step C: the ACDC measurement

[ESTABLISHED] At pixel scale (commit 0caad12) the benefit of removing the label round-trip is
MEMORY, not time:

- Wall clock: null result. Three interleaved cold runs per variant at one iteration and one
  noise level. The edited variant's entire range sat inside the unedited variant's, whose own
  run-to-run spread was about four times the nominal median delta. Reported as no measurable
  effect, not as a small improvement.
- Peak RSS: about 490 MB lower, cleanly separated — the edited variant's maximum was below the
  unedited variant's minimum across all three runs. Consistent with dropping one host-side int64
  copy of the label arrays; the tune set alone is roughly 46M labels, about 371 MB in int64, at
  the full six-noise-level scale.
- Measured on a Ryzen AI 7 350, CPU backend, under light desktop load, with runs interleaved
  between variants rather than grouped. Not comparable to the 5700G figures elsewhere in this
  document — see "Convention: performance figures carry their machine" below.

### Step C: device-commitment risk

[UNVERIFIED] Step C may have introduced a device mismatch that is invisible on CPU.
`Pytorch_wrapper.predict_proba` (`src/utrace/utils/pytorch/model_wrapper.py`) returns its
result without moving it to host — the `.cpu()` call is present only as a trailing,
commented-out no-op:

```python
def predict_proba(self, X):
    X = X.to(self.device)
    with torch.no_grad():
        model_out = self.model(X)
    return flatten_batch(torch.softmax(model_out, dim=1))#.cpu())
```

ACDC moves the network to CUDA when available (`self.device = torch.device(device)`;
`self.model.to(self.device)` in `__init__`), so on such a machine the probabilities come back
device-resident. The dataloader helpers in `src/utrace/utils/pytorch/dataset_wrapper.py`
(`ACDCDataset.__getitem__`, `get_ACDC_dataloader`, `get_ACDC_cal_tun_tst_dataloaders`) move
nothing — labels are built with plain `torch.tensor(...)` and never `.to(device)`'d — so labels
stay in host memory regardless of where the model runs.

`to_jax` (`src/utrace/utils/tensors.py`) checks `isinstance(array_like, np.ndarray)` BEFORE the
`__dlpack__` branch: numpy arrays take `jnp.asarray`, landing uncommitted on JAX's default
compute device; anything else exposing `__dlpack__` (including CPU and CUDA torch tensors)
takes `jax.dlpack.from_dlpack`, which preserves the source device.

Before step C, ACDC's labels were numpy and took `to_jax`'s `jnp.asarray` branch, landing
uncommitted on the default device alongside the probabilities. After step C they are torch
tensors (`y_cal_arr = flatten_batch(y_cal).ravel()`; `tune_y_all = torch.cat(tune_y_list,
dim=0)`) and take the DLPack branch, which preserves the source device — the host, since the
dataloader never moves them. So on a machine where `torch.cuda.is_available()` is true, `p_cal`
(device-resident) and `y_cal_arr` (host-resident) would reach `lac_cal` committed to DIFFERENT
devices. `to_jax`'s own docstring states it does "NOT reconcile mismatches between two genuine
tensors on different devices... that remains the caller's responsibility."

This is an inference from reading the code, not an observation: every machine used in this
refactor has been CPU-only, where all placements coincide and the mismatch cannot appear. No
test covers it.

[UNVERIFIED] Whether an array is committed to a particular device, and whether two arguments
reaching the same jitted call agree, is not observable on a CPU backend — every device is the
same device there. This needs the RTX 3070, which is on a machine that has not been available
during this work.

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

### Convention: performance figures carry their machine

[ESTABLISHED] Measurements in this document come from more than one machine and are not
comparable across them. Every performance figure recorded from now on must name the machine it
was taken on. Retroactively labelled by this pass: the B1 sort-buffer win and the B2 regression
figures (the isolated-section comparison, the call-count sweep, and the golden-run delta) were
taken on a Ryzen 7 5700G workstation; the ACDC step-C figures were taken on a Ryzen AI 7 350
laptop.

The defer-sort win (~357x, 7.2s → 20ms, in the Backlog "defer-sort" entry below) keeps its
existing attribution to an RTX 3070. The `to_jax` direct-vs-indirect figures under "What is
unverified" below were measured on the 5700G; see
`.reports/2026-07-29_phase6_step7_diagnostic_labels_hostcopy_5700G.md`.

[UNVERIFIED] All measurements to date are CPU backend, except the RTX 3070 defer-sort figure
noted above. The open GPU questions — the device-to-host-to-device ratio, whether np.asarray
rejects a torch CUDA tensor, whether the jnp padding pays inside a jit trace, and now the
device-commitment risk from step C — require an RTX 3070 on a third machine. This is a
constraint on where, not only on when.

### Forwarding accessors: temporary scaffolding (removed)

[ESTABLISHED] The forwarding properties for `_N`, `_sorted`, `_conformity_scores_` and the
name-mangled `__alpha`/`__q_hat` existed so that roughly 45 external read sites across the test
suite kept resolving unchanged, which is what made the ~200-line B.5 restructuring (the
module-level `_UQState` and `_ensure_sorted`, plus the rewritten `_calibrate_impl`, `alpha`
setter and `_get_uncertainty_jit_impl`) verifiable by a zero-test-edit criterion rather than
merely plausible. They were always temporary — nothing about the design required `_N` etc. to
keep resolving under those exact names forever, only that they did not need to change AT THAT
STEP — and were removed by the rename batch's second commit (`cfba206`; see "Rename batch"
above): all five properties are gone from `uncertaintyQuantifier.py`, and every read that used
to go through them now reads `self._state.<field>` directly (`_N` -> `_state.N`, `_sorted` ->
`_state.sorted`, `_conformity_scores_` -> `_state.conformity_scores`, `__q_hat` ->
`_state.q_hat`, `__alpha` -> `_state.alpha`).

The `_new_api` suffixes and `NEW_API_BASELINE_DIR` naming, flagged alongside these as rename
candidates, were also renamed by the batch (see "Rename batch" above). The third item flagged
alongside them — consolidating the two import-property tests
(`tests/core/test_x64_is_enabled.py` and `tests/core/test_import_properties.py`, currently
separate files grouped by subject — see "Test conventions" below) — was NOT done by the rename
batch; the two files remain separate. It stays open, no longer blocked on the collected-test-ID
constraint that held back the other two (that constraint applied while Phase 6 steps were still
using an empty-diff acceptance criterion; the rename batch's own criterion, recorded above,
tolerates a declared, walked ID diff, so a future consolidation could use the same pattern).

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

### What is unverified [UNVERIFIED]

The items below are open questions about GPU behaviour. Where a CPU measurement exists it is
labelled and attributed; what remains unverified is the extrapolation to GPU.

- [ESTABLISHED] Measured on the 5700G, CPU backend
  (`.reports/2026-07-29_phase6_step7_diagnostic_labels_hostcopy_5700G.md`, Q7): the direct
  `to_jax` path runs flat at ~0.04 ms/call regardless of array size — 0.0410 ms at 500
  elements, 0.0398 at 12k, 0.0399 at 2M — while the indirect path scales with it: 0.0804 ms,
  0.0852 ms and 6.98 ms respectively, a 175x ratio at ACDC pixel scale. The flatness is
  consistent with genuine DLPack zero-copy doing no data movement.
  On a GPU backend the data must reach the device regardless, so the relevant comparison is
  device→host→device versus device→device, which is a DIFFERENT ratio and has not been
  measured on any GPU backend. Do not read a CPU ~175x figure as applying to GPU hardware.
- [UNVERIFIED] Whether `np.asarray()` on a torch CUDA tensor raises, and therefore whether the
  current `calibrate` rejects GPU-resident labels outright rather than merely
  copying them inefficiently. This is the strongest single argument for step A if true, and it
  is an inference from known torch behaviour that nobody has run. No CUDA device was available
  in this (or any prior) diagnostic environment.
- [UNVERIFIED] Whether the real cost is the copy or the per-batch device→host synchronisation
  that `np.asarray()` on a jax array forces. In the ACDC streaming pattern this would be a sync
  barrier per batch, which an isolated microbenchmark cannot see because it calls
  `block_until_ready` anyway.
- [ESTABLISHED, was UNVERIFIED] Whether jitting `_calibrate_impl` pays at all, independent of
  vmap, is answered on CPU for the marginal (`classes=None`) slice only — see "Step D, marginal
  slice" above for the full numbers, machine, and design findings. Short version: yes, it pays,
  from the very first call, at the batch sizes MNIST-family scripts use (~2-3x end-to-end through
  the public `calibrate()`, more in isolation); the win shrinks to near-parity (~1.08x) at ACDC
  pixel scale, where it remains genuinely open pending GPU measurement. The class-conditional path
  (with vmap not yet in the picture) is untested and untouched — this item's "independent of
  vmap" framing is answered only for the branch that has no class filter to interact with vmap in
  the first place.
- [UNVERIFIED] Whether returning jnp scalars instead of numpy scalars from
  `get_uncertainty` breaks any script. `float()`, `np.isnan()` and pandas all accept
  jnp scalars, so this is expected to be soft, but it must be checked against the scripts
  rather than assumed.

## Backlog (does not block the phases)

- `get_uncertainty_grid_from_proba`: alpha search by grid, as a method separate from the binary search (kept to investigate differences). Pending.
- `tuning_stability(probs, y, n_splits)`: diagnostic for tuning-set size adequacy (runs the search on disjoint subsets and reports spread). This is the formalization of the "L random splits" scheme from the paper.
- Golden test with a trained model (current ones use an untrained model: reproducible but in a degenerate regime, unstable alphas).
- [DONE] Packaging cleanup (post-Phase 6): see "Phase 6 step 3c — packaging cleanup [RESOLVED]" and its second pass, "Phase 6 step 3c, second pass — reshaping extras around usage [RESOLVED]", both below. Note that torch was already absent from `[project].dependencies` before either cleanup - it only appeared in the optional-dependency groups (first pass) / dependency groups (current). What used to keep torch mandatory was that the core imported `flatten_batch` at module level; Phase 6 step 5 removed that import, and `tests/core/test_import_properties.py::test_core_does_not_import_torch` now guards the property.
- **[SUPERSEDED]** monai is in the `dev` group and depends on `torch` unconditionally — recorded
  at the time as an unwanted side effect, since `torch` was NOT otherwise a `dev`-group member,
  so a plotting-only dev environment paid for `torch` (then pinned to `2.11.0`, per the dry-run
  that established this) purely as monai's transitive dependency. The second packaging pass (see
  "Phase 6 step 3c, second pass" above) made `torch` a direct, intentional `dev`-group member —
  every contributor environment needs it for `tests/integration/torch/` regardless of monai — so
  the "pays for torch it didn't ask for" framing no longer applies; monai no longer changes
  whether `torch` is installed, only that it's one more package alongside it. Only
  `scripts/ACDC_example.py` uses monai anywhere in the repo. Giving monai its own extra (so a
  contributor not running ACDC doesn't pay for it specifically) remains a live, smaller idea; not
  acted on. **[RESOLVED]** monai removed from the dev group entirely rather than given its own extra; it
lives only in the `examples` extra now. Verified in a clean venv: syncing dev plus viz gives
120 passing tests with monai absent from the installed set.
- **sklearn as a test dependency, reported not acted on.** `scikit-learn` is declared in all
  three dependency groups (`dev`, `dev-cuda13`, `dev-rocm7`), and its only reachable use anywhere
  in `src/`, `tests/` or `scripts/` is
  `src/utrace/utils/pytorch/model_wrapper.py:4`'s `from sklearn.base import BaseEstimator` —
  which carries its own `# TODO: not needed anymore` comment from the author. That import is
  exercised by the test suite only because `tests/integration/torch/test_golden_mnist.py` imports
  `Pytorch_wrapper` from `model_wrapper.py`. If the import is removed, `scikit-learn` stops being
  a test dependency (and could drop out of the three groups entirely, pending confirmation
  nothing else picks it up). Flagged per instruction, not acted on — the comment's author should
  confirm the "not needed anymore" claim before the import is actually removed.
  [ESTABLISHED] Not removable, and the comment is wrong. `BaseEstimator` is used as a base class
— `class Pytorch_wrapper(nn.Module, BaseEstimator)` — so removing the import raises NameError
at class-definition time. It is paired with `__sklearn_is_fitted__`, the special method
sklearn's check_is_fitted looks for, so the inheritance is deliberate scikit-learn
compatibility. scikit-learn therefore stays a test dependency. The `# TODO: not needed
anymore` comment on that import is stale and actively misleading — it produced one wrong
backlog item already. Removing the comment, not the import, is the correct follow-up.
- Performance benchmark per phase.
- Accidental public surface: `scores/__init__.py`'s `from .jax_impl import *` re-exports every public module-level name in `jax_impl.py` (no `__all__` there), so `jnp` and `jit` are reachable as `utrace.scores.jnp` / `utrace.scores.jit` without that being a decided API surface. Discovered when removing the unused `jax_print` import — see "Rename batch" > "Accidental public surface" above. Add an explicit `__all__` to `jax_impl.py` (or switch `scores/__init__.py` to named imports) to close it. Not done.
- Buffer/padding design for high-volume regimes (segmentation): the fixed-size `_max_N` buffer must currently be sized per class by hand. Consider a design that scales without manual sizing (without reintroducing variable shapes / JAX recompilation).
- [DONE] `_max_N` overflow guard (Phase 6 step 1): `_calibrate_impl` now checks capacity before writing, in both branches, raising `ValueError` instead of allowing an out-of-bounds write. The pre-fix diagnostic found the failure was not one mode but three distinct behaviours, empirically reproduced:
  1. Batched write whose start index (`_N`) is already `>= _max_N`: JAX's `.at[].set()` silently dropped the update; `_N` still incremented regardless; neither reading `conformity_scores_` afterward nor setting `alpha` raised anything — the only fully silent failure of the three, and the one a caller relying on repeated `batched=True` streaming calibration would hit first.
  2. Batched write straddling the boundary (starts in-bounds, overruns past `_max_N`): raised `ValueError` from JAX broadcasting (the in-bounds slice clips shorter than the incoming values).
  3. Non-batched write with `num_scores > _max_N` in a single call: same `ValueError` as (2).
  Both branches of `_calibrate_impl` now check `_N + num_scores <= _max_N` (batched) / `num_scores <= _max_N` (non-batched) before any state mutation, so all three cases now raise consistently instead of only two of three doing so. Cross-references the manual buffer-sizing item above — a design that scales sizing automatically would still need this guard as a backstop.

- force_non_empty_sets is silently ignored in the new prediction path. The jit _predict_sets does not implement it, and predict (renamed from predict_from_proba by the rename batch) accepts the parameter but does not pass it through. The legacy _predict_sets (initial commit) honored it (y_sets[arange, y_pred] = True). This is behavior lost in the jit migration. Harmless for callers passing False, but a latent bug for any script relying on force_non_empty_sets=True.

- [RESOLVED] The global batched branch of _calibrate_impl concatenated conformity scores into the buffer without re-sorting (.at[_N:_N+num].set with no np.sort), while the non-batched and per-class batched branches do sort. _masked_quantile_higher assumes an ascending-sorted buffer, so the tuning quantile (q_hat) became non-monotonic in alpha when calibrating global+batched, breaking the binary search for U (it failed to converge; U  collapsed to 0 or oscillated). Fix: sort the concatenation, matching the per-class branch.
  - The _masked_quantile_higher unit test did not catch this because it is fed an already-sorted array: the bug was in the integration (calibration violating the sort precondition), not in the function itself.
  - [RESOLVED] Coverage gap: no test exercises the global+batched path. `tests/core/test_deferred_sort_buffer.py::test_property_sorts_on_read` calibrates `classes=None` with `batched=True` across 3 separate calls and, after reading `conformity_scores_`, asserts the valid prefix is ascending-sorted; `test_no_sort_between_batches` covers the same global+batched shape (4 calls) and asserts no sort happens between batches. Together they cover the previously-uncovered global (`classes=None`) batched path (commit 4852c3b).

- [RESOLVED] Per-class calibration double-counted _N: the trailing _N update ran unconditionally and overwrote the correct `_N = total` set inside the per-class branch, adding the last class's num_scores on top (e.g. N=66 for a 60-sample calibration). Fix: move the _N update into the global branch only. Also switched per-class accounting to a per-class count (_class_N) and fixed _class_scores initialization (was np.empty(_max_N), garbage). classes=[full list] now matches classes=None (commit 1a2c8a).

- [RESOLVED] to_jax device mismatch on GPU backends. to_jax routed any object with __dlpack__ through jax.dlpack.from_dlpack; numpy arrays implement __dlpack__, so numpy label arrays landed on CPU (DLPack preserves host origin) while CUDA torch probability tensors landed on GPU. The jitted score (lac_cal) then received its two arguments on different devices and raised "Received incompatible devices for jitted computation". The numpy DLPack path also emitted a "buffer is not aligned, creating a copy" warning (neither zero-copy nor correct-device). Invisible on the CPU backend; only reproduces with a GPU JAX backend. Fix: check isinstance(np.ndarray) BEFORE the __dlpack__ branch, route numpy via jnp.asarray (lands on JAX default compute device); DLPack kept only for genuine framework tensors. Device contract documented in the to_jax docstring: preserve device for tensors, normalize host arrays to the default compute device, do not reconcile mismatches between two genuine tensors. Validated on CUDA (to_jax(numpy)->GPU, to_jax(cuda tensor)->GPU, MNIST_example --extra=cuda matches CPU results). NOTE: the test suite runs on CPU and does NOT exercise this path; it only guards against regression.

- [ESTABLISHED] `index-strategy = "unsafe-best-match"` no longer exists: it was removed as an
incidental part of the extras reshape (fe21f31), not as a verified decision. Every extra and
group resolves without it against the three explicit indexes, so its removal is confirmed
correct after the fact.

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

- [Phase 6] Zero-copy in tuning: `get_uncertainty`'s (renamed from `get_uncertainty_from_proba`) body does `np.asarray(to_jax(...))`, forcing a host copy and negating DLPack zero-copy on the tuning path; `calibrate` / `predict` (renamed from `calibrate_from_proba` / `predict_from_proba`) keep zero-copy. Make tuning consume the jnp array directly (see the adjacent bare `# TODO: ... espera numpy` comment — no symbol name to anchor to; that comment and the call sit at `uncertaintyQuantifier.py:435-436` as of HEAD `53b1e8d`, but re-verify by symbol/grep rather than trusting that number after further commits — it has now moved twice, from `417-418` as of `ebc5ddb`, to `355-356` as of `a0ea8f6`, to `435-436` as of `53b1e8d` (the B.5 state-extraction commit added ~150 lines earlier in the file), each time purely from unrelated line-count changes, never from this call site itself being touched. Re-confirmed still present as of this pass; perf impact is UNMEASURED (the RTX 3070 GPU benchmark above measured the calibration path, not the tuning/uncertainty path).

- Disconnected `transform` parameter in MNIST_example.py: main() receives a `transform`  argument but the noise injection (~:176) uses a hardcoded `AddGaussianNoise`, ignoring it — so the __main__ transform_str dispatch (AWGN/RandomPerspective/ElasticTransform) currently has no effect on the experiment; AWGN is always applied. Likely a remnant of the lambda->class migration done to support num_workers>0 (a lambda transform is not picklable   and breaks multi-worker DataLoaders). To resolve: decide whether to reconnect the transform  sweep (as other scripts do) or whether fixed-AWGN is intentional for this script. If  reconnecting, note the three transforms have different signatures (AddGaussianNoise(0., n), RandomPerspective(n, 1), ElasticTransform(n)), so the swept parameter must be mapped per signature — this is a behavior change, warranting its own commit and revalidation. Separate from the I/O refactor.

- [RESOLVED, step C] Labels passed to what was then the *_from_proba API used to go through .numpy() in the canonical recipe and in all example scripts (e.g. flatten_batch(y).ravel().numpy().astype(int)), violating the "do not call .cpu().numpy() on values feeding the core" rule and, after the to_jax fix, forcing a host->device hop. Fixed across three commits (32a1309, 6a630e5, 0caad12): labels now stay backend tensors into calibrate / get_uncertainty / predict (renamed from calibrate_from_proba / get_uncertainty_from_proba / predict_from_proba by the rename batch), taking the DLPack zero-copy path. The stale claim in this entry — "the six call sites are now marked `# COMPAT`" — undercounted (nine, not six; see "Step C: the `# COMPAT` marker post-mortem" above) and is no longer true regardless: grepping `# COMPAT` in scripts/ now returns nothing. Downstream label indexing (coverage counts, masks, get_coverage) was verified against tensor labels by the commits' own equivalence runs, not by the test suite, which does not cover the scripts.

- to_jax DLPack unaligned-copy: even for genuine tensors the DLPack path can emit "buffer is not aligned ... Creating a copy", so zero-copy is not guaranteed. Decide whether to make such copies VISIBLE (warn/error) rather than silent. Connects to the existing to_jax device-handling backlog item. Perf/observability task.

- User-configurable target device for to_jax (like torch's device=): host arrays currently go to JAX's default compute device; a future API should let the user choose. The current fix is written so the default-device path is the single point a future device= would generalize.

- Noise-sweep scripts rebuild the dataset (and DataLoader) inside the iteration loop, partly to reshuffle the split per iteration and partly to change the noise level. Reconstructing the full dataset per iteration is wasteful — only the noise (and the split) need to change, not the 60000-sample base. Optimization: instantiate the base dataset (and loader) ONCE outside the loop, and inside the loop either mutate the transform's sigma (transform.std sigma — valid because AddGaussianNoise reads self.std in __call__, not __init__) or reassign it (dataset.transform = AddGaussianNoise(0., sigma)). IMPORTANT: the random_split must STAY inside the loop (with a varying generator) to preserve per-iteration reshuffling — only the dataset/loader construction moves out. Caveat: mutating transform.std from the main process only propagates with num_workers=0; with spawn/fork workers, each worker holds its own copy and the loader would need rebuilding (ties into the num_workers decision). Applies to several sweep scripts (MNIST_class_conditional, and others with a noise sweep). Behavior-adjacent — revalidate numbers after the change. Its own diagnostic + commit.

### GPU / scalability (example scripts)

- [DONE] precompute_proba helper consolidated (since renamed to precompute_softmax by the rename batch; still duplicated between scripts/_common.py and scripts/setsize_analysis.py — see "Rename batch" above and the deduplication item below). scripts/_common.py provides precompute_softmax(loader, classifier) returning raw torch tensors (torch.cat of softmax output and labels, no conversion) so the softmax output takes the zero-copy DLPack path into to_jax. Adopted in the six MNIST-like scripts (MNIST_example, MNIST_class_conditional, MNIST_test_coverage, MNIST_test_convergence, convergence_analysis, data_size_analysis). ACDC_example (pixel-scale segmentation) and setsize_analysis (non-batched calibration use) intentionally keep their own logic — setsize_analysis defines its own precompute_softmax rather than importing the shared one. Faithful dedup: in the five scripts whose labels were numpy, a temporary compatibility line `flatten_batch(y).ravel().numpy().astype(int)` tagged `# COMPAT` preserves current behavior; convergence_analysis already consumed tensor labels and has no COMPAT line.
- Forward-pass batch vs jit padding are SEPARATE knobs; do not tie them. The DataLoader batch size only chunks the model forward (no effect on results — the tune set is re-concatenated and passed whole to get_uncertainty). max_batch_size is the jit padding and must be >= the materialized tune set. On an 8GB GPU the OOMs were ALWAYS in the model forward (predict_proba), never in the utrace core/tuning. Scripts pin max_batch_size to a hardcoded constant (e.g. 12000) tied to the 0.2 tune split of 60k MNIST; prefer deriving it (ceil(tune_split * len(dataset)) + margin) instead of a magic number.
- Whether to deduplicate precompute_softmax, still independently defined in both scripts/_common.py and scripts/setsize_analysis.py (functionally identical since step C). The rename batch's caller-locals commit renamed both definitions independently and deliberately did NOT deduplicate them — that remains a separate, undecided decision.
- [DONE for now] DataLoader num_workers set to 0 in MNIST_class_conditional_example (the only script that used workers; ACDC already used 0, the rest default to 0). Reason: num_workers>0 forks, and forking after JAX has initialized its threads can deadlock (generic fork-with-multithreading hazard, not a JAX bug). Resolved at the root by disabling workers. Deferred alternatives if workers are wanted back (e.g. if data loading becomes the bottleneck rather than the GPU forward pass): (a) spawn start method — robust but pays interpreter startup cost, requires picklable transforms (already satisfied: lambda->AddGaussianNoise) and the __main__ guard (already present); (b) lazy JAX initialization so all worker forks happen before XLA threads start — fragile/non-deterministic, NOT recommended. Decide by measuring workers=4 vs 0 wall-time on GPU first (the per-class script is likely GPU-forward-bound, so workers may add little). A permanent user-facing note belongs in docs/ (future), since this affects anyone using torch DataLoaders alongside the package — MIGRATION.md is process log only.
- 8GB VRAM is a hard constraint, not a bug: the per-class script (10 CPs) runs but only just fits with small forward batches. Not something to "fix"; scripts should scale by config.

## Phase 6 diagnostic findings (historical — resolved by steps 3a/3b except where marked)

- [RESOLVED, step 3b] Running with `USE_JAX=false` used to raise `ImportError` at package-import time, reproduced directly at the time (`USE_JAX=false uv run --extra=cpu python -c "import utrace"`):
  ```
  File "src/utrace/__init__.py", line 2, in <module>
      from .uncertaintyQuantifier import UncertaintyQuantifier
  File "src/utrace/uncertaintyQuantifier.py", line 16, in <module>
      from .utils import _masked_quantile_higher, _bucket_size
  ImportError: cannot import name '_masked_quantile_higher' from 'utrace.utils'
  ```
  `utils/__init__.py` used to define `_masked_quantile_higher` only inside `if USE_JAX:`. Consequently, `scores/numpy_impl.py` was unreachable code: no configuration could load it — the package failed to import before `scores/__init__.py`'s own `if USE_JAX:` branch would even matter. `USE_JAX` no longer exists anywhere in `src/`; `_masked_quantile_higher` is exported unconditionally; `scores/numpy_impl.py` was deleted (step 3b). Kept here as the historical record of why this diagnostic made the removal behaviour-preserving by construction, not merely by luck.
- [RESOLVED, step 3a] Consequence at the time: `score='aps'` was non-functional in every reachable configuration (`scores/jax_impl.py`'s `aps`/`aps_cal` both unconditionally raised `NotImplementedError()`; the numpy implementation that did implement them could never load, per the point above). Resolved: `score='aps'` now raises `ValueError` explicitly at construction, naming `'lac'` as the supported value; the `aps`/`aps_cal` stub functions were removed from `scores/jax_impl.py` in step 3b.
- [RESOLVED, step 3b] `src/utrace/.env.example` used to ship `USE_JAX=False` while the actual development `src/utrace/.env` in this checkout had `USE_JAX=True`; a fresh clone following the repo's own example file would have gotten a package that raised `ImportError` on `import utrace`. `.env.example` was deleted; `USE_JAX` has no effect anywhere now, so there is no longer a variable for an example file to get wrong.
- [RESOLVED, step 3b — moot] `scores/numpy_impl.py`'s `lac_cal` was not actually numpy-typed despite the module name (it called `.cpu().numpy()` on the indexed result, which only works on a torch tensor). This was unreachable code even before removal, so it never bit in practice; the file was deleted outright.
- [RESOLVED, step 3b] x64 precision's ordering concern — `src/utrace/__init__.py` imports `.uncertaintyQuantifier` BEFORE calling `jax.config.update("jax_enable_x64", True)`, which worked only because nothing at import time of `uncertaintyQuantifier.py` computes a float64 array before the flag is set — was preserved when the call became unconditional (the `if USE_JAX:` guard was removed, not the ordering). Now directly guarded by `tests/core/test_x64_is_enabled.py::test_x64_is_enabled`.

### Phase 6 step 3c — packaging cleanup [RESOLVED]

Resolved across three commits after Phase 6. `jax` and `numpy` moved into base
`[project].dependencies` (the previous shape declared `jax` only inside the
`cpu`/`cuda`/`cuda124`/`rocm`/`rocm5` extras, so a plain `pip install utrace` produced a package
that raised on `import utrace` — a package-level version of the same "reached unconditionally
vs. reached only when asked for" problem described for matplotlib/pandas under "Import
structure" below). `matplotlib`, `pandas`, `torch` and `torchvision` moved to extras; the
`cpu`/`cuda`/`cuda124`/`rocm`/`rocm5` extra names were retired in favour of
`viz`/`torch`/`cuda13`/`rocm7-local`.

What the blocking investigation found, corrected against what was recorded at the time:

- [ESTABLISHED] `python-dotenv` was inert, exactly as recorded — `config.py`, its only
  consumer, was deleted in step 3b. It is now removed from `[project].dependencies` outright.
  The `uv.lock` re-resolve that used to fail (see below) now succeeds, so the blocker is gone,
  not worked around.
- [ESTABLISHED] Three more dependencies had zero import sites anywhere in `src/`, `tests/` or
  `scripts/` and are now removed too: `scikit-image`, `tqdm`, `flax`. Worth recording the irony
  on `flax`: its only mention anywhere in the repo was the comment in `_UQState`'s docstring
  explaining that step B.5 used a plain `NamedTuple` *instead of* it — the dependency stayed
  declared for a design that had already been explicitly rejected. Dropping it also removed
  roughly twenty packages that existed only to support it (`optax`, the `orbax-*` ecosystem,
  `etils`, `msgpack`, `humanize`, `treescope`, and others transitive to those).
- [ESTABLISHED] The typo'd `torch-rocm` index (`eexplicit = true`) is gone — the rocm6.4 index
  it pointed at was retired outright, superseded by a ROCm 7.2 index, so the postmortem note
  about it is now purely historical, not a description of the current file. `uv.lock` is
  derivable from `pyproject.toml` again: a plain `uv lock` against the corrected file succeeds
  cleanly with no hand-patching. The deliberate desync recorded above is over.
- [ESTABLISHED] `jax[rocm]` never existed as an extra, exactly as the original diagnostic found
  (`uv lock` warned at the time: "the package `jax==0.9.2` does not have an extra named
  `rocm`"). The correct name, confirmed by resolving it directly, is `jax[rocm7-local]` — and
  per JAX's own installation docs, it requires ROCm already present on the host or container;
  JAX ships no extra that installs ROCm itself.
- [ESTABLISHED] The transitive triton package under ROCm 7.2 is named `triton-rocm`, not
  `pytorch-triton-rocm`. The old name was correct for the rocm6.4-era wheels this repo used to
  target, which is exactly why the original blocked note recorded it — but current torch under
  ROCm 7.2 renamed it. Confirmed by resolving against the real index: the resolver's own
  dependency line named `triton-rocm` directly, and omitting it from the extra reproduced the
  exact failure this note predicted — the resolver backing off through successive torch
  versions, downloading full wheels for metadata, before giving up on a years-old release.
- [ESTABLISHED] Routing `triton-rocm` in `[tool.uv.sources]` was not enough on its own to fix
  that failure. `uv` requires an extra-scoped source to point at a package the extra declares
  DIRECTLY — `triton-rocm` had to be added to the `rocm7-local` extra's own dependency list, not
  merely given an index route, before it would resolve.
- [ESTABLISHED, corrected] The CUDA index was independently stale, but not quite in the way
  first assumed: cu128 wheels have **not** disappeared — `torch==2.11.0+cu128` still resolves —
  but cu128 is not maintained past that release, and the current stable torch (2.13.0, matching
  what the `cuda13` extra's cu130 index carries) has no cu128 build. So cu128 was stale as an
  index *choice* (it would silently cap the resolved torch version well below current), not
  stale as in "gone." The repo now targets cu130 under the renamed `cuda13` extra instead.

**Superseded by the reshape below.** The shape this section describes — a `torch` extra,
`cuda13`/`rocm7-local` extras that themselves carried torch, and a "point pip at
`--extra-index-url`" workaround for it — was itself replaced by a second packaging pass shortly
after landing. The dependency-cleanup findings above (dead packages removed, index typos fixed,
`triton-rocm` naming, cu128 staleness) remain true of the current file; the extras table and the
pip-routing analysis do not. See the next section.

### Phase 6 step 3c, second pass — reshaping extras around usage [RESOLVED]

[ESTABLISHED] The shape landed by the first packaging pass (previous section) declared torch
directly inside the GPU extras. That inverts the package's primary use case: someone who already
has a working PyTorch build for their hardware and wants uncertainty quantification on top of
it. Installing `utrace[cuda13]` into such an environment would pull in, and could replace, their
existing torch. Commit `fe21f31` ("Reshape the extras around how the package is actually used")
removed the `torch` extra outright and rebuilt the extras around who installs what:

| extra | declares | torch? |
|---|---|---|
| (base, `[project].dependencies`) | jax, numpy | no |
| `cuda13` | jax[cuda13] | no |
| `rocm7-local` | jax[rocm7-local] | no |
| `viz` | matplotlib, pandas | no |
| `examples` | torch, torchvision, monai, nibabel | yes (CPU build, routed via `[tool.uv.sources]`) |

`utrace.utils.pytorch`'s helpers do not need `utrace` to install torch; they need it present. If
it is not, importing them raises a plain `ImportError` — accepted as the honest failure mode
rather than something to paper over.

Contributors (who need torch to run `tests/integration/torch/`) now get it from one of three
mutually conflicting dependency groups instead of an extra: `dev` (default, routes to the `cpu`
index), `dev-cuda13` (routes to `cu130`), `dev-rocm7` (routes to `rocm7.2`, plus `triton-rocm`).
The test command changed accordingly: `uv run --extra=torch --extra=viz pytest ...` became
`uv run --extra=viz pytest ...` — torch now arrives from the default `dev` group, not an extra.

[ESTABLISHED] Verified directly in this pass, not assumed from the commit message:
- **Group-scoped `[tool.uv.sources]` routing is genuine.** Reading `uv.lock` itself (not
  checking exit codes) shows separate `torch`/`torchvision` entries with distinct
  `source.registry` values keyed by marker: `https://download.pytorch.org/whl/cpu` for
  `group-6-utrace-dev` and `extra-6-utrace-examples`, `.../whl/cu130` for
  `group-6-utrace-dev-cuda13`, `.../whl/rocm7.2` for `group-6-utrace-dev-rocm7`. All four routes
  resolve to the same torch/torchvision version (2.13.0 / 0.28.0) — no version skew between
  them, confirming the drift fix below actually landed everywhere.
- **`utrace[cuda13]` does not touch an existing torch.** `pip install "utrace[cuda13]" --dry-run`
  against a venv with `torch==2.13.0+cpu` already installed shows a "Would install" list of
  `jax`, `jax-cuda13-pjrt`, `jax-cuda13-plugin`, `jaxlib`, the `nvidia-*` CUDA runtime packages,
  `numpy`, `scipy`, `opt_einsum` and `utrace` itself — torch and torchvision appear nowhere in
  it, because `cuda13` never declares them as a dependency. A real install over that same venv
  (not a dry-run) is recorded in `.reports/2026-08-14_packaging_redesign_docs.md`, confirming the
  installed torch build and its files are unchanged before and after.
- **`rocm7-local` resolves cleanly and stays torch-free.** `uv pip install --dry-run
  utrace[rocm7-local]` adds exactly two packages beyond base — `jax-rocm7-pjrt`,
  `jax-rocm7-plugin` — matching the extras table above.

[ESTABLISHED] **The failure that shaped the `examples` scoping.** The first shape attempted for
`examples` routed `torch`/`torchvision` to the `cpu` index unconditionally (unscoped by extra or
group). `uv lock` rejected it outright: conflicting indexes for `torch` across all marker
environments, because the GPU dev groups (`dev-cuda13`, `dev-rocm7`) already route the same
package name to different indexes, and an unscoped route collides with a scoped one in any
resolution fork where both are active simultaneously. The fix actually shipped: scope the
`examples` extra's route to `extra = "examples"` specifically (as the `dev`/`dev-cuda13`/
`dev-rocm7` routes are scoped to their own groups), and declare `examples` conflicting with the
two GPU groups — `dev-cuda13`, `dev-rocm7` — but NOT with `dev`, since `dev` and `examples` both
want the same `cpu` index and have nothing to fight over.

[ESTABLISHED] **Torch version drift, found and fixed.** Before this pass, `uv.lock`'s `cpu`-routed
torch was pinned at `2.11.0` (left over from earlier work), while the `cu130` route had already
moved to `2.13.0` — confirmed by reading the lock at the prior commit (`89a5411`) directly: the
`cpu`-registry `torch` entries read `version = "2.11.0"` / `"2.11.0+cpu"`, while the `cu130`-registry
entry already read `version = "2.13.0+cu130"`. `uv`'s minimal-change resolution had preserved the
`2.11.0` pin on `cpu` rather than advancing it, which would have left CPU contributors testing
against a different torch than CUDA contributors. A scoped
`uv lock --upgrade-package torch --upgrade-package torchvision` moved all routes to 2.13.0 /
0.28.0 uniformly (`setuptools` also moved 81.0.0 → 82.0.1 as a side effect of the same scoped
resolve — confirmed by diffing `uv.lock` at both commits). The golden `.npy` baselines were
verified unchanged across the upgrade (recorded in
`.reports/2026-08-14_packaging_redesign_execution.md`), which matters because those baselines
were originally generated under the older torch — an upgrade that silently shifted numerics
would have invalidated them without any test failing to say so.

### uv finding: dependency-groups and extra-scoped index routing don't mix

[ESTABLISHED] An unmarked (plain) requirement on a package inside a `[dependency-groups]`
member conflicts with extra-scoped `[tool.uv.sources]` routing for that same package name,
because `uv` treats dependency-group members as active in every resolution fork simultaneously
— including forks where an extra-scoped route also claims the same package. Reproduced
directly: declaring `torch` as a plain `dev`-group member, with `[tool.uv.sources]` routing
`torch` per-extra plus an unconditional fallback route for everything else, makes `uv lock` fail
with "Requirements contain conflicting indexes for package `torch` in all marker environments."
This general finding still holds and is why the current file routes torch with a **group-scoped**
source per dependency group (`group = "dev"`, `group = "dev-cuda13"`, `group = "dev-rocm7"`)
rather than an extra-scoped route plus an unconditional fallback.

**Superseded consequence.** At the time this was written, the fix was to keep `torch`/
`torchvision` OUT of the `dev` group entirely and supply them via a `torch` extra instead, which
is why the test command briefly required `uv run --extra=torch --extra=viz pytest ...`. The
second packaging pass (see "Phase 6 step 3c, second pass" above) replaced that with group-scoped
sources, so `torch`/`torchvision` ARE now plain members of `dev`/`dev-cuda13`/`dev-rocm7` — the
group-scoped source resolves the same conflict this section describes, without needing to keep
torch out of the group. The test command is `uv run --extra=viz pytest ...`; `--extra=torch` no
longer exists.

### Import structure: matplotlib and pandas deferred out of `import utrace`

[ESTABLISHED] `import utrace` used to reach `matplotlib` and `pandas` through `utils/utils.py`'s
module-level imports (`utils/__init__.py`'s `from .utils import *` pulls in the whole module
for `_bucket_size`, which `uncertaintyQuantifier.py` imports), and reached `matplotlib` a second
time, independently, through `utils/pytorch/helpers.py`'s own module-level import — reached
whenever a caller imports `flatten_batch`, e.g. `tests/integration/torch/test_golden_mnist.py`.
Both matplotlib imports and the one pandas import are now deferred into the bodies of the
specific functions that use them (`plot_scores` and `class_wise_performance` in
`utils/utils.py`; `view_classify` in `utils/pytorch/helpers.py`) instead of sitting at module
scope. `tests/core/test_import_properties.py` guards both properties by subprocess, matching
its existing pattern for torch-absence.

[ESTABLISHED] The annotation trap: Python evaluates a function's annotations when the `def`
statement executes, not when the function is called, so moving an import into a function body
does not help if a signature in the same module annotates a parameter with the now-deferred
name. `plot_scores`'s `ax` parameter was annotated `ax: plt.Axes`; quoting it as a string
forward reference (`ax: "plt.Axes"`) resolved it, chosen deliberately over
`from __future__ import annotations`, which would have changed evaluation semantics for every
annotation in the module rather than just the one that needed it. `utils/pytorch/helpers.py`
was checked for the same pattern and does not have it — none of its functions carry type
annotations at all, so no annotation-handling decision was needed there.

## Canonical migration recipe (new API)

Apply to each script. The canonical alpha-search method is the **binary** one (`get_uncertainty`,
renamed from `get_uncertainty_from_proba` by the rename batch — see "Rename batch" above), which
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
- `tests/core/test_x64_is_enabled.py` and `tests/core/test_import_properties.py` assert properties of the package as an importable artifact (global JAX x64 config, torch absent from `sys.modules` after `import utrace`) rather than API behavior; they are grouped by that subject, not by mechanism — the torch-absence check runs in a subprocess because the rest of the integration suite imports torch into the main test process.

## Decisions to respect (do NOT "fix" without confirming)

- In the coverage test scripts, `uq.alpha = U` (setting alpha to the U value, not the tuned alpha) is INTENTIONAL: it is part of the alignment tests between U and (1-Cov). It may be changed to `alpha` in the future, but it is not a bug.

- **SUPERSEDED**: The per-class branch with a *full* class list was equivalent to `classes=None` (commit 1a2c8a); with a *partial* list it already calibrated group-conditionally on the listed classes only. We have formalized this as group semantics: `classes=[labels]` calibrates on the subpopulation whose label ∈ labels (group-conditional coverage); `classes=None` is marginal. Multiple classes are fully supported under these semantics, and the internal per-class buffers (`_class_scores`, `_class_N`, `_class_alphas`, `_class_q_hats`) have been removed.

## Post-migration analysis (open items)

1. U-vs-alpha for the two Appendix-A scripts (`MNIST_test_coverage.py`, `MNIST_test_convergence.py`): both now use `U` (not the tuned alpha) on BOTH the prediction threshold (`cp.alpha = U`) and the BetaBinom null parameter (`a_p = U_mean`), per the `uq.alpha = U` decision above. Whether the tuned alpha is preferable instead is an open question — revisit once, for both scripts together, not independently.
2. BetaBinom null fragility: in both Appendix-A scripts, `Nr` (= `Nv`) is taken from only the last loop iteration's test-set size, while per-iteration sizes vary by ~1 sample due to `random_split`'s remainder rounding. The null distribution's trial count is therefore a (very close) approximation, not exact, across all recorded iterations.
3. GPU validation for per-class calibration: the per-class calibration path (classes=[...]) has not been fully validated on a GPU backend end-to-end; only the global path (classes=None, MNIST_example) has a clean GPU run. MNIST_class_conditional ran on GPU only with ad-hoc batch tuning (a probe, not the final structure). This remains an open GPU-validation item.

## Legacy method state (discovered during ACDC migration)

**Update (rename batch): `calibrate`, `predict`, and bare `get_uncertainty` are no longer absent
names.** This whole section predates the rename batch and describes their state as of Phase 6.
The rename batch's third commit (`70fa11e`) reintroduced `calibrate` and `predict` as the
current public API (renamed from `calibrate_from_proba`/`predict_from_proba`), and bare
`get_uncertainty` as the current public alpha-search method (renamed from
`get_uncertainty_from_proba` — a different symbol from the `get_uncertainty` removed below,
which no longer exists under any name). Where this section says a script "does NOT run" because
it calls one of these three names, that is now **wrong in the dangerous direction**: such a
script DOES run, against the new implementation, silently passing raw model/data input where
softmax output is expected. See "Rename batch" above for the full explanation and the accepted
risk. `get_uncertainty_jit` and the `model=` parameter remain genuinely, permanently absent —
nothing reused those.

The legacy subset exercised by the former legacy golden — `calibrate`, `get_uncertainty_jit`, `predict` — plus the `model=` constructor parameter, were all **removed** (Phase 6 steps 5-6; confirmed absent from `src/utrace/uncertaintyQuantifier.py` as of that time — grepping any of the four in `src/` returned nothing then, and `UncertaintyQuantifier(model=x)` still raises `TypeError` from Python itself today, with no compatibility shim). As of the rename batch, `calibrate` and `predict` exist again as unrelated methods (see update above); `get_uncertainty_jit` remains absent.

Two further legacy methods were found broken/orphaned during the ACDC migration and were **removed** earlier (Phase 6 step 2; confirmed absent from `src/utrace/uncertaintyQuantifier.py` as of that time):
- `get_uncertainty` (no suffix) called `self._predict_sets`, which was never defined as a method anywhere in the class (only as a same-named module-level function taking no `self`) → raised `AttributeError` at runtime on every call. Removed. As of the rename batch, `get_uncertainty` exists again as the renamed `get_uncertainty_from_proba` — an unrelated, working implementation; the removal above describes the pre-Phase-6 broken method, not the current one.
- `get_uncertainty_opt` was model-bound (called `self.model.predict_proba`, had no `*_from_proba` counterpart) and raised `AttributeError` if the UQ was constructed without `model=`, making it unusable under the no-model migration. Removed, together with its sole helper `get_U` — which had exactly one call site in the entire repo (inside `get_uncertainty_opt` itself) and so had no other consumer to preserve. Both remain absent; neither name was reused.

`fit_opt`, `predict_opt`, `fit` remain absent entirely — unrelated to this pass; they were never part of the current `UncertaintyQuantifier` and were never something Phase 6 needed to remove.

Consequence, corrected for the rename batch: a script relying on `get_uncertainty_opt`, `get_U`, `get_uncertainty_jit`, the removed `model=` parameter, or `fit`/`fit_opt`/`predict_opt` still does NOT run against the current package — those names and that parameter remain genuinely absent, and calling them still raises `AttributeError`/`TypeError`. A script relying on `calibrate`, `predict`, or bare `get_uncertainty` (with raw model/data input, the pre-Phase-6 calling convention) now DOES run, but against the current softmax-input implementation — it will not raise, and its output should not be trusted without checking the arguments actually being passed. Such scripts (convergence_analysis, data_size_analysis, setsize_analysis, MNIST_test_coverage, MNIST_test_convergence) are full rewrites — validate them against the paper, not a prior local run, and confirm they are calling the current API with softmax-output arguments, not merely that they run without error. Tests passing does NOT imply these scripts are correct against the current API; the test suite covers the core, not the scripts.

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
- `fit` was the former name of `calibrate` in the pre-Phase-6 legacy API — note that, as of the rename batch, `calibrate` is ALSO the current public method name, an unrelated implementation reintroduced by renaming `calibrate_from_proba` (see "Rename batch" above and the correction in "Legacy method state" above). `*_opt` were "optimized" variants; part of their logic was folded into the main methods. These scripts, in their pre-Phase-4 form, were written against the legacy API and did NOT run as-is against the post-Phase-6 package: migrating them meant rewriting them against the `*_from_proba` API of the time (since further renamed by the rename batch). Note: `fit`, `fit_opt`, and `predict_opt` are fully absent from the current `UncertaintyQuantifier` (not merely deprecated) and always have been — nothing reused these three. `get_uncertainty_opt` and its helper `get_U` are likewise fully absent (removed, Phase 6 step 2) and were never reused. `get_uncertainty_jit` (removed, Phase 6 steps 5-6) is also still fully absent. By contrast, `calibrate`, `predict`, and bare `get_uncertainty` — all three named in this table's "Legacy methods used" column — are NOT fully absent from the current `UncertaintyQuantifier`: the rename batch reintroduced all three, with unrelated softmax-input implementations. This column remains a historical record of what each script called before its Phase 4 rewrite, and is still not a claim about what exists in the current class — but as of the rename batch, three of its names now collide with current, working method names rather than resolving to nothing.
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

- **Core** (`uncertaintyQuantifier`, alpha-search functions, masked quantile): imports ONLY numpy + jax. NEVER imports torch, onnx, or any backend, and NEVER imports from the backend subpackages below. Data flow is always: user code → backend wrapper → softmax output → core (`calibrate`/`predict`/`get_uncertainty`).
- **`utrace.utils.pytorch.*`**: everything that touches torch — `Pytorch_wrapper`, example models, dataset loaders, transforms, and any helper that needs torch (e.g. `flatten_batch` / `unflatten` if they operate on torch tensors).
- **`utrace.utils.onnx.*`**: analogous, for the ONNX backend.
- **`utrace.utils`** (root): only truly backend-agnostic helpers (pure numpy).
- The JAX backend's GPU acceleration is available via the **optional extras** `cuda13` and
  `rocm7-local` (neither declares torch — see "Phase 6 step 3c, second pass" above). The PyTorch
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

This rule guides both the script migration (Phase 4) and the packaging cleanup tracked in the backlog and Phase 6 — the goal recorded here as "make torch an optional extra" is what the first packaging pass did; the second pass (see "Phase 6 step 3c, second pass" above) replaced the `torch` extra with the `examples` extra plus contributor dependency groups, for the reasons given there. The underlying rule — core installable and importable without torch — is unchanged by which mechanism supplies torch to whom.

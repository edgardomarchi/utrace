# U-TraCE — Findings

Verified findings from the backend-agnostic-core refactor, kept regardless of the refactor's status: measured results, negative results, uv/packaging discoveries, and post-mortems. These outlive the refactor. Split out of MIGRATION.md on 2026-08-19 (see `.reports/2026-08-19_docs_restructure.md`).

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

### Accidental public surface [RESOLVED]

[ESTABLISHED] `src/utrace/scores/__init__.py` does `from .jax_impl import *`, and
`scores/jax_impl.py` defines no `__all__`, so every public (non-underscore) module-level name
bound in `jax_impl.py` is re-exported through `utrace.scores`. Before the dead-code commit this
included `jax_print` (`utrace.scores.jax_print`, reachable with zero consumers anywhere in the
repo); it still includes `jnp` and `jit` — `utrace.scores.jnp` and `utrace.scores.jit` are
reachable today, without anyone having decided that as an API surface. Verified by execution
during the batch (`from utrace import scores; 'jax_print' in dir(scores)` returned `True` before
the dead-code commit), not inferred from reading the wildcard import.

[RESOLVED] `scores/jax_impl.py` now declares `__all__ = ["lac", "lac_cal"]` (landed in the
three-cleanups commit), closing the wildcard re-export: `jnp` and `jit` are no longer
reachable through `utrace.scores`.

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
scalability (example scripts)" in BACKLOG.md) were not a reliable inventory of the conversions step C
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

This was an inference from reading the code, not an observation, when written: every machine used
in this refactor up to that point had been CPU-only, where all placements coincide and the
mismatch cannot appear.

**[ESTABLISHED, confirmed and fixed]** The RTX 3070 has since been available and answered this
directly. `.reports/2026-08-20_gpu_verification_3070.md` (Q1) reproduced the crash exactly as
predicted, both synthetically and against the real ACDC model and data: `calibrate()` raised
`ValueError: Received incompatible devices for jitted computation` on the first calibration
batch when probabilities are GPU-resident and labels are host-resident. It fails loud, not
silently — no wrong numbers were produced. The fix landed in `calibrate()` in commit `7f140ea`
("Reconcile argument devices in calibrate", see `.reports/2026-08-21_stepE_device_coherence.md`),
and the analogous defect in `get_uncertainty` (found by the 2026-08-21 docs audit, H4b) was fixed
separately in Batch 1 (`.reports/2026-08-21_batch1_defect_fixes.md`). `predict` takes no `y` and
was never exposed. Both fixes have GPU-only regression tests
(`tests/core/test_calibrate_device_reconciliation.py`,
`tests/core/test_get_uncertainty_device_reconciliation.py`).

### Convention: performance figures carry their machine

[ESTABLISHED] Measurements in this document come from more than one machine and are not
comparable across them. Every performance figure recorded from now on must name the machine it
was taken on. Retroactively labelled by this pass: the B1 sort-buffer win and the B2 regression
figures (the isolated-section comparison, the call-count sweep, and the golden-run delta) were
taken on a Ryzen 7 5700G workstation; the ACDC step-C figures were taken on a Ryzen AI 7 350
laptop.

The defer-sort win (~357x, 7.2s → 20ms, the defer-sort entry below, in this document) keeps its
existing attribution to an RTX 3070. The `to_jax` direct-vs-indirect figures under "What is
unverified" below were measured on the 5700G; see
`.reports/2026-07-29_phase6_step7_diagnostic_labels_hostcopy_5700G.md`.

[UPDATED] Most measurements were CPU backend at the time this was written; the RTX 3070 has since
produced several reports (`.reports/2026-08-20_gpu_verification_3070.md`,
`.reports/2026-08-20_gpu_packaging_fixes.md`, `.reports/2026-08-21_stepE_device_coherence.md`,
`.reports/2026-08-21_gpu_main_verification.md`, `.reports/2026-08-21_gpu_measurements_acdc.md`).
Of the open GPU questions listed here: the device-to-host-to-device ratio is now measured
(`2026-08-20_gpu_verification_3070.md`, Q3: 32.03x at 2M elements for CUDA-origin tensors, not
the CPU-measured 175x — a different comparison; the direct `to_jax` path is not flat on GPU
either, unlike on CPU); whether `np.asarray` rejects a CUDA tensor is confirmed yes (same report,
Q2, `TypeError`); and the device-commitment risk from step C is confirmed real and has since been
fixed (see "Step C: device-commitment risk" above). [UNVERIFIED] still open: whether the jnp-native
padding measured as a CPU regression under "Measured negative result: jnp-native padding" in
MIGRATION.md pays off inside a jit trace on GPU — no report has measured this specific question.

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
(`tests/core/test_x64_is_enabled.py` and `tests/core/test_import_properties.py`, at the time
separate files grouped by subject — see "Test conventions" in CONTRIBUTING.md) — was NOT done by the rename
batch, and stayed open for a time; it was done later, by a separate commit
(`2011758`, "Consolidate the import-property tests into one file") that predates this pass —
`tests/core/test_x64_is_enabled.py` no longer exists, and `test_x64_is_enabled` lives in
`tests/core/test_import_properties.py` alongside the rest. Recorded here as corrected, not as
newly done by this pass — this document had not been updated to reflect it until now.

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
  device→host→device versus device→device — a DIFFERENT ratio. **[ESTABLISHED]** now measured:
  `.reports/2026-08-20_gpu_verification_3070.md` (Q3) found 32.03x at 2M elements for
  CUDA-origin tensors (5.34x for CPU-origin tensors at the same size) — do not read the CPU
  ~175x figure as applying to GPU hardware; the direct `to_jax` path's flatness does not hold on
  GPU either, it scales mildly with size.
- **[ESTABLISHED]** Whether `np.asarray()` on a torch CUDA tensor raises, and therefore whether
  the pre-step-A `calibrate` would have rejected GPU-resident labels outright rather than merely
  copying them inefficiently. Confirmed yes: `.reports/2026-08-20_gpu_verification_3070.md` (Q2)
  — `np.asarray(t_cuda)` raises `TypeError: can't convert cuda:0 device type tensor to numpy`.
  The current DLPack-based path correctly accepts CUDA-resident labels when they agree with the
  probability tensor's device (mismatched devices raise instead — see "Step C:
  device-commitment risk" above).
- [UNVERIFIED] Whether the real cost is the copy or the per-batch device→host synchronisation
  that `np.asarray()` on a jax array forces. In the ACDC streaming pattern this would be a sync
  barrier per batch, which an isolated microbenchmark cannot see because it calls
  `block_until_ready` anyway.
- [ESTABLISHED] Whether jitting `_calibrate_impl` pays at all, independent of vmap, for the
  marginal (`classes=None`) slice — see "Step D, marginal slice" in MIGRATION.md for the full
  numbers, machine, and design findings. On CPU: yes, it pays, from the very first call, at the
  batch sizes MNIST-family scripts use (~2-3x end-to-end through the public `calibrate()`, more
  in isolation), shrinking to near-parity (~1.08x) at ACDC pixel scale. **Since answered on GPU
  too**: `.reports/2026-08-20_gpu_verification_3070.md` (Q4/Q5) measured the same marginal-write
  jit on an RTX 3070 and found the win does not shrink to parity there — it stabilizes around
  ~1.3x even at pixel scale (B=2,000,000; ~1.3-1.5x depending on whether measured as a per-call
  steady-state ratio or total streaming wall-clock), a real, resolved separation, not overlapping
  noise. The class-conditional path (with vmap not yet in the picture) remains untested and
  untouched on both backends — this item's "independent of vmap" framing is answered only for
  the branch that has no class filter to interact with vmap in the first place.
- [UNVERIFIED] Whether returning jnp scalars instead of numpy scalars from
  `get_uncertainty` breaks any script. `float()`, `np.isnan()` and pandas all accept
  jnp scalars, so this is expected to be soft, but it must be checked against the scripts
  rather than assumed.

## CI and tooling (post-Phase-6 infrastructure)

[ESTABLISHED] Five pieces of work landed here, recorded together since they arrived in sequence
and the later ones build on the earlier: a CI workflow (the project had none before); a Python
3.11 syntax defect in `scripts/`, found once CI could check for it; ruff adoption, rung 1; the
pytest dependency-group split; and a `scripts/` lint cleanup that brought `scripts/` under the
new lint job. None of the five is a phase and nothing here changes the Phase 6 step list or the
step ladder in MIGRATION.md.

### Continuous integration

[ESTABLISHED] The project had no CI until now; everything had been verified by hand across three
machines. `.github/workflows/tests.yml` now runs on every push and on pull requests against
`main`:
- `lock-check` — `uv lock --check`, catching a `uv.lock` that has drifted from `pyproject.toml`
  before anything else runs, without resolving or installing anything.
- `base-import` — installs only the base package (no extras, no dependency group) and asserts
  `import utrace` succeeds with torch absent from `sys.modules`. Matrixed across Python 3.11,
  3.12, 3.13, 3.14.
- `core` — runs `tests/core/`, matrixed across the same four Python versions; also runs a
  `compileall` pass over `scripts/` as one of its steps (see below).
- `integration` — runs `tests/integration/torch/`, matrixed across the same four versions.
- `lint` — `ruff check src/ tests/ scripts/` (default ruleset), a single run, not matrixed.

[ESTABLISHED] `uv sync --locked` and `uv sync --frozen` are not synonyms, and the difference is
the entire point of the drift job. Verified against a copy of the project with simulated drift (a
dependency added to a copy of `pyproject.toml` without re-locking): `--frozen` installs the stale
lock silently and exits 0; `--locked` and `uv lock --check` both exit 1. A lock that has drifted
from the pyproject has bitten this project before — the check exists because of that history, not
speculatively.

[ESTABLISHED] Third-party GitHub Actions (`actions/checkout`, `astral-sh/setup-uv`) are pinned by
commit SHA rather than by tag, because a tag is a movable pointer to third-party code that runs
with access to the repository. `.github/dependabot.yml` keeps the pins current (monthly, on the
`github-actions` ecosystem; it also has a provisional `uv`-ecosystem block for `uv.lock` itself,
flagged in its own comment as newer, less-proven Dependabot support, to be dropped if it turns out
not to work). `uv` itself is deliberately NOT pinned to a specific release in the workflow: it
comes from its own release process rather than running third-party code with repository access in
the same sense, and `uv.lock` already fixes every dependency version regardless of which `uv`
release resolves it.

### The Python 3.11 syntax defect scripts/ carried

[ESTABLISHED] `scripts/data_size_analysis.py` used nested identical quote characters inside an
f-string — PEP 701 syntax, requiring Python 3.12 — while `requires-python` declares `>=3.11`. The
script did not compile on the oldest Python version the project claims to support. Fixed by
commit `76fc223` ("Fix f-string syntax that broke scripts/ on Python 3.11").

[ESTABLISHED] Two existing nets both missed it, neither one built to cover the specific
combination that hid it:
- CI runs the test suite on 3.11 and passed regardless, because no test in `tests/` imports
  anything under `scripts/` — a syntax error in a script is invisible to `pytest` collection.
- The same script had already run successfully, twice, during the step C equivalence check (see
  "Verification method for unvalidated scripts" in CONTRIBUTING.md; the run itself is recorded in
  `.reports/2026-07-31_stepC_group1_5700G.md`) — but in a local environment using this project's
  development-default Python (3.14 throughout the other sessions in this refactor; that specific
  report does not print the interpreter version, so this is not a direct quote of it, only the
  best-supported inference), where the PEP 701 syntax is valid and the bug cannot appear.

Neither net covered "this file" and "the oldest supported interpreter" at the same time. A
`compileall` pass over `scripts/`, now part of CI (see above), catches this whole class of defect
in a second, without importing or running anything — it would have caught this one immediately,
and is exactly why it was added.

### ruff adoption, rung 1

[ESTABLISHED] ruff replaces `black`, `isort`, `pydocstyle`, `pylint`, and `pylint-pytest` in the
`dev`/`dev-cuda13`/`dev-rocm7` dependency groups — none of which anything in the repo actually
invoked: no pre-commit config, no CI lint job (until this pass), no Makefile target. `mypy` stays;
it is a type checker, not a linter, and ruff does not replace it.

[ESTABLISHED] Rung 1 is ruff's default ruleset, chosen on measurement rather than preference: the
default ruleset finds 130 issues repo-wide and 41 in `src/`, and its safe autofixes touch only
import ordering and one type-hint modernization. Golden `.npy` baselines byte-identical before and
after; collected test-ID diff empty.

[ESTABLISHED] Four findings were suppressed rather than fixed, each with the reason recorded
inline at the point of suppression, because otherwise they will look like oversights to a later
reader:
- The broad `except Exception` in `to_jax` is the middle tier of a deliberate three-tier fallback
  (DLPack, then a cpu/numpy fallback, then a generic one) — swallowing the exception and falling
  through to the next tier is the mechanism, not an omission.
- `plot_scores`'s `ax: "plt.Axes"` parameter is a quoted forward reference, added on purpose when
  matplotlib's import was deferred out of `import utrace` (see "Import structure: matplotlib and
  pandas deferred out of `import utrace`" below) — the annotation cannot name `plt` unquoted,
  since `plt` is not bound at module scope until the function body runs.
- `pytest-expecter`'s `expect(x) == y` performs its assertion as a side effect inside `__eq__`, so
  the bare comparison expression IS the intended API, not a mistake. **No longer live**: this
  suppressed a per-file ignore for `src/utrace/tests/test_utils.py`, the only file that used the
  idiom. Batch 1 (`.reports/2026-08-21_batch1_defect_fixes.md`) removed that file, its per-file
  ignore, and the `pytest-expecter` dependency itself as dead — `pyproject.toml`'s
  `[tool.ruff.lint.per-file-ignores]` now names only `__init__.py`. Left here as a record of a
  suppression that once existed, not a description of the current file.
- `__init__.py` re-exports get a per-file `F401` ignore — the standard convention for a package's
  public-API surface, not a suppression of a real finding.

[ESTABLISHED] Two findings were real and are fixed:
- `max_batch_size` was annotated `int` while defaulting to `None` — the same implicit-Optional
  defect mypy had reported as the single standing error on `uncertaintyQuantifier.py` since the
  first Phase 6 diagnostic. Now annotated `int | None`; `mypy` reports zero errors on the file.
- `Pytorch_wrapper.__init__` defaulted `classes` to `np.arange(4)`, a mutable default evaluated
  once at class-definition time and assigned to `self.classes_` without a copy — every instance
  constructed without an explicit `classes=` would have shared the same array object. The bug was
  latent, not live: every one of the nine construction sites across `src/`, `tests/`, and
  `scripts/` passes `classes=` explicitly, and nothing anywhere mutates `classes_` in place.

[INTENT] Two further rungs were measured and deliberately not taken:
- pydocstyle would triple the finding count to 364, with 188 in `src/`, and less than half of
  those are auto-fixable — the remainder is writing docstrings, not running a fixer. Best
  sequenced after any classifier/regressor split (see "Architecture / design direction" in MIGRATION.md):
  docstrings written now, against the current single-class shape, would be written twice.
- `ruff format` (replacing `black`) would change 159 lines in `uncertaintyQuantifier.py` alone.
  All-or-nothing per file, low semantic risk, large diff. Best done when the affected files are
  not about to be rewritten, since formatting first would multiply the noise in any later diff.

### The pytest dependency-group split

[ESTABLISHED] `pytest` lived only inside the three `dev`/`dev-cuda13`/`dev-rocm7` groups, all of
which bundle torch, so no environment could run the suite without installing it — and the CI
`core` job paid that download for tests that never import torch. A `test` group now holds
`pytest` and its plugins; the three `dev*` groups include it via PEP 735 `{include-group =
"test"}` rather than duplicating the package list, verified by actually installing from each
route, not only by resolving.

[ESTABLISHED] `pytest-cov` had to move with it, non-obviously. `pyproject.toml`'s `addopts` block
passes `--cov-report=...` and `--no-cov-on-fail` unconditionally, on every invocation, and
`--no-cov` — the canonical test command's own flag — is itself a `pytest-cov`-provided option.
Without the plugin, `pytest` fails at argument-parsing time, before collection starts, on every
invocation including the canonical `--no-cov` one. Leaving `pytest-cov` behind in `dev` would have
made the new `test` group unusable for its only purpose. `coveragespace`, which uploads coverage
reports rather than hooking into `pytest` itself, stays in `dev`.

[ESTABLISHED, re-measured 2026-09-02] A clean environment synced with the `test` group and the
`viz` extra — no `dev` group, no torch — runs `tests/core/` at 113 passed, 5 skipped: the one
test that asks for torch and does not find it (`pytest.importorskip("torch")`, by design), plus
two GPU-only tests in each of `test_calibrate_device_reconciliation.py` and
`test_get_uncertainty_device_reconciliation.py`, inert without GPU-backed jax hardware. The
figure was already stale before Batch 1: `test_calibrate_device_reconciliation.py` (2 skips)
predates it, added by `7f140ea` ("Reconcile argument devices in calibrate"), which landed after
this "113 passed, 1 skipped" line was written. Batch 1
(`.reports/2026-08-21_batch1_defect_fixes.md`) added the other file, the remaining 2 skips. The
passed count (113) is unchanged throughout — none of the four device-reconciliation tests run on
a backend without GPU-backed jax.

### scripts/ lint cleanup

[ESTABLISHED] ruff found 53 findings in `scripts/` at the start of this pass, mostly dead
instrumentation — summary statistics computed, assigned, and never printed, saved, or plotted.
Worth recording precisely, because a smaller figure (25) had circulated earlier: 25 was the count
for seven of the eight touched files — an earlier ruff-adoption pass had deliberately excluded
`scripts/ACDC_example.py` from an autofix run because its `I001`/`F811` findings were entangled in
one hunk, and that file's remaining 28 findings were never separately addressed until this pass.
All 53 are now resolved; `scripts/` is under the CI `lint` job (see above).

[ESTABLISHED] Two cases a blind autofix would have gotten wrong are the argument for
grep-verifying every deletion by hand rather than trusting a fixer:
- A deleted variable in `MNIST_test_convergence.py` was referenced by a commented-out alternative
  implementation directly below it. The comment was removed along with the variable, rather than
  being left pointing at a name that no longer exists.
- Removing `test_loader` in `data_size_analysis.py` orphaned `test_dataset`, one element of a
  `random_split` tuple whose other elements stay live. It was renamed with a leading underscore
  rather than deleted, keeping the `random_split` call — and therefore its RNG consumption —
  identical.

[ESTABLISHED] No test covers `scripts/`, so the suite passing proved nothing about this cleanup.
Verification used the step C equivalence procedure (see "Verification method for unvalidated
scripts" in CONTRIBUTING.md) on two of the eight touched scripts, including the one with the cascading deletion
above; both produced byte-identical `.npy` output before and after.

- [DONE] Packaging cleanup (post-Phase 6): see "Phase 6 step 3c — packaging cleanup [RESOLVED]" and its second pass, "Phase 6 step 3c, second pass — reshaping extras around usage [RESOLVED]", both below. Note that torch was already absent from `[project].dependencies` before either cleanup - it only appeared in the optional-dependency groups (first pass) / dependency groups (current). What used to keep torch mandatory was that the core imported `flatten_batch` at module level; Phase 6 step 5 removed that import, and `tests/core/test_import_properties.py::test_core_does_not_import_torch` now guards the property.
- **[SUPERSEDED]** monai is in the `dev` group and depends on `torch` unconditionally — recorded
  at the time as an unwanted side effect, since `torch` was NOT otherwise a `dev`-group member,
  so a plotting-only dev environment paid for `torch` (then pinned to `2.11.0`, per the dry-run
  that established this) purely as monai's transitive dependency. The second packaging pass (see
  "Phase 6 step 3c, second pass" below) made `torch` a direct, intentional `dev`-group member —
  every contributor environment needs it for `tests/integration/torch/` regardless of monai — so
  the "pays for torch it didn't ask for" framing no longer applies; monai no longer changes
  whether `torch` is installed, only that it's one more package alongside it. Only
  `scripts/ACDC_example.py` uses monai anywhere in the repo. Giving monai its own extra (so a
  contributor not running ACDC doesn't pay for it specifically) remains a live, smaller idea; not
  acted on. **[RESOLVED]** monai removed from the dev group entirely rather than given its own extra; it
lives only in the `examples` extra now. Verified in a clean venv: syncing dev plus viz gives
120 passing tests with monai absent from the installed set.
- [DONE] `_max_N` overflow guard (Phase 6 step 1): `_calibrate_impl` now checks capacity before writing, in both branches, raising `ValueError` instead of allowing an out-of-bounds write. The pre-fix diagnostic found the failure was not one mode but three distinct behaviours, empirically reproduced:
  1. Batched write whose start index (`_N`) is already `>= _max_N`: JAX's `.at[].set()` silently dropped the update; `_N` still incremented regardless; neither reading `conformity_scores_` afterward nor setting `alpha` raised anything — the only fully silent failure of the three, and the one a caller relying on repeated `batched=True` streaming calibration would hit first.
  2. Batched write straddling the boundary (starts in-bounds, overruns past `_max_N`): raised `ValueError` from JAX broadcasting (the in-bounds slice clips shorter than the incoming values).
  3. Non-batched write with `num_scores > _max_N` in a single call: same `ValueError` as (2).
  Both branches of `_calibrate_impl` now check `_N + num_scores <= _max_N` (batched) / `num_scores <= _max_N` (non-batched) before any state mutation, so all three cases now raise consistently instead of only two of three doing so. Cross-references the manual buffer-sizing item in BACKLOG.md — a design that scales sizing automatically would still need this guard as a backstop.

- [RESOLVED] The global batched branch of _calibrate_impl concatenated conformity scores into the buffer without re-sorting (.at[_N:_N+num].set with no np.sort), while the non-batched and per-class batched branches do sort. _masked_quantile_higher assumes an ascending-sorted buffer, so the tuning quantile (q_hat) became non-monotonic in alpha when calibrating global+batched, breaking the binary search for U (it failed to converge; U  collapsed to 0 or oscillated). Fix: sort the concatenation, matching the per-class branch.
  - The _masked_quantile_higher unit test did not catch this because it is fed an already-sorted array: the bug was in the integration (calibration violating the sort precondition), not in the function itself.
  - [RESOLVED] Coverage gap: no test exercises the global+batched path. `tests/core/test_deferred_sort_buffer.py::test_property_sorts_on_read` calibrates `classes=None` with `batched=True` across 3 separate calls and, after reading `conformity_scores_`, asserts the valid prefix is ascending-sorted; `test_no_sort_between_batches` covers the same global+batched shape (4 calls) and asserts no sort happens between batches. Together they cover the previously-uncovered global (`classes=None`) batched path (commit 4852c3b).

- [RESOLVED] Per-class calibration double-counted _N: the trailing _N update ran unconditionally and overwrote the correct `_N = total` set inside the per-class branch, adding the last class's num_scores on top (e.g. N=66 for a 60-sample calibration). Fix: move the _N update into the global branch only. Also switched per-class accounting to a per-class count (_class_N) and fixed _class_scores initialization (was np.empty(_max_N), garbage). classes=[full list] now matches classes=None (commit 1a2c8a).

- [RESOLVED] to_jax device mismatch on GPU backends. to_jax routed any object with __dlpack__ through jax.dlpack.from_dlpack; numpy arrays implement __dlpack__, so numpy label arrays landed on CPU (DLPack preserves host origin) while CUDA torch probability tensors landed on GPU. The jitted score (lac_cal) then received its two arguments on different devices and raised "Received incompatible devices for jitted computation". The numpy DLPack path also emitted a "buffer is not aligned, creating a copy" warning (neither zero-copy nor correct-device). Invisible on the CPU backend; only reproduces with a GPU JAX backend. Fix: check isinstance(np.ndarray) BEFORE the __dlpack__ branch, route numpy via jnp.asarray (lands on JAX default compute device); DLPack kept only for genuine framework tensors. Device contract documented in the to_jax docstring: preserve device for tensors, normalize host arrays to the default compute device, do not reconcile mismatches between two genuine tensors. Validated on CUDA (to_jax(numpy)->GPU, to_jax(cuda tensor)->GPU, MNIST_example --extra=cuda matches CPU results). NOTE: the test suite runs on CPU and does NOT exercise this path; it only guards against regression.

- [ESTABLISHED] `index-strategy = "unsafe-best-match"` no longer exists: it was removed as an
incidental part of the extras reshape (fe21f31), not as a verified decision. Every extra and
group resolves without it against the three explicit indexes, so its removal is confirmed
correct after the fact.

- [RESOLVED] Two quantile implementations: np.nanquantile(scores[:_N], method='higher') in the alpha setter vs _masked_quantile_higher (jax, assumes sorted input, used by the tuning path via _q_hat_from_alpha) in the tuning path. Unified toward `_masked_quantile_higher` in the alpha setter (commit 0f8512f). Prerequisite (a) numerical equivalence: verified by the committed test `tests/core/test_alpha_setter_quantile_equiv.py`, which pins exact float64 equality between the two paths across N in {1,2,5,10,100,600,1200,5000,50000} and 502 alpha values each, including tied buffers and the cap boundary. Prerequisite (b) sort precondition: the setter now reads `conformity_scores_` through the lazy-sort property (commit 4852c3b), so the sortedness precondition is satisfied by the defer-sort mechanism itself rather than a per-call assertion.
_masked_quantile_higher is called only from the tuning fori_loop, which is why it does not sort internally: re-sorting unchanging data on each of the ~30 iterations would be wasteful. The sort precondition is the price of that optimization; prefer a sortedness assertion in _calibrate_impl (cheap, once per alibration) over sorting inside the loop.

- [RESOLVED]/[DONE] Performance: _calibrate_impl sorts the full buffer on every batch (O(N log N) per batch). For large datasets (ACDC) it would be better to sort once when calibration is finalized, not per batch. Implemented as defer-sort (commit 4852c3b): new scores are written into the fixed buffer at `_conformity_scores_[_N:_N+num_scores]` without per-batch sorting; a single `jnp.sort` runs lazily on first read of `conformity_scores_`, gated by the `_sorted` flag. MEASURED on RTX 3070 (CPU-unmeasured): streaming calibration ~357x faster than the prior sort-per-batch design at ~2M-score ACDC scale (7.2s → 20ms).

- [RESOLVED, step C] Labels passed to what was then the *_from_proba API used to go through .numpy() in the canonical recipe and in all example scripts (e.g. flatten_batch(y).ravel().numpy().astype(int)), violating the "do not call .cpu().numpy() on values feeding the core" rule and, after the to_jax fix, forcing a host->device hop. Fixed across three commits (32a1309, 6a630e5, 0caad12): labels now stay backend tensors into calibrate / get_uncertainty / predict (renamed from calibrate_from_proba / get_uncertainty_from_proba / predict_from_proba by the rename batch), taking the DLPack zero-copy path. The stale claim in this entry — "the six call sites are now marked `# COMPAT`" — undercounted (nine, not six; see "Step C: the `# COMPAT` marker post-mortem" above) and is no longer true regardless: grepping `# COMPAT` in scripts/ now returns nothing. Downstream label indexing (coverage counts, masks, get_coverage) was verified against tensor labels by the commits' own equivalence runs, not by the test suite, which does not cover the scripts.

- [DONE] precompute_proba helper consolidated (since renamed to precompute_softmax by the rename batch; still duplicated between scripts/_common.py and scripts/setsize_analysis.py — see "Rename batch" above and the deduplication item in BACKLOG.md). scripts/_common.py provides precompute_softmax(loader, classifier) returning raw torch tensors (torch.cat of softmax output and labels, no conversion) so the softmax output takes the zero-copy DLPack path into to_jax. Adopted in the six MNIST-like scripts (MNIST_example, MNIST_class_conditional, MNIST_test_coverage, MNIST_test_convergence, convergence_analysis, data_size_analysis). ACDC_example (pixel-scale segmentation) and setsize_analysis (non-batched calibration use) intentionally keep their own logic — setsize_analysis defines its own precompute_softmax rather than importing the shared one. Faithful dedup: in the five scripts whose labels were numpy, a temporary compatibility line `flatten_batch(y).ravel().numpy().astype(int)` tagged `# COMPAT` preserves current behavior; convergence_analysis already consumed tensor labels and has no COMPAT line.
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
- [RESOLVED, step 3b] x64 precision's ordering concern — `src/utrace/__init__.py` imports `.uncertaintyQuantifier` BEFORE calling `jax.config.update("jax_enable_x64", True)`, which worked only because nothing at import time of `uncertaintyQuantifier.py` computes a float64 array before the flag is set — was preserved when the call became unconditional (the `if USE_JAX:` guard was removed, not the ordering). Now directly guarded by `tests/core/test_import_properties.py::test_x64_is_enabled` (consolidated
from its own former file, `test_x64_is_enabled.py` — see "Forwarding accessors" above).

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
| `MNIST_class_conditional_example.py` | 11, 12 | calibrate, get_uncertainty_jit, predict | Migrated (Phase 4) |
| `MNIST_example.py` | 9, 10 | calibrate, get_uncertainty_opt, predict | Migrated (Phase 4) |
| `ACDC_example.py` | 13–16, tables B1/B2 | calibrate, get_uncertainty, predict | Migrated (Phase 4; pending numerical validation against the paper) |
| `convergence_analysis.py` | 7(b) | fit, get_uncertainty | Migrated (Phase 4) |
| `data_size_analysis.py` | 7(a,c) | fit, get_uncertainty_opt | Migrated (Phase 4) |
| `setsize_analysis.py` | 4, 5 | fit, get_uncertainty, predict | Migrated (Phase 4) |
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


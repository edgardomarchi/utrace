# U-TraCE — Backlog

Open work only. Split out of MIGRATION.md on 2026-08-19 (see `.reports/2026-08-19_docs_restructure.md`). Entries resolved or superseded before the split were filtered out; where the resolution carried a finding worth keeping, it moved to FINDINGS.md instead.

## Backlog (does not block the phases)

- `get_uncertainty_grid_from_proba`: alpha search by grid, as a method separate from the binary search (kept to investigate differences). Pending.
- `tuning_stability(probs, y, n_splits)`: diagnostic for tuning-set size adequacy (runs the search on disjoint subsets and reports spread). This is the formalization of the "L random splits" scheme from the paper.
- Golden test with a trained model (current ones use an untrained model: reproducible but in a degenerate regime, unstable alphas).
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
- Buffer/padding design for high-volume regimes (segmentation): the fixed-size `_max_N` buffer must currently be sized per class by hand. Consider a design that scales without manual sizing (without reintroducing variable shapes / JAX recompilation).
- force_non_empty_sets is silently ignored in the new prediction path. The jit _predict_sets does not implement it, and predict (renamed from predict_from_proba by the rename batch) accepts the parameter but does not pass it through. The legacy _predict_sets (initial commit) honored it (y_sets[arange, y_pred] = True). This is behavior lost in the jit migration. Harmless for callers passing False, but a latent bug for any script relying on force_non_empty_sets=True.

### TODO: make device handling in to_jax() explicit (deferred)

`to_jax()` (utils/tensors.py) currently handles a device mismatch silently.
For a CUDA tensor with a CPU JAX backend, the DLPack path raises, the exception is caught and logged at `debug` level, and execution falls through to a `.cpu().numpy()` copy to host. The result is correct, but the host transfer is invisible: a user who believes they are running zero-copy on GPU is silently paying a copy on every call, with no visible signal.

This contradicts the intended design goal (use the user's backend device by default, with an option to specify the device explicitly).

When addressing device handling (separate task, own branch / design discussion):
- Make a device mismatch explicit rather than silent — a visible warning, an error, or a `device=` parameter controlling the behavior.
- Narrow the `except` around the DLPack call: it currently catches all exceptions and routes them to `logger.debug`, which also hides non-device failures (unsupported dtype, version mismatch, malformed array). Catch only the expected device/DLPack exceptions and let the rest propagate.

Out of scope for the current script-migration work. Recorded here so the context is not lost.

- [Phase 6] Zero-copy in tuning: `get_uncertainty`'s (renamed from `get_uncertainty_from_proba`) body does `np.asarray(to_jax(...))`, forcing a host copy and negating DLPack zero-copy on the tuning path; `calibrate` / `predict` (renamed from `calibrate_from_proba` / `predict_from_proba`) keep zero-copy. Make tuning consume the jnp array directly (see the adjacent bare `# TODO: ... espera numpy` comment — no symbol name to anchor to; that comment and the call sit at `uncertaintyQuantifier.py:435-436` as of HEAD `53b1e8d`, but re-verify by symbol/grep rather than trusting that number after further commits — it has now moved twice, from `417-418` as of `ebc5ddb`, to `355-356` as of `a0ea8f6`, to `435-436` as of `53b1e8d` (the B.5 state-extraction commit added ~150 lines earlier in the file), each time purely from unrelated line-count changes, never from this call site itself being touched. Re-confirmed still present as of this pass; perf impact is UNMEASURED (the RTX 3070 GPU benchmark in FINDINGS.md measured the calibration path, not the tuning/uncertainty path).

- Disconnected `transform` parameter in MNIST_example.py: main() receives a `transform`  argument but the noise injection (~:176) uses a hardcoded `AddGaussianNoise`, ignoring it — so the __main__ transform_str dispatch (AWGN/RandomPerspective/ElasticTransform) currently has no effect on the experiment; AWGN is always applied. Likely a remnant of the lambda->class migration done to support num_workers>0 (a lambda transform is not picklable   and breaks multi-worker DataLoaders). To resolve: decide whether to reconnect the transform  sweep (as other scripts do) or whether fixed-AWGN is intentional for this script. If  reconnecting, note the three transforms have different signatures (AddGaussianNoise(0., n), RandomPerspective(n, 1), ElasticTransform(n)), so the swept parameter must be mapped per signature — this is a behavior change, warranting its own commit and revalidation. Separate from the I/O refactor.

- to_jax DLPack unaligned-copy: even for genuine tensors the DLPack path can emit "buffer is not aligned ... Creating a copy", so zero-copy is not guaranteed. Decide whether to make such copies VISIBLE (warn/error) rather than silent. Connects to the existing to_jax device-handling backlog item. Perf/observability task.

- User-configurable target device for to_jax (like torch's device=): host arrays currently go to JAX's default compute device; a future API should let the user choose. The current fix is written so the default-device path is the single point a future device= would generalize.

- Noise-sweep scripts rebuild the dataset (and DataLoader) inside the iteration loop, partly to reshuffle the split per iteration and partly to change the noise level. Reconstructing the full dataset per iteration is wasteful — only the noise (and the split) need to change, not the 60000-sample base. Optimization: instantiate the base dataset (and loader) ONCE outside the loop, and inside the loop either mutate the transform's sigma (transform.std sigma — valid because AddGaussianNoise reads self.std in __call__, not __init__) or reassign it (dataset.transform = AddGaussianNoise(0., sigma)). IMPORTANT: the random_split must STAY inside the loop (with a varying generator) to preserve per-iteration reshuffling — only the dataset/loader construction moves out. Caveat: mutating transform.std from the main process only propagates with num_workers=0; with spawn/fork workers, each worker holds its own copy and the loader would need rebuilding (ties into the num_workers decision). Applies to several sweep scripts (MNIST_class_conditional, and others with a noise sweep). Behavior-adjacent — revalidate numbers after the change. Its own diagnostic + commit.

### GPU / scalability (example scripts)

- Forward-pass batch vs jit padding are SEPARATE knobs; do not tie them. The DataLoader batch size only chunks the model forward (no effect on results — the tune set is re-concatenated and passed whole to get_uncertainty). max_batch_size is the jit padding and must be >= the materialized tune set. On an 8GB GPU the OOMs were ALWAYS in the model forward (predict_proba), never in the utrace core/tuning. Scripts pin max_batch_size to a hardcoded constant (e.g. 12000) tied to the 0.2 tune split of 60k MNIST; prefer deriving it (ceil(tune_split * len(dataset)) + margin) instead of a magic number.
- Whether to deduplicate precompute_softmax, still independently defined in both scripts/_common.py and scripts/setsize_analysis.py (functionally identical since step C). The rename batch's caller-locals commit renamed both definitions independently and deliberately did NOT deduplicate them — that remains a separate, undecided decision.
- [DONE for now] DataLoader num_workers set to 0 in MNIST_class_conditional_example (the only script that used workers; ACDC already used 0, the rest default to 0). Reason: num_workers>0 forks, and forking after JAX has initialized its threads can deadlock (generic fork-with-multithreading hazard, not a JAX bug). Resolved at the root by disabling workers. Deferred alternatives if workers are wanted back (e.g. if data loading becomes the bottleneck rather than the GPU forward pass): (a) spawn start method — robust but pays interpreter startup cost, requires picklable transforms (already satisfied: lambda->AddGaussianNoise) and the __main__ guard (already present); (b) lazy JAX initialization so all worker forks happen before XLA threads start — fragile/non-deterministic, NOT recommended. Decide by measuring workers=4 vs 0 wall-time on GPU first (the per-class script is likely GPU-forward-bound, so workers may add little). A permanent user-facing note belongs in docs/ (future), since this affects anyone using torch DataLoaders alongside the package — MIGRATION.md is process log only.
- 8GB VRAM is a hard constraint, not a bug: the per-class script (10 CPs) runs but only just fits with small forward batches. Not something to "fix"; scripts should scale by config.
## Post-migration analysis (open items)

1. U-vs-alpha for the two Appendix-A scripts (`MNIST_test_coverage.py`, `MNIST_test_convergence.py`): both now use `U` (not the tuned alpha) on BOTH the prediction threshold (`cp.alpha = U`) and the BetaBinom null parameter (`a_p = U_mean`), per the `uq.alpha = U` decision in CONTRIBUTING.md ("Decisions to respect"). Whether the tuned alpha is preferable instead is an open question — revisit once, for both scripts together, not independently.
2. BetaBinom null fragility: in both Appendix-A scripts, `Nr` (= `Nv`) is taken from only the last loop iteration's test-set size, while per-iteration sizes vary by ~1 sample due to `random_split`'s remainder rounding. The null distribution's trial count is therefore a (very close) approximation, not exact, across all recorded iterations.
3. GPU validation for per-class calibration: the per-class calibration path (classes=[...]) has not been fully validated on a GPU backend end-to-end; only the global path (classes=None, MNIST_example) has a clean GPU run. MNIST_class_conditional ran on GPU only with ad-hoc batch tuning (a probe, not the final structure). This remains an open GPU-validation item.

## Research questions (candidates for a second publication — NOT implementation tasks)

These are open research questions the refactor surfaced, recorded here because this is where the
work happens, not because any of them is scheduled work. Nobody should pick one of these up as a
chore; each needs measurement or proof that has not been done, and some needs data or hardware
this project does not currently have.

A second publication is anticipated alongside the package's planned regression support (the
current public API — `calibrate`, `predict`, `get_uncertainty` — was already named task-agnostically
for this, per "Naming convention now in force" in FINDINGS.md), and the questions below are
candidates for it.

1. **Numerical quantisation as a contribution to the reported uncertainty.**
   [ESTABLISHED, from source] The conformity-score buffer is `jnp.float64`
   throughout (`_UQState.conformity_scores`, `uncertaintyQuantifier.py`). `lac`/`lac_cal`
   (`scores/jax_impl.py`) return `1 - smx`, so scores lie in `[0,1]`, which float32 would
   represent with room to spare. `_masked_quantile_higher` (`utils/utils_jax.py`) does not
   interpolate: it computes `ceil(q*(n-1))`, clips to `[0, n_valid-1]`, and indexes into the
   sorted buffer — so it returns an element of the array, not a blend of two, and the precision
   of the returned quantile is the precision of the scores themselves.
   The risk in float32 is therefore not rounding of the quantile value but TIES: two adjacent
   scores collapsing to the same float32 value, changing which index the ceil/clip selects.
   Coverage's marginal guarantee depends on the ORDERING of the scores, not their values, so the
   question is concrete and measurable: how many adjacent score pairs tie or invert under
   float32, how that propagates to `q_hat`, to empirical coverage, and to prediction-set size —
   compared against the Monte Carlo error of coverage itself, which scales as `1/sqrt(N)` and is
   tiny at the N this project reaches (see item 3). In a metrological setting this is a
   contribution to the uncertainty budget that has to be declared, not optional polish.
   Caveat on the memory framing: halving the buffer's dtype would halve its footprint, but the
   one full-scale ACDC OOM actually measured (`.reports/2026-08-21_stepE_device_coherence.md`,
   `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.03GiB`) was in the tuning-set
   padding/masking array inside `_get_uncertainty_jit_impl`, not in the conformity-score buffer
   itself — so "float32 buffer is the difference between the full ACDC dataset fitting and not"
   is plausible but not demonstrated by anything measured so far; the buffer's own memory
   footprint at full scale has not been isolated.

2. **The computational cost of pixel-scale uncertainty quantification.**
   [ESTABLISHED] This refactor measured jitting's effect across two CPU architectures and one
   GPU — not three CPU architectures as sometimes summarized; the documented machines are a
   Ryzen 7 5700G workstation, a Ryzen AI 7 PRO 350 laptop (both CPU backend), and an RTX 3070
   (GPU, one figure). The mechanism established ("Step D, marginal slice" in MIGRATION.md,
   measured on the Ryzen AI 7 PRO 350): jitting the buffer write amortises dispatch overhead, and
   the advantage shrinks as the batch grows and compute comes to dominate — **~13.5x at B=500**,
   **~2.3x at B=12000**, **~1.08x (near parity) at ACDC pixel scale (B=65000)**
   (`.reports/2026-08-18_stepD_jit_marginal_diagnostic.md`). Separately, the defer-sort design
   measured **~357x faster streaming calibration at ~2M-score ACDC scale (7.2s → 20ms)** on the
   RTX 3070 (FINDINGS.md, "defer-sort" entry).
   None of this characterises end-to-end per-pixel UQ overhead in a full segmentation pipeline at
   the scale a clinical user would run — that is the open question: for medical image
   segmentation, where one image is tens of thousands of pixels per class, what is the practical
   cost of adding UQ to a pipeline, and nobody appears to have characterised it yet.

3. **Streaming calibration under a bounded memory budget.**
   [ESTABLISHED] The fixed-size buffer with a masked quantile lets calibration run over tens of
   millions of scores without holding them all — the ACDC majority class alone reached
   **N≈67.5M at full scale** (`.reports/2026-08-21_stepE_device_coherence.md`). On the RTX 3070
   (8GB), the full ~150-patient dataset produced two distinct, separately-reproduced OOMs: the
   model forward pass at `BATCH_SIZE=200` (`torch.OutOfMemoryError`), and — independent of that —
   the tuning-set padding/masking array in `_get_uncertainty_jit_impl` at `BATCH_SIZE=20`
   (`jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate
   2.03GiB`). Both tracebacks are captured verbatim in that report.
   The open question is statistical, not engineering: how many scores does the quantile actually
   need for a given alpha and a given coverage tolerance, and at what point does a bounded buffer
   stop being sufficient? [UNVERIFIED] Whether the buffer's own `jnp.sort` (JAX arrays are
   immutable, so a sort necessarily allocates a fresh array rather than sorting in place) adds a
   separately significant amount on top of the measured tuning-padding OOM has not been isolated
   — only the tuning-padding contribution was directly measured.

4. **The trained model as a calibrated instrument.**
   [ESTABLISHED] The project is PTB-funded (Physikalisch-Technische Bundesanstalt, Germany's
   national metrology institute — see README.md), and FINDINGS.md's "Rename batch" section
   records the paper's argument that treating a model's scaled logits as an approximation to a
   probability distribution is a conceptual error — the reason the public API dropped `proba`
   from its method names. The implementation now exists, runs on GPU, and has been exercised on
   real ACDC medical-imaging data end-to-end (`.reports/2026-08-21_stepE_device_coherence.md`).
   One specific claim in the source conversation for this entry — that the distinction between
   metrological traceability of outputs and data lineage was contributed to a BIPM working group
   ("BIPM TG-IA") — could not be substantiated: it does not appear anywhere in MIGRATION.md,
   FINDINGS.md, CONTRIBUTING.md, README.md, or the git history searched for this pass. It is
   **not** asserted here and should not be treated as established until someone who can confirm
   it directly does so.
   What is open, grounded only in what's confirmed above: the gap between the traceability
   argument and a demonstration of it running on real data is smaller than it was, but what
   would still be needed to close it into a publishable claim — statistically, and not just as a
   working pipeline — remains unstated.


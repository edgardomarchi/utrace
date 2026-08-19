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
- Accidental public surface: `scores/__init__.py`'s `from .jax_impl import *` re-exports every public module-level name in `jax_impl.py` (no `__all__` there), so `jnp` and `jit` are reachable as `utrace.scores.jnp` / `utrace.scores.jit` without that being a decided API surface. Discovered when removing the unused `jax_print` import — see "Rename batch" > "Accidental public surface" in FINDINGS.md. Add an explicit `__all__` to `jax_impl.py` (or switch `scores/__init__.py` to named imports) to close it. Not done.
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

- [Phase 6] Zero-copy in tuning: `get_uncertainty`'s (renamed from `get_uncertainty_from_proba`) body does `np.asarray(to_jax(...))`, forcing a host copy and negating DLPack zero-copy on the tuning path; `calibrate` / `predict` (renamed from `calibrate_from_proba` / `predict_from_proba`) keep zero-copy. Make tuning consume the jnp array directly (see the adjacent bare `# TODO: ... espera numpy` comment — no symbol name to anchor to; that comment and the call sit at `uncertaintyQuantifier.py:435-436` as of HEAD `53b1e8d`, but re-verify by symbol/grep rather than trusting that number after further commits — it has now moved twice, from `417-418` as of `ebc5ddb`, to `355-356` as of `a0ea8f6`, to `435-436` as of `53b1e8d` (the B.5 state-extraction commit added ~150 lines earlier in the file), each time purely from unrelated line-count changes, never from this call site itself being touched. Re-confirmed still present as of this pass; perf impact is UNMEASURED (the RTX 3070 GPU benchmark above measured the calibration path, not the tuning/uncertainty path).

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


# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- `scripts/btorch_MNIST_test.py` gains `--json PATH` (writes the plotted numbers), `--iterations N`
  (repetitions of the noise sweep) and `--seed N` (a reproducible run). Without them its behaviour
  is unchanged.

### Fixed
- `scripts/btorch_MNIST_test.py` now moves the Bayesian model and the test batch to the selected
  device. Before, on a GPU machine only the training used the GPU and the Monte Carlo loop silently
  ran on the CPU.

### Documentation
- `notebooks/periodic_verification_demo.py` (jupytext) commissions one frozen `utrace` instrument on
  an MNIST classifier and verifies it periodically under contrast loss and AWGN: what `U`,
  `U_empirical` and `U_delta` say against the measured error, the drift tests and the label-free
  alarms, and what recalibrating hides.
- `notebooks/acdc_periodic_verification.py` (jupytext) does the same for a pre-trained cardiac MRI
  segmentation model on the ACDC data, with one frozen quantifier per class, under AWGN, blur and a
  bias field, and states the bound at patient level, the unit of independence.
- The two notebooks are not part of the installed package.

## [0.3.0] - 2026-10-17

Upgrading from 0.2.x: three changes can affect existing code. Labels outside `[0, K)`, and
floating-point labels that are not finite and integral, now raise `ValueError` instead of being
silently aliased or truncated; string and object labels raise `ValueError` instead of `TypeError`.
`get_uncertainty` without an evaluation set now emits a `TuningDataReuseWarning`, which fails under
`-W error`; pass `eval_softmax`/`eval_y` or filter the warning. `scipy>=1.13` is a new declared
dependency.

### Added
- `UncertaintyQuantifier.tune_alpha(softmax, y, max_iters=30)`: selects alpha on a tuning set and
  returns it, without measuring `U` there. Same search, status and `SearchStatusWarning` as
  `get_uncertainty`.
- `UncertaintyQuantifier.uncertainty_at(softmax, y, alpha)`: measures `U = 1 - EC * (1 - alpha)` at a
  given alpha on an evaluation set, and returns an `UncertaintyEvaluation` with `U`, `EC` and the
  counts `n_covered`, `n_valid`, `n_succ` (the inputs of the confidence bound below). The
  evaluation set can be any size; it is not limited by `max_batch_size`. `U` is `nan`, not `1.0`, if
  it holds no sample of the calibrated class group. Raises `ValueError` for an uncalibrated
  instance, a row-count or column-count mismatch, or an alpha outside `[1/(N+1), 1]`.
- `get_uncertainty(..., *, eval_softmax=None, eval_y=None)`: when both are given, alpha is selected on
  the tuning set and `U` is measured on the evaluation set, so the two estimates no longer share
  data. Giving only one raises `ValueError`. The return value is still `(U, alpha)`.
- `utrace.UncertaintyEvaluation` and `utrace.TuningDataReuseWarning`.
- `uncertainty_at(softmax, y, alpha, *, delta=None)`: with `delta` in `(0, 1)` it returns `U_delta`, a
  one-sided upper confidence bound at level `1 - delta` on `U* = 1 - e * p` (`p` the coverage,
  `e = E[1/|C| | covered]` at that alpha), which bounds the model's error probability; any other
  `delta` raises `ValueError`. It is
  `1 - EC_lower * coverage_lower` with a Clopper-Pearson bound on `p` and a lower bound on `e`,
  each at level `delta/2` (union bound). The bound on `e` is `EC` minus the smaller of a Hoeffding
  margin and an empirical Bernstein margin (Maurer and Pontil (2009), "Empirical Bernstein Bounds
  and Sample Variance Penalization", COLT, arXiv:0907.3740, Theorem 4), each one-sided at `delta/4`;
  the empirical Bernstein margin is much smaller when the sets are mostly singletons, and over the
  grid examined the minimum is at most about 9 % wider than a Hoeffding margin at `delta/2`. It needs independent, representative evaluation samples and
  the paper's uniform-selection assumption; the code checks neither. `U` stays the point estimate
  `1 - EC * (1 - alpha)`.
- `UncertaintyEvaluation` gains the fields `coverage`, `U_empirical` (`1 - EC * coverage`), `delta`,
  `coverage_lower`, `EC_lower`, `U_delta`, appended after the original five with defaults. `coverage`
  and `U_empirical` are always computed; the bound fields are `None` without `delta`.
- `scipy>=1.13` is now a declared dependency (jax already required it). It is imported only when
  `delta` is given; `import utrace` does not load it.

### Changed
- Labels outside `[0, K)` (`K` the number of softmax columns) now raise `ValueError` in `calibrate`,
  `get_uncertainty`/`tune_alpha` and `uncertainty_at`. They used to alias silently to the last class
  (`-1` wraps, anything `>= K` clamps), giving a wrong coverage and `EC`. `calibrate` also rejects
  `classes` outside `[0, K)`. There is no `ignore_index`.
- Labels with a floating dtype must now be finite and integral in `calibrate`,
  `get_uncertainty`/`tune_alpha` (tuning and evaluation labels) and `uncertainty_at`; `0.9`,
  `-0.5`, `nan` or `inf` raise `ValueError` naming the first offending value and its count. They
  used to be truncated toward zero by the integer cast (`0.9` became class 0, `-0.5` became 0 and
  passed the range check). Integral floats such as `2.0` give exactly the same results as `2`.
  Boolean labels are accepted as 0/1; complex, object and string labels raise `ValueError` (string
  and object arrays used to raise `TypeError` from the jax conversion).
- The tuning path now checks the softmax trailing shape against the one `calibrate` recorded, and a
  batched `calibrate` with a different trailing shape raises `ValueError`.
- `get_uncertainty` called without an evaluation set now emits a `TuningDataReuseWarning` on every
  call: `U` is then estimated on the same samples that selected alpha, which correlates the errors in
  alpha and in `U`. Returned values are unchanged. The warning is a `UserWarning`, independent of
  `SearchStatusWarning`; silence it with
  `warnings.filterwarnings("ignore", category=TuningDataReuseWarning)`, or avoid it by passing an
  evaluation set.

### Documentation
- `README.md` no longer calls `U` a guaranteed upper bound: `U` is a point estimate with no
  confidence level, and `U_delta` holds only under the stated independence and uniform-selection
  assumptions.
- `notebooks/mnist_method_evaluation.py` (jupytext) evaluates `tune_alpha`, `uncertainty_at` and
  `U_delta` on a trained MNIST model; a `notebooks` dependency group runs it. The notebook is not
  part of the installed package.

## [0.2.0] - 2026-09-24

Upgrading from 0.1.x: two changes can break existing code. `predict()` no longer accepts
`force_non_empty_sets` — remove the argument if you pass it; it already had no effect. Assigning
`alpha` below `1/(N+1)` now raises `ValueError` instead of silently clipping — assign `1/(N+1)`
explicitly instead.

### Fixed
- `get_uncertainty` no longer crashes when passed a device-resident (e.g. GPU) label array while
  the softmax argument is also device-resident. The label array is now routed through the same
  device-reconciliation path already used for the softmax argument.
- The alpha search in `get_uncertainty` no longer returns an invalid result when the base model's
  error rate is below what the calibration set can resolve, roughly `1/(N+1)` for `N` calibration
  scores. Previously the search could walk to `alpha = 2**-max_iters` (about 9.3e-10 at the default
  `max_iters=30`) and return a `U` that was not a bound. The search now operates on `[1/(N+1), 1]`
  and reports its outcome in the new `search_status_` attribute:
  - `CONVERGED`: the ordinary case.
  - `FLOOR_LIMITED`: the target is already met at the floor `1/(N+1)`. `U` is determined by the
    size of the calibration set, not by the model.
  - `INFEASIBLE`: no alpha in the domain meets the target. The trivial bound `U = 1.0` is
    returned, with `alpha = 1.0`. That alpha is the boundary of the search domain, not an
    operating point: passed to `predict()` it produces the smallest possible prediction sets, the
    opposite reading of `U = 1.0`.

  A `SearchStatusWarning` is emitted whenever the status is not `CONVERGED`.
- The alpha search now returns the smallest feasible alpha it visited, with `U` evaluated at that
  same point. It previously returned its last iterate, feasible or not. Because the search
  criterion changes in steps at the grid points `k/(N+1)`, the result depended on the parity of
  `max_iters`. The shift is one order statistic of the calibration scores: a difference of 0.097
  in `U` was observed with `N = 18`; at larger `N` it is correspondingly smaller.
- Assigning `alpha` below `1/(N+1)` now raises `ValueError`. Previously it clipped the quantile
  level and stored the out-of-range value, so `alpha` did not match the quantile in use. Code that
  relied on the clip should assign `1/(N+1)` explicitly; assigning exactly that value does not
  raise.

### Changed
- `utrace.utils.get_coverage` no longer prints to stdout when coverage exceeds 0.99; the message
  is emitted through the module logger at INFO level.

### Removed
- `predict()`'s `force_non_empty_sets` parameter. It was accepted and documented as functional but
  had no effect: the underlying jit-migrated prediction path never implemented it, so the value
  was discarded for every caller regardless of what was passed.
- Test scaffolding (`src/utrace/tests/`) that was being shipped inside the built wheel. It tested a
  function that no longer exists in `src/` and could not run under the project's current test
  configuration.

### Added
- `utrace.SearchStatus` and `utrace.SearchStatusWarning`.
- `UncertaintyQuantifier.search_status_`: the outcome of the most recent `get_uncertainty` call.
  It is `None` before any search has run, after `reset()`, and after a call whose tuning set
  contains no sample of the calibrated class group.

### Documentation
- `CONTRIBUTING.md` documents three departures from the published paper: the search criterion (an
  unconditional proxy for the paper's conditioned criterion, needed for bisection; `U` itself is
  unchanged), a quantile index that is conservative by one order statistic, and the search domain
  and outcomes above.
- A documentation correction pass brought several project documents back in line with the current
  state of the repository.

## [0.1.0] - 2026-08-21

Initial release.

[Unreleased]: https://github.com/edgardomarchi/utrace/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/edgardomarchi/utrace/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/edgardomarchi/utrace/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edgardomarchi/utrace/releases/tag/v0.1.0

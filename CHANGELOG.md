# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/edgardomarchi/utrace/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/edgardomarchi/utrace/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edgardomarchi/utrace/releases/tag/v0.1.0

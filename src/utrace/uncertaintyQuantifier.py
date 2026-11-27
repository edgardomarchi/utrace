# ruff: noqa: N999 - renaming this module (uncertaintyQuantifier.py -> snake_case) is a real
# refactor, not a trivial fix: 5 other files import it by module path (src/utrace/__init__.py,
# tests/core/test_calibrate_jit_marginal.py, tests/core/test_search_uncertainty.py,
# tests/integration/torch/test_golden_mnist.py, tests/core/test_alpha_setter_quantile_equiv.py),
# all of which would need updating in the same change. Deferred, not fixed here.
"""Conformal predictor wrapper.
"""

import logging
import warnings
from enum import Enum
from functools import partial
from typing import Literal, NamedTuple

import jax
import numpy as np
from jax import jit, lax
from jax import numpy as jnp

from .bounds import Bound, SampleStats, resolve_bound
from .budget import BudgetComponent, ComponentKind, UncertaintyBudget
from .scores.objects import Score, resolve_score
from .utils import _bucket_size, _masked_quantile_higher
from .utils.tensors import to_jax

logger = logging.getLogger(__name__)


class SearchStatus(str, Enum):
    """Outcome of the alpha search run by `get_uncertainty`.

    A `str` subclass so `uq.search_status_ == "converged"` works without importing
    this enum. See `_search_uncertainty` for how each is derived.

    The tuning criterion is the score's: mean prediction-set size `<= 1` for `'lac'`,
    mean overlap ratio over covered samples `>= target_ratio` for `'abs_error'`.

    Describes the alpha search on the tuning set only. When `get_uncertainty` is given
    an evaluation set (`eval_softmax`/`eval_y`), the `U` mentioned below is measured on
    that set at the alpha the search returned, not on the tuning set.
    """
    CONVERGED = "converged"
    """The search bracketed the tuning criterion at an alpha strictly
    above the domain floor `1/(N+1)`. `alpha`/`U` are the smallest feasible alpha
    visited and the conditioned mean evaluated at it."""
    FLOOR_LIMITED = "floor_limited"
    """The tuning criterion is already met at the domain floor
    `1/(N+1)` itself -- the calibration set is too small to resolve a tighter
    alpha. `alpha` is exactly the floor; `U` is evaluated there."""
    INFEASIBLE = "infeasible"
    """No alpha in `[1/(N+1), 1]` visited during the search met the tuning
    criterion. `alpha = 1.0` and `U = 1.0` exactly (the trivial bound: at
    alpha=1 the `(1 - alpha)` factor in the U formula is zero). The returned
    `alpha` is the search domain's boundary, not a usable operating point for
    `predict()` -- see `get_uncertainty`'s docstring for why reading it
    alongside `U = 1.0` is easy to misread."""


class SearchStatusWarning(UserWarning):
    """Raised by `get_uncertainty` when `search_status_` is not `CONVERGED`.

    A dedicated subclass (not a bare `UserWarning`) so callers can filter
    specifically on this condition, e.g.
    `warnings.filterwarnings("error", category=SearchStatusWarning)`.
    """


class TuningDataReuseWarning(UserWarning):
    """Raised by `get_uncertainty` when no evaluation set is given.

    Without `eval_softmax`/`eval_y`, `U` is estimated on the same samples that
    selected alpha, so the two estimates are correlated. Not a subclass of
    `SearchStatusWarning`: that category reports the outcome of the alpha search,
    this one reports where `U` was measured, and the two are filtered independently
    (e.g. `warnings.filterwarnings("ignore", category=TuningDataReuseWarning)`).
    """


class UncertaintyEvaluation(NamedTuple):
    """Result of `UncertaintyQuantifier.uncertainty_at`: `U`, the counts it is built from, and
    optionally a confidence bound on it.

    All fields refer to the evaluation set, at the alpha that was passed in.

    `pi` below is the per-sample efficiency of the score: `1/|C|` for `'lac'`,
    `|C ∩ B| / |C|` with the tolerance window `B = y_hat ± tol` for `'abs_error'`.

    `U = 1 - EC * (1 - alpha)` is a POINT ESTIMATE (the paper's eq. (6), with the nominal
    coverage `1 - alpha`). It carries no confidence level. `U_empirical = 1 - EC * coverage`
    is the same expression with the coverage measured on the evaluation set. `U_delta`
    (only when `uncertainty_at` was given `delta`) is a one-sided upper confidence bound,
    at level `1 - delta`, on the population value `U* = 1 - e * p`, where `p` is the
    coverage and `e = E[pi | covered]` at this alpha. By the paper's theorem `U*` bounds
    the model's error probability, so `U_delta` does too, at level `1 - delta`, UNDER TWO
    ASSUMPTIONS the code cannot check: (1) the evaluation samples are independent and
    representative of the data the model is used on (pixels of one image, for example, are
    not independent); (2) the paper's uniform-selection assumption.

    Fields
    ------
    U : np.float64
        Point estimate, see above. `nan` when the evaluation set has no sample of the
        calibrated group, or when `alpha` was `nan` (nothing was evaluated).
    EC : np.float64
        Mean of `pi` over valid, covered samples (0.0 if there are none; `nan`
        only when `alpha` was `nan`).
    n_covered : int
        Valid samples whose true label (or target) is in the prediction set (or interval).
    n_valid : int
        Samples belonging to the calibrated group (all samples if `classes` is None).
    n_succ : int
        Valid, covered samples (with a non-empty set, for `'lac'`): the denominator of `EC`.
    coverage : np.float64 or None
        `n_covered / n_valid`; `nan` if `n_valid == 0` or `alpha` was `nan`.
    U_empirical : np.float64 or None
        `1 - EC * coverage`; `nan` under the same conditions.
    delta : float or None
        The `delta` that was passed, as given (`None` if none).
    coverage_lower : np.float64 or None
        Clopper-Pearson lower bound on the coverage `p`, at level `delta/2`; `0.0` if
        `n_covered == 0`. `None` when `delta` is `None`.
    EC_lower : np.float64 or None
        Lower bound on `e` at level `delta/2`, clipped at 0. By default `EC` minus the smaller
        of a Hoeffding margin and an empirical Bernstein margin (Maurer and Pontil, 2009,
        Theorem 4), each one-sided at `delta/4` (`pi` lies in `[0, 1]`; see
        `uncertainty_at`); `0.0` if `n_succ == 0`; only the Hoeffding margin if `n_succ == 1`.
        `uncertainty_at(..., bound=...)` replaces that default. `None` when `delta` is `None`.
    U_delta : np.float64 or None
        `1 - EC_lower * coverage_lower`. The two `delta/2` levels are fixed (not
        data-dependent) and combine by a union bound, valid although both come from the
        same evaluation set. `None` when `delta` is `None`.

    When `alpha` was `nan` nothing is evaluated: the three counts are 0 and every float
    field (the bound fields too, if `delta` was given) is `nan`.
    """
    U: np.float64
    EC: np.float64
    n_covered: int
    n_valid: int
    n_succ: int
    coverage: np.float64 | None = None
    U_empirical: np.float64 | None = None
    delta: float | None = None
    coverage_lower: np.float64 | None = None
    EC_lower: np.float64 | None = None
    U_delta: np.float64 | None = None


def _check_delta(delta) -> float | None:
    """`None`, or `delta` as a float in the open interval (0, 1); `ValueError` otherwise."""
    if delta is None:
        return None
    try:
        d = float(delta)
    except (TypeError, ValueError):
        raise ValueError(f"delta must be None or a real number in (0, 1), got {delta!r}.") from None
    if not (0.0 < d < 1.0):
        raise ValueError(f"delta must be None or a real number in (0, 1), got {delta!r}.")
    return d


def _confidence_bound(
    EC, n_covered: int, n_valid: int, n_succ: int, sum_sq_dev, delta: float,
    bound: Bound | None = None, values=None,
):
    """`(coverage_lower, EC_lower, U_delta)` as float64 scalars, on the host.

    Clopper-Pearson lower bound on the coverage at level `delta/2`; lower bound on
    `e = E[pi | covered]` at level `delta/2`, built from two one-sided margins at level `delta/4`
    each, of which the smaller is taken; combined as `1 - EC_lower * coverage_lower` (union
    bound over the three events). `pi` is the score's per-sample efficiency: `1/|C|` for `LAC`,
    `|C ∩ B| / |C|` for `AbsResidual`; it lies in `[0, 1]`. With `n = n_succ` and `z = pi`:

    - Hoeffding: `m_H = sqrt(ln(4/delta) / (2 n))`.
    - Empirical Bernstein, Maurer & Pontil (2009), "Empirical Bernstein Bounds and Sample
      Variance Penalization", COLT, arXiv:0907.3740, Theorem 4 (one-sided, `n >= 2`) at failure
      level `delta/4`: `m_EB = sqrt(2 V_n ln(8/delta) / n) + 7 ln(8/delta) / (3 (n - 1))`, with
      `V_n = sum_sq_dev / (n - 1)` the unbiased sample variance of `z`; `sum_sq_dev` is the
      centred sum of squares `sum((z - mean(z))^2)` over the `n_succ` covered samples (clipped
      at 0 here).

    `EC_lower = max(0, EC - min(m_H, m_EB))`; `n_succ == 0` gives 0, `n_succ == 1` (no sample
    variance) uses `m_H` alone. Which margin is smaller depends on the data through `V_n`: both
    events are fixed in advance and hold with probability `>= 1 - delta/4`, so taking the
    smaller is a union bound, not a data-dependent choice of level. Requires `n_valid > 0`.
    scipy is imported here, not at module level, so `import utrace` and every `delta=None`
    call stay scipy-free.

    The above is the default, `bound=None` (`bounds.DEFAULT_BOUND`, `Tightest((Hoeffding(),
    EmpiricalBernstein()))` at `delta/2`, so each member at `delta/4`), and reproduces the
    formulas bit for bit. Any other `Bound` replaces the `EC` term: `EC_lower = max(0,
    bound.lower(SampleStats(n_succ, EC, sum_sq_dev, values), delta/2))`, with `0` for
    `n_succ == 0` and `-inf` (an undefined bound) also giving 0. `values` are the `n_succ` values
    of `pi` in evaluation order, required when `bound.needs_values`. The coverage term is
    Clopper-Pearson whatever the bound.
    """
    if n_covered == 0:
        coverage_lower = np.float64(0.0)
    else:
        from scipy.special import betaincinv
        coverage_lower = np.float64(
            betaincinv(float(n_covered), float(n_valid - n_covered + 1), delta / 2.0)
        )
    bound = resolve_bound(bound)
    if n_succ == 0:
        EC_lower = np.float64(0.0)
    else:
        if bound.needs_values and values is None:
            raise ValueError(f"bound {bound.describe()!r} needs the per-sample values; none were given.")
        stats = SampleStats(n=int(n_succ), mean=EC, sum_sq_dev=sum_sq_dev, values=values)
        EC_lower = np.float64(max(0.0, float(bound.lower(stats, delta / 2.0))))
    return coverage_lower, EC_lower, np.float64(1.0 - EC_lower * coverage_lower)


def _labels_to_jax(y, where: str, score: Score) -> jax.Array:
    """`to_jax(y)` with `score.check_label_dtype` applied, without casting.

    Host input (numpy, lists) is checked before conversion, because `to_jax` raises
    `TypeError` on a string or object array; framework tensors (torch, jax) are checked
    after it, on the converted array.
    """
    if isinstance(y, np.ndarray) or not hasattr(y, "__dlpack__"):
        y = np.asarray(y)
        score.check_label_dtype(y, where)
        return to_jax(y)
    y = to_jax(y)
    score.check_label_dtype(y, where)
    return y


def _resolve_tol_alias(tol, eps, where: str):
    """`tol`, after resolving its deprecated alias `eps` (removed in 0.5.0).

    `eps` given alone emits one `DeprecationWarning` (attributed to the caller of the public
    method that called this) and is returned as `tol`; both given is a `TypeError`.
    """
    if eps is None:
        return tol
    if tol is not None:
        raise TypeError(f"{where}: got both tol and eps; eps is the deprecated alias of tol, pass tol only.")
    warnings.warn(
        f"{where}: eps is deprecated and will be removed in utrace 0.5.0; use tol instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return eps

# Internal int32 status codes used inside the jitted search (jit cannot return a
# Python str/Enum). `_get_uncertainty_jit_impl` maps these back to `SearchStatus`
# after the jitted call returns. Order matches SearchStatus's declaration order
# and the priority order in which outcomes are selected (see
# `_search_uncertainty`): floor eligibility is checked first, then
# whether the loop ever visited a feasible point.
_STATUS_CONVERGED = 0
_STATUS_FLOOR_LIMITED = 1
_STATUS_INFEASIBLE = 2
_STATUS_CODE_TO_ENUM = {
    _STATUS_CONVERGED: SearchStatus.CONVERGED,
    _STATUS_FLOOR_LIMITED: SearchStatus.FLOOR_LIMITED,
    _STATUS_INFEASIBLE: SearchStatus.INFEASIBLE,
}


@partial(jit, static_argnames=["score"])
def _predict_sets(output: jnp.ndarray, q_hat: np.float64, score: Score) -> tuple[jnp.ndarray, jnp.ndarray]:
    """What `UncertaintyQuantifier.predict` returns at `q_hat`: `score.predict` of
    `score.region` -- `(argmax, boolean sets)` for `LAC`, `(lower, upper)` for `AbsResidual`.

    Parameters
    ----------
    output : jnp.ndarray
        Model output: softmax `(n, K)` for `LAC`, point predictions `(n,)` for `AbsResidual`.
    q_hat : jnp.float64
        Calibrated quantile of the conformity scores.
    score : Score
        The score object (static).
    """
    return score.predict(output, score.region(output, q_hat))

@jit
def _q_hat_from_alpha(cs_padded: jnp.ndarray,
                      n_cs: jnp.ndarray,
                      alpha: jnp.ndarray) -> jnp.ndarray:
    """cs_padded already comes with N valid entries an padded with inf.

    For alpha < 1/(n_cs+1) the level exceeds 1 and is clipped to 1.0, so q_hat
    saturates at the maximum calibration score -- the same q_hat the search
    would get by evaluating exactly at the domain floor 1/(n_cs+1). This clip
    is safe to keep even though the search is no longer supposed to CLAIM
    coverage for any alpha below that floor: `_search_uncertainty`
    never returns a sub-floor alpha (it clamps the floor as its own lower
    bound and reports `SearchStatus.FLOOR_LIMITED`/`INFEASIBLE` instead), so a
    sub-floor evaluation reaching this function is only ever an interior probe
    the bisection takes on its way to a valid answer, never the answer itself.
    """
    q_level = jnp.ceil((n_cs + 1) * (1.0 - alpha)) / n_cs
    q_level = jnp.minimum(q_level, 1.0)
    return _masked_quantile_higher(cs_padded, n_cs, q_level)


@partial(jit, static_argnames=["score"])
def _evaluate_alpha(
    alpha: jnp.ndarray,
    y: jnp.ndarray,
    smx: jnp.ndarray,
    valid_mask: jnp.ndarray,
    cs_padded: jnp.ndarray,
    n_cs: jnp.ndarray,
    score: Score,
    params=(),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates the search's metric and conditioned efficiency at a single alpha.

    Factored out of `_search_uncertainty`'s loop body so the exact same evaluation
    (q_hat -> `score.evaluate`) runs both once at the domain floor `1/(n_cs+1)` and,
    unchanged, on every bisection iterate -- the floor check and the loop must agree on
    what "feasible" means (`score.feasible`, the same rule in both places).

    Returns
    -------
    metric : jnp.ndarray
        What the search steers by. `LAC`: mean prediction-set size over `valid_mask`
        samples, feasible when `<= 1` -- the unconditional mean set size vs. 1, not the
        paper's conditioned criterion (Section 3.2); see CONTRIBUTING.md, "Departures from
        the published paper", for why. `AbsResidual`: mean of `pi` over valid covered
        samples, feasible when `>= target_ratio`.
    EC : jnp.ndarray
        Mean of `pi` over valid, covered samples (`1/|C|` for `LAC`) -- the efficiency term
        `U = 1 - EC * (1 - alpha)` is built from.
    """
    q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha)
    return score.evaluate(y, smx, q_hat, valid_mask, params)

@partial(jit, static_argnames=["score", "max_iters"])
def _search_uncertainty(
    y: jnp.ndarray,                  # (n,) labels or targets
    smx: jnp.ndarray,                # (n, K) softmax output, or (n,) point predictions
    valid_mask: jnp.ndarray,         # (n,) bool - True: sample from selected class(es)
    cs_padded: jnp.ndarray,          # (m,) calibration scores
    n_cs: jnp.ndarray,               # (1,) number of calibration scores
    max_iters: int,
    score: Score,
    params=(),                       # per-call parameters (pytree): () for LAC, TolParams for AbsResidual
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Binary search for the smallest alpha whose search metric (`score.evaluate`, over
    `valid_mask` samples) meets the target (`score.feasible`), on the domain `[1/(n_cs+1), 1]`.

    For `LAC` the metric is the mean prediction-set size and the target `<= 1`; for
    `AbsResidual` it is the mean overlap ratio `pi` over covered samples and the target
    `>= target_ratio`. In both, the feasible alphas form an upper interval of the domain (sets
    shrink, and the overlap ratio grows, as alpha grows), so one search serves both: the step
    goes up where `score.step_up(metric)` (the complement of `score.feasible` for every non-NaN
    metric) and down otherwise. The same exact rule decides the floor check, the step and the
    feasibility of every iterate, so no iterate can be recorded as feasible while the floor
    check or the step treats the same metric value as infeasible.

    Fixes two defects a 2026-09-22 diagnostic confirmed in the previous,
    domain-`[0,1]` version of this search:

    1. Below `1/(n_cs+1)` the quantile level `_q_hat_from_alpha` computes
       exceeds 1 and is clipped, so `q_hat` saturates at the maximum
       calibration score and stops responding to alpha -- the old search had
       no floor and would walk straight to `2**-max_iters` in that regime,
       returning a U that is not a valid bound (coverage there is actually
       `n_cs/(n_cs+1)`, not `1-alpha`; see the diagnostic, Hypothesis 1).
    2. The old search returned whatever the *last* loop iterate happened to
       be, feasible or not. Because the criterion is a step function (jumps
       at the quantile's discrete order-statistic grid), bisection can
       oscillate across the jump near convergence, and whether the final
       iterate lands on the feasible or infeasible side depends on
       `max_iters`' parity -- observed to shift U by up to ~0.1 on this
       repo's own golden configuration, in ordinary (non-saturated) cases,
       not only the sub-floor regime (diagnostic, Hypothesis 2).

    **The evaluated sequence of alphas the loop visits is bit-for-bit
    unchanged from the old search** (same `sign`/`alpha_proposed`/
    `out_of_bounds`/`will_freeze` update, still bisecting nominally over
    `[0,1]`, not an explicit bracket on `[1/(n_cs+1),1]`) -- sub-floor
    iterates are now harmless (see `_q_hat_from_alpha`'s docstring) because
    they can never be the value this function returns; only what gets
    *selected* as the final answer changes. This is deliberate: reformulating
    the loop as bisection directly on `[1/(n_cs+1),1]` would change every
    evaluated midpoint and move every golden value for no statistical gain.
    The loop starts from `score.init_metric`, a metric value for which the first step goes
    down (alpha 1.0 -> 0.5), and carries the metric of the last committed iterate, as before.

    Three mutually exclusive outcomes (see `SearchStatus`), decided in this
    priority order:

    1. `FLOOR_LIMITED` -- the predicate is already feasible at the floor
       `1/(n_cs+1)` itself. By monotonicity of `q_hat` in alpha
       (`_q_hat_from_alpha`'s level is non-increasing in alpha), the floor is
       then the smallest feasible alpha in the whole domain. Returns
       `alpha = floor`, `U` evaluated at that same alpha.
    2. `CONVERGED` -- otherwise: the loop visited at least one feasible
       alpha. Returns the *smallest* feasible alpha visited (tracked
       explicitly through the loop, not read off the final iterate) and `U`
       evaluated at that same alpha.
    3. `INFEASIBLE` -- the loop never visited a feasible alpha. Returns
       `alpha = 1.0`, `U = 1.0` exactly -- the value the `U` formula itself
       gives at the domain's upper boundary (`(1 - alpha)` is exactly zero
       there), not an ad hoc sentinel.

    Returns
    -------
    alpha : jnp.ndarray
    U : jnp.ndarray
    status : jnp.ndarray
        int32 code; see `_STATUS_CODE_TO_ENUM` for the mapping to
        `SearchStatus`, applied host-side by `_get_uncertainty_jit_impl`
        (jit cannot return a Python `str`/`Enum`).
    """
    floor = 1.0 / (n_cs.astype(jnp.float64) + 1.0)
    setsize_floor, EC_floor = _evaluate_alpha(
        floor, y, smx, valid_mask, cs_padded, n_cs, score, params
    )
    floor_feasible = score.feasible(setsize_floor, params)

    init_state = (
        jnp.asarray(1.0),        # alpha: last committed iterate (drives `sign`, unchanged role)
        jnp.asarray(1.0),        # delta
        jnp.asarray(score.init_metric(params)),  # setsize: last committed iterate's search metric
        jnp.asarray(0.0),        # EC_yt: last committed iterate's conditioned mean
        jnp.asarray(False),      # frozen: alpha is out from [0,1]
        jnp.asarray(jnp.inf),    # best_alpha: smallest feasible alpha visited so far
        jnp.asarray(0.0),        # best_EC: EC_yt evaluated at best_alpha
        jnp.asarray(False),      # any_feasible: whether any iterate has been feasible
    )

    def body(i, state):
        alpha, delta, setsize, EC_yt, frozen, best_alpha, best_EC, any_feasible = state

        # Update -- IDENTICAL to the pre-fix search; do not change (see docstring).
        delta_new      = delta / 2.0
        sign           = jnp.where(score.step_up(setsize, params), 1.0, -1.0)
        alpha_proposed = alpha + sign * delta_new

        # Freeze if out of bounds
        out_of_bounds = (alpha_proposed < 0.0) | (alpha_proposed > 1.0)
        will_freeze   = frozen | out_of_bounds

        # If frozen, we do not update
        alpha_next = jnp.where(will_freeze, alpha, alpha_proposed)
        delta_next = jnp.where(will_freeze, delta, delta_new)

        setsize_curr, EC_yt_curr = _evaluate_alpha(
            alpha_next, y, smx, valid_mask, cs_padded, n_cs, score, params
        )

        # If frozen:
        setsize_next = jnp.where(will_freeze, setsize, setsize_curr)
        EC_yt_next   = jnp.where(will_freeze, EC_yt, EC_yt_curr)

        # Track the smallest feasible alpha visited (and its EC), regardless
        # of freeze -- this is the "last feasible" fix for Defect 2. Every
        # iterate the loop visits is a candidate, not just the final one.
        feasible_here      = score.feasible(setsize_curr, params)
        should_update_best = feasible_here & (alpha_next < best_alpha)
        best_alpha_next     = jnp.where(should_update_best, alpha_next, best_alpha)
        best_EC_next        = jnp.where(should_update_best, EC_yt_curr, best_EC)
        any_feasible_next   = any_feasible | feasible_here

        return (alpha_next, delta_next, setsize_next, EC_yt_next, will_freeze,
                best_alpha_next, best_EC_next, any_feasible_next)

    (_, _, _, _, _, best_alpha_f, best_EC_f, any_feasible_f) = lax.fori_loop(
        0, max_iters, body, init_state
    )

    alpha_out = jnp.where(
        floor_feasible, floor, jnp.where(any_feasible_f, best_alpha_f, 1.0)
    )
    EC_out = jnp.where(
        floor_feasible, EC_floor, jnp.where(any_feasible_f, best_EC_f, 0.0)
    )
    U_out = 1.0 - EC_out * (1.0 - alpha_out)
    status = jnp.where(
        floor_feasible,
        jnp.int32(_STATUS_FLOOR_LIMITED),
        jnp.where(any_feasible_f, jnp.int32(_STATUS_CONVERGED), jnp.int32(_STATUS_INFEASIBLE)),
    )
    return alpha_out, U_out, status


@jit
def _calibrate_write_jit(buffer: jnp.ndarray, start, scores: jnp.ndarray) -> jnp.ndarray:
    """Writes precomputed conformity `scores` into `buffer` at offset `start` under
    jit, replacing the marginal (classes=None) write path's eager `.at[].set()`.

    Deliberately takes already-computed `scores` rather than `y`/`smx` and the
    score function: fusing the score computation into this same trace was tried
    first and reverted, because `self.cal_score_` (e.g. `lac_cal`) is itself a
    standalone `@jit` function whose OWN `_cache_size()` is asserted directly by
    tests/core/test_label_dtype_canonicalisation.py -- calling it from inside
    another jit trace inlines its body without incrementing that counter (JAX
    does not go through a nested function's normal dispatch/cache path when
    already tracing), which silently broke that test's assertion. Keeping
    `cal_score_` as a separate, top-level call preserves its existing cache
    semantics unchanged; this function then only replaces the write, which the
    diagnostic and execution measurements found to be the dominant separate-
    dispatch cost (an isolated eager `.at[].set()` cost ~40ms on first use,
    against ~30ms for `lac_cal` alone) -- so most of the win survives un-fused.

    `start` must be passed as a concrete Python int (or a scalar built from one)
    at every call site, never as a value read back from a device array inside a
    hot loop -- JAX's jit cache keys on abstract shape/dtype, not on the concrete
    value of a non-static argument, so distinct `start` values reuse the same
    compiled trace (verified: cache stays at 1 across 45 distinct offsets in a
    streaming sequence; see the diagnostic and execution reports). The write
    itself uses `lax.dynamic_update_slice_in_dim` rather than the `buffer.at[
    start:start+size].set(...)` slice syntax used elsewhere in this file: the
    slice form fails outright when `start` is traced (`IndexError: Slice entries
    must be static integers`), while `dynamic_update_slice_in_dim` traces
    correctly because the update size is static even though the offset is not.
    """
    return lax.dynamic_update_slice_in_dim(
        buffer, jnp.asarray(scores, dtype=jnp.float64), start, axis=0
    )


class _UQState(NamedTuple):
    """Mutable calibration state for UncertaintyQuantifier, threaded explicitly
    through the methods below instead of being mutated on `self` in place.

    A plain NamedTuple: JAX registers NamedTuple subclasses as PyTrees
    automatically, so this needs no flax dependency and no hand-written
    tree_flatten/tree_unflatten. Every field here is MUTABLE STATE (changes as
    calibration proceeds); configuration (`_max_N`, `label_dtype_`,
    `cal_score_`, `score_`, `classes`, `_classes_jax`, `_max_batch_size`)
    stays on the class and is deliberately excluded, which is also why no
    field of this NamedTuple needs to be static.
    """
    N: int
    conformity_scores: jnp.ndarray
    sorted: bool
    alpha: np.float64
    q_hat: np.float64


def _ensure_sorted(state: _UQState) -> _UQState:
    """Sorts the FULL conformity-score buffer if it is not already sorted.

    Pure: returns a new state rather than mutating one in place. Sorting the
    full (max_N,) buffer rather than just [:N] is deliberate -- the region
    beyond N is +inf-padded (see UncertaintyQuantifier.reset() and
    _calibrate_impl), and +inf entries sort to the tail regardless, so this
    is bit-identical to sorting the variable-length prefix while keeping the
    sort's input shape fixed at (max_N,) across every call, instead of one
    XLA compilation per distinct N.
    """
    if state.sorted:
        return state
    return state._replace(conformity_scores=jnp.sort(state.conformity_scores), sorted=True)


@partial(jit, static_argnames=["score"])
def _evaluate_at_alpha(
    alpha: jnp.ndarray,
    y: jnp.ndarray,
    smx: jnp.ndarray,
    valid_mask: jnp.ndarray,
    cs_padded: jnp.ndarray,
    n_cs: jnp.ndarray,
    score: Score,
    params=(),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """`U`, `EC`, the counts behind them, at one fixed alpha, on one dataset.

    `EC` comes from `_evaluate_alpha`, called unchanged. The per-sample coverage and `pi`
    are recomputed here (same `_q_hat_from_alpha` call, then `score.per_sample`) for the
    counts `_evaluate_alpha` does not return; both calls sit in one trace, so XLA is free
    to share the work. `U` is formed inside this jit on purpose: the same expression
    evaluated eagerly on the host differs from the jitted one by one ulp in a
    noticeable fraction of cases.

    Returns `(U, EC, n_covered, n_valid, n_succ, sum_sq_dev)`; `U` is `nan` when
    `n_valid == 0` (without that guard an empty group would silently give `U = 1.0`).
    `sum_sq_dev` is the centred sum of squares `sum((z - mean(z))^2)` of `z = pi` over the
    `n_succ` samples `EC` averages over (valid, covered and, for `LAC`, non-empty; the mean is
    formed here, in the same trace), for the empirical Bernstein bound; it is exactly 0 when
    all those `pi` are equal (singleton sets for `LAC`; always, for `AbsResidual`).
    It is internal and does not affect the other five outputs.
    """
    _, EC = _evaluate_alpha(alpha, y, smx, valid_mask, cs_padded, n_cs, score, params)
    q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha)
    is_covered, succ, z = score.per_sample(y, smx, q_hat, valid_mask, params)

    n_valid = valid_mask.sum()
    n_covered = (is_covered & valid_mask).sum()
    n_succ = succ.sum()
    mean_z = z.sum() / jnp.maximum(n_succ, 1)
    sum_sq_dev = jnp.where(succ, (z - mean_z) ** 2, 0.0).sum()
    U = jnp.where(n_valid > 0, 1.0 - EC * (1.0 - alpha), jnp.nan)
    return U, EC, n_covered, n_valid, n_succ, sum_sq_dev


@partial(jit, static_argnames=["score"])
def _evaluate_samples_at_alpha(
    alpha: jnp.ndarray,
    y: jnp.ndarray,
    smx: jnp.ndarray,
    valid_mask: jnp.ndarray,
    cs_padded: jnp.ndarray,
    n_cs: jnp.ndarray,
    score: Score,
    params=(),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The per-sample values behind `_evaluate_at_alpha`'s sums, at one fixed alpha.

    Returns `(covered, succ, z)`, each of shape `(n,)` (padded rows included): `covered` is
    `is_covered & valid_mask`, `succ` the samples `EC` averages over, `z` their `pi` (0
    elsewhere) -- `score.per_sample` at the same `q_hat`, unchanged. A separate function, not a
    static flag on `_evaluate_at_alpha`, so the default path's compiled trace, outputs and cache
    are untouched; `uncertainty_at` calls it only for a bound that needs the values
    (`Bound.needs_values`), in addition to `_evaluate_at_alpha`, which still gives every field of
    the result.
    """
    q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha)
    is_covered, succ, z = score.per_sample(y, smx, q_hat, valid_mask, params)
    return is_covered & valid_mask, succ, z


@partial(jit, static_argnames=["score"])
def _budget_sums_at_alpha(
    alpha: jnp.ndarray,
    y: jnp.ndarray,
    smx: jnp.ndarray,
    valid_mask: jnp.ndarray,
    cs_padded: jnp.ndarray,
    n_cs: jnp.ndarray,
    score: Score,
    params=(),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The in-trace sums behind `uncertainty_budget`'s structural terms, at one fixed alpha.

    Over the valid samples, with `B` the score's acceptance region (`score.accepted`) and `C` the set
    or interval at `q_hat`: `n_in_B = #{y in B}`, `n_in_B_not_C = #{y in B, y not in C}` and
    `sum_selection = sum 1{y in C} (1{y in B} - pi)`. `pi` is `score.per_sample`'s `z`, which on a
    valid covered sample is its `pi` for both scores (a covered `LAC` set is never empty). Nothing
    per sample leaves the trace.
    """
    q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha)
    is_covered, _, z = score.per_sample(y, smx, q_hat, valid_mask, params)
    acc = score.accepted(y, smx, params) & valid_mask
    cov = is_covered & valid_mask
    n_in_B = acc.sum()
    n_in_B_not_C = (acc & ~cov).sum()
    sum_selection = jnp.where(cov, acc.astype(jnp.float64) - z, 0.0).sum()
    return n_in_B, n_in_B_not_C, sum_selection
# The `method` text of the budget's score-dependent components, by `Score.name`.
_BUDGET_TEXT = {
    "lac": {"error": "top-1 error on the evaluation set",
            "accepted_not_covered": "empty set, top-1 answer correct",
            "selection": "multi-label sets: 1/|C| against P(top-1 correct) (uniform selection)"},
    "abs_error": {"error": "|y - y_hat| > tol on the evaluation set",
                  "accepted_not_covered": "|y - y_hat| <= tol but y outside the interval",
                  "selection": "covered: 1{|y - y_hat| <= tol} - pi (uniform selection)"},
    None: {"error": "y outside the acceptance region B",
           "accepted_not_covered": "y in B, not in C",
           "selection": "covered: 1{y in B} - pi (uniform selection)"},
}


class _EvalContext(NamedTuple):
    """What `uncertainty_budget` reuses from an evaluation: the padded arguments of the jitted
    evaluation functions (`alpha, y, output, valid_mask, cs_padded, n_cs, score, params`) and the
    number of unpadded rows `B`."""
    eval_args: tuple
    B: int


class UncertaintyQuantifier:
    """Wrapper for uncertainty quantification using U-TraCE.

    Everything task-specific is held by a score object (`self.score_`, see
    `utrace.scores.objects`): `'lac'`/`LAC()` for classification from a softmax output,
    `'abs_error'`/`AbsResidual()` for regression from point predictions.

    Parameters
    ----------
    classes : Union[list[int], np.ndarray, None], optional
        labels defining the conditioning group; instantiate one object per class/group; None → marginal calibration.
    score : str or Score, optional
        The score, by default 'lac'.
    """
    def __init__(self, N: int = 1000,
                 classes: list[int] | np.ndarray | None = None,
                 score: Literal['lac', 'abs_error'] | Score = 'lac',
                 max_batch_size: int | None = None):
        """Wrapper for uncertainty quantification using U-TraCE.

        Parameters
        ----------
        N : int, default=1000
            Maximum number of calibration scores to retain.
        classes : list[int] or array, optional
            labels defining the conditioning group; instantiate one object per class/group; None → marginal calibration.
            Only for scores that support it (`LAC`); `ValueError` otherwise.
        score : {'lac', 'abs_error'} or Score, default='lac'
            `'lac'` (classification, `LAC`) or `'abs_error'` (regression with the absolute
            residual, `AbsResidual`), or a `Score` instance. Available as `self.score_`.
        max_batch_size : int, optional
            Fixed padding size for input batches. See _get_uncertainty_jit_impl.
        """
        self.score_ = resolve_score(score)
        if classes is not None and not self.score_.uses_classes:
            raise ValueError(
                f"classes={classes!r} is not supported by score {self.score_.name!r}: "
                f"class-conditional calibration needs class labels. Omit classes."
            )
        self.classes = classes
        self._classes_jax = jnp.asarray(classes) if classes is not None else None
        self._max_batch_size = max_batch_size

        self.score_name = self.score_.name
        # A top-level call in calibrate, outside any trace (see _calibrate_write_jit), and an
        # instance attribute so it can be replaced for inspection.
        self.cal_score_ = self.score_.cal_score
        # Declared by the score, not hardcoded at the boundary: hardcoding an integer cast
        # in calibrate would silently truncate continuous regression targets.
        self.label_dtype_ = self.score_.label_dtype

        self._max_N = N
        self.reset()

    def reset(self):
        """Resets the scoores and alpha."""
        self._state = _UQState(
            N=0,
            conformity_scores=jnp.full(self._max_N, jnp.inf, dtype=jnp.float64),
            sorted=True,  # +inf buffer is trivially sorted; no lazy sort needed on empty read
            alpha=np.float64('nan'),
            q_hat=np.float64('nan'),
        )
        # Fitted attribute (sklearn-style trailing underscore), not part of
        # _UQState: it is never traced inside jit, only set host-side after a
        # `get_uncertainty` call returns. None means "no search has run since
        # the last reset()" -- distinct from any SearchStatus member.
        self.search_status_ = None
        # Shape of one input row as seen by `calibrate` (softmax.shape[1:]); lets
        # `uncertainty_at` reject an evaluation set with a different number of columns.
        self._item_shape = None

        logger.debug("UQ reset.")

    @property
    def alpha(self) -> np.float64:
        """The alpha value used for the conformal prediction stage."""
        return self._state.alpha

    @alpha.setter
    def alpha(self, alpha: np.float64):
        """Sets the alpha value and calculates the q_hat level based on the current conformity scores.

        Raises
        ------
        ValueError
            If the model has not been calibrated yet, or if `alpha` is below
            `1/(N+1)` for the current calibration set size `N`. Below that
            floor no quantile level in this calibration set can express the
            requested alpha (the level `ceil((N+1)(1-alpha))/N` would exceed
            1); the largest coverage guarantee this calibration set can back
            is `N/(N+1)`, obtained by setting `alpha = 1/(N+1)` explicitly (the
            same value `get_uncertainty` returns with `search_status_ ==
            SearchStatus.FLOOR_LIMITED`). This replaces a previous
            warn-and-silently-clip behaviour that stored the caller's
            out-of-range `alpha` in `self.alpha` while actually using `q_hat`
            for a different, unstated alpha (`1/(N+1)`) -- changed as part of
            the 2026-09-22 fix. Comparing `alpha` directly against `1/(N+1)` (a single
            division), rather than checking whether the computed quantile
            level exceeds 1, is deliberate: the level formula's `ceil`/
            multiply chain can round to a value fractionally above 1 exactly
            at `alpha == 1/(N+1)` due to floating-point error, which would
            wrongly raise on the exact floor value `get_uncertainty` itself
            returns in the `FLOOR_LIMITED` case.
        """
        if self._state.N == 0:
            raise ValueError("The model must be calibrated before setting alpha.")

        N = self._state.N
        floor = np.float64(1.0) / np.float64(N + 1)
        new_alpha = np.float64(alpha)
        if new_alpha < floor:
            raise ValueError(
                f"alpha={new_alpha!r} is below 1/(N+1)={floor!r} for this "
                f"calibration set (N={N}). No quantile level in a calibration "
                f"set of this size can resolve an alpha smaller than that "
                f"floor; the largest coverage guarantee it can back is "
                f"N/(N+1). Use a larger calibration set, or set "
                f"alpha={floor!r} explicitly."
            )

        q_level = np.divide(np.ceil((N + 1) * (1 - new_alpha)), N, dtype=np.float64)
        # Floating-point safety net only, not a behavioural clip: alpha >= floor
        # already guarantees q_level <= 1 mathematically (mod the same rounding
        # noise the check above avoids relying on); _masked_quantile_higher
        # would clip its index to the same effect regardless (see its docstring).
        q_level = np.minimum(q_level, np.float64(1.0))
        logger.debug("'q_level' set to %f for alpha %f and N %d", q_level, new_alpha, N)
        self._state = _ensure_sorted(self._state)
        logger.debug("Conformity scores: %s", self._state.conformity_scores[:N])
        new_q_hat = np.float64(
            _masked_quantile_higher(self._state.conformity_scores, jnp.int32(N), q_level)
        )
        self._state = self._state._replace(alpha=new_alpha, q_hat=new_q_hat)
        logger.debug("'q_hat' set to %f for alpha %f", new_q_hat, new_alpha)

    @property
    def conformity_scores_(self):
        """Calibration buffer: sorted ascending in [:_N], +inf padding in [_N:].

        Lazy-sorts the valid prefix on first read after a calibration write.
        The sort itself is the pure _ensure_sorted(); this property performs
        the (legal, outside-jit) self._state reassignment and returns the
        buffer. Single-threaded access assumed per instance.
        """
        self._state = _ensure_sorted(self._state)
        return self._state.conformity_scores

    def calibrate(self, softmax, y, batched: bool = False):
        """
        Calibrate the conformal predictor with precomputed model output.

        Parameters
        ----------
        softmax : array-like, shape (n_samples, n_classes) or (n_samples,)
            Model output: a softmax output for `'lac'`, point predictions for `'abs_error'`.
            Accepts any array that implements DLPack (jax, numpy, torch, tensorflow, ...).
            Zero-copy when possible.
        y : array-like, shape (n_samples,)
            `'lac'`: integer class labels; boolean labels are read as 0/1, floating labels
            are accepted only if every value is finite and integral (`2.0`).
            `'abs_error'`: finite integer or floating targets.
        batched : bool, default=False
            If True, append to existing calibration scores instead of replacing.

        Notes
        -----
        `preds` and `y` may arrive committed to different devices (e.g. a GPU-resident
        model output alongside host-resident labels); this method reconciles them by
        moving `y` to `softmax`'s device before scoring, rather than raising.

        Raises `ValueError` if `batched` and the trailing shape of `softmax` differs from the
        one already calibrated; then, after that shape check, if `y` fails the score's dtype
        check (`'lac'`: a floating dtype with a non-finite or non-integral value, which the
        integer cast would truncate toward zero, or a dtype that is not integer, boolean or
        floating; `'abs_error'`: a dtype that is not integer or floating, or a non-finite
        value); then, for `'abs_error'`, if `softmax` is not a 1-D array of point predictions
        with as many entries as `y` has (an `(n, 1)` column is rejected, not reshaped); then,
        for `'lac'`, if a label is outside `[0, K)` with `K` the number of
        softmax columns (labels outside it would otherwise alias silently to the last class),
        or if `classes` is not within `[0, K)`.
        """
        softmax = to_jax(softmax)
        item_shape = tuple(softmax.shape[1:])
        # Shape check first, label checks second: a column mismatch keeps its own message.
        if batched and self._item_shape is not None and item_shape != self._item_shape:
            raise ValueError(
                f"softmax has shape {item_shape} per row, but this instance was calibrated "
                f"with {self._item_shape}; a batched calibration must keep the same shape."
            )
        y_arr = _labels_to_jax(y, "calibrate", self.score_).astype(self.label_dtype_)
        self.score_.check_output(softmax, y_arr, "calibrate")
        K = item_shape[0] if len(item_shape) == 1 else None
        self.score_.check_label_range(y_arr, K, "calibrate")
        if self.classes is not None:
            self.score_.check_label_range(np.asarray(self.classes), K, "calibrate (classes)")
        # to_jax converts each argument independently and does NOT reconcile devices
        # (see its docstring) -- two genuine framework tensors sourced from different
        # devices (e.g. a CUDA-resident softmax output alongside host-resident labels,
        # the shape ACDC's per-batch loop produces once the model runs on GPU) reach
        # here still committed to their own devices, and the jitted score function
        # raises "Received incompatible devices for jitted computation" if left as-is.
        # Reconcile here, at the one place both arguments are about to feed the same
        # jit call: move y_arr to softmax's device rather than the reverse, since
        # softmax carries n_classes columns per sample and y_arr carries one -- moving
        # the labels is the smaller transfer whichever device softmax landed on
        # (including the common case where both are already on the same device,
        # where devices() equality makes this a no-op).
        if y_arr.devices() != softmax.devices():
            y_arr = jax.device_put(y_arr, next(iter(softmax.devices())))
        self._calibrate_impl(softmax, y_arr, batched=batched)
        # Recorded only once the calibration succeeded, so a failed call leaves it as it was.
        self._item_shape = item_shape

    def _calibrate_impl(self, smx: jnp.ndarray, y: jnp.ndarray, batched: bool = False):
        """Calibrates the conformal predictor with the given data.

        Parameters
        ----------
        smx : np.ndarray
            Softmax output or point predictions for calibration.
        y : np.ndarray
            Target labels for calibration.
        batched : bool, optional
            For batched calibration; appends new scores to the buffer. By default False
        """
        old_N = self._state.N 

        if self._classes_jax is None:
            # Marginal path: jitted buffer write (_calibrate_write_jit) in place
            # of the eager `.at[].set()` below. cal_score_ stays a standalone,
            # top-level call -- see _calibrate_write_jit's docstring for why it
            # is not fused into the same trace. Deliberately NOT shared with the
            # class-conditional branch below -- see the step-D diagnostic and
            # execution reports: sharing would make the class-conditional path
            # retrace on every distinct filtered batch size, which it pays
            # nothing for today.
            scores = self.cal_score_(y, smx)
            num_scores = len(scores)
            if batched:
                if old_N + num_scores > self._max_N:
                    raise ValueError(
                        f"Batched calibration buffer overflow: current _N={old_N} + "
                        f"num_scores={num_scores} exceeds _max_N={self._max_N}. "
                        f"N is set at construction time."
                    )
                # Append new scores at offset _N without sorting (lazy sort deferred to property getter).
                new_conformity_scores = _calibrate_write_jit(
                    self._state.conformity_scores, old_N, scores
                )
            else:
                if num_scores > self._max_N:
                    raise ValueError(
                        f"Non-batched calibration buffer overflow: num_scores={num_scores} "
                        f"exceeds _max_N={self._max_N} (current _N={old_N}). "
                        f"N is set at construction time."
                    )
                # Non-batched: reset buffer to +inf and write scores at offset 0 without sorting.
                fresh_buffer = jnp.full((self._max_N,), jnp.inf, dtype=jnp.float64)
                new_conformity_scores = _calibrate_write_jit(fresh_buffer, 0, scores)
        else:
            mask = jnp.isin(y, self._classes_jax)
            y = y[mask]
            smx = smx[mask]

            scores = self.cal_score_(y, smx)
            num_scores = len(scores)
            if batched:
                if old_N + num_scores > self._max_N:
                    raise ValueError(
                        f"Batched calibration buffer overflow: current _N={old_N} + "
                        f"num_scores={num_scores} exceeds _max_N={self._max_N}. "
                        f"N is set at construction time."
                    )
                # Append new scores at offset _N without sorting (lazy sort deferred to property getter).
                new_conformity_scores = self._state.conformity_scores.at[
                    old_N:old_N + num_scores
                ].set(jnp.asarray(scores, dtype=jnp.float64))
            else:
                if num_scores > self._max_N:
                    raise ValueError(
                        f"Non-batched calibration buffer overflow: num_scores={num_scores} "
                        f"exceeds _max_N={self._max_N} (current _N={old_N}). "
                        f"N is set at construction time."
                    )
                # Non-batched: reset buffer to +inf and write scores at offset 0 without sorting.
                new_conformity_scores = jnp.full(
                    (self._max_N,), jnp.inf, dtype=jnp.float64
                ).at[:num_scores].set(jnp.asarray(scores, dtype=jnp.float64))

        # valid prefix is unsorted after any write; getter will sort on next read
        new_sorted = False if num_scores > 0 else self._state.sorted
        new_N = old_N + num_scores if batched else num_scores

        logger.debug("Conformity scores shape: %s, used: %d", new_conformity_scores.shape, old_N)

        self._state = self._state._replace(
            N=new_N,
            conformity_scores=new_conformity_scores,
            sorted=new_sorted,
        )

        if self.classes is not None and new_N == 0:
            logger.warning("No calibration scores for the requested class group %s after calibration.", self.classes)

    def predict(self, softmax) -> tuple[np.ndarray, np.ndarray]:
        """Prediction sets or intervals at the current `self.alpha`, from precomputed output.

        Parameters
        ----------
        softmax : array-like, shape (n_samples, n_classes) or (n_samples,)
            Model output (softmax for `'lac'`, point predictions for `'abs_error'`).
            Accepts any DLPack-compatible array.

        Raises `ValueError` for `'abs_error'` if `softmax` is not a 1-D array.

        Returns
        -------
        `'lac'`:
            y_pred : np.ndarray, shape (n_samples,)
                Predicted class labels (argmax).
            y_sets : np.ndarray, shape (n_samples, n_classes)
                Boolean prediction sets.
        `'abs_error'`:
            lower, upper : np.ndarray, shape (n_samples,)
                The interval `[y_hat - q_hat, y_hat + q_hat]`.
        """
        softmax = to_jax(softmax)
        self.score_.check_output(softmax, None, "predict")
        first, second = _predict_sets(softmax, self._state.q_hat, self.score_)
        return np.array(first), np.array(second)

    def get_uncertainty(self, softmax, y, max_iters: int = 30, *,
                        eval_softmax=None, eval_y=None,
                        tol=None, target_ratio=None, eps=None,
                        ) -> tuple[np.float64, np.float64]:
        """Estimate model uncertainty over a tuning set via conformal prediction.

        Searches for the smallest alpha that meets the score's tuning criterion, using ALL
        provided samples as a single tuning set: for `'lac'`, mean prediction-set size
        `<= 1`; for `'abs_error'`, mean overlap ratio `pi = |C ∩ B| / |C|` over covered
        samples `>= target_ratio`, with the tolerance window `B = y_hat ± tol`. The estimate
        converges to the true error probability with the tuning set size (conformal
        guarantee); it does NOT require batching.

        **Where `U` is measured.** `U = 1 - EC * (1 - alpha)` needs `EC`, the mean of
        `pi` (`1/|C|` for `'lac'`) over covered samples, evaluated at the selected alpha. Without
        `eval_softmax`/`eval_y`, `EC` is measured on the tuning set itself, the same
        samples that selected alpha. Because the selected alpha is a function of
        that data, the errors in alpha and in `EC` are correlated and compound
        instead of averaging out, so this call emits a `TuningDataReuseWarning`.
        Pass an independent evaluation set to avoid it:

            U, alpha = uq.get_uncertainty(tune_probs, tune_y,
                                          eval_softmax=eval_probs, eval_y=eval_y)

        This is `tune_alpha(tune_probs, tune_y)` followed by
        `uncertainty_at(eval_probs, eval_y, alpha).U` (with the same `tol`); call those two
        directly to also get `EC` and the counts. The evaluation set must be disjoint from
        both the calibration and the tuning sets for the guarantee to hold; this method
        cannot check that.

        This method does not set self.alpha or self.q_hat -- the caller decides
        whether and how to apply the returned alpha. It is NOT fully free of
        side effects, though: it reads the conformity-score buffer through the
        same lazy-sort path as conformity_scores_, so if the buffer has not
        been read since the last calibration write, this call sorts it and
        updates self._state accordingly (self._state.sorted flips to True). That
        mutation is idempotent and value-preserving -- it does not change the
        buffer's contents or this method's return value, only the internal
        representation used to compute it -- but callers should not assume
        this method leaves self._state completely untouched. It also sets
        `self.search_status_` (see below) as a genuine, intended side effect.
        To use the returned alpha for subsequent predictions, set it explicitly:

            U, alpha = uq.get_uncertainty(tune_probs, tune_y)
            uq.alpha = alpha                  # explicit, caller's decision
            y_pred, y_sets = uq.predict(test_probs)

        The search runs on the domain `[1/(N+1), 1]`, where `N` is the number
        of calibration scores currently held, and returns one of three
        outcomes -- see `SearchStatus` for the full description of each:

        - `SearchStatus.CONVERGED`: the ordinary case. `alpha`/`U` are the
          smallest feasible alpha the search visited (tuning criterion met
          over the tuning set) and `U` evaluated at that same alpha.
        - `SearchStatus.FLOOR_LIMITED`: the tuning criterion is already met at
          the domain floor `alpha = 1/(N+1)` -- this calibration set is too
          small to resolve a smaller alpha; `U` is evaluated at the floor.
          Enlarge the calibration set to look for a tighter alpha.
        - `SearchStatus.INFEASIBLE`: no alpha in `[1/(N+1), 1]` reaches the
          target; `alpha = 1.0`, `U = 1.0` exactly (the trivial bound). Read
          `U` and `alpha` here in OPPOSITE directions: `U = 1.0` reports that
          no bound was actually obtained (the worst possible value), while
          the same `alpha = 1.0` passed to `predict()` produces the SMALLEST
          possible prediction sets (the quantile level is 0, so `q_hat` is
          the minimum calibration score) -- the opposite of what "no bound
          found" would suggest. This `alpha` is the boundary of the search
          domain, returned because the formula demands some value there, not
          a usable operating point for `predict()`; do not set
          `self.alpha = alpha` after an `INFEASIBLE` result and expect it to
          mean anything about this tuning set's actual error rate.

        The outcome of the most recent call is recorded on `self.search_status_`
        (a `SearchStatus`, or `None` if no search has run since construction
        or the last `reset()` -- also `None` after a call whose tuning set had
        no sample in `self.classes`, since that call returns `(nan, nan)`
        without the search running at all). When it is not `CONVERGED`, this method also
        emits a `SearchStatusWarning` (a `UserWarning` subclass, filterable
        independently of other warnings) describing why, via `warnings.warn`
        -- not the module logger, specifically so it is visible under default
        Python warning configuration; the earlier `alpha` setter's
        `logger.warning` on this same condition was how the underlying defect
        went unnoticed in practice.

        Parameters
        ----------
        softmax : array-like, shape (n_tuning, n_classes) or (n_tuning,)
            Model output for the tuning set (softmax for `'lac'`, point predictions for
            `'abs_error'`). Any DLPack-compatible array. All samples are used; the caller
            controls the tuning set size by choosing how many samples to pass.
        y : array-like, shape (n_tuning,)
            Labels or targets for the tuning set, under the same rule as in `calibrate`.
        max_iters : int, default=30
            Maximum iterations for the binary search over alpha.
        eval_softmax : array-like, optional
            Model output of an independent evaluation set, on which `U` is measured
            at the selected alpha. Keyword-only. Must be given together with `eval_y`
            (giving only one raises `ValueError`). Its size is independent of
            `max_batch_size`.
        eval_y : array-like, shape (n_eval,), optional
            Labels or targets of the evaluation set, under the same rule as `y`.
        tol : float, optional
            `'abs_error'` only, and required there: the half-width of the tolerance window
            `B = y_hat ± tol`, in the units of the target; finite and `> 0`. `ValueError`
            if given with `'lac'`.
        target_ratio : float, optional
            `'abs_error'` only: the search target for the mean overlap ratio, in `(0, 1]`;
            default 1.0 (`C ⊆ B`). `ValueError` if given with `'lac'`.
        eps : float, optional
            Deprecated alias of `tol` (`DeprecationWarning`; removed in 0.5.0). Giving both
            `tol` and `eps` raises `TypeError`.

        Returns
        -------
        U : float
            Estimated uncertainty. See `SearchStatus` for how it is derived
            in each of the three outcomes. With an evaluation set, `U` is measured
            on it at the returned alpha; it is `nan` if the evaluation set has no
            sample of `self.classes`, and `nan` (with alpha) if the tuning set has none.
        alpha : float
            The alpha found by the search. NOT applied to the object -- see
            above. `self.search_status_` records which of the three outcomes
            produced it.

        Notes
        -----
        Passing a tuning set in batches and averaging per-batch alphas is
        statistically incorrect (alpha is a nonlinear function of the data).
        `tuning_stability()`, which would assess tuning-set-size adequacy by
        running the search on disjoint subsets and reporting the spread, is
        listed in BACKLOG.md as future work -- it does not exist yet in this
        version.
        """
        tol = _resolve_tol_alias(tol, eps, "get_uncertainty")
        if (eval_softmax is None) != (eval_y is None):
            raise ValueError(
                "eval_softmax and eval_y must be given together (or both omitted); "
                f"got eval_softmax={'None' if eval_softmax is None else 'given'}, "
                f"eval_y={'None' if eval_y is None else 'given'}."
            )

        if eval_softmax is None:
            params = self.score_.resolve_params(
                tol=tol, target_ratio=target_ratio, where="get_uncertainty"
            )
            U, alpha = self._search_single_dataset(softmax, y, max_iters, params)
            warnings.warn(
                "get_uncertainty: no evaluation set was given, so U is being estimated "
                "on the same samples that selected alpha. The two estimates are "
                "correlated and their errors compound. Pass eval_softmax and eval_y "
                "(an independent evaluation set) to measure U on separate data, or "
                "filter TuningDataReuseWarning to silence this.",
                TuningDataReuseWarning,
                stacklevel=2,
            )
            return U, alpha

        alpha = self.tune_alpha(softmax, y, max_iters=max_iters, tol=tol, target_ratio=target_ratio)
        if np.isnan(alpha):
            return np.float64('nan'), np.float64('nan')
        return self.uncertainty_at(eval_softmax, eval_y, alpha, tol=tol).U, alpha

    def _search_single_dataset(self, softmax, y, max_iters, params):
        """Alpha search and `U` on the one tuning set: the pre-evaluation-set body of
        `get_uncertainty`, moved here unchanged so `tune_alpha` can share it without
        emitting `TuningDataReuseWarning`. `params` is the score's resolved per-call
        parameter pytree (`self.score_.resolve_params`)."""
        # y goes through to_jax first, exactly like softmax on the line above -- a raw
        # device-resident tensor (e.g. a CUDA torch tensor) fails np.asarray() directly
        # (TypeError: can't convert cuda:0 device type tensor to numpy), but to_jax's
        # DLPack conversion produces a jax array that np.asarray() *can* consume via the
        # array protocol, with an implicit device-to-host copy. This mirrors calibrate's
        # own to_jax(y) call; it does not change _get_uncertainty_jit_impl, which still
        # receives, and still rebuilds, plain host numpy arrays exactly as before.
        # _labels_to_jax is to_jax plus the score's label dtype check, which must precede
        # the cast (score.prepare_labels).
        where = "get_uncertainty/tune_alpha"
        softmax = np.asarray(to_jax(softmax))
        y_arr = self.score_.prepare_labels(
            np.asarray(_labels_to_jax(y, where, self.score_)), softmax, where
        )
        # Host-side validation, outside _get_uncertainty_jit_impl: columns first, then labels.
        self._check_item_shape(softmax)
        self.score_.check_label_range(y_arr, self._label_range(), where)
        return self._get_uncertainty_jit_impl(softmax, y_arr, max_iters=max_iters, params=params)

    def _check_item_shape(self, smx) -> None:
        """`ValueError` if the trailing shape of `smx` differs from the one `calibrate` saw.
        Skipped when it is not known (`_item_shape is None`)."""
        if self._item_shape is not None and tuple(smx.shape[1:]) != self._item_shape:
            raise ValueError(
                f"softmax has shape {tuple(smx.shape[1:])} per row, but this instance was "
                f"calibrated with {self._item_shape}."
            )

    def _label_range(self) -> int | None:
        """`K` for the score's label range check: `_item_shape[0]`, or `None` (check skipped)
        when the instance has no recorded shape or it is not of length 1. Scores without a
        label range (`AbsResidual`) ignore it."""
        if self._item_shape is not None and len(self._item_shape) == 1:
            return self._item_shape[0]
        return None

    def _get_uncertainty_jit_impl(self, smx, y, max_iters=30, params=()):
        """Builds a fixed-size, class-masked, zero-padded batch from variable-size
        input, then calls the jitted `_search_uncertainty` on it.

        Despite the name, this method itself is NOT `@jit`-decorated -- "jit" in
        the name describes what it feeds, not what it is. `_search_uncertainty`
        IS jitted and would recompile on every distinct input shape it saw
        directly (the root cause Phase 2 eliminated for the calibration path);
        padding `smx`/`y` to a shape-stable size here means `_search_uncertainty`
        compiles once per shape it actually sees: always the same shape if
        `max_batch_size` was set at construction, otherwise once per bucket
        produced by `_bucket_size` (nearest multiple of 256 at or above the batch
        size) -- fewer distinct shapes than raw batch sizes would produce, but
        not literally "once" unless `max_batch_size` is set. The per-call parameters
        (`params`: `tol` and the target for `AbsResidual`) are traced, so new values
        do not recompile either.

        Also reads (and, if not already sorted, sorts) the conformity-score
        buffer via `_ensure_sorted` -- the same lazy-sort mutation
        `conformity_scores_` and `get_uncertainty` perform; see
        `get_uncertainty`'s docstring for why this isn't fully side-effect-free.

        Also sets `self.search_status_` and, when it is not
        `SearchStatus.CONVERGED`, emits a `SearchStatusWarning` -- both
        host-side, after the jitted call returns (a jit trace cannot warn).
        The early `return nan, nan` below, for an empty class group, sets
        `self.search_status_ = None` rather than leaving whatever value a
        previous call left there: no search ran for THIS call, and a stale
        status from an earlier, unrelated call would misreport why (or
        whether) this one's `nan, nan` happened.

        Parameters
        ----------
        smx : array-like, shape (B, K) or (B,)
            Model output; B varies from call to call.
        y : array-like, shape (B,)
            Labels or targets, already in the score's canonical form.
        max_iters : int, default=30
            Maximum iterations for the binary search over alpha, forwarded to
            `_search_uncertainty`.
        params : pytree, default=()
            The score's resolved per-call parameters.
        """
        B = y.shape[0]

        if self.classes is not None:
            valid = np.isin(np.asarray(y), np.asarray(self.classes))
        else:
            valid = np.ones(B, dtype=bool)

        if not valid.any():
            self.search_status_ = None
            return np.float64('nan'), np.float64('nan')

        if self._max_batch_size is not None:
            target_size = self._max_batch_size
            if B > target_size:
                raise ValueError(
                    f"Batch size {B} exceeds max_batch_size={target_size}. "
                    f"Increase max_batch_size at construction, or pass smaller batches."
                )
        else:
            target_size = _bucket_size(B)
        y_dtype = np.dtype(self.label_dtype_)
        y_arr = np.asarray(y).astype(y_dtype)
        y_padded   = np.zeros(target_size, dtype=y_dtype)
        p_padded   = np.zeros((target_size,) + smx.shape[1:], dtype=np.float64)
        mask_padded = np.zeros(target_size, dtype=bool)

        y_padded[:B]    = y_arr
        p_padded[:B]    = np.asarray(smx)
        mask_padded[:B] = valid                       # solo válidos reales en True

        y_safe = np.where(mask_padded, y_padded, 0)

        y_j    = jnp.asarray(y_safe)
        p_j    = jnp.asarray(p_padded, dtype=jnp.float64)
        mask_j = jnp.asarray(mask_padded)

        self._state = _ensure_sorted(self._state)
        cs_padded = self._state.conformity_scores
        n_cs = jnp.int32(self._state.N)
        N = self._state.N

        alpha, U, status_code = _search_uncertainty(
            y_j, p_j, mask_j, cs_padded, n_cs, max_iters, self.score_, params
        )
        alpha = np.float64(alpha)
        U = np.float64(U)
        status = _STATUS_CODE_TO_ENUM[int(status_code)]
        self.search_status_ = status

        metric, op, target = self.score_.describe_target(params)
        if status is SearchStatus.FLOOR_LIMITED:
            floor = 1.0 / (N + 1)
            warnings.warn(
                f"get_uncertainty: the tuning target ({metric} {op} {target}) "
                f"is already met at the domain floor alpha=1/(N+1)={floor!r} for this "
                f"calibration set (N={N}). U={U!r} is determined by the calibration "
                f"set size, not by the model -- a larger calibration set may allow a "
                f"tighter alpha.",
                SearchStatusWarning,
                stacklevel=2,
            )
        elif status is SearchStatus.INFEASIBLE:
            warnings.warn(
                f"get_uncertainty: no alpha in [1/(N+1), 1] (N={N}) brought the {metric} "
                f"to {op} {target} on this tuning set. Returning the "
                f"trivial bound alpha=1.0, U=1.0.",
                SearchStatusWarning,
                stacklevel=2,
            )

        return U, alpha

    def uncertainty_budget(self, softmax, y, alpha, *, delta, tol=None, bound=None) -> UncertaintyBudget:
        """`uncertainty_at` with `delta`, and the exact split of its `U_delta` into the model's error
        and what the method adds to it.

        With `B` the score's acceptance region (`{argmax}` for `'lac'`, `y_hat ± tol` for
        `'abs_error'`), `C` the prediction set or interval at `alpha`, `pi` the per-sample efficiency
        (`1/|C|`, or `|C ∩ B| / |C|`), and means over the valid evaluation samples (those of
        `self.classes`, or all):

            U_delta = error + accepted_not_covered + selection + coverage + efficiency

        - `error = mean 1{y not in B}` (kind `model`): the model's error on this set, top-1 error for
          `'lac'`, `|y - y_hat| > tol` for `'abs_error'`;
        - `accepted_not_covered = mean 1{y in B, y not in C}` (`structural`): acceptable answers the
          set leaves out; for `'lac'` exactly the correct answers whose set is empty;
        - `selection = mean 1{y in C} (1{y in B} - pi)` (`structural`): for `'lac'` the multi-label
          term, `sum_{k>=2} P(covered, |C| = k) (P(top-1 correct | covered, |C| = k) - 1/k)`. It is
          NOT clipped: on a finite sample it can be slightly negative. Its expectation is `>= 0`
          under the paper's uniform-selection assumption (then `E[1{y in B} | y in C] >= E[pi | y in
          C]`), so a clearly negative value is evidence against that assumption;
        - `coverage = EC (coverage - coverage_lower)` (`statistical`, Clopper-Pearson at
          `delta/2`) and `efficiency = coverage_lower (EC - EC_lower)` (`statistical`, `bound` at
          `delta/2`).

        The first three sum to `U_empirical - error`, the last two to `U_delta - U_empirical`;
        exact in real arithmetic, equal to rounding in floating point. For `'lac'` they are the
        terms of the usual decomposition of `U_delta - err_eval` (empty set with a correct answer,
        multi-label, coverage, `EC`). `error_upper` is the direct Clopper-Pearson upper bound on the
        error probability at level `1 - delta`, a comparator, not a component.

        The structural sums are formed inside the jitted evaluation (no per-sample array leaves
        the device), plus the per-sample path of `uncertainty_at` when `bound` needs the values.
        Same assumptions as `U_delta`: independent, representative evaluation samples.

        Parameters
        ----------
        softmax, y, alpha, tol, bound
            As in `uncertainty_at`.
        delta : float
            Required: a real number in `(0, 1)`.

        Returns
        -------
        UncertaintyBudget
            `evaluation` is exactly `uncertainty_at(softmax, y, alpha, tol=tol, delta=delta,
            bound=bound)`. With `alpha` `nan`, or no valid evaluation sample, `components` is empty
            and `error`, `error_upper` are `nan`.

        Raises
        ------
        ValueError
            If `tol` does not suit the score, if `delta` is `None` or not a real number in `(0, 1)`,
            if `bound` is an unknown string (`TypeError` if not a string or `Bound`), and under every
            condition `uncertainty_at` raises on its data.
        NotImplementedError
            If the score does not implement `Score.accepted`.
        """
        where = "uncertainty_budget"
        params = self.score_.resolve_params(tol=tol, target_ratio=None, where=where)
        if delta is None:
            raise ValueError(f"{where}: delta is required, a real number in (0, 1), got None.")
        try:
            delta = _check_delta(delta)
        except ValueError:
            raise ValueError(f"{where}: delta must be a real number in (0, 1), got {delta!r}.") from None
        bound = resolve_bound(bound)
        evaluation, ctx = self._evaluate_impl(softmax, y, alpha, params, delta, bound, where)
        nan = float("nan")
        if ctx is None or evaluation.n_valid == 0:
            return UncertaintyBudget(evaluation=evaluation, components=(), error=nan, error_upper=nan)

        n_in_B, n_in_B_not_C, sum_selection = _budget_sums_at_alpha(*ctx.eval_args)
        n = evaluation.n_valid
        n_in_B, n_in_B_not_C = int(n_in_B), int(n_in_B_not_C)
        k_err = n - n_in_B
        error = k_err / n
        from scipy.special import betaincinv
        error_upper = float(betaincinv(k_err + 1.0, float(n - k_err), 1.0 - delta)) if k_err < n else 1.0
        EC, cov, cov_lo, EC_lo = (float(evaluation.EC), float(evaluation.coverage),
                                  float(evaluation.coverage_lower), float(evaluation.EC_lower))
        text = _BUDGET_TEXT.get(self.score_.name, _BUDGET_TEXT[None])
        components = (
            BudgetComponent("error", error, ComponentKind.MODEL, text["error"]),
            BudgetComponent("accepted_not_covered", n_in_B_not_C / n, ComponentKind.STRUCTURAL,
                            text["accepted_not_covered"]),
            BudgetComponent("selection", float(sum_selection) / n, ComponentKind.STRUCTURAL,
                            text["selection"]),
            BudgetComponent("coverage", EC * (cov - cov_lo), ComponentKind.STATISTICAL,
                            "Clopper-Pearson at delta/2"),
            BudgetComponent("efficiency", cov_lo * (EC - EC_lo), ComponentKind.STATISTICAL,
                            f"{bound.describe()} at delta/2"),
        )
        return UncertaintyBudget(evaluation=evaluation, components=components,
                                 error=error, error_upper=error_upper)

    def tune_alpha(self, softmax, y, max_iters: int = 30, *,
                   tol=None, target_ratio=None, eps=None) -> np.float64:
        """Select alpha on a tuning set, without measuring `U` on it.

        Runs the same search as `get_uncertainty` on the given data and returns only
        alpha. Use it with `uncertainty_at` to measure `U` on a separate evaluation
        set (`get_uncertainty(..., eval_softmax=..., eval_y=...)` is exactly this
        composition).

        Like `get_uncertainty`, it does not set `self.alpha` or `self.q_hat`, sorts the
        conformity-score buffer lazily, sets `self.search_status_` and emits a
        `SearchStatusWarning` when the search is not `CONVERGED`. It does not emit
        `TuningDataReuseWarning`: selecting alpha on a tuning set is its purpose.

        Raises `ValueError` first if `tol`/`target_ratio` do not suit the score (see
        `get_uncertainty`). Then, if `y` fails the score's dtype check (see `calibrate`);
        for `'abs_error'`, if the output is not a 1-D array of point predictions of the same
        shape `(n,)` as the targets; if the
        columns of `softmax` differ from those `calibrate` saw (when known); and, for
        `'lac'`, if a label is outside `[0, K)` (`K` the number of columns), the column check
        running first. `get_uncertainty` raises the same on its tuning set and, through
        `uncertainty_at`, on its evaluation set.

        Parameters
        ----------
        softmax : array-like, shape (n_tuning, n_classes) or (n_tuning,)
            Model output for the tuning set. Any DLPack-compatible array.
        y : array-like, shape (n_tuning,)
            Labels or targets for the tuning set, under the same rule as in `calibrate`.
        max_iters : int, default=30
            Maximum iterations for the binary search over alpha.
        tol, target_ratio, eps : optional, keyword-only
            As in `get_uncertainty`.

        Returns
        -------
        alpha : np.float64
            The selected alpha; `nan` if the tuning set has no sample of
            `self.classes` (in which case `search_status_` is `None`). See
            `SearchStatus` for what the value means in each outcome; in particular
            `INFEASIBLE` returns the domain boundary `1.0`, not an operating point.
        """
        tol = _resolve_tol_alias(tol, eps, "tune_alpha")
        params = self.score_.resolve_params(tol=tol, target_ratio=target_ratio, where="tune_alpha")
        _, alpha = self._search_single_dataset(softmax, y, max_iters, params)
        return alpha

    def uncertainty_at(self, softmax, y, alpha, *, tol=None, delta=None, bound=None,
                       eps=None) -> UncertaintyEvaluation:
        """Measure `U` at a given alpha on an evaluation set.

        Computes the point estimate `U = 1 - EC * (1 - alpha)` where `EC` is the mean of
        `pi` over the evaluation samples that are covered at `alpha`: `pi = 1/|C|` for
        `'lac'`, `pi = |C ∩ B| / |C|` with `B = y_hat ± tol` for `'abs_error'`. Pair it with
        `tune_alpha`, on a different dataset, so that `alpha` is not a function of the data
        it is measured on. Nothing here checks that the sets are disjoint.

        `coverage = n_covered / n_valid` and `U_empirical = 1 - EC * coverage` are always
        returned. With `delta`, it also returns `U_delta`: a one-sided upper confidence bound
        at level `1 - delta` on the population value `U* = 1 - e * p` (`p` the coverage and
        `e = E[pi | covered]` at this alpha), which by the paper's theorem bounds the
        model's error probability. It is `1 - EC_lower * coverage_lower`, with a
        Clopper-Pearson lower bound on `p` at level `delta/2` and a lower bound on `e`
        (clipped at 0) at level `delta/2`, combined by a union bound (valid although both use
        the same evaluation set). The bound on `e` is `EC - min(m_H, m_EB)`, with
        `m_H = sqrt(ln(4/delta) / (2 n))` (Hoeffding) and
        `m_EB = sqrt(2 V ln(8/delta) / n) + 7 ln(8/delta) / (3 (n - 1))` (empirical Bernstein,
        Maurer and Pontil (2009), "Empirical Bernstein Bounds and Sample Variance
        Penalization", COLT, arXiv:0907.3740, Theorem 4), where `n = n_succ` and `V` is the
        unbiased sample variance of `pi` over the covered samples; each margin is one-sided
        at level `delta/4`. With `n_succ == 1` only `m_H` is used. That is the default bound on `e`;
        `bound=` replaces it (see below), the coverage term staying Clopper-Pearson. The bound holds only if (1) the evaluation samples are
        independent and representative of the data the model is used on, and (2) the paper's
        uniform-selection assumption holds; the code can check neither. scipy is imported
        only when `delta` is given.

        Does not change `self.alpha`, `self.q_hat` or `self.search_status_`. Like
        `get_uncertainty` it may sort the conformity-score buffer lazily. Evaluation
        sets of any size are accepted: they are padded to the next multiple of 256
        (not to `max_batch_size`), so each distinct padded size compiles once; new `tol`
        values do not recompile.

        Parameters
        ----------
        softmax : array-like, shape (n_eval, n_classes) or (n_eval,)
            Model output for the evaluation set. Any DLPack-compatible array.
        y : array-like, shape (n_eval,)
            Labels or targets for the evaluation set, under the same rule as in `calibrate`.
            Samples of `self.classes` are selected from THESE labels, not from any
            tuning-set mask.
        alpha : float
            Must lie in `[1/(N+1), 1]` for the `N` calibration scores held. `nan`
            (what `tune_alpha` returns for an empty tuning group) is passed through:
            nothing is evaluated and `U` is `nan`.
        tol : float, optional
            `'abs_error'` only, and required there: the half-width of `B = y_hat ± tol`
            (finite, `> 0`). `ValueError` if given with `'lac'`.
        delta : float, optional
            `None` (default: no bound is computed, `scipy` is not imported) or a real
            number in the open interval `(0, 1)`: the bound's error level.
        bound : Bound or str, optional
            The lower confidence bound on `e` used for `EC_lower`, applied at level `delta/2`: a
            `utrace.bounds.Bound` instance, or `'hoeffding'` / `'empirical_bernstein'`. `None`
            (default) is `bounds.DEFAULT_BOUND`, `Tightest((Hoeffding(), EmpiricalBernstein()))`:
            the two margins above, each at `delta/4`, bit for bit. Requires `delta`. A bound whose
            `needs_values` is True receives the per-sample `pi` of the covered samples, in
            evaluation order.
        eps : float, optional
            Deprecated alias of `tol` (`DeprecationWarning`; removed in 0.5.0).

        Returns
        -------
        UncertaintyEvaluation
            `U`, `EC` and the counts `n_covered`, `n_valid`, `n_succ`; see its
            docstring. `U` is `nan` (never `1.0`) if the evaluation set has no sample
            of `self.classes`. At `alpha = 1.0` (the `INFEASIBLE` boundary) `U` is
            exactly `1.0`.

        Raises
        ------
        TypeError
            If both `tol` and `eps` are given.
        ValueError
            If `tol` does not suit the score (checked first), if `delta` is not `None` or a
            real number in `(0, 1)`, if `bound` is given without `delta` or is an unknown string
            (`TypeError` if it is neither a string nor a `Bound`), if `y` fails the score's dtype check (see `calibrate`;
            checked before the remaining conditions), for `'abs_error'` if predictions and
            targets are not both of shape `(n,)`, if the instance is not calibrated, the row
            counts of `softmax` and `y` differ, the columns of `softmax` differ from those
            seen by `calibrate` (checked before the labels), for `'lac'` if a label is
            outside `[0, K)` (`K` the number of columns), or if `alpha` is outside
            `[1/(N+1), 1]`.
        """
        tol = _resolve_tol_alias(tol, eps, "uncertainty_at")
        params = self.score_.resolve_params(tol=tol, target_ratio=None, where="uncertainty_at")
        delta = _check_delta(delta)
        if bound is not None and delta is None:
            raise ValueError("uncertainty_at: bound= needs delta; got delta=None.")
        bound = resolve_bound(bound)
        evaluation, _ = self._evaluate_impl(softmax, y, alpha, params, delta, bound, "uncertainty_at")
        return evaluation

    def _evaluate_impl(self, softmax, y, alpha, params, delta, bound, where: str):
        """The body of `uncertainty_at` after its argument checks, shared with `uncertainty_budget`.

        Returns `(evaluation, ctx)`: the `UncertaintyEvaluation`, and an `_EvalContext` holding the
        padded jit arguments and the unpadded row count (`None` when `alpha` is `nan`, where nothing is
        evaluated). `where` names the public method in the error messages.
        """
        if np.isnan(alpha):
            nan = np.float64('nan')
            return UncertaintyEvaluation(
                U=nan, EC=nan, n_covered=0, n_valid=0, n_succ=0,
                coverage=nan, U_empirical=nan, delta=delta,
                coverage_lower=None if delta is None else nan,
                EC_lower=None if delta is None else nan,
                U_delta=None if delta is None else nan,
            ), None

        # Same conversion chain as the tuning path (see _search_single_dataset): to_jax
        # on both, then np.asarray, so a device-resident tensor never reaches numpy
        # directly and no committed-device mismatch reaches the jit.
        smx = np.asarray(to_jax(softmax))
        y_arr = self.score_.prepare_labels(
            np.asarray(_labels_to_jax(y, where, self.score_)), smx, where
        )

        N = self._state.N
        if N == 0:
            raise ValueError(f"The model must be calibrated before calling {where}.")
        if smx.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"softmax has {smx.shape[0]} rows but y has {y_arr.shape[0]} labels."
            )
        self._check_item_shape(smx)
        self.score_.check_label_range(y_arr, self._label_range(), where)
        floor = np.float64(1.0) / np.float64(N + 1)
        alpha_f = np.float64(alpha)
        if not (floor <= alpha_f <= 1.0):
            raise ValueError(
                f"alpha={alpha_f!r} is outside [1/(N+1), 1] = [{floor!r}, 1.0] for this "
                f"calibration set (N={N})."
            )

        B = y_arr.shape[0]
        if self.classes is not None:
            valid = np.isin(y_arr, np.asarray(self.classes))
        else:
            valid = np.ones(B, dtype=bool)

        y_dtype = np.dtype(self.label_dtype_)
        target_size = _bucket_size(max(B, 1))
        y_padded = np.zeros(target_size, dtype=y_dtype)
        p_padded = np.zeros((target_size,) + smx.shape[1:], dtype=np.float64)
        mask_padded = np.zeros(target_size, dtype=bool)
        y_padded[:B] = y_arr.astype(y_dtype)
        p_padded[:B] = smx
        mask_padded[:B] = valid
        y_safe = np.where(mask_padded, y_padded, 0)

        self._state = _ensure_sorted(self._state)
        eval_args = (
            jnp.asarray(alpha_f, dtype=jnp.float64),
            jnp.asarray(y_safe),
            jnp.asarray(p_padded, dtype=jnp.float64),
            jnp.asarray(mask_padded),
            self._state.conformity_scores,
            jnp.int32(N),
            self.score_,
            params,
        )
        U, EC, n_covered, n_valid, n_succ, sum_sq_dev = _evaluate_at_alpha(*eval_args)
        U, EC = np.float64(U), np.float64(EC)
        n_covered, n_valid, n_succ = int(n_covered), int(n_valid), int(n_succ)
        ctx = _EvalContext(eval_args=eval_args, B=B)
        if n_valid == 0:
            nan = np.float64('nan')
            return UncertaintyEvaluation(
                U=U, EC=EC, n_covered=n_covered, n_valid=n_valid, n_succ=n_succ,
                coverage=nan, U_empirical=nan, delta=delta,
                coverage_lower=None if delta is None else nan,
                EC_lower=None if delta is None else nan,
                U_delta=None if delta is None else nan,
            ), ctx
        coverage = np.float64(n_covered / n_valid)
        U_empirical = np.float64(1.0 - EC * coverage)
        if delta is None:
            return UncertaintyEvaluation(
                U=U, EC=EC, n_covered=n_covered, n_valid=n_valid, n_succ=n_succ,
                coverage=coverage, U_empirical=U_empirical,
            ), ctx
        values = None
        if bound.needs_values and n_succ > 0:
            _, succ, z = _evaluate_samples_at_alpha(*eval_args)
            succ, z = np.asarray(succ)[:B], np.asarray(z)[:B]
            values = z[succ].astype(np.float64)          # pi of the covered samples, in evaluation order
        coverage_lower, EC_lower, U_delta = _confidence_bound(
            EC, n_covered, n_valid, n_succ, sum_sq_dev, delta, bound, values
        )
        return UncertaintyEvaluation(
            U=U, EC=EC, n_covered=n_covered, n_valid=n_valid, n_succ=n_succ,
            coverage=coverage, U_empirical=U_empirical, delta=delta,
            coverage_lower=coverage_lower, EC_lower=EC_lower, U_delta=U_delta,
        ), ctx

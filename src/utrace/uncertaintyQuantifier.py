# ruff: noqa: N999 - renaming this module (uncertaintyQuantifier.py -> snake_case) is a real
# refactor, not a trivial fix: 5 other files import it by module path (src/utrace/__init__.py,
# tests/core/test_calibrate_jit_marginal.py, tests/core/test_search_uncertainty.py,
# tests/integration/torch/test_golden_mnist.py, tests/core/test_alpha_setter_quantile_equiv.py),
# all of which would need updating in the same change. Deferred, not fixed here.
"""Conformal predictor wrapper.
"""

import logging
import warnings
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Literal, NamedTuple

import jax
import numpy as np
from jax import jit, lax
from jax import numpy as jnp

from .scores import lac, lac_cal, abs_error, abs_error_cal


from .utils import _bucket_size, _masked_quantile_higher
from .utils.tensors import to_jax

logger = logging.getLogger(__name__)


class SearchStatus(str, Enum):
    """Outcome of the alpha search run by `get_uncertainty`.

    A `str` subclass so `uq.search_status_ == "converged"` works without importing
    this enum. See `_search_uncertainty` for how each is derived.
    """
    CONVERGED = "converged"
    """The search bracketed the target (mean set size == 1) at an alpha strictly
    above the domain floor `1/(N+1)`. `alpha`/`U` are the smallest feasible alpha
    visited and the conditioned mean evaluated at it."""
    FLOOR_LIMITED = "floor_limited"
    """The target is already met (mean set size <= 1) at the domain floor
    `1/(N+1)` itself -- the calibration set is too small to resolve a tighter
    alpha. `alpha` is exactly the floor; `U` is evaluated there."""
    INFEASIBLE = "infeasible"
    """No alpha in `[1/(N+1), 1]` visited during the search brought the mean set
    size to <= 1. `alpha = 1.0` and `U = 1.0` exactly (the trivial bound: at
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


@partial(jit, static_argname = ["scores_fn", "task"])
def _predict_sets(smx: jnp.ndarray, q_hat: np.float64, score_fn: Callable,
                  task: str = "classification") -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Construct classification prediction sets or regression intervals 
    bounds.

    Parameters
    ----------
    smx : np.ndarray
        Softmax output for each class for classification.
        y_hat output for regression
    q_hat : jnp.float64
        Calibrated quantile level.
    score_fn : Callable
        The scoring function to use.
    task: str, optional
        The task type ("classification or "regression").
    
    Returns
    -------
    y_pred : jnp.ndarray
        The predicted class labels.
    y_sets : jnp.ndarray
        The sets of labels as a boolean array.
    """
    if task == "classification":
        y_pred = jnp.argmax(smx, axis=1)  # -1 for tensorflow
        scores = score_fn(smx)
        y_sets = scores <= q_hat
        return y_pred, y_sets
    else:
        # score_fn is the continuous region constructor
        # score_fn(y_hat, q_hat) -> (lower, upper)
        lower, upper = score_fn(smx, q_hat)
        return lower, upper

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


@jit
def _intersection_length(lower_cp: jnp.ndarray, upper_cp:jnp.ndarray,
                         lower_tol: jnp.ndarray, upper_tol: jnp.ndarray) -> jnp.ndarray:
    """Computes the geometric intersection between the Conformal
        prediction intervals and the tolerance bounds."""
    left = jnp.maximum(lower_cp, lower_tol)
    right = jnp.minimum(upper_cp, upper_tol)
    return jnp.maximum(0.0, right - left)

@partial(jit, static_argnames=["score_fn"])
def clf_risk_fn(y: jnp.ndarray, smx: jnp.ndarray, q_hat: jnp.ndarray,
                valid_mask: jnp.ndarray, 
                score_fn: Callable, 
                aux_data: tuple = ()) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Classification risk function trgeting 'lac'."""
    _, prediction_sets = _predict_sets(smx, q_hat, score_fn = score_fn)

    n_valid = jnp.maximum(valid_mask.sum(), 1)
    set_sizes = prediction_sets.sum(axis = 1)

    # 1. Steering Metric: Average set size across valid samples
    setsize_curr = jnp.where(valid_mask, set_sizes, 0.0).sum()/n_valid

    # 2. Efficiency Metric (E): Average 1/|S| over valid covered samples
    is_covered = prediction_sets[jnp.arange(y.shape[0]), y]
    mask_succ = is_covered & (set_sizes > 0) & valid_mask
    safe_sizes = jnp.where(mask_succ, set_sizes, 1.0)
    inv_succ = jnp.where(mask_succ, 1.0/safe_sizes, 0.0)
    n_succ = mask_succ.sum()

    EC_yt_curr = inv_succ.sum()/jnp.maximum(n_succ, 1)
    return setsize_curr, EC_yt_curr

@partial(jit, static_argnames=["score_fn"])
def reg_risk_fn(y: jnp.ndarray, y_hat: jnp.ndarray, q_hat: jnp.ndarray, 
                valid_mask: jnp.ndarray, score_fn: Callable,
                aux_data: tuple) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Regression Risk function targeting 'abs_error'.
    """
    lower_tol, upper_tol = y_hat - aux_data[0], y_hat + aux_data[1]

    lower_cp, upper_cp = _predict_sets(y_hat, q_hat, score_fn = score_fn,
                                         task = "regression")

    L = jnp.maximum(upper_cp - lower_cp, 1e-8)
    delta = _intersection_length(lower_cp, upper_cp, lower_tol, upper_tol)

    covered = (y >= lower_cp) & (y <= upper_cp)
    valid_covered = covered & valid_mask
    n_valid_covered = valid_covered.sum()
    bounded_values = jnp.minimum(1.0, delta/L)

    ratio = jnp.where(
        n_valid_covered > 0,
        (bounded_values*valid_covered).sum()/jnp.maximum(n_valid_covered,1),
        0.0)

    # For regression, steering metric and efficiency metric are both ratio
    return ratio, ratio 

@partial(jit, static_argnames=["score_fn"])
def _evaluate_alpha(
    alpha: jnp.ndarray,
    y: jnp.ndarray,
    smx: jnp.ndarray,
    valid_mask: jnp.ndarray,
    cs_padded: jnp.ndarray,
    n_cs: jnp.ndarray,
    score_fn: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates the search's feasibility predicate and conditioned efficiency
    at a single alpha.

    Factored out of `_search_uncertainty`'s loop body so the exact
    same evaluation (q_hat -> prediction sets -> mean set size / conditioned
    mean 1/|C|) runs both once at the domain floor `1/(n_cs+1)` and, unchanged,
    on every bisection iterate -- the floor check and the loop must agree on
    what "feasible" means.

    Returns
    -------
    setsize : jnp.ndarray
        Mean prediction-set size over `valid_mask` samples. Feasible means
        `setsize <= 1.0`; this is the criterion the search steers by
        (unconditional mean set size vs. 1, not the paper's conditioned
        criterion (Section 3.2) -- see CONTRIBUTING.md, "Departures from the
        published paper", for why).
    EC : jnp.ndarray
        Mean of `1/|C|` over valid, covered samples -- the efficiency term
        `U = 1 - EC * (1 - alpha)` is built from.
    """
    n_valid = jnp.maximum(valid_mask.sum(), 1)
    q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha)
    _, prediction_sets = _predict_sets(smx, q_hat, score_fn=score_fn)

    set_sizes = prediction_sets.sum(axis=1)
    setsize = jnp.where(valid_mask, set_sizes, 0.0).sum() / n_valid

    is_covered = prediction_sets[jnp.arange(y.shape[0]), y]
    mask_succ = is_covered & (set_sizes > 0) & valid_mask
    safe_sizes = jnp.where(mask_succ, set_sizes, 1)
    inv_succ = jnp.where(mask_succ, 1.0 / safe_sizes, 0.0)
    n_succ = mask_succ.sum()
    EC = inv_succ.sum() / jnp.maximum(n_succ, 1)

    return setsize, EC

@partial(jit, static_argnames = ["risk_fn", "score_fn", "max_iters"])
def _search_uncertainty( 
    y: jnp.ndarray,                 # (n,) Target Values or labels.
    smx: jnp.ndarray,               # (n, K) softmax output (clf) or y_hat (reg)
    valid_mask: jnp.ndarray,        # (n,) bool - True: sample from selected class(es) (clf) or valid samples (reg)
    cs_padded: jnp.ndarray,         # (m,) calibration scores
    n_cs: jnp.ndarray,              # (1,) number of calibration socres
    max_iters: int, 
    score_fn: Callable,
    target_ratio: float = 1.0,
    aux_data: tuple = (),           # () for clf, (lower_tol, upper_tol) for reg.
    risk_fn: Callable = clf_risk_fn,
    direction: float = 1.0          # +1.0 if metric decreases w/alpha (clf), -1.0 if it increases (reg)  
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Binary search for the alpha whose mean prediction-set size (over
    `valid_mask` samples) equals 1, on the domain `[1/(n_cs+1), 1]`.

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

    Three mutually exclusive outcomes (see `SearchStatus`), decided in this
    priority order:

    1. `FLOOR_LIMITED` -- the predicate is already feasible (mean set size
       `<= 1`) at the floor `1/(n_cs+1)` itself. By monotonicity of `q_hat`
       in alpha (`_q_hat_from_alpha`'s level is non-increasing in alpha, so
       mean set size is non-increasing in alpha too), the floor is then the
       smallest feasible alpha in the whole domain. Returns `alpha = floor`,
       `U` evaluated at that same alpha.
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

    """RAY
     init_alpha = jnp.asarray(0.5, dtype = jnp.float64)
        init_q_hat = _q_hat_from_alpha(cs_padded, n_cs, init_alpha)
        init_metric, init_E = risk_fn(y, preds, init_q_hat, valid_mask, score_fn, aux_data)
        init_state = (
    
            init_alpha,
            #jnp.asarray(0.5),   # alpha.
            jnp.asarray(0.5),   # delta.
            #jnp.asarray(0.0),   # metric: setsize for clf, ratio for reg.
            init_metric,
            #jnp.asarray(0.0),   # E_curr: eficiency metric EC_yt for clf, ratio for reg.
            init_E,
            jnp.asarray(False), # frozen: alpha is out from [0, 1]   
        )
    """

    floor = 1.0 / (n_cs.astype(jnp.float64) + 1.0)
    setsize_floor, EC_floor = _evaluate_alpha(
            floor, y, smx, valid_mask, cs_padded, n_cs, score_fn
    )
    floor_feasible = setsize_floor <= 1.0

    init_state = (
        jnp.asarray(1.0),       # alpha: last committed iterate (drives 'sign', unchange role)
        jnp.asarray(1.0),       # delta
        jnp.asarray(0.0),       # setsize: last commited iterate's mean set size
        jnp.asarray(0.0),       # Ec_yt: last committed iterate's conditioned mean
        jnp.asarray(False),     # frozen: alpha is out from [0,1]
        jnp.asarray(jnp.inf),   # best_alpha: smallest feasible alpha visited so far
        jnp.asarray(0.0),       # best_EC: EC_yt evaluated at best_alpha
        jnp.asarray(False),     # any_feasible: whether any iterate has been feasible.
    )

    def body(i, state):
        alpha, delta, setsize, EC_yt, frozen, best_alpha, best_EC, any_feasible = state

        # Update -- IDENTICAL to the pre-fix search; do not change (see docstring).
        delta_new = delta/2.0
        sign = jnp.where(setsize > 1.0, 1.0, -1.0)
        alpha_proposed = alpha + sign*delta_new

        # Freeze if out of bounds
        out_of_bounds = (alpha_proposed < 0.0) | (alpha_proposed > 1.0)
        will_freeze = frozen | out_of_bounds

        # If frozen, we do not update
        alpha_next = jnp.where(will_freeze, alpha, alpha_proposed)
        delta_next = jnp.where(will_freeze, delta, delta_new)

        setsize_curr, EC_yt_curr = _evaluate_alpha(
            alpha_next, y, smx, valid_mask, cs_padded, n_cs, score_fn
        )

        # If frozen:
        setsize_next = jnp.where(will_freeze, setsize, setsize_curr)
        EC_yt_next = jnp.where(will_freeze, EC_yt, EC_yt_curr)

        # Track the smallest feasible alpha visited (and its EC), regardless
        # of freeze -- this is the "last feasible" fix for Defect 2. Every
        # iterate the loop visits is a candidate, not just the final one.
        feasible_here = setsize_curr <= 1.0
        should_update_best = feasible_here & (alpha_next < best_alpha)
        best_alpha_next = jnp.where(should_update_best, alpha_next, best_alpha)
        best_EC_next = jnp.where(should_update_best, EC_yt_curr, best_EC)
        any_feasible_next = any_feasible | feasible_here

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

    U_out = 1.0 - EC_out*(1.0 - alpha_out)
    status = jnp.where(
        floor_feasible,
        jnp.int32(_STATUS_FLOOR_LIMITED),
        jnp.where(any_feasible_f, jnp.int32(_STATUS_CONVERGED), jnp.int32(_STATUS_INFEASIBLE)),
    )

    """RAY
    def body(i, state):
        alpha, delta, metric, EC_yt, frozen = state
    
        # Update:
        delta_new = delta/2.0
    
        # Higher alpha shrinks sets/intervals -> metric decreases
        sign = jnp.where(metric >= target_ratio - 1e-6, direction, -direction)
        alpha_proposed = alpha + sign*delta_new
        
        # Freeze if out of bounds
        out_of_bounds = (alpha_proposed < 0.0) | (alpha_proposed > 1.0)
        will_freeze = frozen | out_of_bounds
            
        # If frozen, we do not update
        alpha_next = jnp.where(will_freeze, alpha, alpha_proposed)
        delta_next = jnp.where(will_freeze, delta, delta_new)
            
       
        # Quantile estimation
        q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha_next)
            
        metric_curr, EC_yt_curr = risk_fn(y, preds, q_hat, valid_mask, score_fn, aux_data)
            
        metric_next = jnp.where(will_freeze, metric, metric_curr)
        EC_yt_next = jnp.where(will_freeze, EC_yt, EC_yt_curr)
        return (alpha_next, delta_next, metric_next, EC_yt_next, will_freeze)
            
    alpha_f, _, _, E_f, _ = lax.fori_loop(0, max_iters, body, init_state)
    U = 1.0 - E_f*(1-alpha_f)
    return alpha_f, U

    """
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


class UncertaintyQuantifier:
    """Wrapper for uncertainty quantification using U-TraCE.

    Parameters
    ----------
    classes : Union[list[int], np.ndarray, None], optional
        labels defining the conditioning group; instantiate one object per class/group; None → marginal calibration.
    score : Literal['lac'], optional
        The scoring function to use, by default 'lac'
    """
    def __init__(self, N: int = 1000,
                 classes: list[int] | np.ndarray | None = None,
                 score: Literal['lac', 'abs_error'] = 'lac',
                 max_batch_size: int | None = None):
        """Wrapper for uncertainty quantification using U-TraCE.

        Parameters
        ----------
        N : int, default=1000
            Maximum number of calibration scores to retain.
        classes : list[int] or array, optional
            labels defining the conditioning group; instantiate one object per class/group; None → marginal calibration.
        score : {'lac', 'abs_error'}, default='lac'
            Scoring function for nonconformity. For now, only 'lac' for clf and 'abs_error' for reg is supported.
        max_batch_size : int, optional
            Fixed padding size for input batches. See _get_uncertainty_jit_impl.
        """
        self.classes = classes
        self._classes_jax = jnp.asarray(classes) if classes is not None else None
        self._max_batch_size = max_batch_size

        self.score_name = score

        
        match score:
            case 'lac':
                self.cal_score_ = lac_cal
                self.score_ = lac
                # Declared by the score family, not hardcoded at the boundary: a future
                # regression score would declare a float dtype here, and hardcoding an
                # integer cast in calibrate would silently truncate continuous
                # targets.
                self.label_dtype_ = jnp.int32
                self._task = "classification"
            case 'aps':
                raise ValueError(
                    "score='aps' is not implemented in the JAX backend. "
                    "The only implementation of APS lived in the numpy "
                    "backend, which is unreachable in the current "
                    "configuration. This is a known gap, not a typo. "
                    "'lac' is the supported value."
                )
            case 'abs_error':
                self.cal_score_ = abs_error_cal
                self.score_ = abs_error
                self.label_dtype_ = jnp.float64
                self._task = "regression"

            case _:
                raise ValueError(
                    f"Unknown score {score!r}. The supported value is 'lac' (clf) and 'abs_error' (reg)."
                )

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
        Calibrate the conformal predictor with precomputed softmax output.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target labels or values.

        softmax : array-like, shape (n_samples, n_classes) or (n_samples,)
            Predicted outputs (softmax output for classification, or point
            estimation for regression). Accepts any array type that implements
            DLPack (jax, numpy, torch, tensorflow, ...). Zero-copy when possible.
        batched : bool, default=False
            If True, append to existing calibration scores instead of replacing.

        Notes
        -----
        `preds` and `y` may arrive committed to different devices (e.g. a GPU-resident
        model output alongside host-resident labels); this method reconciles them by
        moving `y` to `softmax`'s device before scoring, rather than raising.
        """
        y_arr = to_jax(y).astype(self.label_dtype_)
        smx_arr = to_jax(softmax)

        if y_arr.devices() != smx_arr.devices():
            y_arr = jax.device_put(y_arr, next(iter(smx_arr.devices())))
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
        self._calibrate_impl(smx_arr, y_arr, batched=batched)

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
        """Predict class labels/values and prediction sets/intervals from precomputed output.

        Parameters
        ----------
        preds : array-like, shape (n_samples, n_classes)
            Predicted class/labels output. Accepts any DLPack-compatible array.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels.
        y_sets : np.ndarray, shape (n_samples, n_classes)
            Boolean prediction sets.
        """
        softmax = to_jax(softmax)
        y_pred, y_sets = _predict_sets(softmax, self._state.q_hat, score_fn = self.score_, 
                                       task = self._task)
        return np.array(y_pred), np.array(y_sets)

    def get_uncertainty(self, softmax, y, max_iters: int = 30, eps: float = 0.1,
                        target_ratio: float = 1.0) -> tuple[np.float64, np.float64]:
        """Estimate model uncertainty over a tuning set via conformal prediction.

        Searches for the alpha that yields the target average prediction-set size,
        using ALL provided samples as a single tuning set. The estimate converges
        to the true error probability with the tuning set size (conformal
        guarantee); it does NOT require batching.

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
          smallest feasible alpha the search visited (mean prediction-set
          size <= 1 over the tuning set) and `U` evaluated at that same
          alpha.
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
        softmax : array-like, shape (n_tuning, n_classes)
            Predicted softmax output for the tuning set. Any DLPack-compatible array.
            All samples are used; the caller controls the tuning set size by
            choosing how many samples to pass.
        y : array-like, shape (n_tuning,)
            Integer class labels for the tuning set.
        max_iters : int, default=30
            Maximum iterations for the binary search over alpha.

        Returns
        -------
        U : float
            Estimated uncertainty. See `SearchStatus` for how it is derived
            in each of the three outcomes.
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
    
        # TODO: _get_uncertainty_jit_impl espera numpy (lo convierte a jnp adentro)
        # y goes through to_jax first, exactly like softmax on the line above -- a raw
        # device-resident tensor (e.g. a CUDA torch tensor) fails np.asarray() directly
        # (TypeError: can't convert cuda:0 device type tensor to numpy), but to_jax's
        # DLPack conversion produces a jax array that np.asarray() *can* consume via the
        # array protocol, with an implicit device-to-host copy. This mirrors calibrate's
        # own to_jax(y) call; it does not change _get_uncertainty_jit_impl, which still
        # receives, and still rebuilds, plain host numpy arrays exactly as before.
        softmax = np.asarray(to_jax(softmax))
        y_arr = np.asarray(to_jax(y)).flatten()
        if self._task == "classification":
            y_arr = y_arr.astype(int)
        else:
            y_arr = y_arr.astype(float)
            if softmax.ndim > 1 and softmax.shape[1] == 1:
                softmax = softmax.flatten()

        return self._get_uncertainty_jit_impl(softmax, y_arr, max_iters=max_iters, eps = eps,
                                              target_ratio = target_ratio)

    def _get_uncertainty_jit_impl(self, smx, y, max_iters=30, eps = 0.1, target_ratio = 1.0):
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
        not literally "once" unless `max_batch_size` is set.

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
        smx : array-like, shape (B, K)
            Softmax output; B varies from call to call.
        y : array-like, shape (B,)
            Integer class labels.
        max_iters : int, default=30
            Maximum iterations for the binary search over alpha, forwarded to
            `_search_uncertainty`.
        """
        B = y.shape[0]
        K = smx.shape[1] if smx.ndim > 1 else 1

        # 1. máscara de validez: muestra real (siempre True aquí, B es el real)
        #    AND pertenece a la clase de interés
        if self._task == "classification" and self.classes is not None:
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
        # arrays paddeados con valores arbitrarios (se enmascaran)
        y_dtype = np.int32 if self._task == "classification" else np.float64 
        y_arr = np.asarray(y).astype(y_dtype)

        y_padded   = np.zeros(target_size, dtype=y_dtype)
        mask_padded = np.zeros(target_size, dtype=bool)

        if smx.ndim > 1:
            p_padded = np.zeros((target_size, K), dtype = np.float64)
        else:
            p_padded = np.zeros(target_size, dtype = np.float64)

        y_padded[:B]    = y_arr
        p_padded[:B]    = np.asarray(smx)
        mask_padded[:B] = valid                       # solo válidos reales en True

        # 3. y_safe: índices en rango incluso en padding (clase 0)
        fill_val = 0 if self._task == "classification" else 0.0 
        y_safe = np.where(mask_padded, y_padded, fill_val)

        # 4. a jnp y al JIT
        y_j    = jnp.asarray(y_safe)
        p_j    = jnp.asarray(p_padded, dtype=jnp.float64)
        mask_j = jnp.asarray(mask_padded)

        self._state = _ensure_sorted(self._state)
        cs_padded = self._state.conformity_scores
        n_cs = jnp.int32(self._state.N)
        N = self._state.N

        if self._task == "classification":
            # For classification: empty aux_data, defaults to clf_risk_fn
            alpha, U, status_code = _search_uncertainty(y_j, p_j, mask_j, cs_padded, n_cs,
                                           target_ratio = 1.0,
                                           score_fn = self.score_,
                                           aux_data = (),
                                           max_iters = max_iters,
                                           risk_fn = clf_risk_fn,
                                           direction = 1.0) # metric decreases with alpha
        else:
            # For regression pass,  eps  and direction = -1.0
            # Ensure target_ratio < 1.0 when callig get_uncertainty
            alpha, U, status_code = _search_uncertainty(y_j, p_j, mask_j, cs_padded, n_cs,
                                           target_ratio = target_ratio,
                                           score_fn = self.score_,
                                           aux_data = (eps, eps),
                                           max_iters = max_iters,
                                           risk_fn = reg_risk_fn,
                                           direction = -1.0)

        # CHEQUEAR PARA REGESSION.

        alpha = np.float64(alpha)
        U = np.float64(U)
        status = _STATUS_CODE_TO_ENUM[int(status_code)]
        self.search_status_ = status

        if status is SearchStatus.FLOOR_LIMITED:
            floor = 1.0 / (N + 1)
            warnings.warn(
                f"get_uncertainty: the tuning target (mean prediction-set size <= 1) "
                f"is already met at the domain floor alpha=1/(N+1)={floor!r} for this "
                f"calibration set (N={N}). U={U!r} is determined by the calibration "
                f"set size, not by the model -- a larger calibration set may allow a "
                f"tighter alpha.",
                SearchStatusWarning,
                stacklevel=2,
            )
        elif status is SearchStatus.INFEASIBLE:
            warnings.warn(
                f"get_uncertainty: no alpha in [1/(N+1), 1] (N={N}) brought the mean "
                f"prediction-set size to <= 1 on this tuning set. Returning the "
                f"trivial bound alpha=1.0, U=1.0.",
                SearchStatusWarning,
                stacklevel=2,
            )

        return U, alpha


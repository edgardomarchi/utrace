"""Conformal predictor wrapper.
"""

import logging
from typing import Literal, Union, Callable, NamedTuple

import numpy as np
from jax import numpy as jnp
from jax import jit, lax

from functools import partial

from .scores import lac, lac_cal
from .utils import _masked_quantile_higher, _bucket_size
from .utils.tensors import to_jax

logger = logging.getLogger(__name__)

@partial(jit, static_argnames=["score_fn"])
def _predict_sets(smx:jnp.ndarray, q_hat: np.float64, 
                  score_fn: Callable) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predicts the class labels and sets of labels for the input data X.

    Parameters
    ----------
    smx : np.ndarray
        Softmax output for each class.
    q_hat : jnp.float64
        Calibrated quantile level.
    score_fn : Callable
        The scoring function to use.

    Returns
    -------
    y_pred : jnp.ndarray
        The predicted class labels.
    y_sets : jnp.ndarray
        The sets of labels as a boolean array.
    """
    y_pred = jnp.argmax(smx, axis=1)  # -1 for tensorflow
    scores = score_fn(smx)
    y_sets = scores <= q_hat
    
    return y_pred, y_sets

@jit
def _q_hat_from_alpha(cs_padded: jnp.ndarray,
                      n_cs: jnp.ndarray,
                      alpha: jnp.ndarray) -> jnp.ndarray:
    """cs_padded already comes with N valid entries an padded with inf."""
    q_level = jnp.ceil((n_cs + 1) * (1.0 - alpha)) / n_cs
    q_level = jnp.minimum(q_level, 1.0)
    return _masked_quantile_higher(cs_padded, n_cs, q_level)

@partial(jit, static_argnames=["score_fn", "max_iters"])
def _search_uncertainty(
    y: jnp.ndarray,                  # (n,) filtered labels
    smx: jnp.ndarray,                # (n, K) softmax output
    valid_mask: jnp.ndarray,         # (n,) bool - True: sample from selected class(es)
    cs_padded: jnp.ndarray,          # (m,) calibration scores
    n_cs: jnp.ndarray,               # (1,) number of calibration scores
    max_iters: int,
    score_fn: Callable,
):

    init_state = (
        jnp.asarray(1.0),    # alpha
        jnp.asarray(1.0),    # delta
        jnp.asarray(0.0),    # setsize
        jnp.asarray(0.0),    # EC_yt
        jnp.asarray(False),  # frozen: alpha is out from [0,1]
    )

    n_valid = valid_mask.sum()

    def body(i, state):
        alpha, delta, setsize, EC_yt, frozen = state

        # Update:
        delta_new      = delta / 2.0
        sign           = jnp.where(setsize > 1.0, 1.0, -1.0)
        alpha_proposed = alpha + sign * delta_new

        # Freeze if out of bounds
        out_of_bounds = (alpha_proposed < 0.0) | (alpha_proposed > 1.0)
        will_freeze   = frozen | out_of_bounds

        # If frozen, we do not update
        alpha_next = jnp.where(will_freeze, alpha, alpha_proposed)
        delta_next = jnp.where(will_freeze, delta, delta_new)

        # Predict
        q_hat = _q_hat_from_alpha(cs_padded, n_cs, alpha_next)
        _, prediction_sets = _predict_sets(smx, q_hat, score_fn=score_fn)

        set_sizes    = prediction_sets.sum(axis=1)
        setsize_curr = jnp.where(valid_mask, set_sizes, 0.0).sum() / jnp.maximum(n_valid, 1)

        is_covered = prediction_sets[jnp.arange(y.shape[0]), y]
        mask_succ  = is_covered & (set_sizes > 0) & valid_mask
        safe_sizes = jnp.where(mask_succ, set_sizes, 1)
        inv_succ   = jnp.where(mask_succ, 1.0 / safe_sizes, 0.0)
        n_succ     = mask_succ.sum()
        EC_yt_curr = inv_succ.sum() / jnp.maximum(n_succ, 1)

        # If frozen:
        setsize_next = jnp.where(will_freeze, setsize, setsize_curr)
        EC_yt_next   = jnp.where(will_freeze, EC_yt, EC_yt_curr)

        return (alpha_next, delta_next, setsize_next, EC_yt_next, will_freeze)

    alpha_f, _, _, EC_yt_f, _ = lax.fori_loop(0, max_iters, body, init_state)

    U = 1.0 - EC_yt_f * (1.0 - alpha_f)
    return alpha_f, U

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
                 classes: Union[list[int], np.ndarray, None] = None,
                 score: Literal['lac'] = 'lac',
                 max_batch_size: int = None):
        """Wrapper for uncertainty quantification using U-TraCE.

        Parameters
        ----------
        N : int, default=1000
            Maximum number of calibration scores to retain.
        classes : list[int] or array, optional
            labels defining the conditioning group; instantiate one object per class/group; None → marginal calibration.
        score : {'lac'}, default='lac'
            Scoring function for nonconformity. For now, only 'lac' is supported.
        max_batch_size : int, optional
            Fixed padding size for input batches. See _get_uncertainty_jit_impl.
        """
        self.classes = classes
        self._classes_jax = jnp.asarray(classes) if classes is not None else None
        self._max_batch_size = max_batch_size

        match score:
            case 'lac':
                self.cal_score_ = lac_cal
                self.score_ = lac
                # Declared by the score family, not hardcoded at the boundary: a future
                # regression score would declare a float dtype here, and hardcoding an
                # integer cast in calibrate would silently truncate continuous
                # targets.
                self.label_dtype_ = jnp.int32
            case 'aps':
                raise ValueError(
                    "score='aps' is not implemented in the JAX backend. "
                    "The only implementation of APS lived in the numpy "
                    "backend, which is unreachable in the current "
                    "configuration. This is a known gap, not a typo. "
                    "'lac' is the supported value."
                )
            case _:
                raise ValueError(
                    f"Unknown score {score!r}. The supported value is 'lac'."
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

        logger.debug("UQ reset.")

    @property
    def alpha(self) -> np.float64:
        """The alpha value used for the conformal prediction stage."""
        return self._state.alpha

    @alpha.setter
    def alpha(self, alpha: np.float64):
        """Sets the alpha value and calculates the q_hat level based on the current conformity scores."""
        # n = self.conformity_scores_.shape[0]
        if self._state.N == 0:
            raise ValueError("The model must be calibrated before setting alpha.")

        q_level = np.divide(np.ceil((self._state.N + 1) * (1 - alpha)), self._state.N, dtype=np.float64)
        if q_level > 1.0:
            logger.warning("'q_level' > 1.0, setting to 1.0 - Scores size: %d (< 1/alpha???) - alpha %f", self._state.N, alpha)
            q_level = np.float64(1.0)
        new_alpha = np.float64(alpha)
        logger.debug("'q_level' set to %f for alpha %f and N %d", q_level, new_alpha, self._state.N)
        self._state = _ensure_sorted(self._state)
        logger.debug("Conformity scores: %s", self._state.conformity_scores[:self._state.N])
        # Cap preserved: _masked_quantile_higher clips silently, but we keep
        # explicit q_level <= 1.0 to match historical semantics and warning.
        new_q_hat = np.float64(
            _masked_quantile_higher(self._state.conformity_scores, jnp.int32(self._state.N), q_level)
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
        """Calibrate the conformal predictor with precomputed softmax output.
        
        Parameters
        ----------
        softmax : array-like, shape (n_samples, n_classes)
            Predicted class softmax output. Accepts any array type that implements
            DLPack (jax, numpy, torch, tensorflow, ...). Zero-copy when possible.
        y : array-like, shape (n_samples,)
            Integer class labels.
        batched : bool, default=False
            If True, append to existing calibration scores instead of replacing.
        """
        softmax = to_jax(softmax)
        y_arr = to_jax(y).astype(self.label_dtype_)
        self._calibrate_impl(softmax, y_arr, batched=batched)

    def _calibrate_impl(self, smx, y, batched: bool = False):
        """Calibrates the conformal predictor with the given data.

        Parameters
        ----------
        smx : np.ndarray
            Softmax output for calibration.
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
        
    def predict(self, softmax, force_non_empty_sets: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict class labels and prediction sets from precomputed softmax output.
        
        Parameters
        ----------
        softmax : array-like, shape (n_samples, n_classes)
            Predicted class softmax output. Accepts any DLPack-compatible array.
        force_non_empty_sets : bool, default=False
            If True, ensure the predicted class is always included in the set.
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels.
        y_sets : np.ndarray, shape (n_samples, n_classes)
            Boolean prediction sets.
        """
        softmax = to_jax(softmax)
        y_pred, y_sets = _predict_sets(softmax, self._state.q_hat, score_fn=self.score_)
        return np.array(y_pred), np.array(y_sets)

    def get_uncertainty(self, softmax, y, max_iters: int = 30) -> tuple[np.float64, np.float64]:
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
        this method leaves self._state completely untouched. To use the
        returned alpha for subsequent predictions, set it explicitly:

            U, alpha = uq.get_uncertainty(tune_probs, tune_y)
            uq.alpha = alpha                  # explicit, caller's decision
            y_pred, y_sets = uq.predict(test_probs)

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
            Estimated uncertainty.
        alpha : float
            The alpha found by the search. NOT applied to the object.

        Notes
        -----
        Passing a tuning set in batches and averaging per-batch alphas is
        statistically incorrect (alpha is a nonlinear function of the data).
        To assess whether your tuning set size is adequate, use tuning_stability(),
        which runs the search on disjoint subsets and reports the spread.
        """
    
        # TODO: _get_uncertainty_jit_impl espera numpy (lo convierte a jnp adentro)
        softmax = np.asarray(to_jax(softmax))
        y_arr = np.asarray(y).flatten().astype(int)
        return self._get_uncertainty_jit_impl(softmax, y_arr, max_iters=max_iters)

    def _get_uncertainty_jit_impl(self, smx, y, max_iters=30):
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
        K = smx.shape[1]

        # 1. máscara de validez: muestra real (siempre True aquí, B es el real)
        #    AND pertenece a la clase de interés
        if self.classes is not None:
            valid = np.isin(np.asarray(y), np.asarray(self.classes))
        else:
            valid = np.ones(B, dtype=bool)

        if not valid.any():
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
        y_arr = np.asarray(y).astype(np.int32)
        y_padded   = np.zeros(target_size, dtype=np.int32)
        p_padded   = np.zeros((target_size, K), dtype=np.float64)
        mask_padded = np.zeros(target_size, dtype=bool)

        y_padded[:B]    = y_arr
        p_padded[:B]    = np.asarray(smx)
        mask_padded[:B] = valid                       # solo válidos reales en True

        # 3. y_safe: índices en rango incluso en padding (clase 0)
        y_safe = np.where(mask_padded, y_padded, 0)

        # 4. a jnp y al JIT
        y_j    = jnp.asarray(y_safe)
        p_j    = jnp.asarray(p_padded, dtype=jnp.float64)
        mask_j = jnp.asarray(mask_padded)

        self._state = _ensure_sorted(self._state)
        cs_padded = self._state.conformity_scores
        n_cs = jnp.int32(self._state.N)

        alpha, U = _search_uncertainty(y_j, p_j, mask_j, cs_padded, n_cs, max_iters, self.score_)
        return np.float64(U), np.float64(alpha)
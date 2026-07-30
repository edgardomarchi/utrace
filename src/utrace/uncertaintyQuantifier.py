"""Conformal predictor wrapper.
"""

import logging
from typing import Literal, Union, Callable

import numpy as np
from jax import numpy as jnp
from jax import jit, lax

from functools import partial

from .scores import lac, lac_cal
from .utils import _masked_quantile_higher, _bucket_size
from .utils.tensors import to_jax

logger = logging.getLogger(__name__)

@partial(jit, static_argnames=["score_fn"])
def _predict_sets(y_pred_proba:jnp.ndarray, q_hat: np.float64, 
                  score_fn: Callable) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predicts the class labels and sets of labels for the input data X.

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Predicted probabilities for each class.
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
    y_pred = jnp.argmax(y_pred_proba, axis=1)  # -1 for tensorflow
    scores = score_fn(y_pred_proba)
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
    y_pred_proba: jnp.ndarray,       # (n, K) probabilities
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
        _, prediction_sets = _predict_sets(y_pred_proba, q_hat, score_fn=score_fn)

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
                # integer cast in calibrate_from_proba would silently truncate continuous
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
        self._N = 0 #TODO: Count the number of trully used samples
        self._max_N = N
        self.reset()



    def reset(self):
        """Resets the scoores and alpha."""
        self._conformity_scores_ = jnp.full(self._max_N, jnp.inf, dtype=jnp.float64)
        self._sorted = True  # +inf buffer is trivially sorted; no lazy sort needed on empty read
        self.__alpha:np.float64 = np.float64('nan')
        self.__q_hat:np.float64 = np.float64('nan')

        self._N = 0

        logger.debug("UQ reset.")


    @property
    def alpha(self) -> np.float64:
        """The alpha value used for the conformal prediction stage."""
        return self.__alpha
    
    @alpha.setter
    def alpha(self, alpha: np.float64):
        """Sets the alpha value and calculates the q_hat level based on the current conformity scores."""
        # n = self.conformity_scores_.shape[0]
        if self._N == 0:
            raise ValueError("The model must be calibrated before setting alpha.")
        
        q_level = np.divide(np.ceil((self._N + 1) * (1 - alpha)), self._N, dtype=np.float64)
        if q_level > 1.0:
            logger.warning("'q_level' > 1.0, setting to 1.0 - Scores size: %d (< 1/alpha???) - alpha %f", self._N, alpha)
            q_level = np.float64(1.0)
        self.__alpha = np.float64(alpha)
        logger.debug("'q_level' set to %f for alpha %f and N %d", q_level, self.__alpha, self._N)
        logger.debug("Conformity scores: %s", self.conformity_scores_[:self._N])
        # Cap preserved: _masked_quantile_higher clips silently, but we keep
        # explicit q_level <= 1.0 to match historical semantics and warning.
        self.__q_hat = np.float64(
            _masked_quantile_higher(self.conformity_scores_, jnp.int32(self._N), q_level)
        )
        logger.debug("'q_hat' set to %f for alpha %f", self.__q_hat, self.__alpha)

    @property
    def conformity_scores_(self):
        """Calibration buffer: sorted ascending in [:_N], +inf padding in [_N:].

        Lazy-sorts the valid prefix on first read after a calibration write.
        Single-threaded access assumed per instance.
        """
        if not self._sorted:
            # Sort the FULL buffer, not just [:self._N]: the region beyond
            # self._N is +inf-padded (see reset() and _calibrate_impl), and
            # +inf entries sort to the tail regardless, so this is
            # bit-identical to sorting the variable-length prefix while
            # keeping the sort's input shape fixed at (self._max_N,) across
            # every call, instead of one XLA compilation per distinct _N.
            self._conformity_scores_ = jnp.sort(self._conformity_scores_)
            self._sorted = True
        return self._conformity_scores_

    def calibrate_from_proba(self, y_pred_proba, y, batched: bool = False):
        """Calibrate the conformal predictor with precomputed probabilities.
        
        Parameters
        ----------
        y_pred_proba : array-like, shape (n_samples, n_classes)
            Predicted class probabilities. Accepts any array type that implements
            DLPack (jax, numpy, torch, tensorflow, ...). Zero-copy when possible.
        y : array-like, shape (n_samples,)
            Integer class labels.
        batched : bool, default=False
            If True, append to existing calibration scores instead of replacing.
        """
        y_pred_proba = to_jax(y_pred_proba)
        y_arr = to_jax(y).astype(self.label_dtype_)
        self._calibrate_impl(y_pred_proba, y_arr, batched=batched)

    def _calibrate_impl(self, y_pred_proba, y, batched: bool = False):
        """Calibrates the conformal predictor with the given data.

        Parameters
        ----------
        y_pred_proba : np.ndarray
            Predicted probabilities for calibration.
        y : np.ndarray
            Target labels for calibration.
        batched : bool, optional
            For batched calibration; appends new scores to the buffer. By default False
        """
        if self._classes_jax is not None:
            mask = jnp.isin(y, self._classes_jax)
            y = y[mask]
            y_pred_proba = y_pred_proba[mask]

        scores = self.cal_score_(y, y_pred_proba)
        num_scores = len(scores)
        if batched:
            if self._N + num_scores > self._max_N:
                raise ValueError(
                    f"Batched calibration buffer overflow: current _N={self._N} + "
                    f"num_scores={num_scores} exceeds _max_N={self._max_N}. "
                    f"N is set at construction time."
                )
            # Append new scores at offset _N without sorting (lazy sort deferred to property getter).
            self._conformity_scores_ = self._conformity_scores_.at[
                self._N:self._N + num_scores
            ].set(jnp.asarray(scores, dtype=jnp.float64))
        else:
            if num_scores > self._max_N:
                raise ValueError(
                    f"Non-batched calibration buffer overflow: num_scores={num_scores} "
                    f"exceeds _max_N={self._max_N} (current _N={self._N}). "
                    f"N is set at construction time."
                )
            # Non-batched: reset buffer to +inf and write scores at offset 0 without sorting.
            self._conformity_scores_ = jnp.full(
                (self._max_N,), jnp.inf, dtype=jnp.float64
            ).at[:num_scores].set(jnp.asarray(scores, dtype=jnp.float64))

        if num_scores > 0:
            self._sorted = False  # valid prefix is unsorted; getter will sort on next read

        logger.debug("Conformity scores shape: %s, used: %d", self._conformity_scores_.shape, self._N)

        self._N = self._N + num_scores if batched else num_scores

        if self.classes is not None and self._N == 0:
            logger.warning("No calibration scores for the requested class group %s after calibration.", self.classes)
        
    def predict_from_proba(self, y_pred_proba, force_non_empty_sets: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict class labels and prediction sets from precomputed probabilities.
        
        Parameters
        ----------
        y_pred_proba : array-like, shape (n_samples, n_classes)
            Predicted class probabilities. Accepts any DLPack-compatible array.
        force_non_empty_sets : bool, default=False
            If True, ensure the predicted class is always included in the set.
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels.
        y_sets : np.ndarray, shape (n_samples, n_classes)
            Boolean prediction sets.
        """
        y_pred_proba = to_jax(y_pred_proba)
        y_pred, y_sets = _predict_sets(y_pred_proba, self.__q_hat, score_fn=self.score_)
        return np.array(y_pred), np.array(y_sets)

    def get_uncertainty_from_proba(self, y_pred_proba, y, max_iters: int = 30) -> tuple[np.float64, np.float64]:
        """Estimate model uncertainty over a tuning set via conformal prediction.

        Searches for the alpha that yields the target average prediction-set size,
        using ALL provided samples as a single tuning set. The estimate converges
        to the true error probability with the tuning set size (conformal
        guarantee); it does NOT require batching.

        This method is PURE: it does not modify the object's state. In particular,
        it does not set self.alpha or self.q_hat. To use the returned alpha for
        subsequent predictions, set it explicitly:

            U, alpha = uq.get_uncertainty_from_proba(tune_probs, tune_y)
            uq.alpha = alpha                  # explicit, caller's decision
            y_pred, y_sets = uq.predict_from_proba(test_probs)

        Parameters
        ----------
        y_pred_proba : array-like, shape (n_tuning, n_classes)
            Predicted probabilities for the tuning set. Any DLPack-compatible array.
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
        y_pred_proba = np.asarray(to_jax(y_pred_proba))
        y_arr = np.asarray(y).flatten().astype(int)
        return self._get_uncertainty_jit_impl(y_pred_proba, y_arr, max_iters=max_iters)

    def _get_uncertainty_jit_impl(self, y_pred_proba, y, max_iters=30):
        """y_pred_proba: (B, K) array (jnp/np), B variable.
           y:            (B,)   int labels.
        Internally pads to a fixed shape so the jitted search compiles once."""
        B = y.shape[0]
        K = y_pred_proba.shape[1]

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
        p_padded[:B]    = np.asarray(y_pred_proba)
        mask_padded[:B] = valid                       # solo válidos reales en True

        # 3. y_safe: índices en rango incluso en padding (clase 0)
        y_safe = np.where(mask_padded, y_padded, 0)

        # 4. a jnp y al JIT
        y_j    = jnp.asarray(y_safe)
        p_j    = jnp.asarray(p_padded, dtype=jnp.float64)
        mask_j = jnp.asarray(mask_padded)

        cs_padded = self.conformity_scores_
        n_cs = jnp.int32(self._N)

        alpha, U = _search_uncertainty(y_j, p_j, mask_j, cs_padded, n_cs, max_iters, self.score_)
        return np.float64(U), np.float64(alpha)
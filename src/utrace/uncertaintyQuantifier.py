"""Conformal predictor wrapper.
"""

import logging
from typing import Literal, Union, Callable

import numpy as np
from jax import numpy as jnp
from jax import jit, lax

from functools import partial

from .scores import aps, aps_cal, lac, lac_cal
from .utils import flatten_batch, _masked_quantile_higher
from .utils.tensors import to_jax

from .config import USE_JAX

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

@partial(jit, static_argnames=["score_fn"])
def get_U(y: jnp.ndarray, y_pred_proba: jnp.ndarray, 
          q_hat: np.float64, alpha: np.float64,
          score_fn: Callable):
    
    _, prediction_sets = _predict_sets(y_pred_proba, q_hat, score_fn=score_fn)

    is_covered = prediction_sets[jnp.arange(len(y)), y]
    set_sizes  = prediction_sets.sum(axis=1)
    
    # --- p1_hat = E[1/k | success] ---
    mask_succ  = is_covered & (set_sizes > 0)
    safe_sizes = jnp.where(mask_succ, set_sizes, 1)
    inv_succ   = jnp.where(mask_succ, 1.0 / safe_sizes, 0.0)
    n_succ     = mask_succ.sum()
    p1_hat     = inv_succ.sum() / n_succ

    return p1_hat * (1 - alpha)


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
        setsize_curr = jnp.nanmean(set_sizes.astype(jnp.float64))

        is_covered = prediction_sets[jnp.arange(y.shape[0]), y]
        mask_succ  = is_covered & (set_sizes > 0)
        safe_sizes = jnp.where(mask_succ, set_sizes, 1)
        inv_succ   = jnp.where(mask_succ, 1.0 / safe_sizes, 0.0)
        n_succ     = mask_succ.sum()
        EC_yt_curr = inv_succ.sum() / jnp.maximum(n_succ, 1)

        # --- 4. si congelamos, conservamos los valores anteriores -
        setsize_next = jnp.where(will_freeze, setsize, setsize_curr)
        EC_yt_next   = jnp.where(will_freeze, EC_yt,   EC_yt_curr)

        return (alpha_next, delta_next, setsize_next, EC_yt_next, will_freeze)

    alpha_f, _, _, EC_yt_f, _ = lax.fori_loop(0, max_iters, body, init_state)

    U = 1.0 - EC_yt_f * (1.0 - alpha_f)
    return alpha_f, U

class UncertaintyQuantifier:
    """Wrapper for uncertainty quantification using U-TraCE.

    Parameters
    ----------
    model : Any
        (deprecated) A trained model with a `predict_proba` method.
    classes : Union[list[int], np.ndarray, None], optional
        List or array of class labels
    score : Literal['lac','aps'], optional
        The scoring function to use, by default 'lac'
    """
    def __init__(self, model,
                 N: int = 1000,
                 classes:Union[list[int], np.ndarray, None]=None,
                 score:Literal['lac','aps']='lac'):
        self.model = model
        self.classes = classes
        match score:
            case 'lac':
                self.cal_score_ = lac_cal
                self.score_ = lac
            case 'aps':
                self.cal_score_ = aps_cal
                self.score_ = aps
            case _:
                self.cal_score_ = lac_cal
                self.score_ = lac
        self._N = 0 #TODO: Count the number of trully used samples
        self._max_N = N
        self.reset()



    def reset(self):
        """Resets the scoores and alpha."""
        self.conformity_scores_ = jnp.full(self._max_N, jnp.inf, dtype=jnp.float64)
        self.__alpha:np.float64 = np.float64('nan')
        self.__q_hat:np.float64 = np.float64('nan')

        self._class_alphas:np.ndarray = np.zeros_like(self.classes, dtype=np.float64) if self.classes is not None else np.empty(self._max_N)
        self._class_q_hats:np.ndarray = np.zeros_like(self.classes, dtype=np.float64) if self.classes is not None else np.empty(self._max_N)
        self._class_scores:list[np.ndarray] = [np.empty(self._max_N) for _ in self.classes] if self.classes is not None else []

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
        self.__q_hat = np.nanquantile(
        np.asarray(self.conformity_scores_[:self._N]), q_level, method='higher')
        logger.debug("'q_hat' set to %f for alpha %f", self.__q_hat, self.__alpha)     
    

    def calibrate(self, X, y, batched:bool=False):
        """Legacy entry point: Calibrates the conformal predictor with the given data.

        Parameters
        ----------
        X : np.ndarray
            Input features for calibration.
        y : np.ndarray
            Target labels for calibration.
        batched : bool, optional
            For batched calibration; concatenates new scores with prvious ones. By default False
        """
        logger.debug('Calibrating with input shape: %s', X.shape)

        y = flatten_batch(y).ravel().numpy().astype(int)  # added .numpy() #TODO: generalize API
        num_samples = len(y)
        logger.debug('Fitting with %d samples', num_samples)
        y_pred_proba = self.model.predict_proba(X)

        if USE_JAX:
            y = to_jax(y)
            y_pred_proba = to_jax(y_pred_proba)
        
        self._calibrate_impl(y_pred_proba, y, batched)


    def _calibrate_impl(self, y_pred_proba, y, batched: bool = False):
        """Calibrates the conformal predictor with the given data.

        Parameters
        ----------
        y_pred_proba : np.ndarray
            Predicted probabilities for calibration.
        y : np.ndarray
            Target labels for calibration.
        batched : bool, optional
            For batched calibration; concatenates new scores with prvious ones. By default False
        """
        num_scores = 0
        
        # Classes
        if self.classes is not None:
            total_scores = 0
            for c_idx, C in enumerate(self.classes):
                logger.debug("Calibrating for class %d", C)
                
                scores = self.cal_score_(y[y==C], y_pred_proba[y==C])
                num_scores = len(scores)
                if batched:
                    self._class_scores[c_idx] = np.sort(
                                                np.concatenate([self._class_scores[c_idx][:self._N],
                                                                np.array(scores)]
                                                ))
                else:
                    self._class_scores[c_idx] = np.sort(np.array(scores))
                if self._class_scores[c_idx].size == 0:
                    logger.warning("No scores for class %d after calibration.", C)
                total_scores += num_scores
                
            logger.debug("Total conformity scores shape before class calibration: %s", self.conformity_scores_.shape)
            sorted_all = np.sort(np.concatenate(self._class_scores))  # numpy, como antes
            self.conformity_scores_ = self.conformity_scores_.at[:self._N + total_scores].set(
                jnp.asarray(sorted_all, dtype=jnp.float64))
            logger.debug("Total conformity scores shape after class calibration: %s", self.conformity_scores_.shape)

        else:   
            scores = self.cal_score_(y, y_pred_proba)
            num_scores = len(scores)
            if batched:
                # If batched calibration, we need to concatenate the conformity scores for each batch
                self.conformity_scores_ = self.conformity_scores_.at[self._N:self._N + num_scores].set(
                jnp.asarray(np.asarray(scores), dtype=jnp.float64))
            else:
                #self.conformity_scores_ = scores
                srt = np.sort(np.asarray(scores))
                self.conformity_scores_ = jnp.full((self._max_N,), jnp.inf,
                                                    dtype=jnp.float64
                ).at[:num_scores].set(jnp.asarray(srt, dtype=jnp.float64))
            
            logger.debug("Conformity scores shape: %s, used: %d", self.conformity_scores_.shape, self._N)

        #Update number of scores
        self._N += num_scores if batched else 0




    def predict(self, X:np.ndarray, force_non_empty_sets:bool=False) -> tuple[np.ndarray, np.ndarray]:
        """Predicts the class labels and sets of labels for the input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data for prediction.
        force_non_empty_sets : bool, optional
            If True, ensures that the predicted class is included in the set, by default False.

        Returns
        -------
        y_pred : np.ndarray
            The predicted class labels.
        y_sets : np.ndarray
            The sets of labels as a boolean array.
        """
        y_pred_proba = self.model.predict_proba(X) # added .cpu().numpy
        y_pred_proba = to_jax(y_pred_proba)
        y_pred, y_sets = _predict_sets(y_pred_proba, self.__q_hat, score_fn=self.score_)

        return np.array(y_pred), np.array(y_sets)
    


    def get_uncertainty_opt(self, X, y) -> tuple[np.float64, np.float64]:
        """Calculates the overall uncertainty of the model predictions.
        
        This method uses a intelligent grid search-like approach to find the optimal alpha value
        that yields the minimum upper bound for model uncertatinty.
        
        Parameters
        ----------
        X : np.ndarray
            Input data for prediction.
        y : np.ndarray
            Target labels for prediction.
        Returns
        -------
        U, alpha : float
            The uncertainty of the model predictions and the alpha of the CP found.
        """
        
        y_pred_proba = self.model.predict_proba(X)
        y = y.numpy().flatten().astype(int)
        logger.debug(" Computing model uncertainty with: 'X' shape: %s, 'y' shape: %s\n, class(es): %s, N: %d",
                     X.shape, y.shape, self.classes, self._N)

        # if self.classes is not None:
        #     K = len(self.model.classes_)
            

        if self.classes is not None:
            valid_indexes = np.isin(y, np.array(self.classes))  #type: ignore
        else:
            valid_indexes = np.ones(len(y), dtype=bool)

        if USE_JAX:
            y = to_jax(y[valid_indexes])
            y_pred_proba = to_jax(y_pred_proba[valid_indexes])

        best_alpha = np.float64('nan')
        
        max_lower_bound = np.float64(0.0) # This represents P(y=y_t), or 1 - U

        for j,score in enumerate(self.conformity_scores_[:self._N]):
            
            q_hat = score
            alpha = 1 - (j + 1) / (self._N + 1)

            lower_bound = get_U(y, y_pred_proba, q_hat, alpha, self.score_)

            # Update bound and alpha if better
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_alpha = np.float64(alpha)

        self.alpha = best_alpha

        logger.debug("Best alpha: %f - Min upper uncertainty bound: %f\n", best_alpha, 1-max_lower_bound)
        return 1-max_lower_bound, best_alpha
    

    def get_uncertainty(self, X, y, max_iters = 30) -> tuple[np.float64, np.float64]:
        """Calculates the uncertainty of the model predictions.
        
        This method uses a binary search-like approach to find the optimal alpha value
        that yields the average target set size of the predicted sets.
        
        Parameters
        ----------
        X : np.ndarray
            Input data for prediction.
        y : np.ndarray
            Target labels for prediction.
        max_iters : int, optional
            Maximum number of iterations for the search, by default 20
        Returns
        -------
        U, alpha : float
            The uncertainty of the model predictions and the alpha of the CP found.
        """
        
        y_pred_proba = self.model.predict_proba(X)
        y = y.numpy().flatten().astype(int)
        logger.debug("'X' shape: %s, 'y' shape: %s", X.shape, y.shape)

        if self.classes is not None:
            valid_indexes = np.isin(y, np.array(self.classes))  #type: ignore
        else:
            valid_indexes = np.ones(len(y), dtype=bool)

        y_f = y[valid_indexes]

        if not valid_indexes.any():
            logger.warning("No valid indexes found for class(es) %s", self.classes)
            return np.float64('nan'), np.float64('nan')
        logger.debug("Valid indexes shape: %s", valid_indexes.shape)

        setsize = np.float64(0.0)
        setsize_std = np.float64(0.0)
        alpha = np.float64(1.0)
        delta = np.float64(1.0)

        it = 0
        alphas: list[np.float64] = []
        EC_yt = np.float64(0.0)
        while (it < max_iters):
            
            delta = delta/2
            if setsize > 1.0:
                alpha += delta
            else:
                alpha -= delta

            
            if alpha < 0.0 or alpha > 1.0:
                logger.error("Alpha out of bounds: %s - Iter: %d",alphas, it)
                break

            self.alpha = alpha
            # Store the alpha value for debugging
            alphas.append(alpha)

            # Predict to evaluate the average set size
            y_p, y_s = self._predict_sets(y_pred_proba.cpu().numpy(), force_non_empty_sets=False)

            # Filter out the ouputs that are not in the classes
            y_p = y_p[valid_indexes]
            y_s = y_s[valid_indexes]

            # Calculate E[1/|C| | y_t in C(x_t)]
            mask = y_s[np.arange(len(y_s)), y_f]
            y_sf = y_s[mask]
            y_nsf = y_s[np.logical_not(mask)]
            setsizes = y_sf.sum(axis=1)
            if setsizes.shape[0] == 0:
                logger.warning("No sets found for class(es) %s at alpha %f", self.classes, alpha)
                return np.float64(1.0), alpha
            miss_setsizes = y_nsf.sum(axis=1)
            logger.debug("y_sf shape: %s - setsizes shape:\n%s ", y_sf.shape, setsizes.shape)

            try:
                setsize = np.nanmean(y_s.sum(axis=1), dtype=np.float64)
                EC_yt = np.nanmean(1/setsizes, dtype=np.float64)
                setsize_std = setsizes.std()

                #EC_nyt = np.nanmean(1/(len(self.classes) - miss_setsizes), dtype=np.float64)
                #logger.debug("EC_yt: %f - EC_nyt: %f", EC_yt, EC_nyt)


            except ValueError:
                logger.error("Error calculating E[1/|C| | y_t in C(x_t)]:\n"
                             "Input shape: %s - Output sets shape: %s - "
                             "Valid indexes shape: %s.\n",
                             y.shape, y_s.shape, valid_indexes.shape)
                break

            logger.debug("Iteration %d - alpha: %f - delta: %f - Set size: %f", it, alpha, delta, setsize)
            it += 1

        logger.debug("Found alpha %f for class(es) %s with average set-size %f and std %f.",
                      self.alpha, self.classes, setsize, setsize_std)
        logger.debug("Alphas analyzed: %s", alphas)

        p_tc = EC_yt*(1-self.alpha) #+ EC_nyt*self.alpha (*)
        U = 1 - p_tc
        logger.debug("Model U: %f - EC_yt: %f.", U, EC_yt)
        return U, self.alpha


    def get_uncertainty_jit(self, X, y, max_iters=30):
        y_pred_proba = self.model.predict_proba(X).cpu().numpy()
        y = y.numpy().flatten().astype(int)
        return self._get_uncertainty_jit_impl(y_pred_proba, y, max_iters)
    
    
    def _get_uncertainty_jit_impl(self, y_pred_proba, y, max_iters=30):
        """Core uncertainty estimation logic.
        """
        valid = (np.isin(y, self.classes) if self.classes is not None
                else np.ones_like(y, dtype=bool))
        if not valid.any():
            return np.float64('nan'), np.float64('nan')
    
        y_f = jnp.asarray(y[valid])
        p_f = jnp.asarray(y_pred_proba[valid])
        #cs  = jnp.asarray(self.conformity_scores_[:self._N])

        cs_padded = self.conformity_scores_
        n_cs = jnp.asarray(self._N)
    
        alpha, U = _search_uncertainty(y_f, p_f, cs_padded, n_cs,max_iters, self.score_)
        return float(U), float(alpha)
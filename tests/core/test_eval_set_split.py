"""Tests for the separate evaluation set: `tune_alpha`, `uncertainty_at`, and
`get_uncertainty(..., eval_softmax=..., eval_y=...)`.

Added tests only -- this file does not modify any existing test. The pinned
single-dataset values live in `test_get_uncertainty_single_dataset_pinned.py`.
"""
import inspect
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from utrace import (
    SearchStatus,
    SearchStatusWarning,
    TuningDataReuseWarning,
    UncertaintyEvaluation,
    UncertaintyQuantifier,
)
from utrace.uncertaintyQuantifier import _evaluate_at_alpha

K = 10


def _confident(rng, n):
    y = rng.integers(0, K, n)
    smx = np.full((n, K), 1e-4)
    smx[np.arange(n), y] = 1 - 1e-4 * (K - 1)
    return y, smx


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _near_uniform(rng, n):
    return rng.integers(0, K, n), rng.dirichlet(np.ones(K) * 0.05, n)


_MAKERS = {"confident": _confident, "interior": _interior, "near_uniform": _near_uniform}


def _build(kind, N, classes=None, seed=0, n_tune=None, n_eval=None, **uq_kwargs):
    """Calibrated instance plus tuning and evaluation data (independent draws)."""
    rng = np.random.default_rng(seed)
    make = _MAKERS[kind]
    cal_y, cal_smx = make(rng, N * (5 if classes else 1))
    tune = make(rng, n_tune or N)
    ev = make(rng, n_eval or N)
    uq = UncertaintyQuantifier(N=N * 5, classes=classes, **uq_kwargs)
    uq.calibrate(cal_smx, cal_y)
    return uq, tune, ev


def _quiet_get_uncertainty(uq, smx, y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return uq.get_uncertainty(smx, y, **kw)


def _bytes(x):
    return np.float64(x).tobytes()


# ----- 1. evaluation path == manual composition ----------------------------------

@pytest.mark.parametrize("classes", [None, [3]])
def test_eval_path_equals_manual_composition(classes):
    uq, (ty, ts), (ey, es) = _build("interior", 300, classes, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, alpha = uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=ey)
        alpha_m = uq.tune_alpha(ts, ty)
    U_m = uq.uncertainty_at(es, ey, alpha_m).U
    assert _bytes(alpha) == _bytes(alpha_m)
    assert _bytes(U) == _bytes(U_m)


# ----- 2. eval == tune reproduces the single-dataset U bitwise ---------------------

@pytest.mark.parametrize("classes", [None, [3]])
@pytest.mark.parametrize("kind,N", [("interior", 300), ("interior", 1200),
                                     ("confident", 42), ("confident", 96)])
def test_eval_equal_to_tune_matches_single_dataset_bitwise(kind, N, classes):
    uq, (ty, ts), _ = _build(kind, N, classes, seed=1)
    U_single, alpha = _quiet_get_uncertainty(uq, ts, ty)
    status = uq.search_status_
    U_eval = uq.uncertainty_at(ts, ty, alpha).U
    assert _bytes(U_single) == _bytes(U_eval), (status, U_single, U_eval)


def test_bitwise_grid_covers_converged_and_floor_limited():
    """Guards the guard: the parametrised test above must actually exercise both
    statuses, otherwise its bitwise claim is narrower than it reads."""
    seen = set()
    for kind, N in [("interior", 300), ("interior", 1200), ("confident", 42), ("confident", 96)]:
        for classes in (None, [3]):
            uq, (ty, ts), _ = _build(kind, N, classes, seed=1)
            _quiet_get_uncertainty(uq, ts, ty)
            seen.add(uq.search_status_)
    assert {SearchStatus.CONVERGED, SearchStatus.FLOOR_LIMITED} <= seen


# ----- 4. warning ------------------------------------------------------------------

def test_warning_category_is_independent_of_search_status_warning():
    assert issubclass(TuningDataReuseWarning, UserWarning)
    assert not issubclass(TuningDataReuseWarning, SearchStatusWarning)
    assert not issubclass(SearchStatusWarning, TuningDataReuseWarning)


def test_warning_fires_on_default_path_and_points_at_caller():
    uq, (ty, ts), _ = _build("interior", 300, seed=2)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        uq.get_uncertainty(ts, ty)
    reuse = [w for w in rec if issubclass(w.category, TuningDataReuseWarning)]
    assert len(reuse) == 1
    assert reuse[0].filename == __file__
    assert reuse[0].lineno == line
    assert "same samples that selected alpha" in str(reuse[0].message)


def test_warning_fires_on_every_default_call():
    uq, (ty, ts), _ = _build("interior", 300, seed=2)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        uq.get_uncertainty(ts, ty)
        uq.get_uncertainty(ts, ty)
    assert sum(issubclass(w.category, TuningDataReuseWarning) for w in rec) == 2


def test_warning_fires_for_empty_class_group_too():
    uq, (ty, ts), _ = _build("interior", 300, [3], seed=2)
    keep = ty != 3
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        U, alpha = uq.get_uncertainty(ts[keep], ty[keep])
    assert np.isnan(U) and np.isnan(alpha)
    assert any(issubclass(w.category, TuningDataReuseWarning) for w in rec)


def test_warning_is_silenced_by_filtering_its_own_category():
    uq, (ty, ts), _ = _build("interior", 300, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=TuningDataReuseWarning)
        uq.get_uncertainty(ts, ty)  # a converged case: no SearchStatusWarning either


def test_search_status_filter_does_not_silence_it_and_vice_versa():
    uq, (ty, ts), _ = _build("interior", 300, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=SearchStatusWarning)
        with pytest.raises(TuningDataReuseWarning):
            uq.get_uncertainty(ts, ty)


def test_warning_absent_with_eval_set_and_from_tune_alpha():
    uq, (ty, ts), (ey, es) = _build("interior", 300, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TuningDataReuseWarning)
        uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=ey)
        uq.tune_alpha(ts, ty)
        uq.uncertainty_at(es, ey, 0.2)


def test_tune_alpha_still_emits_search_status_warning():
    uq, (ty, ts), _ = _build("confident", 42, seed=0)
    with pytest.warns(SearchStatusWarning):
        alpha = uq.tune_alpha(ts, ty)
    assert uq.search_status_ == SearchStatus.FLOOR_LIMITED
    assert alpha == pytest.approx(1.0 / (uq._state.N + 1), rel=0, abs=1e-15)


# ----- tune_alpha / uncertainty_at side effects -------------------------------------

def test_tune_alpha_returns_alpha_of_get_uncertainty_and_does_not_apply_it():
    uq, (ty, ts), _ = _build("interior", 300, seed=3)
    _, alpha_g = _quiet_get_uncertainty(uq, ts, ty)
    alpha_t = uq.tune_alpha(ts, ty)
    assert isinstance(alpha_t, np.float64)
    assert _bytes(alpha_t) == _bytes(alpha_g)
    assert np.isnan(uq.alpha) and np.isnan(uq._state.q_hat)


def test_uncertainty_at_changes_no_public_state():
    uq, (ty, ts), (ey, es) = _build("interior", 300, seed=3)
    uq.tune_alpha(ts, ty)
    uq.alpha = 0.3
    before = (uq.alpha, uq._state.q_hat, uq.search_status_)
    res = uq.uncertainty_at(es, ey, 0.1)
    assert isinstance(res, UncertaintyEvaluation)
    assert (uq.alpha, uq._state.q_hat, uq.search_status_) == before


def test_tune_alpha_empty_class_group_is_nan_and_status_none():
    uq, (ty, ts), _ = _build("interior", 300, [3], seed=3)
    keep = ty != 3
    assert np.isnan(uq.tune_alpha(ts[keep], ty[keep]))
    assert uq.search_status_ is None


def test_nan_alpha_returns_nan_result_without_evaluating():
    uq, _, (ey, es) = _build("interior", 300, seed=3)
    _evaluate_at_alpha.clear_cache()
    res = uq.uncertainty_at(es, ey, np.float64("nan"))
    assert np.isnan(res.U) and np.isnan(res.EC)
    assert (res.n_covered, res.n_valid, res.n_succ) == (0, 0, 0)
    assert _evaluate_at_alpha._cache_size() == 0


def test_get_uncertainty_returns_nan_pair_when_tuning_group_is_empty():
    uq, (ty, ts), (ey, es) = _build("interior", 300, [3], seed=3)
    keep = ty != 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        U, alpha = uq.get_uncertainty(ts[keep], ty[keep], eval_softmax=es, eval_y=ey)
    assert np.isnan(U) and np.isnan(alpha)


# ----- 5. argument validation -------------------------------------------------------

def test_only_one_of_eval_arguments_raises():
    uq, (ty, ts), (ey, es) = _build("interior", 300, seed=5)
    with pytest.raises(ValueError, match="together"):
        uq.get_uncertainty(ts, ty, eval_softmax=es)
    with pytest.raises(ValueError, match="together"):
        uq.get_uncertainty(ts, ty, eval_y=ey)


def test_eval_arguments_are_keyword_only():
    uq, (ty, ts), (ey, es) = _build("interior", 300, seed=5)
    with pytest.raises(TypeError):
        uq.get_uncertainty(ts, ty, 30, es, ey)


def test_row_count_mismatch_raises():
    uq, _, (ey, es) = _build("interior", 300, seed=5)
    with pytest.raises(ValueError, match="rows"):
        uq.uncertainty_at(es[:-1], ey, 0.2)


def test_column_mismatch_raises():
    uq, _, (ey, es) = _build("interior", 300, seed=5)
    with pytest.raises(ValueError, match="calibrated with"):
        uq.uncertainty_at(es[:, :K - 1], ey, 0.2)


def test_uncalibrated_instance_raises():
    uq = UncertaintyQuantifier(N=100)
    rng = np.random.default_rng(0)
    y, smx = _interior(rng, 50)
    with pytest.raises(ValueError, match="calibrated"):
        uq.uncertainty_at(smx, y, 0.2)


def test_alpha_outside_valid_domain_raises():
    uq, _, (ey, es) = _build("interior", 300, seed=5)
    floor = 1.0 / (uq._state.N + 1)
    for bad in (floor / 2, 0.0, -0.1, 1.0 + 1e-9, 2.0):
        with pytest.raises(ValueError, match=r"outside \[1/\(N\+1\), 1\]"):
            uq.uncertainty_at(es, ey, bad)
    uq.uncertainty_at(es, ey, floor)  # both ends of the domain are accepted
    uq.uncertainty_at(es, ey, 1.0)


def test_reset_forgets_the_calibrated_shape():
    uq, _, (ey, es) = _build("interior", 300, seed=5)
    uq.reset()
    with pytest.raises(ValueError, match="calibrated"):
        uq.uncertainty_at(es, ey, 0.2)


# ----- 6. class-conditional ----------------------------------------------------------

def test_eval_set_without_the_group_gives_nan_not_one():
    uq, _, (ey, es) = _build("interior", 300, [3], seed=6)
    keep = ey != 3
    res = uq.uncertainty_at(es[keep], ey[keep], 0.2)
    assert np.isnan(res.U)
    assert (res.n_valid, res.n_covered, res.n_succ) == (0, 0, 0)
    assert res.EC == 0.0


def test_group_mask_is_built_from_the_evaluation_labels():
    uq, (ty, _), (ey, es) = _build("interior", 300, [3], seed=6)
    assert (ty == 3).sum() != (ey == 3).sum()  # a reused tuning mask would give the wrong count
    res = uq.uncertainty_at(es, ey, 0.2)
    assert res.n_valid == (ey == 3).sum()
    # selecting the group by hand gives the same numbers: the mask is label-based
    keep = ey == 3
    res_hand = UncertaintyQuantifier(N=1500, classes=[3])
    res_hand._state = uq._state
    res_hand._item_shape = uq._item_shape
    again = res_hand.uncertainty_at(es[keep], ey[keep], 0.2)
    assert _bytes(again.U) == _bytes(res.U) and again.n_valid == res.n_valid


def test_marginal_n_valid_is_every_sample():
    uq, _, (ey, es) = _build("interior", 300, seed=6)
    assert uq.uncertainty_at(es, ey, 0.2).n_valid == len(ey)


# ----- 7. compilation -----------------------------------------------------------------

def test_one_compilation_across_alpha_types_and_calibration_sizes():
    _evaluate_at_alpha.clear_cache()
    for n_cal in (200, 700):
        rng = np.random.default_rng(n_cal)
        cal_y, cal_smx = _interior(rng, n_cal)
        ey, es = _interior(rng, 300)
        uq = UncertaintyQuantifier(N=1000)  # same buffer shape, different n_cs
        uq.calibrate(cal_smx, cal_y)
        for alpha in (0.1, np.float64(0.2), jnp.asarray(0.3), jnp.asarray(np.float64(0.4)),
                      np.float32(0.5), 0.6):
            uq.uncertainty_at(es, ey, alpha)
    assert _evaluate_at_alpha._cache_size() == 1


# ----- 8. INFEASIBLE with an evaluation set ---------------------------------------------

def test_infeasible_with_eval_set_gives_exactly_one_and_counts():
    uq, (ty, ts), (ey, es) = _build("near_uniform", 10, seed=0)
    with pytest.warns(SearchStatusWarning):
        U, alpha = uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=ey)
    assert uq.search_status_ == SearchStatus.INFEASIBLE
    assert alpha == 1.0
    assert U == 1.0
    res = uq.uncertainty_at(es, ey, alpha)
    assert res.U == 1.0  # exact: the (1 - alpha) factor is zero
    assert res.n_valid == len(ey)
    assert res.n_covered >= res.n_succ >= 0


# ----- 9. counts ---------------------------------------------------------------------------

@pytest.mark.parametrize("classes", [None, [3]])
@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5, 0.9])
def test_n_covered_equals_n_succ(alpha, classes):
    """A covered sample has its true label in the prediction set, so the set has at
    least one element: `is_covered & valid` implies `set_sizes > 0`. Hence the
    `n_succ` mask (`is_covered & set_sizes > 0 & valid`) selects the same samples as
    `n_covered`, and the two counts are equal. (Empty sets only occur on UNcovered
    samples.) Checked here rather than assumed."""
    uq, _, (ey, es) = _build("interior", 1200, classes, seed=7, n_eval=2000)
    res = uq.uncertainty_at(es, ey, alpha)
    assert res.n_covered == res.n_succ
    assert 0 < res.n_succ <= res.n_valid


def test_counts_are_consistent_with_predict():
    uq, _, (ey, es) = _build("interior", 1200, seed=7, n_eval=2000)
    alpha = 0.2
    res = uq.uncertainty_at(es, ey, alpha)
    uq.alpha = alpha
    _, sets = uq.predict(es)
    covered = sets[np.arange(len(ey)), ey]
    sizes = sets.sum(axis=1)
    assert res.n_covered == covered.sum()
    assert res.n_valid == len(ey)
    ec = (1.0 / sizes[covered]).mean()
    np.testing.assert_allclose(res.EC, ec, rtol=1e-12)
    np.testing.assert_allclose(res.U, 1 - ec * (1 - alpha), rtol=1e-12)


# ----- 11. evaluation set larger than max_batch_size -----------------------------------------

def test_eval_set_larger_than_max_batch_size_works():
    uq, (ty, ts), (ey, es) = _build("interior", 300, seed=8, n_tune=50, n_eval=700,
                                    max_batch_size=64)
    assert len(ey) > 64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, alpha = uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=ey)
    assert np.isfinite(U) and np.isfinite(alpha)
    # the tuning set is still held to max_batch_size
    with pytest.raises(ValueError, match="max_batch_size"):
        uq.tune_alpha(es, ey)
    # and the result does not depend on max_batch_size
    free, _, _ = _build("interior", 300, seed=8, n_tune=50, n_eval=700)
    assert _bytes(free.uncertainty_at(es, ey, alpha).U) == _bytes(uq.uncertainty_at(es, ey, alpha).U)

"""Tests for the alpha-search fix: the domain floor `1/(N+1)`, the
feasible-alpha return, and `SearchStatus` propagation.

Added tests only -- this file does not modify any existing test.
"""
import warnings

import numpy as np
import pytest

from utrace import SearchStatus, SearchStatusWarning, UncertaintyQuantifier


def _confident_case(N, K=10, seed=0):
    """N calibration + N tuning samples from a highly-confident softmax
    (correct class near 1, all others near 0). This is the exact construction
    the 2026-09-22 diagnostic used to reproduce Defect 1: with the pre-fix
    search, `get_uncertainty` returned `alpha == 2**-max_iters` for N in
    {96, 50, 42} -- values found during use of the package."""
    rng = np.random.default_rng(seed)

    def _make(n):
        y = rng.integers(0, K, size=n)
        smx = np.full((n, K), 1e-4)
        smx[np.arange(n), y] = 1 - 1e-4 * (K - 1)
        return y, smx

    cal_y, cal_smx = _make(N)
    tune_y, tune_smx = _make(N)
    return cal_y, cal_smx, tune_y, tune_smx


def _near_uniform_case(N, K=10, seed=1):
    """Softmax from a LOW-concentration Dirichlet (0.05): sparse, near-random
    mass on a few classes per sample, with frequent near-ties between the top
    two -- unlike a genuinely near-uniform draw (high concentration), this
    keeps prediction sets containing 2+ classes across most of the alpha
    domain, including at/near alpha=1. Parameters were found by an empirical
    search over concentration/K/N/seed for `search_status_ == INFEASIBLE`,
    not derived analytically -- this is the regime this repo's own golden
    test hits for 16 of its 20 configurations (untrained CNN), reproduced
    synthetically here for a fast, torch-free test."""
    rng = np.random.default_rng(seed)
    cal_y = rng.integers(0, K, size=N)
    cal_smx = rng.dirichlet(np.ones(K) * 0.05, size=N)
    tune_y = rng.integers(0, K, size=N)
    tune_smx = rng.dirichlet(np.ones(K) * 0.05, size=N)
    return cal_y, cal_smx, tune_y, tune_smx


def _interior_case(N, K=10, seed=2):
    """Softmax with a real but not overwhelming margin for the true class:
    separated enough to converge to an interior alpha, not saturate at
    either domain boundary. Concentration/boost were picked by trying a
    handful of values and checking `search_status_ == CONVERGED` empirically
    (a low Dirichlet concentration spreads probability mass widely; a modest
    boost keeps the true class only somewhat favoured) -- not derived
    analytically, and not guaranteed to converge for arbitrary N/K/seed."""
    rng = np.random.default_rng(seed)

    def _make(n):
        y = rng.integers(0, K, size=n)
        smx = rng.dirichlet(np.ones(K) * 0.5, size=n)
        smx[np.arange(n), y] += 0.3
        smx /= smx.sum(axis=1, keepdims=True)
        return y, smx

    cal_y, cal_smx = _make(N)
    tune_y, tune_smx = _make(N)
    return cal_y, cal_smx, tune_y, tune_smx


@pytest.mark.parametrize("N", [96, 50, 42])
def test_defect1_regression_floor_limited(N):
    """Below 1/(N+1), the search must report FLOOR_LIMITED with alpha set to
    the floor itself, not walk down to 2**-max_iters (the pre-fix behaviour
    that reproduced the exact alpha found during use of the package, 9.313e-10 ==
    2**-30, for N in {96, 50, 42} -- Hypothesis 1)."""
    cal_y, cal_smx, tune_y, tune_smx = _confident_case(N)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    floor = 1.0 / (N + 1)
    with pytest.warns(SearchStatusWarning):
        U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=30)

    assert alpha == pytest.approx(floor, rel=0, abs=1e-15)
    assert abs(alpha - 2.0 ** -30) > 1e-6, (
        "alpha collapsed to 2**-max_iters -- the pre-fix defect signature"
    )
    assert uq.search_status_ == SearchStatus.FLOOR_LIMITED
    assert uq.search_status_ == "floor_limited"  # str-Enum: string comparison works

    # U must be evaluated AT the floor -- same point as the returned alpha,
    # not a leftover value from wherever the old loop happened to stop.
    EC = (1.0 - U) / (1.0 - alpha)
    assert 0.0 <= EC <= 1.0 + 1e-12
    assert U == pytest.approx(1.0 - EC * (1.0 - floor), rel=0, abs=1e-12)


def test_upward_saturation_is_infeasible():
    """Near-uniform softmax: the search should never bring the mean set size
    to <= 1 anywhere in [1/(N+1), 1]. Must report INFEASIBLE with the *exact*
    trivial bound alpha=1.0, U=1.0 -- not an approximation like the pre-fix
    `alpha ~= 1 - 2**-30` this repo's own golden test returned for 16 of its
    20 calls."""
    N = 10
    cal_y, cal_smx, tune_y, tune_smx = _near_uniform_case(N, seed=0)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    with pytest.warns(SearchStatusWarning):
        U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=30)

    assert uq.search_status_ == SearchStatus.INFEASIBLE
    assert alpha == 1.0
    assert U == 1.0


def test_ordinary_convergence_no_warning():
    """An interior case (real but not overwhelming class separation) should
    CONVERGE, without emitting any SearchStatusWarning."""
    N = 200
    cal_y, cal_smx, tune_y, tune_smx = _interior_case(N, seed=2)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", SearchStatusWarning)
        _U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=30)

    assert uq.search_status_ == SearchStatus.CONVERGED
    floor = 1.0 / (N + 1)
    assert floor < alpha < 1.0


@pytest.mark.parametrize("N,K,seed", [(200, 10, 2), (300, 5, 6), (150, 10, 7)])
def test_feasibility_property_holds_when_converged(N, K, seed):
    """For every CONVERGED result, the mean prediction-set size at the
    returned alpha must be <= 1 (the criterion the search targets) --
    checked through the public API end-to-end (calibrate -> get_uncertainty
    -> alpha= -> predict), not by re-deriving it from internals."""
    cal_y, cal_smx, tune_y, tune_smx = _interior_case(N, K=K, seed=seed)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SearchStatusWarning)
        _U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=30)

    assert uq.search_status_ == SearchStatus.CONVERGED, (
        "fixture drifted out of the interior-convergence regime; adjust seed/N/K"
    )

    uq.alpha = alpha
    _, y_sets = uq.predict(tune_smx)
    mean_size = y_sets.sum(axis=1).mean()
    assert mean_size <= 1.0 + 1e-9


def test_feasibility_property_holds_grid_aligned_edge_case():
    """N+1 a power of two (N=15 -> N+1=16): bisection midpoints (halving from
    1.0) can land exactly on quantile grid points `k/(N+1)`. The feasibility
    property must still hold in that specific alignment."""
    N = 15
    cal_y, cal_smx, tune_y, tune_smx = _interior_case(N, K=6, seed=11)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SearchStatusWarning)
        _U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=30)

    assert uq.search_status_ == SearchStatus.CONVERGED, (
        "fixture drifted out of the interior-convergence regime; adjust seed"
    )

    uq.alpha = alpha
    _, y_sets = uq.predict(tune_smx)
    mean_size = y_sets.sum(axis=1).mean()
    assert mean_size <= 1.0 + 1e-9


@pytest.mark.parametrize("N", [96, 50, 42])
def test_alpha_setter_accepts_exact_floor_from_search(N):
    """The exact numerical trap the setter's floor check must handle: assigning
    `alpha` to precisely the floor value a FLOOR_LIMITED search just returned must
    NOT raise, even though the level formula's ceil/multiply chain can round
    fractionally above 1 exactly at alpha == 1/(N+1)."""
    cal_y, cal_smx, tune_y, tune_smx = _confident_case(N)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    with pytest.warns(SearchStatusWarning):
        _U, alpha = uq.get_uncertainty(tune_smx, tune_y, max_iters=30)
    assert uq.search_status_ == SearchStatus.FLOOR_LIMITED

    uq.alpha = alpha  # must not raise
    assert uq.alpha == alpha


@pytest.mark.parametrize("N", [96, 50, 42])
def test_alpha_setter_raises_below_floor(N):
    cal_y, cal_smx, _, _ = _confident_case(N)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    floor = 1.0 / (N + 1)
    with pytest.raises(ValueError):
        uq.alpha = floor * 0.5


def test_alpha_setter_above_floor_unchanged():
    """Behaviour strictly above the floor is unaffected by the floor check."""
    N = 200
    cal_y, cal_smx, _, _ = _interior_case(N, seed=2)
    uq = UncertaintyQuantifier(N=N, classes=None, score='lac')
    uq.calibrate(cal_smx, cal_y)

    uq.alpha = 0.5  # well above 1/(N+1); must not raise
    assert uq.alpha == 0.5
    assert not np.isnan(uq._state.q_hat)


def test_empty_class_group_resets_status_to_none():
    """A tuning call whose class group has no matching sample returns
    `(nan, nan)` without running the search at all -- `search_status_` must
    become `None` (not silently keep whatever an earlier, unrelated call left
    there), since a stale status would misreport why this call returned nan."""
    N = 200
    K = 10
    C = 3
    cal_y, cal_smx, tune_y, tune_smx = _interior_case(N, K=K, seed=2)
    # calibrate on class C only, so a tuning set containing class C converges normally
    uq = UncertaintyQuantifier(N=N, classes=[C], score='lac')
    mask = cal_y == C
    uq.calibrate(cal_smx[mask], cal_y[mask])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SearchStatusWarning)
        uq.get_uncertainty(tune_smx[tune_y == C], tune_y[tune_y == C], max_iters=30)
    assert uq.search_status_ is not None  # sanity: a real search ran first

    # Now call with a tuning set that has NO sample of class C.
    other = tune_y != C
    U, alpha = uq.get_uncertainty(tune_smx[other], tune_y[other], max_iters=30)

    assert np.isnan(U)
    assert np.isnan(alpha)
    assert uq.search_status_ is None

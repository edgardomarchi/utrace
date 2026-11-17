import itertools
import math

import numpy as np
import jax.numpy as jnp

import pytest

from scipy.stats import beta as scipy_beta


from utrace import UncertaintyQuantifier, UncertaintyEvaluation
from utrace.uncertaintyQuantifier import (
    _confidence_bound,
    _evaluate_at_alpha,
    _q_hat_from_alpha
)
from utrace.uncertaintyQuantifier import TuningDataReuseWarning

def _build_regression(*, N = 300, seed = 0, n_eval = 400, noise = 0.3):
    rng = np.random.default_rng(seed)

    # Calibration data
    y_cal = rng.normal(size = N)
    y_hat_cal = y_cal + rng.normal(scale = noise, size = N)

    uq = UncertaintyQuantifier(N = N, score = "abs_error")
    uq.calibrate(y_hat_cal, y_cal)

    # Independent evaluation data
    y_eval = rng.normal(size = n_eval)
    y_hat_eval = y_eval + rng.normal(scale = noise, size = n_eval)

    return uq, y_hat_eval, y_eval


@pytest.mark.parametrize("n", [1, 2, 5, 10, 100, 1000, 10**5])
@pytest.mark.parametrize("delta", [0.05, 0.01, 0.001])
def test_coverage_lower_matches_scipy_ppf(n, delta):
    for kk in sorted({1, 2, max(1, n//2), max(1, n-1), n}):
        if not 1<= kk <= n:
            continue
        cl, _, _ = _confidence_bound(0.5, kk, n, kk, 0.0, delta)
        ref = scipy_beta.ppf(delta/2, kk, n-kk+1)
        assert cl == pytest.approx(ref, rel = 1e-12, abs = 0)

@pytest.mark.parametrize("n", [1, 7, 100, 10**4])
@pytest.mark.parametrize("delta", [0.2, 0.05, 0.001])
def test_converage_lower_closed_forms(n, delta):
    assert _confidence_bound(0.5, 0, n, 0, 0.0, delta)[0]== 0.0

    cl = _confidence_bound(0.5, n, n, n, 0.0, delta)[0]

    assert cl == pytest.approx((delta/2)**(1/n), rel = 1e-12)

def test_ec_lower_hand_computed_n100_zero_variance():
    # delta = 0.05, n = 100
    m_h = math.sqrt(math.log(80.0)/200.0)
    m_eb = 7.0*math.log(160.0)/(3.0*99.0)

    assert m_eb < m_h

    _, ec_lower, _ = _confidence_bound(0.6, 50, 100, 100, 0.0, 0.05)

    assert ec_lower == pytest.approx(0.6 - m_eb, rel = 1e-14)

    # delta = 0.1, n=50
    m_h = math.sqrt(np.log(40.0)/100.0)
    m_eb = 7.0*math.log(80.0)/(3.0*49.0)

    assert m_h < m_eb

    _, ec_lower, _ = _confidence_bound(0.6, 50, 100, 50, 0.0, 0.1)

    assert ec_lower == pytest.approx(0.6 - m_h, rel = 1e-14)

def test_ec_lower_clipped_at_zero_and_zero_succ():
    assert _confidence_bound(
        0.05, 50, 100, 100, 0.0, 0.05
    )[1] == 0.0

    assert _confidence_bound(
        0.0, 0, 100, 0, 0.0, 0.05
    )[1] == 0.0

    assert _confidence_bound(
        0.0, 0, 100, 0, 0.0, 0.05
    )[2] == 1.0


def test_u_delta_is_one_minus_product():
    cl, el, ud = _confidence_bound(
        0.7, 80, 100, 80, 0.0, 0.05
    )

    assert ud == 1.0 - el * cl


def test_regression_ec_and_coverage_match_numpy_reference():
    uq, y_hat, y = _build_regression(seed = 43)
    alpha = 0.1
    eps = 0.2

    r = uq.uncertainty_at(y_hat, y, alpha, eps = eps, delta = 0.05)

    # Reconstruct the conformal intervals independently.
    N = uq._state.N
    scores = jnp.asarray(uq.conformity_scores_)

    q_hat = float(
        _q_hat_from_alpha(scores, jnp.asarray(N, dtype = jnp.int32), jnp.asarray(alpha, dtype = jnp.float64))
    )
    lower_cp = y_hat - q_hat
    upper_cp = y_hat + q_hat

    lower_tol = y_hat - eps
    upper_tol = y_hat + eps
    covered = (y >= lower_cp) & (y <= upper_cp)
    intersection = np.maximum(0.0, np.minimum(upper_cp, upper_tol) - np.maximum(lower_cp, lower_tol))
    interval_length = np.maximum(upper_cp - lower_cp, 1e-8)

    z = np.minimum(1.0, intersection / interval_length)
    coverage_ref = covered.mean()
    ec_ref = z[covered].mean() if covered.any() else 0.0

    assert r.n_valid == len(y)
    assert r.n_covered == covered.sum()
    assert r.n_succ == covered.sum()

    assert r.coverage == pytest.approx(coverage_ref)
    assert r.EC == pytest.approx(ec_ref)

    assert r.U_empirical == pytest.approx(1.0 - ec_ref*coverage_ref)

@pytest.mark.parametrize("alpha", [0.02, 0.1, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("delta", [0.2, 0.05, 0.001])
def test_regression_bound_orderings(alpha, delta):
    uq, y_hat, y = _build_regression(seed = 43, n_eval = 700)
    floor = 1.0/(uq._state.N+1)
    r = uq.uncertainty_at(y_hat, y, max(alpha, floor), eps = 0.2, delta = delta)

    assert r.delta == delta
    assert r.coverage == pytest.approx(r.n_covered/r.n_valid)
    assert r.n_succ == r.n_covered

    assert r.U_empirical == pytest.approx(1.0 - r.EC*r.coverage)

    assert 0.0 <= r.coverage_lower <= r.coverage
    assert 0.0 <= r.EC_lower <= r.EC

    assert 0.0 <= r.EC_lower <= r.EC

    assert r.U_delta >= r.U_empirical

    assert r.U_delta == pytest.approx(1.0 - r.EC_lower*r.coverage_lower)
    assert r.U_delta <= 1.0

def test_regression_u_delta_decreases_as_delta_grows():
    uq, y_hat, y = _build_regression(seed = 15, n_eval = 700)

    deltas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.7, 0.99]

    vals = [uq.uncertainty_at(y_hat, y, 0.2, eps = 0.2, delta = d).U_delta for d in deltas]

    assert all(
        a>b for a, b in itertools.pairwise(vals))

@pytest.mark.parametrize(
    "bad", [0, 0.0, 1, 1.0, -0.1, 1.5, np.nan, "x", [0.1]],
)
def test_regression_invalid_delta_raises(bad):
    uq, y_hat, y = _build_regression(seed = 4)

    with pytest.raises(ValueError, match = "delta"):
        uq.uncertainty_at(y_hat, y, 0.2, eps = 0.2, delta = bad)

def test_regression_invalid_delta_raises_even_for_nan_alpha():
    uq, y_hat, y = _build_regression(seed = 98)

    with pytest.raises(ValueError, match = "delta"):
        uq.uncertainty_at(y_hat, y, np.nan, eps = 0.2, delta = 2.0)

def test_regression_delta_none_preserves_core_values():
    uq, y_hat, y = _build_regression(seed = 45)
    r0 = uq.uncertainty_at(y_hat, y, 0.2, eps = 0.2)
    r1 = uq.uncertainty_at(y_hat, y, 0.2, eps = 0.2, delta = 0.05)

    assert (r0.delta, r0.coverage_lower, r0.EC_lower, r0.U_delta) == (None,)*4

    for field in ("U", "EC"):
        assert np.float64(getattr(r0, field)).tobytes() == np.float64(getattr(r1, field)).tobytes()

    assert r0[:5] == r1[:5]

    assert isinstance(r0, UncertaintyEvaluation)
    assert len(r0) == 11

def test_regression_nan_alpha_gives_nan_fields():
    uq, y_hat, y = _build_regression(seed=4)

    for delta in (None, 0.05):
        r = uq.uncertainty_at(
            y_hat, y, np.nan, eps=0.2, delta=delta
        )

        assert (r.n_covered, r.n_valid, r.n_succ) == (0, 0, 0)

        for field in ("U", "EC", "coverage", "U_empirical"):
            assert np.isnan(getattr(r, field))

        assert r.delta == delta

        for field in ("coverage_lower", "EC_lower", "U_delta"):
            value = getattr(r, field)
            assert value is None if delta is None else np.isnan(value)

def test_regression_delta_does_not_touch_jit_cache():
    uq, y_hat, y = _build_regression(seed=6)

    uq.uncertainty_at(y_hat, y, 0.2, eps=0.2)

    size = _evaluate_at_alpha._cache_size()

    for delta in (None, 0.05, 0.3):
        uq.uncertainty_at(
            y_hat, y, 0.2, eps=0.2, delta=delta
        )
        uq.uncertainty_at(
            y_hat, y, 0.4, eps=0.2, delta=delta
        )

    assert _evaluate_at_alpha._cache_size() == size
    
def test_regression_individual_bounds_monotonicity():
    uq, y_hat, y = _build_regression(seed = 15, n_eval = 700)
    deltas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.7, 0.99]

    results = [
        uq.uncertainty_at(y_hat, y, 0.2, eps = 0.2, delta = d)
        for d in deltas
    ]

    coverage_lower = [r.coverage_lower for r in results]
    EC_lower = [r.EC_lower for r in results]

    assert all(
        a <= b for a, b in itertools.pairwise(coverage_lower)
    )
    assert all(
        a <= b for a, b in itertools.pairwise(EC_lower)
    )

def test_regression_get_uncertainty_consistency():
    uq, y_hat, y = _build_regression(seed = 57)
    eps = 0.2

    with pytest.warns(TuningDataReuseWarning):
        U, alpha = uq.get_uncertainty(y_hat, y, max_iters = 30, eps = eps,
                                      target_ratio = 1.0)

    r = uq.uncertainty_at(y_hat,y, alpha, eps = eps, delta = None)

    assert np.isfinite(alpha)
    assert U == pytest.approx(r.U, rel = 1e-8, abs = 1e-10)


def test_regression_mismatched_input_shapes_raise():
    uq, y_hat, y = _build_regression(seed = 17)
    with pytest.raises(ValueError):
        uq.uncertainty_at(
            y_hat.reshape(-1, 1), y, alpha = 0.1, eps = 0.2
        )

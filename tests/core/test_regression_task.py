import jax.numpy as jnp
import numpy as np
import warnings

import pytest

from utrace.uncertaintyQuantifier import (
    _intersection_length, reg_risk_fn, _predict_sets, _evaluate_at_alpha
)

import utrace.uncertaintyQuantifier as uq

from utrace.scores import abs_error

@pytest.mark.parametrize(
    "lower_cp, upper_cp, lower_tol, upper_tol, expected",
    [
        # Complete overlap
        (0.0, 10.0, 0.0, 10.0, 10.0),
        # Tolerance interval inside prediction interval
        (0.0, 10.0, 2.0, 6.0, 4.0),
        # Prediction interval inside tolerance interval
        (2.0, 6.0, 0.0, 10.0, 4.0),
        # Partial overlap on the right
        (0.0, 5.0, 3.0, 8.0, 2.0),
        # Partial overlap on the left
        (3.0, 8.0, 0.0, 5.0, 2.0),
        # No overlap
        (0.0, 2.0, 3.0, 5.0, 0.0),
        # Interval touch at one endpoint
        (0.0, 3.0, 3.0, 5.0, 0.0),
    ],
)

def test_intersection_length(
    lower_cp, upper_cp, lower_tol, upper_tol, expected
):
    result = _intersection_length(
        jnp.array(lower_cp), jnp.array(upper_cp),
        jnp.array(lower_tol), jnp.array(upper_tol),
    )

    np.testing.assert_allclose(result, expected)

def test_reg_risk_fn():
    y_hat = jnp.array([5.0, 5.0, 5.0])
    y = jnp.array([5.0, 7.0, 12.0])
    q_hat = 5.0
    valid_mask = jnp.array([True, True, True])

    # Prediction intervals: [0, 10] for every observation.
    # Tolerance intervals: [3, 7] for every observation.
    aux_data = (2.0, 2.0)

    steering, efficiency = reg_risk_fn(
        y, y_hat, q_hat, valid_mask, abs_error, aux_data
    )

    # Observation 1: covered, overlap ratio = 4/10.
    # Observation 2: covered, overlap ratio = 4/10.
    # Observation 3: not covered.

    # Mean over covered observation = 0.4
    np.testing.assert_allclose(steering, 0.4)
    np.testing.assert_allclose(efficiency, 0.4)

def test_reg_risk_fn_no_covered_observations():
    y_hat = jnp.array([5.0, 5.0])
    y = jnp.array([20.0, 30.0])
    valid_mask = jnp.array([True, True])

    result = reg_risk_fn(y, y_hat, 5.0,
                         valid_mask, abs_error, (2.0, 2.0))

    np.testing.assert_allclose(result, (0.0, 0.0))


def test_predict_sets_regression():
    y_hat = jnp.array([2.0, 5.0, 8.0])
    q_hat = 1.5

    lower, upper = _predict_sets(
        y_hat, q_hat, score_fn = abs_error,
        task = "regression"
    )

    np.testing.assert_allclose(lower, [0.5, 3.5, 6.5])
    np.testing.assert_allclose(upper, [3.5, 6.5, 9.5])

def test_reg_risk_fn_ignores_invalid_observations():
    y_hat = jnp.array([5.0, 5.0, 5.0])
    y = jnp.array([5.0, 7.0, 5.0])

    valid_mask = jnp.array([True, False, True])

    sttering, efficiency = reg_risk_fn(
        y, y_hat, 5.0, valid_mask, abs_error,
        (2.0, 2.0)
    )

    # Observation 1 and 3 are valid and covered-
    # Both have overlap ratio 4/10 = 0.4.
    np.testing.assert_allclose(sttering, 0.4)
    np.testing.assert_allclose(efficiency, 0.4)

def test_evaluate_at_alpha_regression():
    alpha = 0.2
    y_hat = jnp.array([5.0, 5.0, 5.0])
    y = jnp.array([5.0, 7.0, 12.0])

    valid_mask = jnp.array([True, True, True])
    cs_padded = jnp.array([5.0, 5.0, 5.0])
    n_cs = 4

    result = _evaluate_at_alpha(alpha, y, y_hat, valid_mask,
                                cs_padded, n_cs, abs_error, reg_risk_fn,
                                (2.0, 2.0), "regression")
    U, EC, n_covered, n_valid, n_succ, sum_sq_dev = result

    # Prediction intervals: [0, 10].
    # Tolerance intervals: [3, 7].

    # First observation: covered, overlap ratio = 0.4
    # Second observation: covered, overalp_ratio =0.4
    # Third observation: not covered.
    np.testing.assert_allclose(U, 0.68)
    np.testing.assert_allclose(EC, 0.4)

    assert int(n_covered) == 2
    assert int(n_valid) == 3
    assert int(n_succ) == 2

    np.testing.assert_allclose(sum_sq_dev, 0.0)

def test_evaluate_At_alpha_regression_ignores_invalid():
    alpha = 0.2

    y_hat = jnp.array([5.0, 5.0, 5.0])
    y = jnp.array([5.0, 7.0, 5.0])

    valid_mask = jnp.array([True, False, True])

    cs_padded = jnp.array([5.0, 5.0, 5.0, 5.0])
    result = _evaluate_at_alpha(alpha, y, y_hat, valid_mask,
                                cs_padded, 4, abs_error,
                                reg_risk_fn, (2.0, 2.0),
                                "regression")
    U, EC, n_covered, n_valid, n_succ, sum_sq_dev = result

    np.testing.assert_allclose(U, 0.68)
    np.testing.assert_allclose(EC, 0.4)

    assert int(n_covered) == 2
    assert int(n_valid) == 2
    assert int(n_succ) == 2

    np.testing.assert_allclose(sum_sq_dev, 0.0)


# Test search uncertainty for regression

def run_mock_search(monkey_patch, metric_fn, *, n_cs = 9, max_iters = 30,
                    target_ratio = 0.8):
    """Run the regression alpha search with a controlled evaluation metric."""

    def fake_evaluate_alpha(alpha, y, smx, valid_mask, cs_padded, n_cs, score_fn,
                            risk_fn, aux_data):
        metric = metric_fn(alpha)
        ec = jnp.asarray(0.4, dtype = jnp.float64)
        return jnp.asarray(metric, dtype = jnp.float64), ec

    monkey_patch.setattr(uq, "_evaluate_alpha", fake_evaluate_alpha)

    # The JIT cache must not retain a trace made with a previous mock.
    uq._search_uncertainty.clear_cache()

    y = jnp.array([1.0, 2.0])
    y_hat = jnp.array([1.1, 2.1])
    valid_mask = jnp.array([True, True])

    cs_padded = jnp.full((n_cs,), 0.5)
    n_cs_array = jnp.asarray(n_cs, dtype = jnp.int32)

    return uq._search_uncertainty(y, y_hat, valid_mask, cs_padded, n_cs_array,
                                  max_iters = max_iters, score_fn = uq.abs_error,
                                  target_ratio = target_ratio, aux_data = (0.1, 0.1),
                                  risk_fn = uq.reg_risk_fn, direction = -1.0)

def test_search_floor_limited(monkeypatch):
    n_cs = 9
    floor = 1.0/(n_cs + 1)
    alpha, U, status = run_mock_search(monkeypatch, metric_fn = lambda alpha:0.9,
                                       n_cs = n_cs)

    np.testing.assert_allclose(alpha, floor)
    np.testing.assert_allclose(U, 1.0 - 0.4*(1.0-floor))

    assert int(status) == uq._STATUS_FLOOR_LIMITED

def test_search_returns_feasible_alpha(monkeypatch):
    n_cs = 9
    alpha, U, status = run_mock_search(
        monkeypatch, metric_fn = lambda a: jnp.where(a >= 0.4, 0.9, 0.2),
    )
    assert 0.4 <= float(alpha) <= 1.0
    assert float(alpha) > 1.0/ (n_cs + 1)

    np.testing.assert_allclose(U, 1.0 - 0.4*(1.0-alpha))
    assert int(status) == uq._STATUS_CONVERGED

def test_search_infeasible(monkeypatch):
    alpha, U, status = run_mock_search(
        monkeypatch, metric_fn = lambda alpha: 0.2,
    )
    np.testing.assert_allclose(alpha, 1.0)
    np.testing.assert_allclose(U, 1.0)

    assert int(status) == uq._STATUS_INFEASIBLE


# Test SearchStatus
@pytest.mark.parametrize(
    "status_code, expected_status, warns",
    [
        (uq._STATUS_CONVERGED, uq.SearchStatus.CONVERGED, False),
        (uq._STATUS_FLOOR_LIMITED, uq.SearchStatus.FLOOR_LIMITED, True),
        (uq._STATUS_INFEASIBLE, uq.SearchStatus.INFEASIBLE, True),
    ]
)

def test_search_status_propagation(monkeypatch, status_code, expected_status, warns):
    uncert_q = uq.UncertaintyQuantifier(N=10, score = "abs_error")
    y_cal = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat_cal = np.array([1.1, 2.1, 3.1, 4.1])

    uncert_q.calibrate(y_hat_cal, y_cal)

    def fake_search(*arg, **kwargs):
        return jnp.array(0.25), jnp.asarray(0.7), jnp.asarray(status_code, dtype = jnp.int32)

    monkeypatch.setattr(uq, "_search_uncertainty", fake_search)

    if warns:
        with pytest.warns(uq.SearchStatusWarning):
            alpha = uncert_q.tune_alpha(y_hat_cal, y_cal,eps = 0.1, target_ratio=0.8)

    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            alpha = uncert_q.tune_alpha(y_hat_cal, y_cal, eps = 1.0, target_ratio = 0.8)

        assert not any(issubclass(w.category, uq.SearchStatusWarning) for w in record)

    assert alpha == pytest.approx(0.25)
    assert uncert_q.search_status_ is expected_status

def test_reset_clears_search_status():
    uncert_q = uq.UncertaintyQuantifier(score="abs_error")

    uncert_q.search_status_ = uq.SearchStatus.CONVERGED

    uncert_q.reset()

    assert uncert_q.search_status_ is None

def test_regression_tune_alpha_runs_real_search():
    rng = np.random.default_rng(43)

    n_cal = 100
    n_tune = 100

    y_cal = rng.normal(size = n_cal)
    y_hat_cal = y_cal + rng.normal(scale = 0.2, size = n_cal)

    y_tune  = rng.normal(size = n_tune)
    y_hat_tune = y_tune + rng.normal(scale = 0.2, size = n_tune)

    uncert_q = uq.UncertaintyQuantifier(N = n_cal, score = "abs_error")
    uncert_q.calibrate(y_hat_cal, y_cal)

    with pytest.warns(
        uq.SearchStatusWarning) if False else pytest.MonkeyPatch.context():
        alpha = uncert_q.tune_alpha(y_hat_tune, y_tune, max_iters = 30,
                                    eps = 0.1, target_ratio = 0.8)

    assert np.isfinite(alpha)
    assert 1.0/(n_cal + 1) <= alpha <= 1.0

    assert uncert_q.search_status_ in (
        uq.SearchStatus.CONVERGED,
        uq.SearchStatus.FLOOR_LIMITED,
        uq.SearchStatus.INFEASIBLE
    )


def test_regression_search_moves_in_correct_direction(monkeypatch):
    def metric_fn(alpha):
        # Overlap decreases as alpha increases.
        return alpha

    alpha, U, status = run_mock_search(
        monkeypatch,
        metric_fn,
        n_cs=9,
        max_iters=30,
    )

    assert alpha == pytest.approx(0.8, abs = 1e-5)
    assert status == uq._STATUS_CONVERGED
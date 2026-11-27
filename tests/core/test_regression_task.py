import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

import utrace.uncertaintyQuantifier as uq
from utrace.scores import AbsResidual, TolParams
from utrace.scores.objects import _intersection_length
from utrace.uncertaintyQuantifier import _evaluate_at_alpha, _predict_sets


def _params(tol, target=1.0):
    """`AbsResidual` per-call parameters, as `resolve_params` builds them (float64 leaves)."""
    return TolParams(jnp.asarray(tol, dtype=jnp.float64), jnp.asarray(target, dtype=jnp.float64))

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
    steering, efficiency = AbsResidual().evaluate(
        y, y_hat, q_hat, valid_mask, _params(2.0)
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

    result = AbsResidual().evaluate(y, y_hat, 5.0, valid_mask, _params(2.0))

    np.testing.assert_allclose(result, (0.0, 0.0))


def test_predict_sets_regression():
    y_hat = jnp.array([2.0, 5.0, 8.0])
    q_hat = 1.5

    lower, upper = _predict_sets(y_hat, q_hat, AbsResidual())

    np.testing.assert_allclose(lower, [0.5, 3.5, 6.5])
    np.testing.assert_allclose(upper, [3.5, 6.5, 9.5])

def test_reg_risk_fn_ignores_invalid_observations():
    y_hat = jnp.array([5.0, 5.0, 5.0])
    y = jnp.array([5.0, 7.0, 5.0])

    valid_mask = jnp.array([True, False, True])

    sttering, efficiency = AbsResidual().evaluate(
        y, y_hat, 5.0, valid_mask, _params(2.0)
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
                                cs_padded, n_cs, AbsResidual(), _params(2.0))
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
                                cs_padded, 4, AbsResidual(), _params(2.0))
    U, EC, n_covered, n_valid, n_succ, sum_sq_dev = result

    np.testing.assert_allclose(U, 0.68)
    np.testing.assert_allclose(EC, 0.4)

    assert int(n_covered) == 2
    assert int(n_valid) == 2
    assert int(n_succ) == 2

    np.testing.assert_allclose(sum_sq_dev, 0.0)


# Test search uncertainty for regression

@dataclass(frozen=True)
class FakeScore(AbsResidual):
    """`AbsResidual` whose search metric is a controlled function of `q_hat` (EC fixed at 0.4)."""
    metric_fn: Callable = None

    def evaluate(self, y, output, q_hat, valid_mask, params):
        return (jnp.asarray(self.metric_fn(q_hat), dtype=jnp.float64),
                jnp.asarray(0.4, dtype=jnp.float64))


def run_mock_search(metric_fn, *, n_cs = 9, max_iters = 30, target_ratio = 0.8):
    """Run the regression alpha search with a controlled search metric.

    The calibration scores are the grid `(i + 1) / n_cs`, so `q_hat` is close to `1 - alpha`.
    """
    y = jnp.array([1.0, 2.0])
    y_hat = jnp.array([1.1, 2.1])
    valid_mask = jnp.array([True, True])

    cs_padded = (jnp.arange(n_cs, dtype = jnp.float64) + 1.0) / n_cs
    n_cs_array = jnp.asarray(n_cs, dtype = jnp.int32)

    return uq._search_uncertainty(y, y_hat, valid_mask, cs_padded, n_cs_array,
                                  max_iters, FakeScore(metric_fn = metric_fn),
                                  _params(0.1, target_ratio))

def test_search_floor_limited():
    n_cs = 9
    floor = 1.0/(n_cs + 1)
    alpha, U, status = run_mock_search(metric_fn = lambda q_hat: 0.9, n_cs = n_cs)

    np.testing.assert_allclose(alpha, floor)
    np.testing.assert_allclose(U, 1.0 - 0.4*(1.0-floor))

    assert int(status) == uq._STATUS_FLOOR_LIMITED

def test_search_returns_feasible_alpha():
    n_cs = 9
    # Feasible where q_hat <= 0.6, i.e. alpha >= 0.4 on this grid.
    alpha, U, status = run_mock_search(
        metric_fn = lambda q_hat: jnp.where(q_hat <= 0.6, 0.9, 0.2),
    )
    q_hat = uq._q_hat_from_alpha((jnp.arange(n_cs, dtype = jnp.float64) + 1.0) / n_cs,
                                 jnp.int32(n_cs), jnp.float64(alpha))
    assert float(q_hat) <= 0.6
    assert 0.39 <= float(alpha) <= 1.0
    assert float(alpha) > 1.0/ (n_cs + 1)

    np.testing.assert_allclose(U, 1.0 - 0.4*(1.0-alpha))
    assert int(status) == uq._STATUS_CONVERGED

def test_search_infeasible():
    alpha, U, status = run_mock_search(metric_fn = lambda q_hat: 0.2)
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
            alpha = uncert_q.tune_alpha(y_hat_cal, y_cal, tol = 0.1, target_ratio=0.8)

    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            alpha = uncert_q.tune_alpha(y_hat_cal, y_cal, tol = 1.0, target_ratio = 0.8)

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

    alpha = uncert_q.tune_alpha(y_hat_tune, y_tune, max_iters = 30,
                                tol = 0.1, target_ratio = 0.8)

    assert np.isfinite(alpha)
    assert 1.0/(n_cal + 1) <= alpha <= 1.0

    assert uncert_q.search_status_ in (
        uq.SearchStatus.CONVERGED,
        uq.SearchStatus.FLOOR_LIMITED,
        uq.SearchStatus.INFEASIBLE
    )


def test_regression_search_moves_in_correct_direction():
    def metric_fn(q_hat):
        # Overlap grows as alpha grows (q_hat shrinks): here metric ~ alpha on the grid.
        return 1.0 - q_hat

    alpha, _U, status = run_mock_search(
        metric_fn,
        n_cs=999,
        max_iters=30,
    )

    # Smallest alpha with 1 - q_hat >= 0.8, up to a few steps of the 1/999 grid (the quantile
    # index is one order statistic above the canonical one; see CONTRIBUTING.md, "Departures").
    assert alpha == pytest.approx(0.8, abs = 5e-3)
    assert status == uq._STATUS_CONVERGED

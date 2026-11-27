"""`Score.accepted`: the acceptance indicator `1{y in B}` of each score. New file."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utrace import LAC, AbsResidual, Score
from utrace.scores.objects import TolParams


def test_lac_accepted_is_argmax_equals_label():
    smx = jnp.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.4, 0.5, 0.1], [0.2, 0.2, 0.6]])
    y = jnp.asarray([0, 1, 1, 2], dtype=jnp.int32)
    got = np.asarray(LAC().accepted(y, smx, ()))
    np.testing.assert_array_equal(got, [True, False, True, True])
    assert got.dtype == bool and got.shape == (4,)


def test_lac_accepted_ties_follow_argmax():
    smx = jnp.asarray([[0.5, 0.5], [0.5, 0.5]])
    got = np.asarray(LAC().accepted(jnp.asarray([0, 1]), smx, ()))
    np.testing.assert_array_equal(got, [True, False])      # argmax takes the first maximum


def test_lac_accepted_random_against_numpy():
    rng = np.random.default_rng(0)
    smx = rng.dirichlet(np.ones(7), 500)
    y = rng.integers(0, 7, 500)
    got = np.asarray(LAC().accepted(jnp.asarray(y), jnp.asarray(smx), ()))
    np.testing.assert_array_equal(got, smx.argmax(1) == y)


def _tol(t):
    return TolParams(tol=jnp.float64(t), target=jnp.float64(1.0))


def test_abs_residual_accepted_closed_window():
    y_hat = jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0])
    y = jnp.asarray([0.25, -0.25, 0.3, 1.0, 0.7])
    got = np.asarray(AbsResidual().accepted(y, y_hat, _tol(0.25)))
    np.testing.assert_array_equal(got, [True, True, False, True, False])   # |y - y_hat| <= tol, edges included
    assert got.dtype == bool


def test_abs_residual_accepted_random_against_numpy():
    rng = np.random.default_rng(1)
    y, yh = rng.normal(size=400), rng.normal(size=400)
    for t in (0.1, 0.5, 2.0):
        got = np.asarray(AbsResidual().accepted(jnp.asarray(y), jnp.asarray(yh), _tol(t)))
        np.testing.assert_array_equal(got, np.abs(y - yh) <= t)


def test_accepted_traces_and_tol_is_not_static():
    f = jax.jit(lambda y, yh, p: AbsResidual().accepted(y, yh, p))
    y, yh = jnp.asarray([0.0, 1.0]), jnp.asarray([0.2, 0.2])
    np.testing.assert_array_equal(np.asarray(f(y, yh, _tol(0.3))), [True, False])
    np.testing.assert_array_equal(np.asarray(f(y, yh, _tol(0.9))), [True, True])
    assert f._cache_size() == 1


def test_default_accepted_raises_naming_uncertainty_budget():
    @dataclass(frozen=True)
    class Minimal(LAC):
        name = "minimal"

        def accepted(self, y, output, params):        # restore the base behaviour
            return Score.accepted(self, y, output, params)

    with pytest.raises(NotImplementedError, match=r"score 'minimal' does not implement accepted.*uncertainty_budget"):
        Minimal().accepted(jnp.zeros(2, dtype=jnp.int32), jnp.ones((2, 2)) / 2, ())


def test_accepted_is_not_abstract():
    assert "accepted" not in Score.__abstractmethods__

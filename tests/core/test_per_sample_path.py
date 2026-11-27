"""The per-sample path of `uncertainty_at`: `_evaluate_samples_at_alpha`, used only for a bound whose
`needs_values` is True. New file.
"""
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from utrace import UncertaintyQuantifier
from utrace.bounds import Bound, Hoeffding
from utrace.uncertaintyQuantifier import (
    _ensure_sorted,
    _evaluate_at_alpha,
    _evaluate_samples_at_alpha,
)
from utrace.utils import _bucket_size

K = 6


@dataclass(frozen=True)
class _Recording(Bound):
    """A test bound that needs the values and records what it received (Hoeffding's number)."""
    name: ClassVar[str] = "recording"
    needs_values: ClassVar[bool] = True
    seen: list = None

    def lower(self, stats, level):
        self.seen.append(stats)
        return Hoeffding().lower(stats, level)


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _clf(classes=None, seed=0, n_eval=777):
    rng = np.random.default_rng(seed)
    cy, cs = _interior(rng, 1500)
    uq = UncertaintyQuantifier(N=1500, classes=classes)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, n_eval)
    return uq, ey, es


def _reg(seed=0, n_eval=611):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=1000 + n_eval)
    yh = y + rng.normal(scale=0.3, size=y.shape)
    uq = UncertaintyQuantifier(N=1000, score="abs_error")
    uq.calibrate(yh[:1000], y[:1000])
    return uq, y[1000:], yh[1000:]


def _args(uq, es, ey, alpha, params=()):
    """The padded arguments `uncertainty_at` passes, rebuilt here (marginal case)."""
    B = len(ey)
    size = _bucket_size(B)
    y_dtype = np.dtype(uq.label_dtype_)
    yp = np.zeros(size, dtype=y_dtype)
    pp = np.zeros((size,) + es.shape[1:])
    m = np.zeros(size, dtype=bool)
    yp[:B], pp[:B], m[:B] = ey, es, True
    uq._state = _ensure_sorted(uq._state)
    return (jnp.asarray(np.float64(alpha)), jnp.asarray(yp), jnp.asarray(pp), jnp.asarray(m),
            uq._state.conformity_scores, jnp.int32(uq._state.N), uq.score_, params)


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5])
def test_values_reproduce_ec_and_sum_sq_dev(alpha):
    uq, ey, es = _clf()
    a = _args(uq, es, ey, alpha)
    _, EC, n_cov, _, n_succ, ssd = _evaluate_at_alpha(*a)
    covered, succ, z = (np.asarray(t) for t in _evaluate_samples_at_alpha(*a))
    assert int(succ.sum()) == int(n_succ) and int(covered.sum()) == int(n_cov)
    v = z[succ]
    assert v.sum() == pytest.approx(float(EC) * int(n_succ), rel=1e-12)
    assert ((v - v.mean()) ** 2).sum() == pytest.approx(float(ssd), rel=1e-9, abs=1e-12)
    assert np.all(z[~succ] == 0.0)
    assert not succ[len(ey):].any() and not covered[len(ey):].any()   # padding never counted


def test_values_reproduce_ec_regression():
    uq, ey, eyh = _reg()
    params = uq.score_.resolve_params(tol=0.2, target_ratio=None, where="test")
    a = _args(uq, eyh, ey, 0.2, params)
    _, EC, _, _, n_succ, _ = _evaluate_at_alpha(*a)
    _, succ, z = (np.asarray(t) for t in _evaluate_samples_at_alpha(*a))
    assert z[succ].sum() == pytest.approx(float(EC) * int(n_succ), rel=1e-12)


@pytest.mark.parametrize("classes", [None, [1, 4]])
def test_needs_values_bound_receives_the_unpadded_covered_values(classes):
    uq, ey, es = _clf(classes=classes)
    rec = _Recording(seen=[])
    r = uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound=rec)
    (stats,) = rec.seen
    assert stats.values is not None and stats.values.dtype == np.float64
    assert stats.values.shape == (r.n_succ,) == (stats.n,)
    assert np.all((stats.values > 0) & (stats.values <= 1))
    assert stats.values.sum() == pytest.approx(float(r.EC) * r.n_succ, rel=1e-12)
    ref = uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound=Hoeffding())
    assert np.float64(r.EC_lower).tobytes() == np.float64(ref.EC_lower).tobytes()


def test_values_are_in_evaluation_order():
    uq, ey, es = _clf(n_eval=300)
    rec = _Recording(seen=[])
    uq.uncertainty_at(es, ey, 0.3, delta=0.05, bound=rec)
    uq.alpha = 0.3                       # predict uses the instance's alpha; same q_hat
    _, S = uq.predict(es)
    S = np.asarray(S)
    sz = S.sum(1)
    cov = S[np.arange(len(ey)), ey] & (sz > 0)
    expected = 1.0 / sz[cov]
    np.testing.assert_array_equal(rec.seen[0].values, expected)


def test_core_fields_identical_with_or_without_values():
    uq, ey, es = _clf()
    r0 = uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound=Hoeffding())
    r1 = uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound=_Recording(seen=[]))
    for f in ("U", "EC", "n_covered", "n_valid", "n_succ", "coverage", "U_empirical", "coverage_lower"):
        assert np.float64(getattr(r0, f)).tobytes() == np.float64(getattr(r1, f)).tobytes(), f


def test_no_covered_sample_does_not_call_the_values_path():
    uq, ey, es = _clf()
    rec = _Recording(seen=[])
    # alpha = 1.0: q_hat is the minimum calibration score, so (almost) nothing is covered
    r = uq.uncertainty_at(es, ey, 1.0, delta=0.05, bound=rec)
    if r.n_succ == 0:
        assert rec.seen == [] and r.EC_lower == 0.0
    else:
        assert len(rec.seen) == 1


def test_default_path_never_compiles_the_values_function():
    uq, ey, es = _clf()
    _evaluate_samples_at_alpha.clear_cache()
    uq.uncertainty_at(es, ey, 0.2, delta=0.05)
    uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound="hoeffding")
    uq.uncertainty_at(es, ey, 0.2)
    assert _evaluate_samples_at_alpha._cache_size() == 0


def test_values_path_does_not_touch_the_default_cache():
    uq, ey, es = _clf()
    uq.uncertainty_at(es, ey, 0.2, delta=0.05)
    size = _evaluate_at_alpha._cache_size()
    for alpha in (0.1, 0.2, 0.4):
        uq.uncertainty_at(es, ey, alpha, delta=0.05, bound=_Recording(seen=[]))
    assert _evaluate_at_alpha._cache_size() == size
    assert _evaluate_samples_at_alpha._cache_size() >= 1

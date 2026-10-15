"""Tests for `uncertainty_at(..., delta=...)`: the Clopper-Pearson / Hoeffding bound on `U`.

New file only. The Monte Carlo check of the guarantee is in
`test_confidence_bound_montecarlo.py`; label validation in `test_label_validation.py`.
"""
import itertools
import math
import warnings

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

from utrace import UncertaintyEvaluation, UncertaintyQuantifier
from utrace.uncertaintyQuantifier import _confidence_bound, _evaluate_at_alpha

K = 10


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _build(N=300, classes=None, seed=0, n_eval=400):
    rng = np.random.default_rng(seed)
    cy, cs = _interior(rng, N * (5 if classes else 1))
    uq = UncertaintyQuantifier(N=N * 5, classes=classes)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, n_eval)
    return uq, ey, es


# ----- 1. Clopper-Pearson ---------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 10, 100, 1000, 10**5, 10**7])
@pytest.mark.parametrize("delta", [0.05, 0.01, 0.001])
def test_coverage_lower_matches_scipy_ppf(n, delta):
    for k in sorted({1, 2, max(1, n // 2), max(1, n - 1), n}):
        if not 1 <= k <= n:
            continue
        cl, _, _ = _confidence_bound(0.5, k, n, max(k, 1), delta)
        ref = scipy_beta.ppf(delta / 2, k, n - k + 1)
        assert cl == pytest.approx(ref, rel=1e-12, abs=0)


@pytest.mark.parametrize("n", [1, 7, 100, 10**4])
@pytest.mark.parametrize("delta", [0.2, 0.05, 0.001])
def test_coverage_lower_closed_forms(n, delta):
    assert _confidence_bound(0.5, 0, n, 0, delta)[0] == 0.0
    cl = _confidence_bound(0.5, n, n, n, delta)[0]
    assert cl == pytest.approx((delta / 2) ** (1 / n), rel=1e-12)


# ----- 2. Hoeffding ----------------------------------------------------------------

def test_ec_lower_hoeffding_hand_computed():
    # sqrt(ln(2/0.05) / (2*100)) = sqrt(ln 40 / 200) = 0.13581015157406195
    _, ec_lower, _ = _confidence_bound(0.6, 50, 100, 100, 0.05)
    assert ec_lower == pytest.approx(0.6 - 0.13581015157406195, rel=1e-14)
    # delta=0.1, n_succ=50: sqrt(ln 20 / 100) = 0.17308183826022852
    _, ec_lower, _ = _confidence_bound(0.6, 50, 100, 50, 0.1)
    assert ec_lower == pytest.approx(0.6 - 0.17308183826022852, rel=1e-14)


def test_ec_lower_clipped_at_zero_and_zero_succ():
    assert _confidence_bound(0.05, 50, 100, 100, 0.05)[1] == 0.0       # margin 0.136 > EC
    assert _confidence_bound(0.0, 0, 100, 0, 0.05)[1] == 0.0           # n_succ == 0
    # Both zero: U_delta is exactly the trivial bound.
    assert _confidence_bound(0.0, 0, 100, 0, 0.05)[2] == 1.0


def test_u_delta_is_one_minus_product():
    cl, el, ud = _confidence_bound(0.7, 80, 100, 80, 0.05)
    assert ud == 1.0 - el * cl


# ----- 3. orderings on real evaluations -------------------------------------------

@pytest.mark.parametrize("classes", [None, [3]])
@pytest.mark.parametrize("alpha", [0.02, 0.1, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("delta", [0.2, 0.05, 0.001])
def test_bound_orderings(classes, alpha, delta):
    uq, ey, es = _build(classes=classes, seed=3, n_eval=700)
    floor = 1.0 / (uq._state.N + 1)
    r = uq.uncertainty_at(es, ey, max(alpha, floor), delta=delta)
    assert r.delta == delta
    assert r.coverage == r.n_covered / r.n_valid
    assert r.U_empirical == 1.0 - r.EC * r.coverage
    assert 0.0 <= r.coverage_lower <= r.coverage
    assert 0.0 <= r.EC_lower <= r.EC
    assert r.U_delta >= r.U_empirical
    assert r.U_delta == 1.0 - r.EC_lower * r.coverage_lower
    assert r.U_delta <= 1.0


def test_u_delta_decreases_as_delta_grows():
    uq, ey, es = _build(seed=5, n_eval=700)
    deltas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.7, 0.99]
    vals = [uq.uncertainty_at(es, ey, 0.2, delta=d).U_delta for d in deltas]
    assert all(a > b for a, b in itertools.pairwise(vals))


@pytest.mark.parametrize("bad", [0, 0.0, 1, 1.0, -0.1, 1.5, np.nan, "x", [0.1]])
def test_invalid_delta_raises(bad):
    uq, ey, es = _build(seed=1)
    with pytest.raises(ValueError, match="delta"):
        uq.uncertainty_at(es, ey, 0.2, delta=bad)


def test_invalid_delta_raises_even_for_nan_alpha():
    uq, ey, es = _build(seed=1)
    with pytest.raises(ValueError, match="delta"):
        uq.uncertainty_at(es, ey, np.nan, delta=2.0)


# ----- delta=None leaves the original fields as they were ---------------------------

def test_delta_none_has_no_bound_fields_and_same_core_values():
    uq, ey, es = _build(seed=2)
    r0 = uq.uncertainty_at(es, ey, 0.2)
    r1 = uq.uncertainty_at(es, ey, 0.2, delta=0.05)
    assert (r0.delta, r0.coverage_lower, r0.EC_lower, r0.U_delta) == (None,) * 4
    assert r0.coverage == r0.n_covered / r0.n_valid and r0.U_empirical is not None
    for f in ("U", "EC"):
        assert np.float64(getattr(r0, f)).tobytes() == np.float64(getattr(r1, f)).tobytes()
    assert r0[:5] == r1[:5]
    assert isinstance(r0, UncertaintyEvaluation) and len(r0) == 11


# ----- 5. nan propagation ------------------------------------------------------------

def test_nan_when_eval_set_lacks_the_class_group():
    uq, ey, es = _build(classes=[3], seed=4)
    keep = ey != 3
    for delta in (None, 0.05):
        r = uq.uncertainty_at(es[keep], ey[keep], 0.2, delta=delta)
        assert r.n_valid == 0 and np.isnan(r.U)
        assert np.isnan(r.coverage) and np.isnan(r.U_empirical)
        assert r.delta == delta
        for f in (r.coverage_lower, r.EC_lower, r.U_delta):
            assert (f is None) if delta is None else np.isnan(f)


def test_nan_alpha_gives_nan_floats_and_zero_counts():
    uq, ey, es = _build(seed=4)
    for delta in (None, 0.05):
        r = uq.uncertainty_at(es, ey, np.nan, delta=delta)
        assert (r.n_covered, r.n_valid, r.n_succ) == (0, 0, 0)
        for f in (r.U, r.EC, r.coverage, r.U_empirical):
            assert np.isnan(f)
        assert r.delta == delta
        for f in (r.coverage_lower, r.EC_lower, r.U_delta):
            assert (f is None) if delta is None else np.isnan(f)


# ----- 9. compile cache --------------------------------------------------------------

def test_delta_does_not_touch_the_jit_cache():
    uq, ey, es = _build(seed=6)
    uq.uncertainty_at(es, ey, 0.2)
    size = _evaluate_at_alpha._cache_size()
    for delta in (None, 0.05, 0.3):
        uq.uncertainty_at(es, ey, 0.2, delta=delta)
        uq.uncertainty_at(es, ey, 0.4, delta=delta)
    assert _evaluate_at_alpha._cache_size() == size


def test_get_uncertainty_return_value_unchanged():
    uq, ey, es = _build(seed=7)
    ty, ts = _interior(np.random.default_rng(99), 300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=ey)
    assert isinstance(out, tuple) and len(out) == 2
    assert math.isfinite(out[0]) and math.isfinite(out[1])

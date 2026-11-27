"""The single-estimand bounds `V` and `W` of `uncertainty_budget`. New file.

`V = pi 1{y in C}` (the paper's `U* = 1 - E[V]`), `W = V + 1{y in B, y not in C}` (keeps the dropped
term). Each is one lower confidence bound at the full `delta`. The Monte Carlo validity check is in
`test_single_estimand_montecarlo.py`.
"""
import math

import numpy as np
import pytest

from utrace import SingleEstimandBound, UncertaintyQuantifier
from utrace.bounds import Betting, EmpiricalBernstein, Hoeffding, SampleStats, Tightest
from utrace.uncertaintyQuantifier import _budget_samples_at_alpha

K = 10
DELTA = 0.05


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _clf(seed=0, classes=None, n_eval=1500):
    rng = np.random.default_rng(seed)
    cy, cs = _interior(rng, 1500)
    uq = UncertaintyQuantifier(N=1500, classes=classes)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, n_eval)
    return uq, ey, es


def _reg(seed=0, n_eval=2000):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=1000 + n_eval)
    yh = y + rng.normal(scale=0.3, size=y.shape)
    uq = UncertaintyQuantifier(N=1000, score="abs_error")
    uq.calibrate(yh[:1000], y[:1000])
    return uq, y[1000:], yh[1000:]


def _per_sample_lac(uq, es, ey, alpha, classes=None):
    uq.alpha = alpha
    _, S = uq.predict(es)
    S = np.asarray(S)
    valid = np.isin(ey, classes) if classes is not None else np.ones(len(ey), bool)
    sz = S.sum(1)
    cov = S[np.arange(len(ey)), ey] & valid
    acc = (es.argmax(1) == ey) & valid
    V = np.where(cov, 1.0 / np.maximum(sz, 1), 0.0)
    W = V + (acc & ~cov)
    return V[valid], W[valid], valid


@pytest.mark.parametrize("classes", [None, [0, 3, 8]])
@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.6])
def test_lac_V_and_W_against_numpy(classes, alpha):
    uq, ey, es = _clf(1, classes=classes)
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    V, W, _ = _per_sample_lac(uq, es, ey, alpha, classes)
    assert b.V.n == b.W.n == len(V) == b.evaluation.n_valid
    assert b.V.mean == pytest.approx(V.mean(), abs=1e-14)
    assert b.W.mean == pytest.approx(W.mean(), abs=1e-14)
    for est, x in ((b.V, V), (b.W, W)):
        ref = max(0.0, Tightest([Hoeffding(), EmpiricalBernstein()]).lower(
            SampleStats(n=len(x), mean=x.mean(), sum_sq_dev=((x - x.mean()) ** 2).sum()), DELTA))
        assert est.lower == pytest.approx(ref, abs=1e-12)
        assert est.U_delta == 1.0 - est.lower


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.6])
def test_values_lie_in_unit_interval_and_events_are_disjoint(alpha):
    uq, ey, es = _clf(2)
    V, W, _ = _per_sample_lac(uq, es, ey, alpha)
    assert np.all((W >= 0) & (W <= 1)) and np.all((V >= 0) & (V <= 1)) and np.all(W >= V)


def test_lac_accepted_not_covered_is_empty_set_with_correct_answer():
    uq, ey, es = _clf(3)
    uq.alpha = 0.3
    _, S = uq.predict(es)
    S = np.asarray(S)
    sz = S.sum(1)
    cov = S[np.arange(len(ey)), ey]
    acc = es.argmax(1) == ey
    np.testing.assert_array_equal(acc & ~cov, (sz == 0) & acc)


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.6])
def test_point_identities(alpha):
    uq, ey, es = _clf(4)
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    ev = b.evaluation
    anc = b.component("accepted_not_covered").value
    assert abs(b.W.mean - (float(ev.coverage) * float(ev.EC) + anc)) < 1e-12
    assert abs((1.0 - b.V.mean) - float(ev.U_empirical)) < 1e-12
    assert b.U_delta_W >= 1.0 - b.W.mean and b.U_delta_V >= 1.0 - b.V.mean


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.6])
@pytest.mark.parametrize("bound", [None, "hoeffding", Betting()])
def test_components_sum_to_their_u_delta(alpha, bound):
    uq, ey, es = _clf(5)
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, bound=bound)
    assert [c.name for c in b.V.components] == ["error", "accepted_not_covered", "selection", "margin"]
    assert [c.name for c in b.W.components] == ["error", "selection", "margin"]
    assert abs(b.V.total - b.U_delta_V) < 1e-12 and abs(b.W.total - b.U_delta_W) < 1e-12
    assert b.V.components[-1].kind == "statistical" and b.V.components[-1].value >= 0
    assert b.U_delta_V - b.U_delta_W == pytest.approx(b.component("accepted_not_covered").value
                                                      + (b.V.components[-1].value - b.W.components[-1].value), abs=1e-12)


def test_full_delta_not_split():
    uq, ey, es = _clf(6)
    b = uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, bound="hoeffding")
    V, _, _ = _per_sample_lac(uq, es, ey, 0.3)
    assert b.V.lower == pytest.approx(max(0.0, V.mean() - math.sqrt(math.log(1 / DELTA) / (2 * len(V)))), abs=1e-12)
    assert "at delta," in b.V.method and "hoeffding" in b.V.method


def test_betting_receives_the_values_in_evaluation_order():
    uq, ey, es = _clf(7)
    b = uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, bound=Betting(order="given"))
    V, W, _ = _per_sample_lac(uq, es, ey, 0.3)
    bet = Betting(order="given")
    assert b.V.lower == max(0.0, bet.lower(SampleStats(n=len(V), mean=V.mean(), values=V), DELTA))
    assert b.W.lower == max(0.0, bet.lower(SampleStats(n=len(W), mean=W.mean(), values=W), DELTA))


def test_values_path_only_for_needs_values_bound():
    uq, ey, es = _clf(8)
    _budget_samples_at_alpha.clear_cache()
    uq.uncertainty_budget(es, ey, 0.3, delta=DELTA)
    uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, bound="hoeffding")
    assert _budget_samples_at_alpha._cache_size() == 0
    uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, bound="betting")
    assert _budget_samples_at_alpha._cache_size() == 1


def test_types_and_naming():
    uq, ey, es = _clf(9)
    b = uq.uncertainty_budget(es, ey, 0.3, delta=DELTA)
    assert isinstance(b.V, SingleEstimandBound) and b.V.estimand == "V" and b.W.estimand == "W"
    assert "paper" in b.V.description and "drops" in b.W.description


def test_nothing_evaluated():
    uq, ey, es = _clf(10)
    b = uq.uncertainty_budget(es, ey, np.nan, delta=DELTA)
    assert b.V is None and b.W is None and np.isnan(b.U_delta_V) and np.isnan(b.U_delta_W)


# ----- regression ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, tol", [(0.5, 0.6), (0.2, 0.2), (0.05, 0.1)])
def test_regression_V_and_W(alpha, tol):
    uq, y, yh = _reg()
    b = uq.uncertainty_budget(yh, y, alpha, tol=tol, delta=DELTA)
    uq.alpha = alpha
    lo, hi = (np.asarray(a) for a in uq.predict(yh))
    cov = (y >= lo) & (y <= hi)
    acc = np.abs(y - yh) <= tol
    L = hi - lo
    pi = np.where(L > 0, np.minimum(1.0, np.clip(np.minimum(hi, yh + tol) - np.maximum(lo, yh - tol), 0, None)
                                    / np.where(L > 0, L, 1)), 1.0)
    V = np.where(cov, pi, 0.0)
    W = V + (acc & ~cov)
    assert b.V.mean == pytest.approx(V.mean(), abs=1e-12)
    assert b.W.mean == pytest.approx(W.mean(), abs=1e-12)
    assert abs(b.V.total - b.U_delta_V) < 1e-12 and abs(b.W.total - b.U_delta_W) < 1e-12
    if tol >= float(L[0]) / 2:    # C inside B: W = 1{|y - y_hat| <= tol}, so 1 - mean W is the error
        assert 1.0 - b.W.mean == pytest.approx(b.error, abs=1e-12)

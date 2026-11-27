"""`UncertaintyQuantifier.uncertainty_budget`: the exact split of `U_delta`. New file.

For `LAC` the terms are checked against `_nb_decompose`, a copy of `decompose` in
`notebooks/periodic_verification_demo.py` (the talk's decomposition figure), so that figure can be
reproduced through the package.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import beta

import utrace
from utrace import (
    BudgetComponent,
    ComponentKind,
    UncertaintyBudget,
    UncertaintyQuantifier,
)
from utrace.bounds import Betting
from utrace.uncertaintyQuantifier import _budget_sums_at_alpha

K = 10
DELTA = 0.05


def _interior(rng, n, conc=0.5, bump=0.3):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * conc, n)
    smx[np.arange(n), y] += bump
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _clf(seed=0, classes=None, n_eval=2000, **kw):
    rng = np.random.default_rng(seed)
    cy, cs = _interior(rng, 1500, **kw)
    uq = UncertaintyQuantifier(N=1500, classes=classes)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, n_eval, **kw)
    return uq, ey, es


def _nb_decompose(r, S, smx, y):
    """`decompose` from notebooks/periodic_verification_demo.py, verbatim in substance."""
    n = len(y)
    sz = S.sum(1)
    covered = S[np.arange(n), y]
    correct = smx.argmax(1) == y
    err = float((~correct).sum() / n)
    empty_correct = float(((sz == 0) & correct).sum() / n)
    multi = float(sum((covered & (sz == k) & correct).sum() - (covered & (sz == k)).sum() / k for k in range(2, K + 1)) / n)
    cov_term = float(r.EC * (r.coverage - r.coverage_lower))
    ec_term = float(r.coverage_lower * (r.EC - r.EC_lower))
    return {"err": err, "dec_empty_correct": empty_correct, "dec_multi": multi,
            "dec_coverage": cov_term, "dec_EC": ec_term}


def _sets(uq, smx, alpha):
    uq.alpha = alpha
    _, S = uq.predict(smx)
    return np.asarray(S)


def _values(b):
    return {c.name: c.value for c in b.components}


@pytest.mark.parametrize("seed, alpha", [(0, 0.05), (0, 0.3), (1, 0.15), (2, 0.6)])
def test_lac_terms_match_the_notebook_decompose(seed, alpha):
    uq, ey, es = _clf(seed)
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    nb = _nb_decompose(b.evaluation, _sets(uq, es, alpha), es, ey)
    v = _values(b)
    assert v["error"] == pytest.approx(nb["err"], abs=1e-15)
    assert v["accepted_not_covered"] == pytest.approx(nb["dec_empty_correct"], abs=1e-15)
    assert v["selection"] == pytest.approx(nb["dec_multi"], abs=1e-12)
    assert v["coverage"] == nb["dec_coverage"]
    assert v["efficiency"] == nb["dec_EC"]


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.6])
@pytest.mark.parametrize("classes", [None, [1, 4, 7]])
def test_components_sum_to_u_delta(alpha, classes):
    uq, ey, es = _clf(3, classes=classes)
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    assert abs(b.total - float(b.U_delta)) < 1e-12
    v = _values(b)
    assert abs((v["error"] + v["accepted_not_covered"] + v["selection"]) - float(b.evaluation.U_empirical)) < 1e-12


def test_class_conditional_terms_against_numpy():
    classes = [2, 5]
    uq, ey, es = _clf(4, classes=classes)
    alpha = 0.3
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    S = _sets(uq, es, alpha)
    valid = np.isin(ey, classes)
    n = valid.sum()
    sz = S.sum(1)
    cov = S[np.arange(len(ey)), ey] & valid
    correct = (es.argmax(1) == ey) & valid
    pi = np.where(sz > 0, 1.0 / np.maximum(sz, 1), 0.0)
    v = _values(b)
    assert b.evaluation.n_valid == n
    assert v["error"] == pytest.approx((valid & ~correct).sum() / n, abs=1e-15)
    assert v["accepted_not_covered"] == pytest.approx((correct & ~cov).sum() / n, abs=1e-15)
    assert v["selection"] == pytest.approx(np.where(cov, correct - pi, 0).sum() / n, abs=1e-12)


def test_evaluation_is_exactly_uncertainty_at():
    uq, ey, es = _clf(5)
    for bound in (None, "hoeffding", Betting()):
        b = uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, bound=bound)
        r = uq.uncertainty_at(es, ey, 0.3, delta=DELTA, bound=bound)
        for f in r._fields:
            x, y_ = getattr(r, f), getattr(b.evaluation, f)
            assert (x == y_) if isinstance(x, int) else np.float64(x).tobytes() == np.float64(y_).tobytes(), f


def test_types_kinds_and_methods():
    uq, ey, es = _clf(6)
    b = uq.uncertainty_budget(es, ey, 0.3, delta=DELTA)
    assert isinstance(b, UncertaintyBudget) and all(isinstance(c, BudgetComponent) for c in b.components)
    assert [c.name for c in b.components] == ["error", "accepted_not_covered", "selection", "coverage", "efficiency"]
    assert [c.kind for c in b.components] == [ComponentKind.MODEL, ComponentKind.STRUCTURAL, ComponentKind.STRUCTURAL,
                                              ComponentKind.STATISTICAL, ComponentKind.STATISTICAL]
    assert b.component("coverage").method == "Clopper-Pearson at delta/2"
    assert b.component("efficiency").method == "tightest(hoeffding, empirical_bernstein) at delta/2"
    assert b.component("selection").kind == "structural"
    assert b.U == b.evaluation.U and b.U_delta == b.evaluation.U_delta
    assert b.error == b.component("error").value
    with pytest.raises(KeyError):
        b.component("nope")
    for name in ("UncertaintyBudget", "BudgetComponent", "ComponentKind"):
        assert getattr(utrace, name) is getattr(utrace.budget, name)
    assert {k.value for k in ComponentKind} == {"model", "structural", "statistical"}


def test_selection_is_not_clipped():
    # In this generator's multi-label sets the top-1 label is right less often than 1/|C|, so the
    # selection term is negative; it must come out as the plain sample mean, not 0.
    uq, ey, es = _clf(7)
    alpha = 0.1
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    nb = _nb_decompose(b.evaluation, _sets(uq, es, alpha), es, ey)
    s = b.component("selection").value
    assert s < -0.01 and s == pytest.approx(nb["dec_multi"], abs=1e-12)
    assert abs(b.total - float(b.U_delta)) < 1e-12


@pytest.mark.parametrize("delta", [0.01, 0.05, 0.3])
def test_error_upper_is_clopper_pearson(delta):
    uq, ey, es = _clf(8)
    b = uq.uncertainty_budget(es, ey, 0.3, delta=delta)
    n = b.evaluation.n_valid
    k = round(b.error * n)
    assert b.error_upper == pytest.approx(float(beta.ppf(1 - delta, k + 1, n - k)), rel=1e-12)
    assert b.error_upper >= b.error


def test_error_upper_when_everything_is_wrong():
    uq, _, es = _clf(9, n_eval=300)
    wrong = (es.argmax(1) + 1) % K
    b = uq.uncertainty_budget(es, wrong, 0.3, delta=DELTA)
    assert b.error == 1.0 and b.error_upper == 1.0


def test_nan_alpha_and_empty_group():
    uq, ey, es = _clf(10)
    b = uq.uncertainty_budget(es, ey, np.nan, delta=DELTA)
    assert b.components == () and np.isnan(b.error) and np.isnan(b.error_upper) and np.isnan(b.total)
    assert np.isnan(b.U_delta)
    uqc, ey, es = _clf(10, classes=[3])
    keep = ey != 3
    b = uqc.uncertainty_budget(es[keep], ey[keep], 0.3, delta=DELTA)
    assert b.components == () and b.evaluation.n_valid == 0 and np.isnan(b.error)


def test_validation_messages():
    uq, ey, es = _clf(11, n_eval=300)
    with pytest.raises(ValueError, match=r"uncertainty_budget: delta is required, a real number in \(0, 1\), got None\."):
        uq.uncertainty_budget(es, ey, 0.3, delta=None)
    with pytest.raises(TypeError):
        uq.uncertainty_budget(es, ey, 0.3)                     # delta is keyword-only and required
    for bad in (0.0, 1.0, "x", -1):
        with pytest.raises(ValueError, match=r"uncertainty_budget: delta must be a real number in \(0, 1\)"):
            uq.uncertainty_budget(es, ey, 0.3, delta=bad)
    with pytest.raises(ValueError, match="Unknown bound"):
        uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, bound="nope")
    with pytest.raises(ValueError, match="takes no tol"):
        uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, tol=0.1)
    fresh = UncertaintyQuantifier(N=10)
    with pytest.raises(ValueError, match="The model must be calibrated before calling uncertainty_budget."):
        fresh.uncertainty_budget(es, ey, 0.3, delta=DELTA)
    with pytest.raises(ValueError, match="outside"):
        uq.uncertainty_budget(es, ey, 1e-9, delta=DELTA)


def test_uncertainty_at_messages_unchanged():
    fresh = UncertaintyQuantifier(N=10)
    rng = np.random.default_rng(0)
    y, s = _interior(rng, 20)
    with pytest.raises(ValueError, match=r"^The model must be calibrated before calling uncertainty_at\.$"):
        fresh.uncertainty_at(s, y, 0.3)


def test_score_without_accepted_raises():
    from dataclasses import dataclass

    from utrace import LAC, Score

    @dataclass(frozen=True)
    class NoAccept(LAC):
        name = "no_accept"

        def accepted(self, y, output, params):
            return Score.accepted(self, y, output, params)

    rng = np.random.default_rng(12)
    cy, cs = _interior(rng, 500)
    uq = UncertaintyQuantifier(N=500, score=NoAccept())
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, 300)
    assert uq.uncertainty_at(es, ey, 0.3, delta=DELTA).U_delta is not None     # works without it
    with pytest.raises(NotImplementedError, match="uncertainty_budget"):
        uq.uncertainty_budget(es, ey, 0.3, delta=DELTA)


# ----- regression -------------------------------------------------------------------------

def _reg(seed=0, n_eval=3000):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=1000 + n_eval)
    yh = y + rng.normal(scale=0.3, size=y.shape)
    uq = UncertaintyQuantifier(N=1000, score="abs_error")
    uq.calibrate(yh[:1000], y[:1000])
    return uq, y[1000:], yh[1000:]


def _reg_reference(uq, y, yh, alpha, tol):
    uq.alpha = alpha
    lo, hi = (np.asarray(a) for a in uq.predict(yh))
    q = (hi - lo) / 2
    cov = (y >= lo) & (y <= hi)
    acc = np.abs(y - yh) <= tol
    pi = np.where(q > 0, np.minimum(1.0, np.minimum(hi, yh + tol) - np.maximum(lo, yh - tol)) / np.where(q > 0, 2 * q, 1), 1.0)
    pi = np.clip(pi, 0, 1)
    n = len(y)
    return {"error": (~acc).sum() / n, "accepted_not_covered": (acc & ~cov).sum() / n,
            "selection": np.where(cov, acc - pi, 0).sum() / n}, float(q[0])


@pytest.mark.parametrize("alpha, tol", [(0.5, 0.6), (0.2, 0.2), (0.05, 0.1)])
def test_regression_terms(alpha, tol):
    uq, y, yh = _reg()
    b = uq.uncertainty_budget(yh, y, alpha, tol=tol, delta=DELTA)
    ref, q_hat = _reg_reference(uq, y, yh, alpha, tol)
    v = _values(b)
    for k in ref:
        assert v[k] == pytest.approx(ref[k], abs=1e-12), k
    assert abs(b.total - float(b.U_delta)) < 1e-12
    if tol >= q_hat:          # C inside B: pi = 1, no selection term, some accepted samples outside C
        assert v["selection"] == pytest.approx(0.0, abs=1e-12) and v["accepted_not_covered"] > 0
    else:                     # B inside C: nothing is accepted outside C
        assert v["accepted_not_covered"] == 0.0


def test_regression_both_regimes_present():
    uq, y, yh = _reg()
    q_small = _reg_reference(uq, y, yh, 0.5, 0.6)[1]
    q_large = _reg_reference(uq, y, yh, 0.05, 0.1)[1]
    assert 0.6 >= q_small and 0.1 < q_large


def test_regression_tol_required():
    uq, y, yh = _reg(n_eval=300)
    with pytest.raises(ValueError, match="tol"):
        uq.uncertainty_budget(yh, y, 0.2, delta=DELTA)


def test_budget_sums_are_one_compilation_across_tol():
    uq, y, yh = _reg(n_eval=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uq.uncertainty_budget(yh, y, 0.2, tol=0.2, delta=DELTA)
        size = _budget_sums_at_alpha._cache_size()
        for t in (0.1, 0.3, 0.7):
            uq.uncertainty_budget(yh, y, 0.2, tol=t, delta=DELTA)
    assert _budget_sums_at_alpha._cache_size() == size

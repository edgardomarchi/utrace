"""`uncertainty_budget(..., units=...)`: per-unit means, then the bound over units. New file.

Reproduces the ACDC notebook's patient-level bound (`notebooks/acdc_periodic_verification.py`:
`v = s / n` per patient, `U_delta_patient = 1 - hedged_lower(v[order])`) on a synthetic patient
structure, with `classes=[k]` selecting the class-`k` pixels and `units=` the patient of each pixel.
"""
import math
import re

import jax.numpy as jnp
import numpy as np
import pytest

from utrace import UncertaintyQuantifier
from utrace.bounds import Betting, Hoeffding

K = 4            # background + 3 classes, as in the ACDC model's output
DELTA = 0.05
CLASS = 1


def _nb_hedged_parts(x, d):
    n = len(x)
    t = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (t + 1)
    s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (t + 1)
    s2_prev = np.r_[0.25, s2[:-1]]
    return np.sqrt(2 * np.log(1 / d) / (n * s2_prev))


def _nb_hedged_lower(x, d=DELTA, c=0.5, iters=50):
    """`hedged_lower` from the ACDC notebook, verbatim in substance."""
    x = np.asarray(x, dtype=float)
    lam_t = _nb_hedged_parts(x, d)
    thr = np.log(1 / d)

    def rejected(m):
        lam = np.minimum(lam_t, c / m)
        return bool(np.cumsum(np.log1p(lam * (x - m))).max() >= thr)

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rejected(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _patients(rng, n_patients, quality):
    """Pixels of n_patients patients; each patient has its own softmax sharpness (dependence within
    a patient). Returns softmax, labels, patient id per pixel."""
    smx, y, pid = [], [], []
    for p in range(n_patients):
        n = int(rng.integers(40, 300))
        lab = rng.integers(0, K, n)
        s = rng.dirichlet(np.ones(K) * 0.6, n)
        s[np.arange(n), lab] += quality[p]
        s /= s.sum(1, keepdims=True)
        smx.append(s), y.append(lab), pid.append(np.full(n, p))
    return np.concatenate(smx), np.concatenate(y), np.concatenate(pid)


@pytest.fixture(scope="module")
def acdc_like():
    rng = np.random.default_rng(20261112)
    q_cal = rng.uniform(0.1, 0.8, 60)
    cs, cy, _ = _patients(rng, 60, q_cal)
    keep = cy == CLASS
    uq = UncertaintyQuantifier(N=int(keep.sum()), classes=[CLASS])
    uq.calibrate(cs[keep], cy[keep])
    tq = rng.uniform(0.1, 0.8, 30)
    ts, ty, _ = _patients(rng, 30, tq)
    alpha = float(uq.tune_alpha(ts[ty == CLASS], ty[ty == CLASS]))
    eq = rng.uniform(0.1, 0.8, 40)
    es, ey, pid = _patients(rng, 40, eq)
    return uq, alpha, es, ey, pid


def _notebook_v(uq, alpha, es, ey, pid):
    """Per-patient `v = s / n` over class-CLASS pixels, as the notebook's `evaluate` builds it."""
    r = ey == CLASS
    uq.alpha = alpha
    _, S = uq.predict(es[r])
    S = np.asarray(S)
    sz = S.sum(1)
    cov = S[:, CLASS]
    inv = np.where(cov, 1.0 / np.maximum(sz, 1), 0.0)
    pos = pid[r]
    n_ev = pid.max() + 1
    n = np.bincount(pos, minlength=n_ev)
    s = np.bincount(pos, inv, n_ev)
    return s / n


def test_reproduces_the_notebook_patient_bound(acdc_like):
    uq, alpha, es, ey, pid = acdc_like
    v = _notebook_v(uq, alpha, es, ey, pid)
    order = np.random.default_rng([20261112, 0, 1]).permutation(len(v))
    expected = 1 - _nb_hedged_lower(v[order])                       # U_delta_patient
    # unit ids whose sorted order is `order`: patient order[j] gets id j
    rank = np.empty(len(v), dtype=int)
    rank[order] = np.arange(len(v))
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, bound=Betting(order="given"), units=rank[pid])
    assert b.W.over == b.V.over == "units" and b.V.n == len(v)
    assert b.U_delta_V == pytest.approx(expected, abs=1e-15)
    assert b.V.mean == pytest.approx(v.mean(), abs=1e-15)
    # with the plain patient ids the bound runs in sorted-id order
    b2 = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, bound=Betting(order="given"), units=pid)
    assert b2.U_delta_V == pytest.approx(1 - _nb_hedged_lower(v), abs=1e-15)


def test_components_sum_and_unit_averaging(acdc_like):
    uq, alpha, es, ey, pid = acdc_like
    for bound in (None, "betting"):
        b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, bound=bound, units=pid)
        assert abs(b.V.total - b.U_delta_V) < 1e-12 and abs(b.W.total - b.U_delta_W) < 1e-12
        assert "units" in b.V.method and "unit-averaged" in b.V.components[0].method
    r = ey == CLASS
    correct = es[r].argmax(1) == CLASS
    err_u = np.bincount(pid[r], ~correct) / np.bincount(pid[r])
    assert b.V.components[0].value == pytest.approx(err_u.mean(), abs=1e-14)
    assert b.V.components[0].value != pytest.approx(b.error, abs=1e-6)     # unit- vs sample-averaged


def test_product_bound_unchanged_by_units(acdc_like):
    uq, alpha, es, ey, pid = acdc_like
    b0 = uq.uncertainty_budget(es, ey, alpha, delta=DELTA)
    b1 = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, units=pid)
    assert b0.evaluation == b1.evaluation
    assert b0.components == b1.components and b0.error == b1.error and b0.error_upper == b1.error_upper


def test_hoeffding_over_units_closed_form(acdc_like):
    uq, alpha, es, ey, pid = acdc_like
    v = _notebook_v(uq, alpha, es, ey, pid)
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, bound=Hoeffding(), units=pid)
    assert b.V.lower == pytest.approx(max(0.0, v.mean() - math.sqrt(math.log(1 / DELTA) / (2 * len(v)))), abs=1e-14)


def test_units_without_valid_samples_are_dropped(acdc_like):
    uq, alpha, es, ey, pid = acdc_like
    units = pid.copy()
    extra = (ey != CLASS) & (pid == 0)
    units[extra] = 999                     # a unit holding only non-class pixels
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, units=units)
    assert b.V.n == 40 and "(1 unit with no valid sample dropped)" in b.V.method


def test_string_and_jax_unit_ids(acdc_like):
    uq, alpha, es, ey, pid = acdc_like
    b = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, units=pid)
    bs = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, units=np.array([f"p{p:03d}" for p in pid]))
    bj = uq.uncertainty_budget(es, ey, alpha, delta=DELTA, units=jnp.asarray(pid))
    assert bs.U_delta_V == b.U_delta_V == bj.U_delta_V and bs.U_delta_W == b.U_delta_W


def test_marginal_with_units():
    rng = np.random.default_rng(3)
    cs, cy, _ = _patients(rng, 30, rng.uniform(0.2, 0.6, 30))
    uq = UncertaintyQuantifier(N=len(cy))
    uq.calibrate(cs, cy)
    es, ey, pid = _patients(rng, 25, rng.uniform(0.2, 0.6, 25))
    b = uq.uncertainty_budget(es, ey, 0.3, delta=DELTA, units=pid)
    assert b.V.n == 25 and abs(b.W.total - b.U_delta_W) < 1e-12


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((10, 2))])
def test_units_shape_validation(acdc_like, bad):
    uq, alpha, es, ey, _ = acdc_like
    msg = (f"uncertainty_budget: units must be a 1-D array with one entry per sample, "
           f"shape (n,) = ({len(ey)},), got shape {bad.shape}.")
    with pytest.raises(ValueError, match=re.escape(msg)):
        uq.uncertainty_budget(es, ey, alpha, delta=DELTA, units=bad)


def test_units_validated_even_for_nan_alpha(acdc_like):
    uq, _, es, ey, _ = acdc_like
    with pytest.raises(ValueError, match="units must be a 1-D array"):
        uq.uncertainty_budget(es, ey, np.nan, delta=DELTA, units=np.zeros(2))

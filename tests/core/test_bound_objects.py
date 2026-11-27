"""Bound objects (`utrace.bounds`) and `bound=` on `uncertainty_at`.

New file. The default bound must reproduce the pre-existing `EC_lower` formula bit for bit; that
formula is frozen below as `_legacy_ec_lower` (a copy of `_confidence_bound`'s body before the bound
objects existed) and compared bytewise.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import utrace
from utrace import UncertaintyQuantifier
from utrace.bounds import (
    DEFAULT_BOUND,
    Bound,
    EmpiricalBernstein,
    Hoeffding,
    SampleStats,
    Tightest,
    resolve_bound,
)
from utrace.uncertaintyQuantifier import _confidence_bound

K = 10


def _legacy_ec_lower(EC, n_succ, sum_sq_dev, delta):
    """`EC_lower` exactly as `_confidence_bound` computed it before `bound=` existed (frozen copy)."""
    if n_succ == 0:
        return np.float64(0.0)
    margin = np.sqrt(np.log(4.0 / delta) / (2.0 * n_succ))
    if n_succ >= 2:
        L = np.log(8.0 / delta)
        V = max(0.0, float(sum_sq_dev)) / (n_succ - 1)
        margin_eb = np.sqrt(2.0 * V * L / n_succ) + 7.0 * L / (3.0 * (n_succ - 1))
        margin = min(margin, margin_eb)
    return np.float64(max(0.0, float(EC) - float(margin)))


def _b(x):
    return np.float64(x).tobytes()


# ----- closed forms, at the level given ------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 10, 1000])
@pytest.mark.parametrize("level", [0.5, 0.05, 1e-4])
def test_hoeffding_closed_form(n, level):
    got = Hoeffding().lower(SampleStats(n=n, mean=0.8), level)
    assert got == pytest.approx(0.8 - math.sqrt(math.log(1 / level) / (2 * n)), rel=1e-14)


@pytest.mark.parametrize("n", [2, 10, 1000])
@pytest.mark.parametrize("level", [0.5, 0.05, 1e-4])
@pytest.mark.parametrize("var", [0.0, 0.01, 0.2])
def test_empirical_bernstein_closed_form(n, level, var):
    ssd = var * (n - 1)
    got = EmpiricalBernstein().lower(SampleStats(n=n, mean=0.7, sum_sq_dev=ssd), level)
    L = math.log(2 / level)
    expected = 0.7 - math.sqrt(2 * var * L / n) - 7 * L / (3 * (n - 1))
    assert got == pytest.approx(expected, rel=1e-13, abs=1e-15)


def test_undefined_bounds_are_minus_inf():
    assert Hoeffding().lower(SampleStats(n=0, mean=0.0), 0.05) == -math.inf
    assert EmpiricalBernstein().lower(SampleStats(n=1, mean=1.0), 0.05) == -math.inf
    assert EmpiricalBernstein().lower(SampleStats(n=0, mean=0.0), 0.05) == -math.inf


def test_empirical_bernstein_clips_negative_sum_of_squares():
    a = EmpiricalBernstein().lower(SampleStats(n=50, mean=0.9, sum_sq_dev=-1e-12), 0.05)
    b = EmpiricalBernstein().lower(SampleStats(n=50, mean=0.9, sum_sq_dev=0.0), 0.05)
    assert _b(a) == _b(b)


def test_tightest_splits_level_evenly_and_takes_the_largest():
    s = SampleStats(n=200, mean=0.9, sum_sq_dev=0.5)
    t = Tightest([Hoeffding(), EmpiricalBernstein()])
    assert t.lower(s, 0.1) == max(Hoeffding().lower(s, 0.05), EmpiricalBernstein().lower(s, 0.05))
    t3 = Tightest([Hoeffding(), EmpiricalBernstein(), Hoeffding()])
    assert t3.lower(s, 0.3) == max(Hoeffding().lower(s, 0.1), EmpiricalBernstein().lower(s, 0.1))
    nested = Tightest([Tightest([Hoeffding()]), EmpiricalBernstein()])
    assert nested.lower(s, 0.1) == t.lower(s, 0.1)


def test_tightest_falls_back_when_a_member_is_undefined():
    s = SampleStats(n=1, mean=1.0)
    assert DEFAULT_BOUND.lower(s, 0.025) == Hoeffding().lower(s, 0.0125)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_level_outside_open_unit_interval_raises(level):
    with pytest.raises(ValueError, match=r"level must lie in \(0, 1\)"):
        Hoeffding().lower(SampleStats(n=5, mean=0.5), level)


# ----- objects: frozen, hashable, equal -------------------------------------------------

def test_bounds_are_frozen_hashable_and_equal_by_value():
    a, b = Tightest([Hoeffding(), EmpiricalBernstein()]), Tightest((Hoeffding(), EmpiricalBernstein()))
    assert a == b == DEFAULT_BOUND and hash(a) == hash(b)
    assert isinstance(a.members, tuple)
    assert Hoeffding() == Hoeffding() and hash(Hoeffding()) == hash(Hoeffding())
    assert Hoeffding() != EmpiricalBernstein()
    with pytest.raises(AttributeError):
        a.members = ()


def test_needs_values_flags():
    assert not Hoeffding().needs_values and not EmpiricalBernstein().needs_values
    assert not DEFAULT_BOUND.needs_values


def test_describe():
    assert DEFAULT_BOUND.describe() == "tightest(hoeffding, empirical_bernstein)"


def test_tightest_validation_messages():
    with pytest.raises(ValueError, match="Tightest needs at least one member bound."):
        Tightest([])
    with pytest.raises(TypeError, match="Tightest members must be Bound instances, got str at position 1."):
        Tightest([Hoeffding(), "hoeffding"])


# ----- resolve_bound -------------------------------------------------------------------

def test_resolve_bound():
    assert resolve_bound(None) is DEFAULT_BOUND
    assert resolve_bound("hoeffding") == Hoeffding()
    assert resolve_bound("empirical_bernstein") == EmpiricalBernstein()
    h = Hoeffding()
    assert resolve_bound(h) is h


def test_resolve_bound_unknown_string():
    with pytest.raises(ValueError, match=r"Unknown bound 'x'\. Supported: 'hoeffding', 'empirical_bernstein'"):
        resolve_bound("x")


def test_resolve_bound_wrong_type():
    with pytest.raises(TypeError, match="bound must be None, a string or a Bound instance, got int."):
        resolve_bound(3)


def test_exports_and_numpy_only_import():
    for name in ("Bound", "Hoeffding", "EmpiricalBernstein", "Tightest", "SampleStats"):
        assert getattr(utrace, name) is getattr(utrace.bounds, name)
    src = Path(utrace.bounds.__file__).read_text()
    assert "import scipy" not in src and "from scipy" not in src and "torch" not in src
    assert issubclass(Hoeffding, Bound)
    assert "torch" not in {m.split(".")[0] for m in sys.modules if m.startswith("utrace")}


# ----- bit-for-bit: default bound against the frozen legacy formula ---------------------

_DELTAS = [1e-6, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.999]
_NS = [0, 1, 2, 3, 10, 100, 7919, 10**6]


def _grid():
    rng = np.random.default_rng(20261127)
    for delta in _DELTAS:
        for n in _NS:
            for EC in (0.0, 1e-9, 0.1, 0.5, 0.9, 1.0, float(rng.uniform())):
                for var in (0.0, 1e-8, 0.01, 0.25):
                    ssd = var * max(n - 1, 0)
                    yield EC, n, ssd, delta
                yield EC, n, -1e-12, delta     # a sum of squares a rounding error below 0
    for _ in range(3000):                       # random draws over the whole domain
        n = int(rng.integers(0, 10**6))
        yield (float(rng.uniform()), n, float(rng.uniform(0, 0.25)) * max(n - 1, 0),
               float(10 ** rng.uniform(-6, math.log10(0.999))))


def test_default_bound_is_bitwise_the_legacy_formula():
    count = 0
    for EC, n, ssd, delta in _grid():
        legacy = _legacy_ec_lower(EC, n, ssd, delta)
        for b in (None, DEFAULT_BOUND, Tightest([Hoeffding(), EmpiricalBernstein()])):
            _, got, _ = _confidence_bound(EC, n, max(n, 1), n, ssd, delta, b)
            assert _b(got) == _b(legacy), (EC, n, ssd, delta, b, got, legacy)
        _, got_pos, _ = _confidence_bound(EC, n, max(n, 1), n, ssd, delta)   # positional, as before
        assert _b(got_pos) == _b(legacy)
        count += 1
    assert count > 3000


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


@pytest.mark.parametrize("classes", [None, [2, 5]])
@pytest.mark.parametrize("delta", [0.01, 0.05, 0.3])
def test_uncertainty_at_default_bound_bitwise_classification(classes, delta):
    rng = np.random.default_rng(3)
    cy, cs = _interior(rng, 2000)
    uq = UncertaintyQuantifier(N=2000, classes=classes)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, 700)
    for alpha in (0.05, 0.2, 0.6):
        r0 = uq.uncertainty_at(es, ey, alpha, delta=delta)
        for b in (DEFAULT_BOUND, Tightest([Hoeffding(), EmpiricalBernstein()])):
            r1 = uq.uncertainty_at(es, ey, alpha, delta=delta, bound=b)
            for f in r0._fields:
                v0, v1 = getattr(r0, f), getattr(r1, f)
                assert (v0 == v1) if isinstance(v0, int) else _b(v0) == _b(v1), f


@pytest.mark.parametrize("delta", [0.01, 0.05])
def test_uncertainty_at_default_bound_bitwise_regression(delta):
    rng = np.random.default_rng(4)
    y = rng.normal(size=1500)
    yh = y + rng.normal(scale=0.3, size=1500)
    uq = UncertaintyQuantifier(N=1000, score="abs_error")
    uq.calibrate(yh[:1000], y[:1000])
    for alpha in (0.1, 0.3):
        r0 = uq.uncertainty_at(yh[1000:], y[1000:], alpha, tol=0.2, delta=delta)
        r1 = uq.uncertainty_at(yh[1000:], y[1000:], alpha, tol=0.2, delta=delta, bound=DEFAULT_BOUND)
        for f in r0._fields:
            v0, v1 = getattr(r0, f), getattr(r1, f)
            assert (v0 == v1) if isinstance(v0, int) else _b(v0) == _b(v1), f


# ----- bound= on uncertainty_at ---------------------------------------------------------

def _uq():
    rng = np.random.default_rng(5)
    cy, cs = _interior(rng, 1000)
    uq = UncertaintyQuantifier(N=1000)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, 500)
    return uq, ey, es


def test_bound_without_delta_raises():
    uq, ey, es = _uq()
    with pytest.raises(ValueError, match=r"uncertainty_at: bound= needs delta; got delta=None\."):
        uq.uncertainty_at(es, ey, 0.2, bound="hoeffding")
    with pytest.raises(ValueError, match="bound= needs delta"):
        uq.uncertainty_at(es, ey, np.nan, bound="hoeffding")


def test_bound_alias_and_unknown_on_uncertainty_at():
    uq, ey, es = _uq()
    with pytest.raises(ValueError, match="Unknown bound"):
        uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound="nope")
    with pytest.raises(TypeError, match="bound must be None"):
        uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound=1.0)


def test_single_member_bounds_at_delta_half():
    uq, ey, es = _uq()
    r = uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound="hoeffding")
    expected = max(0.0, float(r.EC) - math.sqrt(math.log(1 / 0.025) / (2 * r.n_succ)))
    assert r.EC_lower == pytest.approx(expected, rel=1e-14)
    rd = uq.uncertainty_at(es, ey, 0.2, delta=0.05)
    assert r.coverage_lower == rd.coverage_lower                   # coverage stays Clopper-Pearson
    assert r.U_delta == 1.0 - r.EC_lower * r.coverage_lower


def test_nan_alpha_with_bound_gives_nan_fields():
    uq, ey, es = _uq()
    r = uq.uncertainty_at(es, ey, np.nan, delta=0.05, bound="empirical_bernstein")
    assert np.isnan(r.U_delta) and np.isnan(r.EC_lower)


def test_custom_bound_subclass_is_used():
    class Zero(Bound):
        name = "zero"

        def lower(self, stats, level):
            return 0.0

    uq, ey, es = _uq()
    r = uq.uncertainty_at(es, ey, 0.2, delta=0.05, bound=Zero())
    assert r.EC_lower == 0.0 and r.U_delta == 1.0

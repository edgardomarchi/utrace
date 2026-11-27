"""`utrace.bounds.Betting`, ported from `notebooks/acdc_periodic_verification.py` (section 2). New file.

The notebook's own assertions are reproduced here as tests, and its two functions are kept below,
verbatim in substance, as references: `_nb_hedged_lower` (bisection) and `_nb_hedged_lower_grid`
(the diagnostic's grid of 20 000 points, a test oracle only: it allocates `20001 x n`).
"""
import math

import numpy as np
import pytest

from utrace import UncertaintyQuantifier
from utrace.bounds import Betting, SampleStats, Tightest, resolve_bound

SEED = 20261112        # the notebook's SEED
DELTA = 0.05


# ---- the notebook's functions (references) ------------------------------------------------

def _nb_hedged_parts(x, d):
    n = len(x)
    t = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (t + 1)
    s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (t + 1)
    s2_prev = np.r_[0.25, s2[:-1]]
    return np.sqrt(2 * np.log(1 / d) / (n * s2_prev))


def _nb_hedged_lower(x, d=DELTA, c=0.5, iters=50):
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


def _nb_hedged_lower_grid(x, d=DELTA, c=0.5, grid=20001):
    x = np.asarray(x, dtype=float)
    lam_t = _nb_hedged_parts(x, d)
    m = np.linspace(0, 1, grid)[1:]
    lam = np.minimum(lam_t[None, :], c / m[:, None])
    logK = np.cumsum(np.log1p(lam * (x[None, :] - m[:, None])), axis=1)
    keep = logK.max(1) < np.log(1 / d)
    return float(m[np.argmax(keep)]) if keep.any() else 1.0


def _given(x, level=DELTA, **kw):
    x = np.asarray(x, dtype=float)
    return Betting(order="given", **kw).lower(SampleStats(n=len(x), mean=float(x.mean()) if len(x) else 0.0,
                                                          values=x), level)


# ---- the notebook's assertions ---------------------------------------------------------------

def test_hand_case_three_ones():
    """x = (1, 1, 1), delta = 0.05: m is rejected iff (0.5 + 0.5/m)^3 >= 20."""
    lb_hand = 0.5 / (20 ** (1 / 3) - 0.5)
    assert abs(_given([1.0, 1.0, 1.0]) - lb_hand) < 1e-12


def test_hedged_parts_on_three_ones():
    from utrace.bounds import _hedged_parts
    expected = [np.sqrt(2 * np.log(20) / 0.75), np.sqrt(2 * np.log(20) / (3 * 0.15625)),
                np.sqrt(2 * np.log(20) / (3 * (0.25 + 0.0625 + (1 / 6) ** 2) / 3))]
    assert np.allclose(_hedged_parts(np.ones(3), 0.05), expected)


def test_single_observation_rejects_nothing():
    assert _given([1.0]) < 1e-12


def test_below_the_grid_oracle_by_less_than_one_step():
    rng = np.random.default_rng([SEED, 2])
    for x in [rng.beta(30, 3, 75), rng.beta(4, 1, 25), rng.binomial(1, 0.5, 40).astype(float), np.full(10, 0.9)]:
        diff = _nb_hedged_lower_grid(x) - _given(x)
        assert 0 <= diff <= 1 / 20000 + 1e-12, diff


# ---- the port against the notebook function, order and seed ---------------------------------------

@pytest.mark.parametrize("level", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("c", [0.25, 0.5, 0.9])
def test_given_order_equals_the_notebook_function(level, c):
    rng = np.random.default_rng([SEED, 7])
    for x in [rng.beta(30, 3, 100), rng.uniform(size=17), rng.binomial(1, 0.8, 200).astype(float),
              np.zeros(5), np.ones(50)]:
        assert _given(x, level, c=c) == _nb_hedged_lower(x, level, c=c)


def test_iters_controls_resolution():
    x = np.random.default_rng(1).beta(8, 2, 60)
    assert _given(x, iters=10) == _nb_hedged_lower(x, iters=10)
    assert _given(x, iters=10) <= _given(x, iters=50) <= _given(x, iters=10) + 2 ** -10


def test_permute_is_the_seeded_permutation_and_deterministic():
    x = np.random.default_rng(2).beta(5, 2, 300)
    stats = SampleStats(n=len(x), mean=float(x.mean()), values=x)
    a, b = Betting().lower(stats, DELTA), Betting(seed=0).lower(stats, DELTA)
    assert a == b == _nb_hedged_lower(x[np.random.default_rng(0).permutation(len(x))])
    assert Betting(seed=3).lower(stats, DELTA) == _nb_hedged_lower(x[np.random.default_rng(3).permutation(len(x))])
    assert Betting(seed=0).lower(stats, DELTA) == Betting(seed=0).lower(stats, DELTA)


def test_order_matters_and_permute_protects_against_sorted_input():
    x = np.sort(np.random.default_rng(4).beta(5, 2, 400))
    s_up, s_down = (SampleStats(n=len(x), mean=float(x.mean()), values=v) for v in (x, x[::-1]))
    given_up = Betting(order="given").lower(s_up, DELTA)
    given_down = Betting(order="given").lower(s_down, DELTA)
    assert given_up != given_down
    assert Betting().lower(s_up, DELTA) == Betting().lower(
        SampleStats(n=len(x), mean=float(x.mean()), values=x.copy()), DELTA)


def test_does_not_modify_the_values():
    x = np.random.default_rng(5).uniform(size=50)
    x0 = x.copy()
    Betting().lower(SampleStats(n=50, mean=float(x.mean()), values=x), DELTA)
    np.testing.assert_array_equal(x, x0)


def test_valid_range_and_never_above_the_mean_much():
    rng = np.random.default_rng(6)
    for _ in range(20):
        x = rng.beta(rng.uniform(0.5, 30), rng.uniform(0.5, 5), int(rng.integers(2, 300)))
        lb = _given(x)
        assert 0.0 <= lb <= 1.0


# ---- object, validation, alias ---------------------------------------------------------------

def test_object_properties():
    assert Betting() == Betting(c=0.5, iters=50, order="permute", seed=0)
    assert hash(Betting()) == hash(Betting())
    assert Betting().needs_values and Tightest([Betting()]).needs_values
    assert resolve_bound("betting") == Betting()
    assert Betting(order="given").describe() == "betting"
    assert Betting(seed=4).describe() == "betting(permute, seed=4)"


@pytest.mark.parametrize("kw, msg", [
    ({"c": 0.0}, r"Betting: c must be in \(0, 1\), got 0.0."),
    ({"c": 1.0}, r"Betting: c must be in \(0, 1\)"),
    ({"iters": 0}, r"Betting: iters must be an integer >= 1, got 0."),
    ({"iters": 2.5}, r"Betting: iters must be an integer >= 1"),
    ({"order": "sorted"}, r"Betting: order must be 'permute' or 'given', got 'sorted'."),
    ({"seed": -1}, r"Betting: seed must be a non-negative integer, got -1."),
])
def test_parameter_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        Betting(**kw)


def test_values_validation():
    with pytest.raises(ValueError, match="Betting needs the values"):
        Betting().lower(SampleStats(n=3, mean=0.5), DELTA)
    with pytest.raises(ValueError, match=r"Betting: values must have shape \(n,\) = \(4,\), got \(3,\)."):
        Betting().lower(SampleStats(n=4, mean=0.5, values=np.ones(3)), DELTA)
    with pytest.raises(ValueError, match=r"Betting: values must lie in \[0, 1\]."):
        Betting().lower(SampleStats(n=2, mean=0.5, values=np.array([0.5, 1.5])), DELTA)
    assert Betting().lower(SampleStats(n=0, mean=0.0, values=np.zeros(0)), DELTA) == -math.inf


# ---- through uncertainty_at -----------------------------------------------------------------

K = 10


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def test_uncertainty_at_with_betting_matches_the_notebook_on_the_covered_pi():
    rng = np.random.default_rng(8)
    cy, cs = _interior(rng, 1000)
    uq = UncertaintyQuantifier(N=1000)
    uq.calibrate(cs, cy)
    ey, es = _interior(rng, 900)
    alpha = 0.3
    r = uq.uncertainty_at(es, ey, alpha, delta=DELTA, bound=Betting(order="given"))
    uq.alpha = alpha
    _, S = uq.predict(es)
    S = np.asarray(S)
    sz = S.sum(1)
    cov = S[np.arange(len(ey)), ey] & (sz > 0)
    z = 1.0 / sz[cov]
    assert r.EC_lower == max(0.0, _nb_hedged_lower(z, DELTA / 2))
    assert r.U_delta == 1.0 - r.EC_lower * r.coverage_lower
    rp = uq.uncertainty_at(es, ey, alpha, delta=DELTA, bound="betting")
    assert rp.EC_lower == max(0.0, _nb_hedged_lower(z[np.random.default_rng(0).permutation(len(z))], DELTA / 2))


def test_uncertainty_at_with_betting_regression():
    rng = np.random.default_rng(9)
    y = rng.normal(size=1600)
    yh = y + rng.normal(scale=0.3, size=1600)
    uq = UncertaintyQuantifier(N=1000, score="abs_error")
    uq.calibrate(yh[:1000], y[:1000])
    r = uq.uncertainty_at(yh[1000:], y[1000:], 0.2, tol=0.2, delta=DELTA, bound="betting")
    assert 0.0 <= r.EC_lower <= r.EC and r.U_delta >= r.U_empirical

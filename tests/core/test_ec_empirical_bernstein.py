"""Tests for the lower bound on `E[1/|C| | covered]`: `EC - min(m_H, m_EB)`, each at `delta/4`.

`m_H = sqrt(ln(4/delta) / (2 n))` is one-sided Hoeffding; `m_EB = sqrt(2 V ln(8/delta) / n) +
7 ln(8/delta) / (3 (n - 1))` is Maurer and Pontil (2009), "Empirical Bernstein Bounds and Sample
Variance Penalization", COLT, arXiv:0907.3740, Theorem 4, with `V` the unbiased sample variance.

Three groups: (1) hand values, with the formulas typed out here independently of the library and
an assertion of WHICH margin is the smaller; (2) the internal centred sum of squares returned by
`_evaluate_at_alpha` against `numpy.var(ddof=1)`; (3) Monte Carlo checks of the guarantee on a
singleton-dominated and on a high-variance generator (the pre-existing generator, about 61 %
multi-label, is in `test_confidence_bound_montecarlo.py`). Fixed seeds.
"""
import math
import warnings

import numpy as np
import pytest

from utrace import UncertaintyQuantifier
from utrace.uncertaintyQuantifier import (
    _confidence_bound,
    _evaluate_at_alpha,
    _predict_sets,
    _q_hat_from_alpha,
)

K = 10


# ----- 1. hand values ------------------------------------------------------------------

def _m_h(n, delta):
    return math.sqrt(math.log(4.0 / delta) / (2.0 * n))


def _m_eb(n, v, delta):
    big_l = math.log(8.0 / delta)
    return math.sqrt(2.0 * v * big_l / n) + 7.0 * big_l / (3.0 * (n - 1))


# (n, V, delta, winner): `V = 0` at large n; small n; two cases bracketing the crossover at n = 200.
CASES = [
    (10_000, 0.0, 0.05, "EB"),
    (20, 0.04, 0.05, "H"),
    (200, 0.02, 0.05, "EB"),
    (200, 0.06, 0.05, "H"),
    (500, 0.05, 0.05, "EB"),
]


@pytest.mark.parametrize(("n", "v", "delta", "winner"), CASES)
def test_margin_selected_and_ec_lower(n, v, delta, winner):
    ec = 0.9
    m_h, m_eb = _m_h(n, delta), _m_eb(n, v, delta)
    assert (m_eb < m_h) == (winner == "EB")
    expected = ec - min(m_h, m_eb)
    other = ec - max(m_h, m_eb)
    assert 0.0 < other < expected
    _, ec_lower, u_delta = _confidence_bound(ec, n, n, n, v * (n - 1), delta)
    assert ec_lower == pytest.approx(expected, rel=1e-13)
    assert ec_lower != pytest.approx(other, rel=1e-6)
    assert u_delta == 1.0 - ec_lower * _confidence_bound(ec, n, n, n, 0.0, delta)[0]


def test_n_succ_zero_one_two():
    delta = 0.05
    # n_succ == 0: EC_lower is 0 whatever is passed.
    assert _confidence_bound(0.0, 0, 50, 0, 0.0, delta)[1] == 0.0
    assert _confidence_bound(0.0, 0, 50, 0, 3.0, delta)[1] == 0.0
    # n_succ == 1: V is undefined, m_H alone (the sum of squares is ignored). At delta = 0.05
    # m_H(1) = sqrt(ln(80) / 2) = 1.48 exceeds any EC, so the bound is clipped to 0.
    for ss in (0.0, 7.0):
        assert _confidence_bound(0.9, 1, 50, 1, ss, delta)[1] == 0.0
    # n_succ == 2: both margins exist; the range term 7 ln(160) / 3 makes Hoeffding the smaller.
    # Unbiased variance of the pair (1, 0.5) is 0.125, i.e. a centred sum of squares of 0.125.
    assert _m_h(2, delta) < _m_eb(2, 0.125, delta)
    _, el, _ = _confidence_bound(1.0, 2, 50, 2, 0.125, delta)
    assert el == 0.0                           # m_H(2) = 1.48 > EC = 1
    # a case with a non-trivial n_succ == 1 and 2 result needs EC > margin, impossible for
    # delta = 0.05; at delta = 0.9 the margins shrink enough to be visible:
    delta = 0.9
    assert _m_h(1, delta) == pytest.approx(math.sqrt(math.log(4.0 / 0.9) / 2.0), rel=1e-14)
    _, el, _ = _confidence_bound(1.0, 1, 5, 1, 0.0, delta)
    assert el == pytest.approx(1.0 - _m_h(1, delta), rel=1e-13)
    assert _m_h(2, delta) < _m_eb(2, 0.125, delta)
    _, el, _ = _confidence_bound(0.95, 2, 5, 2, 0.125, delta)
    assert el == pytest.approx(0.95 - _m_h(2, delta), rel=1e-13)


def test_negative_sum_of_squares_is_clipped():
    # Numerically negative input behaves as V = 0.
    a = _confidence_bound(0.9, 400, 500, 400, -1e-12, 0.05)[1]
    b = _confidence_bound(0.9, 400, 500, 400, 0.0, 0.05)[1]
    assert a == b


def test_ec_lower_never_below_zero_nor_above_ec():
    for n in (1, 2, 3, 10, 100, 10**4):
        for ss in (0.0, 0.5, 20.0):
            for delta in (0.5, 0.05, 0.001):
                el = _confidence_bound(0.7, n, n, n, ss, delta)[1]
                assert 0.0 <= el <= 0.7


# ----- 2. variance ----------------------------------------------------------------------

def _sharp_mixed(rng, n):
    """Softmax with a spread of set sizes: a sharp and a diffuse component."""
    y = rng.integers(0, K, n)
    conc = np.where(rng.random(n) < 0.5, 0.5, 3.0)[:, None]
    s = rng.gamma(np.broadcast_to(conc, (n, K)), 1.0)
    s /= s.sum(axis=1, keepdims=True)
    s[np.arange(n), y] += np.where(rng.random(n) < 0.5, 0.3, 0.02)
    s /= s.sum(axis=1, keepdims=True)
    return y, s


def _sum_sq_dev(uq, es, ey, alpha, valid):
    uq.uncertainty_at(es, ey, alpha)                     # sorts the buffer lazily
    st = uq._state
    out = _evaluate_at_alpha(
        np.float64(alpha), ey.astype(np.int32), es, valid,
        st.conformity_scores, np.int32(st.N), uq.score_,
    )
    q_hat = _q_hat_from_alpha(st.conformity_scores, np.int32(st.N), np.float64(alpha))
    _, sets = _predict_sets(es, q_hat, score_fn=uq.score_)
    return out, np.asarray(sets)


@pytest.mark.parametrize("classes", [None, [3, 4]])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("alpha", [0.05, 0.3])
def test_sum_sq_dev_matches_numpy_var(classes, seed, alpha):
    rng = np.random.default_rng(100 + seed)
    cy, cs = _sharp_mixed(rng, 400)
    uq = UncertaintyQuantifier(N=400)
    uq.calibrate(cs, cy)
    uq.classes = classes
    ey, es = _sharp_mixed(rng, 600)
    valid = np.isin(ey, classes) if classes is not None else np.ones(600, bool)
    out, sets = _sum_sq_dev(uq, es, ey, max(alpha, 1 / 401), valid)
    sizes = sets.sum(axis=1)
    covered = sets[np.arange(600), ey] & valid & (sizes > 0)
    z = 1.0 / sizes[covered]
    assert int(out[4]) == covered.sum() >= 2
    assert len(set(z.tolist())) > 1                       # genuinely non-constant
    ref = np.var(z, ddof=1)
    got = float(out[5]) / (covered.sum() - 1)
    assert got == pytest.approx(ref, rel=1e-12)


def test_sum_sq_dev_exactly_zero_for_singletons():
    rng = np.random.default_rng(0)
    n = 500
    cy = rng.integers(0, K, n)
    cs = np.full((n, K), 1e-4)
    cs[np.arange(n), cy] = 1 - 1e-4 * (K - 1)
    uq = UncertaintyQuantifier(N=n)
    uq.calibrate(cs, cy)
    ey = rng.integers(0, K, 700)
    es = np.full((700, K), 1e-4)
    es[np.arange(700), ey] = 1 - 1e-4 * (K - 1)
    valid = np.ones(700, bool)
    out, sets = _sum_sq_dev(uq, es, ey, 0.1, valid)
    assert (sets.sum(axis=1) == 1).all() and int(out[4]) == 700
    assert float(out[5]) == 0.0
    # and the library's EC_lower is then governed by the Bernstein range term
    r = uq.uncertainty_at(es, ey, 0.1, delta=0.05)
    assert r.EC == 1.0
    assert r.EC_lower == pytest.approx(1.0 - _m_eb(700, 0.0, 0.05), rel=1e-13)


# ----- 3. Monte Carlo ---------------------------------------------------------------------

M_CHUNK, N_CHUNKS = 100_000, 10
N_EVAL, REPS = 300, 800
ALPHA = 0.2


def _component(rng, n, kind):
    y = rng.integers(0, K, n)
    conc, boost = {"sharp": (0.5, 3.0), "mid": (0.5, 0.3), "wide": (3.0, 0.02)}[kind]
    s = rng.dirichlet(np.ones(K) * conc, n)
    s[np.arange(n), y] += boost
    s /= s.sum(axis=1, keepdims=True)
    return y, s


def _mixture(pi, kind):
    """A sharp softmax, replaced with probability `pi` by a `kind` component."""
    def gen(rng, n):
        y, s = _component(rng, n, "sharp")
        m = rng.random(n) < pi
        if m.any():
            y[m], s[m] = _component(rng, int(m.sum()), kind)
        return y, s
    return gen


GENERATORS = {
    "singleton": _mixture(0.24, "mid"),      # about 0.1 % multi-label among covered sets
    "high_variance": _mixture(0.5, "wide"),  # about 60 % multi-label, sizes spread up to 6
}


def _tolerance(rate, reps):
    return 4.0 * np.sqrt(rate * (1 - rate) / reps)


@pytest.fixture(scope="module", params=sorted(GENERATORS))
def population(request):
    gen = GENERATORS[request.param]
    rng = np.random.default_rng(2025)
    cy, cs = gen(rng, 500)
    uq = UncertaintyQuantifier(N=500)
    uq.calibrate(cs, cy)
    n_cov = n_val = n_succ = 0
    ec_weighted = 0.0
    for _ in range(N_CHUNKS):
        y, s = gen(rng, M_CHUNK)
        r = uq.uncertainty_at(s, y, ALPHA)
        n_cov += r.n_covered
        n_val += r.n_valid
        n_succ += r.n_succ
        ec_weighted += r.EC * r.n_succ
    p, e = n_cov / n_val, ec_weighted / n_succ
    return gen, uq, p, e, 1.0 - e * p


@pytest.mark.parametrize("delta", [0.2, 0.05])
def test_bound_violation_rates_within_nominal(population, delta):
    gen, uq, p, e, u_star = population
    rng = np.random.default_rng(7)
    fail_u = fail_p = fail_e = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(REPS):
            y, s = gen(rng, N_EVAL)
            r = uq.uncertainty_at(s, y, ALPHA, delta=delta)
            fail_u += r.U_delta < u_star
            fail_p += r.coverage_lower > p
            fail_e += r.EC_lower > e
    for name, fails, level in (("U_delta", fail_u, delta),
                               ("coverage_lower", fail_p, delta / 2),
                               ("EC_lower", fail_e, delta / 2)):
        rate = fails / REPS
        assert rate <= level + _tolerance(level, REPS), (name, rate, level)

"""Monte Carlo check of the single-estimand bounds of `uncertainty_budget`. New file.

One instance is calibrated once and alpha is fixed. The population means `E[V]` and `E[W]` are
estimated from 1,000,000 samples of the same generator (standard error about 5e-4, against margins of
several 1e-2 at n = 300). R small evaluation sets are drawn; on each, the bound is violated when its
lower bound exceeds the population mean (`U_delta_V < 1 - E[V]`, `U_delta_W < 1 - E[W]`). The
violation rate is compared with `delta`, allowing four binomial standard deviations, as in the
package's other Monte Carlo tests. Fixed seeds. Classification and regression, default bound and
`Betting`.
"""
import numpy as np
import pytest

from utrace import UncertaintyQuantifier

K = 10
ALPHA = 0.2
DELTA = 0.05
N_EVAL = 300


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _regression(rng, n):
    y = rng.normal(size=n)
    return y + rng.normal(scale=0.3, size=n), y


def _tolerance(rate, reps):
    return 4.0 * np.sqrt(rate * (1 - rate) / reps)


def _population(uq, draw, rng, **kw):
    sv = sw = 0.0
    n = 0
    for _ in range(10):
        out, y = draw(rng, 100_000)
        b = uq.uncertainty_budget(out, y, ALPHA, delta=DELTA, **kw)
        sv += b.V.mean * b.V.n
        sw += b.W.mean * b.W.n
        n += b.V.n
    return sv / n, sw / n


@pytest.fixture(scope="module")
def clf():
    rng = np.random.default_rng(2026)
    cy, cs = _interior(rng, 500)
    uq = UncertaintyQuantifier(N=500)
    uq.calibrate(cs, cy)

    def draw(rng, n):
        y, s = _interior(rng, n)
        return s, y
    return uq, draw, rng, {}, _population(uq, draw, rng)


@pytest.fixture(scope="module")
def reg():
    rng = np.random.default_rng(2027)
    yh, y = _regression(rng, 500)
    uq = UncertaintyQuantifier(N=500, score="abs_error")
    uq.calibrate(yh, y)
    kw = {"tol": 0.25}      # q_hat at alpha 0.2 is about 0.38 > tol: pi < 1, B inside C
    return uq, _regression, rng, kw, _population(uq, _regression, rng, **kw)


@pytest.mark.parametrize("case", ["clf", "reg"])
@pytest.mark.parametrize("bound, reps", [(None, 600), ("betting", 300)])
def test_single_estimand_bounds_violated_at_most_delta(case, bound, reps, request):
    uq, draw, rng, kw, (eV, eW) = request.getfixturevalue(case)
    fail_v = fail_w = 0
    for _ in range(reps):
        out, y = draw(rng, N_EVAL)
        b = uq.uncertainty_budget(out, y, ALPHA, delta=DELTA, bound=bound, **kw)
        fail_v += b.V.lower > eV
        fail_w += b.W.lower > eW
    tol = DELTA + _tolerance(DELTA, reps)
    assert fail_v / reps <= tol, ("V", fail_v / reps)
    assert fail_w / reps <= tol, ("W", fail_w / reps)

"""Monte Carlo check of `utrace.bounds.Betting`: the fraction of draws whose lower bound lies above
the true mean, at `delta = 0.05`. New file.

The laws are the notebook's (`notebooks/acdc_periodic_verification.py`, section 2): a low-variance
skewed Beta(30, 3), a wide Beta(4, 1) and a Bernoulli(0.5), at n = 25 and 100. With R draws per
case, the allowed excess over `delta` is four binomial standard deviations, as in the package's
other Monte Carlo tests. Fixed seeds. The second test runs the bound through `uncertainty_at` on
synthetic classification data, against the population `e = E[pi | covered]`.
"""
import numpy as np
import pytest

from utrace import UncertaintyQuantifier
from utrace.bounds import Betting, SampleStats

DELTA = 0.05
R = 1000


def _tolerance(rate, reps):
    return 4.0 * np.sqrt(rate * (1 - rate) / reps)


@pytest.mark.parametrize("law, mu", [("beta30_3", 30 / 33), ("beta4_1", 0.8), ("bernoulli", 0.5)])
@pytest.mark.parametrize("n", [25, 100])
def test_miss_rate_at_most_delta(law, mu, n):
    rng = np.random.default_rng([20261112, 3, n, len(law)])
    draw = {"beta30_3": lambda: rng.beta(30, 3, n), "beta4_1": lambda: rng.beta(4, 1, n),
            "bernoulli": lambda: rng.binomial(1, 0.5, n).astype(float)}[law]
    misses = 0
    for i in range(R):
        x = draw()
        misses += Betting(seed=i).lower(SampleStats(n=n, mean=float(x.mean()), values=x), DELTA) > mu
    assert misses / R <= DELTA + _tolerance(DELTA, R), misses / R


K = 10
ALPHA = 0.2


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def test_ec_lower_with_betting_is_violated_at_most_half_delta():
    rng = np.random.default_rng(2025)
    cy, cs = _interior(rng, 500)
    uq = UncertaintyQuantifier(N=500)
    uq.calibrate(cs, cy)
    n_succ = 0
    ec_weighted = 0.0
    for _ in range(10):
        y, s = _interior(rng, 100_000)
        r = uq.uncertainty_at(s, y, ALPHA)
        n_succ += r.n_succ
        ec_weighted += r.EC * r.n_succ
    e = ec_weighted / n_succ
    reps, fails = 400, 0
    for _ in range(reps):
        y, s = _interior(rng, 300)
        fails += uq.uncertainty_at(s, y, ALPHA, delta=DELTA, bound="betting").EC_lower > e
    assert fails / reps <= DELTA / 2 + _tolerance(DELTA / 2, reps), fails / reps

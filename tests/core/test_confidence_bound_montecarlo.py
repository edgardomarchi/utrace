"""Monte Carlo check of the confidence bound of `uncertainty_at(..., delta=...)`.

This tests the STATISTICS of the bound (Clopper-Pearson on the coverage, the smaller of a
Hoeffding and an empirical Bernstein margin on `E[1/|C| | covered]`, union bound) on i.i.d.
synthetic data where the population values are known to high precision. It does NOT test the paper's uniform-selection assumption, nor
independence of real evaluation samples; those cannot be checked by a synthetic generator that
satisfies them by construction.

Design. One instance is calibrated once and alpha is fixed. The population coverage `p` and
`e = E[1/|C| | covered]` are estimated from M = 1,000,000 samples drawn from the same
generator (ten chunks of 100,000, accumulated from the counts of `uncertainty_at`). Their
standard errors are about 4e-4 (p) and 5e-4 (e), against bound margins of several 1e-2 at
the evaluation size used (n = 300), so the reference error is negligible relative to what is
being tested. R = 800 small evaluation sets are then drawn and, on each, the fraction where
the bound is violated (`U_delta < U* = 1 - e * p`) is compared with `delta`; the two components
(`coverage_lower > p`, `EC_lower > e`) are checked against `delta/2` each. The allowed excess
over the nominal rate is four binomial standard deviations. Fixed seeds.
"""
import warnings

import numpy as np
import pytest

from utrace import UncertaintyQuantifier

K = 10
M_CHUNK, N_CHUNKS = 100_000, 10
N_EVAL, REPS = 300, 800
ALPHA = 0.2


def _interior(rng, n):
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _tolerance(rate, reps):
    return 4.0 * np.sqrt(rate * (1 - rate) / reps)


@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(2024)
    cy, cs = _interior(rng, 500)
    uq = UncertaintyQuantifier(N=500)
    uq.calibrate(cs, cy)
    n_cov = n_val = n_succ = 0
    ec_weighted = 0.0
    for _ in range(N_CHUNKS):
        y, s = _interior(rng, M_CHUNK)
        r = uq.uncertainty_at(s, y, ALPHA)
        n_cov += r.n_covered
        n_val += r.n_valid
        n_succ += r.n_succ
        ec_weighted += r.EC * r.n_succ
    p, e = n_cov / n_val, ec_weighted / n_succ
    return uq, p, e, 1.0 - e * p


@pytest.mark.parametrize("delta", [0.2, 0.05])
def test_bound_is_violated_at_most_delta_of_the_time(population, delta):
    uq, p, e, u_star = population
    rng = np.random.default_rng(7)
    fail_u = fail_p = fail_e = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(REPS):
            y, s = _interior(rng, N_EVAL)
            r = uq.uncertainty_at(s, y, ALPHA, delta=delta)
            fail_u += r.U_delta < u_star
            fail_p += r.coverage_lower > p
            fail_e += r.EC_lower > e
    for name, fails, level in (("U_delta", fail_u, delta),
                               ("coverage_lower", fail_p, delta / 2),
                               ("EC_lower", fail_e, delta / 2)):
        rate = fails / REPS
        assert rate <= level + _tolerance(level, REPS), (name, rate, level)

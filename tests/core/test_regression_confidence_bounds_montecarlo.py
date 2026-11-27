"""
Monte Carlo check of the confidence bound of
UncertaintyQuantifier.uncertainty_at(..., delta = ...) fro regression.

This test the statistical validity of the confidence bound:
    
    U_delta = 1 - EC_lower*coverage_lower
    
using i.i.d. synthetic regression data with a fixed calibration 
instance and known population distribution.

The population quantities are:
    
    p = P(prediction interval covers the observation)
    e = E[EC_i | observation is covered]
    
where, for regression,
    
    EC_i = min(1, |C_i intersection B_i|/|C_i|)
        
C_I is the prediction interval and B_i is the tolerance interval.
The population uncertanty is 

    U_star = 1 -e*p
        
The test checks:
    
    P(U_delta < U_star)<=delta
            
and the two component guarantees:

    P(coverage_lower > p) <= delta/2
    P(EC_lower > e) <= delta /2
    
The population quantities are estimated using M = 1.000.000
independet observations, accumulated in chinks.
"""


import warnings

import numpy as np
import pytest

from utrace import UncertaintyQuantifier

M_CHUNK = 100_000
N_CHUNK = 10

N_CAL = 500
N_EVAL = 300

REPS = 800
ALPHA = 0.1
EPS_FACTOR = 0.01

DELTAS = [0.2, 0.05]

def _regression_data(rng, n):
    """
    Data generating process:
        X ~ Uniform(-2, 2)
        f(X) = sin(2X) + 0.5*X
        sigma(X) = 0.15 + 0.15*|X|
        Y = f(X) + sigma(X)*w, w ~ N(0,1).

    The regression model is:
    y_hat = f(X) + 0.05*cos(3X)
    """
    x = rng.uniform(-2.0, 2.0, size = n)
    f = np.sin(2.0*x) + 0.5*x
    sigma = 0.15 + 0.15*np.abs(x)
    w = rng.normal(size = n)
    y = f + sigma*w

    y_hat = f + 0.05*np.cos(3.0*x)
    return y_hat, y

def _tolerance(rate, reps):
    """
    Four Binomial standard deviations.
    """
    return 4.0*np.sqrt(rate*(1.0-rate)/reps)

@pytest.fixture(scope="module")
def regression_population():
    """
    Estimate the population coverage, efficiency and uncertainty.
    One fixed calibration instance is used throughout the experiment.
    Population quantities are estimated from N_CHUNKS independent
    chunks of M_CHUNKS observations each.
    """ 

    rng = np.random.default_rng(2024)

    y_hat_cal, y_cal = _regression_data(rng, N_CAL)
    eps = EPS_FACTOR*np.std(y_cal)
    uq = UncertaintyQuantifier(N = N_CAL, score = 'abs_error')
    uq.calibrate(y_hat_cal, y_cal)

    n_cov = 0
    n_val = 0
    n_succ = 0
    ec_weighted = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N_CHUNK):
            y_hat, y = _regression_data(rng, M_CHUNK)
            r = uq.uncertainty_at(
                y_hat, y, ALPHA, tol = eps,
            )
            n_cov += r.n_covered
            n_val += r.n_valid
            n_succ += r.n_succ
            ec_weighted += r.EC*r.n_succ

    assert n_val > 0, "No valid observations in population sample."
    assert n_cov > 0, "No covered observations in population sample."
    assert n_succ > 0, "No successful observations in population sample."

    p = n_cov/n_val
    e = ec_weighted / n_succ
    u_star = 1.0 - e*p
    assert 0.0 <= p <= 1.0
    assert 0.0 <= u_star <= 1.0
    assert 0.0 <= e <= 1.0

    return uq, p, e, u_star, eps

@pytest.mark.parametrize("delta", DELTAS)
def test_regression_bound_is_violated_at_most_delta_of_the_time(
    regression_population, delta,
):
    """
    Check the expiremental violation probability of the confidence bound.
    For each repretition, a fresh evaluation dataset is generated
    independently from the same population distribution.
    
    The calibration instance and ALPHA remain fixed.
    """

    uq, p, e, u_star, eps = regression_population
    rng = np.random.default_rng(4)
    fail_u = 0
    fail_p = 0
    fail_e = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(REPS):
            y_hat, y = _regression_data(rng, N_EVAL)
            r = uq.uncertainty_at(y_hat, y, ALPHA, tol = eps, delta = delta)

            fail_u += r.U_delta < u_star
            fail_p += r.coverage_lower > p 
            fail_e += r.EC_lower > e

    checks = (
        ("U_delta", fail_u, delta),
        ("coverage_lower", fail_p, delta/2.0),
        ("EC_lower", fail_e, delta/2.0)
    )

    for name, fails, level in checks:
        rate = fails/REPS
        tolerance = _tolerance(level, REPS)
        assert rate <= level + tolerance, (
            f"{name} violation rate too high: "
            f"observed = {rate:.4f}, "
            f"nominal = {level: .4f}, "
            f"tolerance = {tolerance:.4f}, "
            f"failures={fails}/{REPS}"
        )

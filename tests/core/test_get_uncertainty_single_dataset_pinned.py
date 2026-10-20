"""Single-dataset `get_uncertainty` is unchanged by the evaluation-set work.

The pinned values are the exact bytes (hex of the float64) of `(U, alpha)` and the
status, captured from `get_uncertainty` BEFORE `tune_alpha`/`uncertainty_at`/the
`eval_*` arguments were added, on the synthetic cases below (the three generators,
marginal and `classes=[3]`, several N, seeds and `max_iters`). They are compared with
`==` on bytes, not with a tolerance. All three `SearchStatus` outcomes are covered.

Added tests only. If a value here changes, the default (no evaluation set) path changed.
"""
import warnings

import numpy as np
import pytest

from utrace import UncertaintyQuantifier

K = 10


def _confident(N, seed):
    rng = np.random.default_rng(seed)

    def mk(n):
        y = rng.integers(0, K, n)
        s = np.full((n, K), 1e-4)
        s[np.arange(n), y] = 1 - 1e-4 * (K - 1)
        return y, s
    return mk(N) + mk(N)


def _near_uniform(N, seed):
    rng = np.random.default_rng(seed)
    cy = rng.integers(0, K, N)
    cs = rng.dirichlet(np.ones(K) * 0.05, N)
    ty = rng.integers(0, K, N)
    ts = rng.dirichlet(np.ones(K) * 0.05, N)
    return cy, cs, ty, ts


def _interior(N, seed):
    rng = np.random.default_rng(seed)

    def mk(n):
        y = rng.integers(0, K, n)
        s = rng.dirichlet(np.ones(K) * 0.5, n)
        s[np.arange(n), y] += 0.3
        s /= s.sum(axis=1, keepdims=True)
        return y, s
    return mk(N) + mk(N)


_GENERATORS = {"confident": _confident, "interior": _interior, "near_uniform": _near_uniform}

PINNED = {
    'interior|None|1200|0|25': ('90bb7e5628d2e23f', '00000080af99df3f', 'converged'),
    'interior|None|96|0|25': ('9a9999b198aae03f', '00000040320ddb3f', 'converged'),
    'interior|[3]|42|0|25': ('1cc7719c4d36e93f', '00000080d145e73f', 'converged'),
    'near_uniform|None|1200|2|30': ('308fad42b731ed3f', '0000004e8318ed3f', 'converged'),
    'near_uniform|None|96|2|30': ('000000035f5ced3f', '000000035f5ced3f', 'converged'),
    'near_uniform|[3]|96|1|30': ('000080d7a370ed3f', '000080d7a370ed3f', 'converged'),
    'confident|None|1200|0|25': ('00b0eaa3af484b3f', '37afeaa3af484b3f', 'floor_limited'),
    'confident|None|300|1|30': ('0007d84a48376b3f', 'ce06d84a48376b3f', 'floor_limited'),
    'confident|None|96|0|30': ('00f8e2ea071d853f', '15f8e2ea071d853f', 'floor_limited'),
    'confident|[3]|1200|2|25': ('003881133881833f', '143881133881833f', 'floor_limited'),
    'confident|[3]|42|1|25': ('989999999999c93f', '9a9999999999c93f', 'floor_limited'),
    'confident|[3]|96|2|30': ('20c7711cc771bc3f', '1cc7711cc771bc3f', 'floor_limited'),
    'near_uniform|[3]|42|1|25': ('000000000000f03f', '000000000000f03f', 'infeasible'),
    'near_uniform|[3]|42|1|30': ('000000000000f03f', '000000000000f03f', 'infeasible'),
    'near_uniform|[3]|42|2|25': ('000000000000f03f', '000000000000f03f', 'infeasible'),
    'near_uniform|[3]|42|2|30': ('000000000000f03f', '000000000000f03f', 'infeasible'),
    'near_uniform|[3]|96|2|25': ('000000000000f03f', '000000000000f03f', 'infeasible'),
    'near_uniform|[3]|96|2|30': ('000000000000f03f', '000000000000f03f', 'infeasible'),
}


@pytest.mark.parametrize("key", sorted(PINNED))
def test_single_dataset_get_uncertainty_is_byte_identical_to_pre_change_capture(key):
    gname, classes, N, seed, max_iters = key.split("|")
    classes = None if classes == "None" else [3]
    N, seed, max_iters = int(N), int(seed), int(max_iters)
    n = N * 5 if classes and gname != "confident" else N
    cy, cs, ty, ts = _GENERATORS[gname](n, seed)

    uq = UncertaintyQuantifier(N=N * 5, classes=classes)
    uq.calibrate(cs, cy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, alpha = uq.get_uncertainty(ts, ty, max_iters=max_iters)

    u_hex, a_hex, status = PINNED[key]
    assert np.float64(U).tobytes().hex() == u_hex
    assert np.float64(alpha).tobytes().hex() == a_hex
    assert uq.search_status_.value == status


def test_pinned_cases_cover_all_three_statuses():
    assert {v[2] for v in PINNED.values()} == {"converged", "floor_limited", "infeasible"}

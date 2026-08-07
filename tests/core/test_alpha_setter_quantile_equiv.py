# CONTRACT: NaN buffers are OUT OF SCOPE.
# LAC conformity scores (1 - softmax[y]) are always finite; the two quantile
# paths are known to diverge on NaN buffers, and that case is deliberately
# excluded here.  The q_level <= 1.0 cap is PART of the contract and is
# replicated below exactly as the alpha setter does it (line 226-228 of
# uncertaintyQuantifier.py).
"""Equivalence test: np.nanquantile(..., method='higher') == _masked_quantile_higher.

Both functions implement arr[ceil(q*(n-1))] clipped to n-1 on a sorted array.
This test pins that they are byte-identical on real calibrated buffers from
UncertaintyQuantifier, across the full operating alpha/N grid.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from utrace import UncertaintyQuantifier
from utrace.utils import _masked_quantile_higher


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_probas(n_samples: int, n_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n_samples, n_classes))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def _make_labels(n_samples: int, n_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    return rng.integers(0, n_classes, n_samples).astype(np.int32)


def _build_calibrated_uq(N: int, n_classes: int = 10, seed: int = 42,
                          tied: bool = False) -> UncertaintyQuantifier:
    """Build a UQ with exactly N calibration scores in the buffer."""
    uq = UncertaintyQuantifier(N=N, classes=None)
    if tied:
        # All rows identical → all LAC scores identical (= 1 - 1/n_classes)
        row = np.ones(n_classes, dtype=np.float64) / n_classes
        probas = np.tile(row, (N, 1))
        y = np.zeros(N, dtype=np.int32)
    else:
        probas = _make_probas(N, n_classes, seed)
        y = _make_labels(N, n_classes, seed)
    uq.calibrate_from_proba(probas, y, batched=False)
    assert uq._state.N == N, f"Expected _state.N={N}, got {uq._state.N}"
    return uq


def _q_level(N: int, alpha: float) -> np.float64:
    """Replicate the setter formula and cap exactly as in uncertaintyQuantifier.py:225-228."""
    q = np.divide(np.ceil((N + 1) * (1 - alpha)), N, dtype=np.float64)
    return np.float64(1.0) if q > 1.0 else q


def _alpha_grid(N: int) -> np.ndarray:
    """Alpha values: dense linspace + explicit edges + cap-triggering alpha."""
    grid = np.linspace(0.001, 0.999, 500)
    edges = np.array([1e-9, 0.5, 1 - 1e-9])
    # alpha < 1/(N+1) triggers the q_level > 1 cap
    alpha_cap = 0.5 / (N + 1)
    return np.unique(np.concatenate([grid, edges, [alpha_cap]]))


# ── parametrize ───────────────────────────────────────────────────────────────

N_VALUES = [1, 2, 5, 10, 100, 600, 1200, 5000, 50000]


@pytest.mark.parametrize("N", N_VALUES)
def test_nanquantile_matches_masked_quantile_higher(N):
    """np.nanquantile(cs[:N], q, method='higher') == _masked_quantile_higher(cs, N, q)
    for every alpha in the operating grid, on a real calibrated buffer.
    """
    uq = _build_calibrated_uq(N, seed=7)
    cs = uq.conformity_scores_  # full padded jnp buffer
    alphas = _alpha_grid(N)

    for alpha in alphas:
        q = _q_level(N, float(alpha))
        oracle = np.nanquantile(np.asarray(cs[:N]), float(q), method='higher')
        candidate = np.float64(
            _masked_quantile_higher(cs, jnp.int32(N), jnp.float64(q))
        )
        np.testing.assert_array_equal(
            candidate, oracle,
            err_msg=f"N={N}, alpha={alpha:.6g}, q_level={q:.17g}: "
                    f"oracle={oracle:.17g}, candidate={candidate:.17g}",
        )


def test_nanquantile_matches_masked_quantile_higher_tied_scores():
    """Same equivalence on a tied (all-equal) score buffer (N=600)."""
    N = 600
    uq = _build_calibrated_uq(N, seed=0, tied=True)
    cs = uq.conformity_scores_
    alphas = _alpha_grid(N)

    for alpha in alphas:
        q = _q_level(N, float(alpha))
        oracle = np.nanquantile(np.asarray(cs[:N]), float(q), method='higher')
        candidate = np.float64(
            _masked_quantile_higher(cs, jnp.int32(N), jnp.float64(q))
        )
        np.testing.assert_array_equal(
            candidate, oracle,
            err_msg=f"tied N={N}, alpha={alpha:.6g}, q_level={q:.17g}: "
                    f"oracle={oracle:.17g}, candidate={candidate:.17g}",
        )

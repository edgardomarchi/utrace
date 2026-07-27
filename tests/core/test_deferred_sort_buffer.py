"""Tests for deferred-sort buffer semantics in _calibrate_impl + conformity_scores_ property.

Invariants under test:
- _sorted flag is False immediately after any calibrate write, True after a getter read.
- The lazy sort produces the same result as np.sort of the raw unsorted scores.
- reset() restores a clean (sorted) empty buffer.
- No sort is triggered between batch writes; sort fires exactly on the first post-loop read.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from utrace import UncertaintyQuantifier


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_probas(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n, k))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def _make_labels(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 99).integers(0, k, n).astype(np.int32)


# ── cases ─────────────────────────────────────────────────────────────────────

def test_property_sorts_on_read():
    """_sorted is False right after calibrate; True after the property is accessed; prefix is sorted."""
    # Non-batched path
    uq = UncertaintyQuantifier(N=200, classes=None)
    uq.calibrate_from_proba(_make_probas(100, 10, 1), _make_labels(100, 10, 1), batched=False)

    assert uq._sorted is False, "_sorted must be False right after calibrate write"
    _ = uq.conformity_scores_
    assert uq._sorted is True, "getter must set _sorted = True"
    cs = np.asarray(uq.conformity_scores_[:uq._N])
    assert np.all(np.diff(cs) >= 0), "valid prefix must be sorted ascending after getter"

    # Batched multi-call path
    uq2 = UncertaintyQuantifier(N=500, classes=None)
    for i in range(3):
        uq2.calibrate_from_proba(_make_probas(50, 10, i), _make_labels(50, 10, i), batched=True)
        assert uq2._sorted is False, f"_sorted must remain False between batches (batch {i})"
    _ = uq2.conformity_scores_
    assert uq2._sorted is True
    cs2 = np.asarray(uq2.conformity_scores_[:uq2._N])
    assert np.all(np.diff(cs2) >= 0), "valid prefix must be sorted ascending after getter (batched)"


def test_deferred_equals_eager_sort():
    """valid prefix after getter == np.sort of raw pre-sort backing store; padding stays +inf."""
    K, BATCH_N, N_BATCHES = 10, 60, 5

    uq = UncertaintyQuantifier(N=400, classes=None)
    for s in range(N_BATCHES):
        uq.calibrate_from_proba(_make_probas(BATCH_N, K, s), _make_labels(BATCH_N, K, s), batched=True)

    N = uq._N
    # Snapshot the raw (unsorted) backing store before the getter touches it.
    raw_before_sort = np.asarray(uq._conformity_scores_[:N]).copy()

    cs = np.asarray(uq.conformity_scores_)  # triggers lazy sort

    np.testing.assert_array_equal(
        cs[:N], np.sort(raw_before_sort),
        err_msg="valid prefix must be exactly np.sort of the pre-sort raw scores",
    )
    assert np.all(np.isinf(cs[N:])), "padding region [N:] must be all +inf"


def test_reset_flag():
    """After reset(), _sorted is True; a conformity_scores_ read does not error and leaves _N=0."""
    uq = UncertaintyQuantifier(N=100, classes=None)
    uq.calibrate_from_proba(_make_probas(50, 10, 5), _make_labels(50, 10, 5), batched=False)
    assert uq._sorted is False  # sanity: dirty after calibrate

    uq.reset()
    assert uq._sorted is True, "reset() must initialise _sorted = True"
    assert uq._N == 0

    buf = uq.conformity_scores_  # must not error on empty buffer
    assert uq._N == 0, "reading the property on empty buffer must not change _N"
    assert jnp.all(jnp.isinf(buf)), "empty buffer must be all +inf"


def test_no_sort_between_batches():
    """_sorted stays False across all batch writes; flips to True only on first post-loop read."""
    uq = UncertaintyQuantifier(N=500, classes=None)
    N_BATCHES = 4
    for i in range(N_BATCHES):
        uq.calibrate_from_proba(_make_probas(40, 10, i), _make_labels(40, 10, i), batched=True)
        assert uq._sorted is False, f"no sort should occur during calibration (batch {i})"

    # First read triggers the lazy sort
    _ = uq.conformity_scores_
    assert uq._sorted is True, "_sorted must flip to True on first post-calibration read"

    # Subsequent reads do not re-sort (flag stays clean)
    _ = uq.conformity_scores_
    assert uq._sorted is True, "_sorted must remain True on subsequent reads (no spurious re-sort)"

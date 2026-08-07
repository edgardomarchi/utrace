"""Tests for _max_N calibration score buffer overflow guards in UncertaintyQuantifier."""

import numpy as np
import pytest
from utrace import UncertaintyQuantifier


def _make_dummy_data(n_samples: int, n_classes: int = 3):
    """Helper to generate dummy probabilities and labels using numpy."""
    rng = np.random.default_rng(42)
    probs = rng.random((n_samples, n_classes))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, n_classes, size=n_samples)
    return probs, labels


def test_overflow_batched_write_past_full_buffer():
    """Case (1): Batched write starting past a full buffer raises ValueError."""
    max_N = 100
    uq = UncertaintyQuantifier(N=max_N)
    probs, labels = _make_dummy_data(max_N)
    uq.calibrate_from_proba(probs, labels, batched=True)
    assert uq._state.N == max_N

    # Next batched write when buffer is already full
    extra_probs, extra_labels = _make_dummy_data(10)
    with pytest.raises(ValueError) as exc_info:
        uq.calibrate_from_proba(extra_probs, extra_labels, batched=True)

    msg = str(exc_info.value)
    # Key numbers must be in message: current _N (100), incoming (10), max_N (100)
    assert "100" in msg
    assert "10" in msg
    # Check that state remains uncorrupted
    assert uq._state.N == max_N


def test_overflow_batched_write_straddling_boundary():
    """Case (2): Batched write straddling the boundary raises ValueError."""
    max_N = 100
    uq = UncertaintyQuantifier(N=max_N)
    probs, labels = _make_dummy_data(80)
    uq.calibrate_from_proba(probs, labels, batched=True)
    assert uq._state.N == 80

    # Incoming batch of 30 straddles 80 + 30 = 110 > 100
    straddle_probs, straddle_labels = _make_dummy_data(30)
    with pytest.raises(ValueError) as exc_info:
        uq.calibrate_from_proba(straddle_probs, straddle_labels, batched=True)

    msg = str(exc_info.value)
    assert "80" in msg
    assert "30" in msg
    assert "100" in msg
    # Verify state remains uncorrupted at N=80
    assert uq._state.N == 80


def test_overflow_non_batched_write_larger_than_max_n():
    """Case (3): Non-batched write larger than _max_N raises ValueError."""
    max_N = 50
    uq = UncertaintyQuantifier(N=max_N)
    large_probs, large_labels = _make_dummy_data(60)

    with pytest.raises(ValueError) as exc_info:
        uq.calibrate_from_proba(large_probs, large_labels, batched=False)

    msg = str(exc_info.value)
    assert "60" in msg
    assert "50" in msg
    # Verify state remains 0
    assert uq._state.N == 0


def test_negative_batched_exact_capacity_fill_succeeds():
    """Negative test: a batched sequence that exactly FILLS the buffer must succeed."""
    max_N = 100
    uq = UncertaintyQuantifier(N=max_N)

    p1, l1 = _make_dummy_data(60)
    uq.calibrate_from_proba(p1, l1, batched=True)
    assert uq._state.N == 60

    p2, l2 = _make_dummy_data(40)
    uq.calibrate_from_proba(p2, l2, batched=True)
    assert uq._state.N == max_N

    # Setting alpha should succeed without error
    uq.alpha = 0.1
    assert not np.isnan(uq._state.q_hat)

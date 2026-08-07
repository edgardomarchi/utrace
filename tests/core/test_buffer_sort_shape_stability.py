"""Pins the shape-stability and correctness invariants of the full-buffer
sort in conformity_scores_ (Step B1): the property now sorts the WHOLE
fixed-size buffer, not the variable-length prefix [:self._state.N], so its output
shape is always (_max_N,) regardless of _state.N.

Honest scope: this test pins the padding invariant, the sortedness of the
valid prefix, and the fixed output shape across several different _state.N
values, including the _state.N==0, _state.N==1, and _state.N==_max_N edges. It does NOT by
itself prove a compilation-count improvement — that is measured once and
reported in the commit, not asserted per-run.
"""
import numpy as np

from utrace import UncertaintyQuantifier


def _make_softmax(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n, k))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def _make_labels(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 99).integers(0, k, n).astype(np.int32)


def _assert_invariants(uq: UncertaintyQuantifier, max_N: int):
    cs = np.asarray(uq.conformity_scores_)
    N = uq._state.N
    assert cs.shape == (max_N,), f"buffer shape must always be (_max_N,)={(max_N,)}, got {cs.shape}"
    valid = cs[:N]
    padding = cs[N:]
    if N > 1:
        assert np.all(np.diff(valid) >= 0), "valid prefix must be sorted ascending"
    assert np.all(np.isinf(padding)), "region beyond _state.N must be all +inf"


def test_shape_stable_across_several_distinct_N():
    """Same instance, several batched calibrations reaching different _state.N
    values: the buffer's shape never changes, and each read is correctly
    sorted/padded."""
    max_N = 500
    uq = UncertaintyQuantifier(N=max_N, classes=None)

    for i, bs in enumerate([30, 45, 20, 60]):
        uq.calibrate(_make_softmax(bs, 8, i), _make_labels(bs, 8, i), batched=True)
        _assert_invariants(uq, max_N)


def test_N_equals_zero():
    """Freshly constructed instance, never calibrated: _state.N == 0."""
    max_N = 300
    uq = UncertaintyQuantifier(N=max_N, classes=None)
    assert uq._state.N == 0
    _assert_invariants(uq, max_N)


def test_N_equals_one():
    max_N = 300
    uq = UncertaintyQuantifier(N=max_N, classes=None)
    uq.calibrate(_make_softmax(1, 5, 0), _make_labels(1, 5, 0), batched=False)
    assert uq._state.N == 1
    _assert_invariants(uq, max_N)


def test_N_equals_max_N():
    """Buffer completely filled: no padding region at all."""
    max_N = 150
    uq = UncertaintyQuantifier(N=max_N, classes=None)
    uq.calibrate(_make_softmax(max_N, 6, 0), _make_labels(max_N, 6, 0), batched=False)
    assert uq._state.N == max_N
    _assert_invariants(uq, max_N)

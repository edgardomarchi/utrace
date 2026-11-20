"""Labels outside `[0, K)` must raise on all three entry points.

Before, `-1` wrapped to the last class and any label `>= K` clamped to it (JAX gather
semantics), silently producing a wrong coverage and `EC`. New file only.
"""
import warnings

import numpy as np
import pytest

from utrace import UncertaintyQuantifier

K = 6


def _data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    smx = rng.dirichlet(np.ones(K) * 0.5, n)
    smx[np.arange(n), y] += 0.3
    smx /= smx.sum(axis=1, keepdims=True)
    return y, smx


def _calibrated(classes=None):
    y, s = _data(600, 1)
    uq = UncertaintyQuantifier(N=600, classes=classes)
    uq.calibrate(s, y)
    return uq


BAD = [-1, K, 255]


@pytest.mark.parametrize("classes", [None, [2]])
@pytest.mark.parametrize("bad", BAD)
def test_calibrate_rejects(classes, bad):
    y, s = _data(300, 2)
    y[5] = bad
    uq = UncertaintyQuantifier(N=600, classes=classes)
    with pytest.raises(ValueError, match=rf"K={K}.*max={max(bad, int(y.max()))}|K={K}.*min=-1"):
        uq.calibrate(s, y)
    assert uq._state.N == 0 and uq._item_shape is None   # failed call leaves no trace


@pytest.mark.parametrize("classes", [None, [2]])
@pytest.mark.parametrize("bad", BAD)
def test_tuning_path_rejects(classes, bad):
    uq = _calibrated(classes)
    y, s = _data(300, 3)
    y[5] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=f"K={K}"):
            uq.get_uncertainty(s, y)
        with pytest.raises(ValueError, match=f"K={K}"):
            uq.tune_alpha(s, y)


@pytest.mark.parametrize("classes", [None, [2]])
@pytest.mark.parametrize("bad", BAD)
def test_uncertainty_at_rejects(classes, bad):
    uq = _calibrated(classes)
    y, s = _data(300, 4)
    y[5] = bad
    with pytest.raises(ValueError, match=f"K={K}"):
        uq.uncertainty_at(s, y, 0.2)
    with pytest.raises(ValueError, match=f"K={K}"):
        uq.uncertainty_at(s, y, 0.2, delta=0.05)


def test_eval_labels_rejected_through_get_uncertainty():
    uq = _calibrated()
    ty, ts = _data(300, 5)
    ey, es = _data(300, 6)
    ey[0] = K
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=f"K={K}"):
            uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=ey)


def test_message_names_min_max_and_K():
    uq = _calibrated()
    y, s = _data(100, 7)
    y[0], y[1] = -3, 9
    with pytest.raises(ValueError) as e:
        uq.uncertainty_at(s, y, 0.2)
    assert "min=-3" in str(e.value) and "max=9" in str(e.value) and f"K={K}" in str(e.value)


def test_valid_boundary_labels_accepted():
    uq = _calibrated()
    y, s = _data(100, 8)
    y[0], y[1] = 0, K - 1
    assert uq.uncertainty_at(s, y, 0.2).n_valid == 100


@pytest.mark.parametrize("bad_classes", [[K], [-1], [0, K + 3]])
def test_calibrate_rejects_classes_outside_range(bad_classes):
    y, s = _data(300, 9)
    uq = UncertaintyQuantifier(N=600, classes=bad_classes)
    with pytest.raises(ValueError, match="classes"):
        uq.calibrate(s, y)


def test_classes_inside_range_accepted():
    _calibrated([0, K - 1])


# ----- shape checks and their order ----------------------------------------------------

def test_column_mismatch_message_wins_over_labels():
    uq = _calibrated()
    y, s = _data(100, 10)
    y[0] = K - 1            # valid for K, out of range for K-1 columns
    with pytest.raises(ValueError, match="calibrated with"):
        uq.uncertainty_at(s[:, :K - 1], y, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="calibrated with"):
            uq.tune_alpha(s[:, :K - 1], y)


def test_tuning_path_wrong_width_raises():
    uq = _calibrated()
    y, s = _data(100, 11)
    y[:] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="calibrated with"):
            uq.get_uncertainty(np.hstack([s, s[:, :1]]), y)


def test_batched_calibrate_with_other_trailing_shape_raises():
    uq = UncertaintyQuantifier(N=600)
    y, s = _data(100, 12)
    uq.calibrate(s, y, batched=True)
    with pytest.raises(ValueError, match="calibrated with"):
        uq.calibrate(s[:, :K - 1], np.zeros(100, dtype=int), batched=True)
    assert uq._state.N == 100 and uq._item_shape == (K,)
    uq.calibrate(s, y, batched=True)     # same shape still appends
    assert uq._state.N == 200


def test_nonbatched_calibrate_may_change_shape():
    uq = UncertaintyQuantifier(N=600)
    y, s = _data(100, 13)
    uq.calibrate(s, y)
    uq.calibrate(s[:, :K - 1], np.minimum(y, K - 2))
    assert uq._item_shape == (K - 1,)


def test_label_check_skipped_when_K_unknown():
    uq = _calibrated()
    uq._item_shape = None            # state filled without calibrate
    y, s = _data(100, 14)
    y[0] = 255
    uq.uncertainty_at(s, y, 0.2)     # no error: K unknown
    uq._item_shape = ()              # 1-D (regression-style) rows: length != 1
    assert uq._label_range() is None


def test_nan_alpha_returns_before_any_validation():
    uq = _calibrated()
    y, s = _data(100, 15)
    y[0] = 255
    assert np.isnan(uq.uncertainty_at(s, y, np.nan).U)

"""Label dtype policy on every entry point that runs the `[0, K)` check.

Before, labels were cast with `astype(int)`/`astype(int32)`, which truncates toward zero:
`0.9` became class 0 and `-0.5` became 0 and passed the range check, silently producing a
wrong coverage and `EC`. Now integer and boolean dtypes are accepted, floating dtypes only
when every value is finite and integral, and any other dtype raises `ValueError`.

The framework-tensor path (checked after `to_jax`) is exercised with jax arrays; torch is
not imported here (tests/core/ is torch-free). New file only.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from utrace import UncertaintyQuantifier

K = 6
ALPHA = 0.2


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


# Each entry point as `call(uq, y) -> result`, with `uq` calibrated except for "calibrate".
def _calibrate(uq, y):
    _, s = _data(len(y), 2)
    uq.calibrate(s, y)
    return np.asarray(uq.conformity_scores_)


def _get_uncertainty_single(uq, y):
    _, s = _data(len(y), 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return uq.get_uncertainty(s, y), uq.search_status_


def _tune_alpha(uq, y):
    _, s = _data(len(y), 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return uq.tune_alpha(s, y), uq.search_status_


def _get_uncertainty_eval_y(uq, y):
    ty, ts = _data(400, 3)
    _, es = _data(len(y), 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return uq.get_uncertainty(ts, ty, eval_softmax=es, eval_y=y)


def _uncertainty_at(uq, y):
    _, s = _data(len(y), 4)
    return tuple(uq.uncertainty_at(s, y, ALPHA, delta=0.05))


ENTRY_POINTS = {
    "calibrate": _calibrate,
    "get_uncertainty": _get_uncertainty_single,
    "tune_alpha": _tune_alpha,
    "get_uncertainty_eval_y": _get_uncertainty_eval_y,
    "uncertainty_at": _uncertainty_at,
}


def _fresh(name, classes=None):
    if name == "calibrate":
        return UncertaintyQuantifier(N=600, classes=classes)
    return _calibrated(classes)


def _same(a, b):
    """Exact equality of nested results, `nan == nan`, compared on bytes for floats."""
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None or not isinstance(a, (np.ndarray, float, np.floating, int)):
        return a == b
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _labels(n=300, seed=5):
    return _data(n, seed)[0]


# --- values ------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.9, -0.5, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("to_array", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_non_integral_or_non_finite_float_rejected(name, to_array, bad):
    y = _labels().astype(np.float64)
    y[7] = bad
    uq = _fresh(name)
    with pytest.raises(ValueError, match="finite and integral"):
        ENTRY_POINTS[name](uq, to_array(y))
    if name == "calibrate":
        assert uq._state.N == 0 and uq._item_shape is None   # failed call leaves no trace


@pytest.mark.parametrize("classes", [None, [2]])
@pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_integral_float_equals_integer(name, float_dtype, classes):
    y = _labels()
    as_int = ENTRY_POINTS[name](_fresh(name, classes), y)
    as_float = ENTRY_POINTS[name](_fresh(name, classes), y.astype(float_dtype))
    as_jax_float = ENTRY_POINTS[name](_fresh(name, classes), jnp.asarray(y, dtype=float_dtype))
    assert _same(as_int, as_float)
    assert _same(as_int, as_jax_float)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_two_point_zero_equals_two(name):
    y = _labels()
    y[:20] = 2
    yf = y.astype(np.float64)
    assert yf[0] == 2.0
    assert _same(ENTRY_POINTS[name](_fresh(name), y), ENTRY_POINTS[name](_fresh(name), yf))


def test_message_names_first_offending_value_and_counts():
    y = _labels().astype(np.float64)
    y[3] = 0.9
    y[10] = -0.5
    y[20] = 0.9
    with pytest.raises(ValueError, match=r"calibrate: .* got 0\.9 \(2 occurrences; 3 invalid labels in total\)"):
        _calibrate(_fresh("calibrate"), y)
    y32 = _labels().astype(np.float32)
    y32[9] = 0.9
    with pytest.raises(ValueError, match=r"tune_alpha: .* got 0\.9 \(1 occurrence;"):
        _tune_alpha(_calibrated(), jnp.asarray(y32))   # float32 shown as 0.9, not 0.89999...
    y = _labels().astype(np.float64)
    y[50] = np.nan
    with pytest.raises(ValueError, match=r"uncertainty_at: .* got nan \(1 occurrence; 1 invalid label in total\)"):
        _uncertainty_at(_calibrated(), y)


@pytest.mark.parametrize("bad", [-1.0, float(K), 255.0])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_integral_out_of_range_float_keeps_range_message(name, bad):
    y = _labels().astype(np.float64)
    y[7] = bad
    with pytest.raises(ValueError, match=rf"labels must lie in \[0, K\) with K={K}") as exc:
        ENTRY_POINTS[name](_fresh(name), y)
    assert "finite and integral" not in str(exc.value)


# --- dtype classes -----------------------------------------------------------------------

INTEGER_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_integer_dtypes_accepted(name, dtype):
    y = _labels()
    assert _same(ENTRY_POINTS[name](_fresh(name), y),
                 ENTRY_POINTS[name](_fresh(name), y.astype(dtype)))


@pytest.mark.parametrize("to_array", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_bool_maps_to_zero_one(name, to_array):
    yb = _labels() % 2 == 1
    assert _same(ENTRY_POINTS[name](_fresh(name), yb.astype(np.int64)),
                 ENTRY_POINTS[name](_fresh(name), to_array(yb)))


@pytest.mark.parametrize("make", [
    lambda y: y.astype(np.complex128),
    lambda y: y.astype(np.complex64),
    lambda y: jnp.asarray(y, dtype=jnp.complex64),
    lambda y: y.astype(object),
    lambda y: y.astype(str),
    lambda y: y.astype(bytes),
    lambda y: [str(v) for v in y],
], ids=["complex128", "complex64", "jax_complex64", "object", "str", "bytes", "list_of_str"])
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_other_dtypes_rejected(name, make):
    y = make(_labels())
    with pytest.raises(ValueError, match="integer, boolean or floating dtype"):
        ENTRY_POINTS[name](_fresh(name), y)


def test_float_list_follows_float_rule():
    y = _labels()
    assert _same(_uncertainty_at(_calibrated(), [float(v) for v in y]),
                 _uncertainty_at(_calibrated(), y))
    bad = [float(v) for v in y]
    bad[0] = 0.5
    with pytest.raises(ValueError, match="finite and integral"):
        _uncertainty_at(_calibrated(), bad)

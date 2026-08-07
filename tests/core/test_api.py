"""Tests for the calibrate/predict/get_uncertainty public API of UncertaintyQuantifier.

These tests verify the contract of each method against itself, using
numpy/jnp inputs only. The equivalence between this API and the legacy
*(X)* methods that require a torch model lives in
tests/integration/torch/test_legacy_equivalence.py.
"""
import numpy as np
import jax.numpy as jnp
import pytest
import warnings

from utrace import UncertaintyQuantifier


# ----- helpers ---------------------------------------------------------

def _make_softmax(n_samples: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """Random softmax-output matrix, each row summing to 1."""
    rng = np.random.default_rng(seed)
    p = rng.random((n_samples, n_classes))
    return p / p.sum(axis=1, keepdims=True)


def _make_labels(n_samples: int, n_classes: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, n_samples).astype(np.int32)


# ----- calibrate --------------------------------------------

class TestCalibrate:

    def test_accepts_numpy_input(self):
        """Calibration should accept plain numpy arrays."""
        softmax = _make_softmax(500, 10)
        y = _make_labels(500, 10)
        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax, y, batched=True)
        assert uq._state.N > 0

    def test_accepts_jnp_input(self):
        """Calibration should equally accept jax.numpy arrays."""
        softmax = jnp.asarray(_make_softmax(500, 10))
        y = jnp.asarray(_make_labels(500, 10))
        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax, y, batched=True)
        assert uq._state.N > 0

    def test_numpy_and_jnp_inputs_give_identical_state(self):
        """Same data via numpy or jnp produces identical conformity_scores_."""
        softmax_np = _make_softmax(500, 10)
        y_np = _make_labels(500, 10)

        uq_np = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq_np.calibrate(softmax_np, y_np, batched=True)

        uq_jnp = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq_jnp.calibrate(jnp.asarray(softmax_np), jnp.asarray(y_np),
                                    batched=True)

        np.testing.assert_array_equal(np.asarray(uq_np.conformity_scores_),
                                      np.asarray(uq_jnp.conformity_scores_))
        assert uq_np._state.N == uq_jnp._state.N

    def test_batched_accumulates_scores(self):
        """Two batched calls should accumulate (final _N = sum of both)."""
        softmax1 = _make_softmax(200, 10, seed=0)
        y1 = _make_labels(200, 10, seed=10)
        softmax2 = _make_softmax(300, 10, seed=1)
        y2 = _make_labels(300, 10, seed=11)

        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax1, y1, batched=True)
        n_after_first = uq._state.N
        uq.calibrate(softmax2, y2, batched=True)

        assert uq._state.N > n_after_first
        # _state.N should be sum of class-3 occurrences across both batches
        expected_n = int((y1 == 3).sum() + (y2 == 3).sum())
        assert uq._state.N == expected_n

    def test_reset_clears_state(self):
        softmax = _make_softmax(500, 10)
        y = _make_labels(500, 10)
        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax, y, batched=True)
        assert uq._state.N > 0

        uq.reset()
        assert uq._state.N == 0
        # padding invariant after reset
        assert jnp.all(jnp.isinf(uq.conformity_scores_))

    def test_conformity_scores_are_sorted_and_padded(self):
        """After calibration, scores[:N] sorted ascending, scores[N:] are +inf."""
        softmax = _make_softmax(500, 10)
        y = _make_labels(500, 10)
        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax, y, batched=True)

        cs = np.asarray(uq.conformity_scores_)
        valid = cs[:uq._state.N]
        padding = cs[uq._state.N:]
        assert np.all(valid[:-1] <= valid[1:]), "valid scores must be sorted ascending"
        assert np.all(np.isinf(padding)), "padding must be +inf"


# ----- predict ----------------------------------------------

class TestPredict:

    @pytest.fixture
    def calibrated_uq(self):
        softmax = _make_softmax(500, 10, seed=0)
        y = _make_labels(500, 10, seed=10)
        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax, y, batched=True)
        uq.alpha = np.float64(0.1)  # set alpha to fix q_hat
        return uq

    def test_returns_pair_of_arrays(self, calibrated_uq):
        softmax = _make_softmax(100, 10, seed=2)
        y_pred, y_sets = calibrated_uq.predict(softmax)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_sets, np.ndarray)

    def test_shapes_match_input(self, calibrated_uq):
        softmax = _make_softmax(100, 10, seed=2)
        y_pred, y_sets = calibrated_uq.predict(softmax)
        assert y_pred.shape == (100,)
        assert y_sets.shape == (100, 10)

    def test_sets_are_boolean(self, calibrated_uq):
        softmax = _make_softmax(100, 10, seed=2)
        _, y_sets = calibrated_uq.predict(softmax)
        assert y_sets.dtype == bool

    def test_y_pred_is_argmax(self, calibrated_uq):
        """The predicted class should be the argmax of the input softmax output."""
        softmax = _make_softmax(100, 10, seed=2)
        y_pred, _ = calibrated_uq.predict(softmax)
        np.testing.assert_array_equal(y_pred, softmax.argmax(axis=1))

    def test_numpy_and_jnp_give_same_sets(self, calibrated_uq):
        softmax_np = _make_softmax(100, 10, seed=2)
        y_pred_n, y_sets_n = calibrated_uq.predict(softmax_np)
        y_pred_j, y_sets_j = calibrated_uq.predict(jnp.asarray(softmax_np))
        np.testing.assert_array_equal(y_pred_n, y_pred_j)
        np.testing.assert_array_equal(y_sets_n, y_sets_j)


# ----- get_uncertainty --------------------------------------

class TestGetUncertainty:

    @pytest.fixture
    def calibrated_uq(self):
        softmax = _make_softmax(500, 10, seed=0)
        y = _make_labels(500, 10, seed=10)
        uq = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
        uq.calibrate(softmax, y, batched=True)
        return uq

    def test_returns_pair_of_floats(self, calibrated_uq):
        softmax = _make_softmax(300, 10, seed=2)
        y = _make_labels(300, 10, seed=12)
        U, alpha = calibrated_uq.get_uncertainty(softmax, y)
        assert isinstance(U, float)
        assert isinstance(alpha, float)

    def test_alpha_in_unit_interval(self, calibrated_uq):
        softmax = _make_softmax(300, 10, seed=2)
        y = _make_labels(300, 10, seed=12)
        _, alpha = calibrated_uq.get_uncertainty(softmax, y)
        assert 0.0 <= alpha <= 1.0

    def test_no_valid_samples_returns_nan(self, calibrated_uq):
        """If no sample matches the class of interest, return (nan, nan)."""
        softmax = _make_softmax(50, 10, seed=2)
        # construct y with no class-3 samples
        y = np.array([c for c in range(10) if c != 3] * 6, dtype=np.int32)[:50]
        assert (y == 3).sum() == 0
        U, alpha = calibrated_uq.get_uncertainty(softmax, y)
        assert np.isnan(U) and np.isnan(alpha)

    def test_numpy_and_jnp_give_same_result(self, calibrated_uq):
        softmax_np = _make_softmax(300, 10, seed=2)
        y_np = _make_labels(300, 10, seed=12)
        U_n, a_n = calibrated_uq.get_uncertainty(softmax_np, y_np)
        U_j, a_j = calibrated_uq.get_uncertainty(
            jnp.asarray(softmax_np), jnp.asarray(y_np))
        np.testing.assert_allclose(U_n, U_j, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a_n, a_j, rtol=1e-12, atol=1e-12)


# ----- constructor deprecation -----------------------------------------

def test_constructor_no_warning_without_model():
    """Default construction (no model) should NOT emit DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # convert to error
        UncertaintyQuantifier(N=100, classes=[0], max_batch_size=64)
        # If a DeprecationWarning fires here, the line above raises.
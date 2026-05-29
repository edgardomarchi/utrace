# tests/core/test_from_proba_api.py
import numpy as np
import jax.numpy as jnp
import pytest
import warnings

from utrace import UncertaintyQuantifier


class _MockModel:
    """Minimal model wrapper that returns fixed precomputed probas.
    Stands in for a real torch/tf model in the legacy API path."""
    def __init__(self, probas):
        self._probas = probas
    def predict_proba(self, X):
        # X here is a torch tensor; we ignore it and return precomputed probas
        # converted to the type the legacy code expects
        import torch
        return torch.from_numpy(self._probas)


def test_calibrate_from_proba_equals_calibrate():
    rng = np.random.default_rng(42)
    K, N_cal = 10, 500
    probas = rng.random((N_cal, K)); probas /= probas.sum(axis=1, keepdims=True)
    y = rng.integers(0, K, N_cal)

    # Legacy API
    import torch
    uq_legacy = UncertaintyQuantifier(
        N=1000, classes=[3], model=_MockModel(probas), max_batch_size=512)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        uq_legacy.calibrate(torch.zeros(N_cal, 1, 28, 28),  # dummy X
                            torch.from_numpy(y), batched=True)

    # New API
    uq_new = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
    uq_new.calibrate_from_proba(probas, y, batched=True)

    np.testing.assert_array_equal(
        np.asarray(uq_legacy.conformity_scores_),
        np.asarray(uq_new.conformity_scores_),
    )
    assert uq_legacy._N == uq_new._N


def test_get_uncertainty_from_proba_equals_legacy():
    rng = np.random.default_rng(42)
    K = 10
    
    # Calibrar ambos con los mismos datos
    probas_cal = rng.random((500, K)); probas_cal /= probas_cal.sum(axis=1, keepdims=True)
    y_cal = rng.integers(0, K, 500)
    
    uq_legacy = UncertaintyQuantifier(N=1000, classes=[3], model=None, max_batch_size=512)
    uq_new    = UncertaintyQuantifier(N=1000, classes=[3], max_batch_size=512)
    
    # Calibrar ambos con la API nueva (más simple, ya verificada por el test anterior)
    uq_legacy.calibrate_from_proba(probas_cal, y_cal, batched=True)
    uq_new.calibrate_from_proba(probas_cal, y_cal, batched=True)

    # Tune
    probas_tune = rng.random((300, K)); probas_tune /= probas_tune.sum(axis=1, keepdims=True)
    y_tune = rng.integers(0, K, 300)

    # Para invocar legacy get_uncertainty_jit, necesitamos un modelo
    import torch
    uq_legacy.model = _MockModel(probas_tune)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        U_l, a_l = uq_legacy.get_uncertainty_jit(
            torch.zeros(300, 1, 28, 28), torch.from_numpy(y_tune))
    
    U_n, a_n = uq_new.get_uncertainty_from_proba(probas_tune, y_tune)

    np.testing.assert_allclose(U_l, U_n, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a_l, a_n, rtol=1e-12, atol=1e-12)


def test_deprecation_warning_on_legacy_calibrate():
    """Using calibrate(X) should emit DeprecationWarning."""
    import torch
    rng = np.random.default_rng(0)
    probas = rng.random((10, 3)); probas /= probas.sum(axis=1, keepdims=True)
    uq = UncertaintyQuantifier(N=100, classes=[0], model=_MockModel(probas), max_batch_size=64)
    
    with pytest.warns(DeprecationWarning, match="calibrate_from_proba"):
        uq.calibrate(torch.zeros(10, 1, 28, 28),
                     torch.from_numpy(rng.integers(0, 3, 10)), batched=True)


def test_constructor_warns_when_model_passed():
    with pytest.warns(DeprecationWarning, match="model"):
        UncertaintyQuantifier(N=100, classes=[0], model=object())
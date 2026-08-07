"""Covers the raw-torch-tensor label input path through calibrate, which no other test exercises: the tests/core/ suite only feeds numpy/jnp arrays (it must stay torch-free by convention), and the example scripts pre-convert torch labels to numpy (`.numpy().astype(int)`) before ever calling calibrate.

IMPORTANT — what this test does and does not prove: it verifies CORRECTNESS for torch input — that calibrating with a torch label tensor produces the same conformity_scores_ and the same q_hat as calibrating with the numpy equivalent of the same data. It does NOT verify that the conversion avoided a host round-trip (i.e. that it went through DLPack zero-copy rather than via a numpy copy). That property is not observable from a CPU backend: a host round-trip and a genuine DLPack transfer of a CPU-resident tensor land on the same device and produce identical values, so nothing here can tell them apart. A green run here is not proof of zero-copy — only proof that the torch-input path yields the right numbers.
"""
import numpy as np
import torch

from utrace import UncertaintyQuantifier


def _make_softmax_float64(n_samples: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """float64 throughout (not torch's float32 default) so the comparison against the numpy path below can be exact equality, not allclose — this is a correctness test, not a precision test."""
    rng = np.random.default_rng(seed)
    p = rng.random((n_samples, n_classes)).astype(np.float64)
    return p / p.sum(axis=1, keepdims=True)


def test_torch_label_tensor_matches_numpy_equivalent():
    n_samples, n_classes = 200, 6
    softmax_np = _make_softmax_float64(n_samples, n_classes)
    label_values = np.random.default_rng(1).integers(0, n_classes, n_samples).astype(np.int32)

    softmax_torch = torch.from_numpy(softmax_np.copy()).to(torch.float64)
    y_torch = torch.from_numpy(label_values.copy())

    uq_torch = UncertaintyQuantifier(N=300, classes=None)
    uq_torch.calibrate(softmax_torch, y_torch, batched=True)

    uq_numpy = UncertaintyQuantifier(N=300, classes=None)
    uq_numpy.calibrate(softmax_np, label_values, batched=True)

    np.testing.assert_array_equal(
        np.asarray(uq_torch.conformity_scores_),
        np.asarray(uq_numpy.conformity_scores_),
    )

    uq_torch.alpha = np.float64(0.1)
    uq_numpy.alpha = np.float64(0.1)

    # No public accessor for q_hat; reached via the internal _state field,
    # matching the existing precedent in tests/core/test_max_n_overflow.py.
    assert uq_torch._state.q_hat == uq_numpy._state.q_hat

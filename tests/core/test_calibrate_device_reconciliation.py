"""Regression test for the step-C device-commitment crash (FINDINGS.md "Step C:
device-commitment risk"; fixed in `calibrate()`, see .reports/2026-08-21_stepE_device_coherence.md).

INERT WITHOUT GPU HARDWARE. This test skips cleanly (not xfail, not silently green)
whenever no GPU-backed jax device is present -- both other development machines and
CI are CPU-only, and the bug this guards against cannot occur when every device is
the same device. A test that reported "passed" on every one of those environments
regardless of whether the reconciliation code ever ran would be worse than no test
at all: it would look like coverage while proving nothing. Run it on GPU hardware
(e.g. `uv run --no-default-groups --group dev-cuda13 --extra=viz --extra=cuda13
pytest tests/core/test_calibrate_device_reconciliation.py -q --no-cov`) to have it
actually exercise the fix.

Uses plain jax arrays placed on explicit devices via `jax.device_put`, not torch
tensors -- the fix lives entirely inside `calibrate()`, downstream of `to_jax`
(`isinstance(array_like, jnp.ndarray)` short-circuits `to_jax` for arrays that are
already jax arrays), so this isolates the reconciliation logic itself without
depending on DLPack/torch behaviour, which is exercised separately by the real
ACDC path (see the report). Keeps this file torch-free, consistent with the
tests/core/ convention (see CLAUDE.md).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utrace import UncertaintyQuantifier

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="No GPU-backed jax device available; inert without GPU hardware (see module docstring).",
)


def _make_softmax(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n, k))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def _make_labels(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 99).integers(0, k, n).astype(np.int32)


def test_mixed_device_calibrate_does_not_raise_and_lands_on_softmax_device():
    """softmax on GPU, y on CPU: must not raise, and the buffer must commit to GPU."""
    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]

    smx = jax.device_put(jnp.asarray(_make_softmax(300, 4, 1)), gpu_device)
    y = jax.device_put(jnp.asarray(_make_labels(300, 4, 1)), cpu_device)
    assert smx.devices() == {gpu_device}
    assert y.devices() == {cpu_device}

    uq = UncertaintyQuantifier(N=1000, classes=None)
    uq.calibrate(smx, y, batched=False)  # must not raise

    assert uq._state.N == 300
    assert uq._state.conformity_scores.devices() == {gpu_device}


def test_mixed_device_calibrate_matches_all_host_and_all_device_bytewise():
    """Mixed-device calibration must produce byte-identical scores to the
    all-host and all-device cases on the same data -- reconciliation must not
    change the result, only where it is computed."""
    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]
    smx_np = _make_softmax(300, 4, 2)
    y_np = _make_labels(300, 4, 2)

    def run(smx_device, y_device):
        uq = UncertaintyQuantifier(N=1000, classes=None)
        smx = jax.device_put(jnp.asarray(smx_np), smx_device)
        y = jax.device_put(jnp.asarray(y_np), y_device)
        uq.calibrate(smx, y, batched=False)
        return np.asarray(uq._state.conformity_scores[: uq._state.N])

    all_host = run(cpu_device, cpu_device)
    all_device = run(gpu_device, gpu_device)
    mixed = run(gpu_device, cpu_device)

    assert np.array_equal(all_host, all_device), "all-host vs all-device must be byte-identical"
    assert np.array_equal(all_host, mixed), "mixed-device must be byte-identical to all-host/all-device"

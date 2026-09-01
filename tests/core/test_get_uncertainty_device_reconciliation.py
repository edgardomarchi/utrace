"""Regression test for the get_uncertainty device-resident-y crash (found by the
2026-08-21 docs audit, H4(b); fixed alongside the same pattern already fixed in
`calibrate()` -- see .reports/2026-08-21_stepE_device_coherence.md and
.reports/2026-08-21_batch1_defect_fixes.md).

Before the fix: `get_uncertainty(softmax, y)` called `np.asarray(y)` directly,
without routing `y` through `to_jax` first (unlike `softmax`, which already did).
A raw device-resident tensor (e.g. a CUDA torch tensor) fails `np.asarray()`
outright with `TypeError: can't convert cuda:0 device type tensor to numpy` --
even when `softmax` agrees on the same device, and even though `calibrate()`
already handled this shape correctly. This is a DIFFERENT failure mode from
`calibrate`'s pre-fix device-mismatch `ValueError`: here there is no mismatch to
detect, `y` alone is simply never converted before numpy touches it.

INERT WITHOUT GPU HARDWARE. This test skips cleanly (not xfail, not silently
green) whenever no GPU-backed jax device is present -- both other development
machines and CI are CPU-only, and `np.asarray()` never sees a device-resident
tensor there, so this bug cannot reproduce. A test that reported "passed" on
every one of those environments regardless of whether the fixed code path ever
ran would be worse than no test at all. Run it on GPU hardware (e.g.
`uv run --no-default-groups --group dev-cuda13 --extra=viz --extra=cuda13
pytest tests/core/test_get_uncertainty_device_reconciliation.py -q --no-cov`)
to have it actually exercise the fix.

Uses plain jax arrays placed on explicit devices via `jax.device_put`, not torch
tensors -- consistent with `test_calibrate_device_reconciliation.py`'s approach
and with the tests/core/ convention of staying torch-free (see CLAUDE.md).
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


def _calibrated_uq(smx_np: np.ndarray, y_np: np.ndarray, device) -> UncertaintyQuantifier:
    uq = UncertaintyQuantifier(N=1000, classes=None)
    smx = jax.device_put(jnp.asarray(smx_np), device)
    y = jax.device_put(jnp.asarray(y_np), device)
    uq.calibrate(smx, y, batched=False)
    return uq


def test_device_resident_y_does_not_raise():
    """softmax and y both GPU-resident: get_uncertainty must not raise."""
    gpu_device = jax.devices("gpu")[0]
    smx_np = _make_softmax(200, 4, 10)
    y_np = _make_labels(200, 4, 10)

    uq = _calibrated_uq(smx_np, y_np, gpu_device)
    smx = jax.device_put(jnp.asarray(smx_np), gpu_device)
    y = jax.device_put(jnp.asarray(y_np), gpu_device)

    U, alpha = uq.get_uncertainty(smx, y, max_iters=10)  # must not raise
    assert np.isfinite(U)
    assert np.isfinite(alpha)


def test_device_resident_y_matches_all_host_and_all_device_bytewise():
    """all-host, all-device, and both mixed-device pairings must agree exactly."""
    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]
    smx_np = _make_softmax(200, 4, 11)
    y_np = _make_labels(200, 4, 11)

    def run(smx_device, y_device):
        uq = _calibrated_uq(smx_np, y_np, smx_device)
        # calibrate() above already reconciles smx/y to smx_device for the buffer write;
        # get_uncertainty is called fresh here with its own explicit device placement.
        smx = jax.device_put(jnp.asarray(smx_np), smx_device)
        y = jax.device_put(jnp.asarray(y_np), y_device)
        return uq.get_uncertainty(smx, y, max_iters=10)

    all_host = run(cpu_device, cpu_device)
    all_device = run(gpu_device, gpu_device)
    mixed_a = run(gpu_device, cpu_device)
    mixed_b = run(cpu_device, gpu_device)

    assert all_host == all_device == mixed_a == mixed_b, (
        f"results differ across device placements: "
        f"all_host={all_host}, all_device={all_device}, mixed_a={mixed_a}, mixed_b={mixed_b}"
    )

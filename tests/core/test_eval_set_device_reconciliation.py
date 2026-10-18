"""Device test for the evaluation arguments of `uncertainty_at` / `get_uncertainty`
(`eval_softmax`, `eval_y`). Mirrors `test_get_uncertainty_device_reconciliation.py`.

INERT WITHOUT GPU HARDWARE. This module skips cleanly (not xfail, not silently green)
whenever no GPU-backed jax device is present: on CPU-only machines and CI a
device-resident tensor never reaches `np.asarray()`, so the defect these tests guard
(`to_jax` skipped on an evaluation argument) cannot reproduce there. Run it on GPU
hardware to have it exercise the code path.

Plain jax arrays placed with `jax.device_put`, not torch tensors, to keep tests/core torch-free.
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


def _data(n, k, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((n, k))
    return (p / p.sum(axis=1, keepdims=True)), rng.integers(0, k, n).astype(np.int32)


def _uq(device):
    smx, y = _data(500, 4, 0)
    uq = UncertaintyQuantifier(N=1000)
    uq.calibrate(jax.device_put(jnp.asarray(smx), device), jax.device_put(jnp.asarray(y), device))
    return uq


def test_device_resident_eval_arguments_do_not_raise():
    gpu = jax.devices("gpu")[0]
    uq = _uq(gpu)
    t_smx, t_y = _data(300, 4, 1)
    e_smx, e_y = _data(300, 4, 2)
    put = lambda a: jax.device_put(jnp.asarray(a), gpu)
    U, alpha = uq.get_uncertainty(put(t_smx), put(t_y), max_iters=10,
                                  eval_softmax=put(e_smx), eval_y=put(e_y))
    assert np.isfinite(U) and np.isfinite(alpha)


@pytest.mark.parametrize("delta", [None, 0.05])
def test_eval_result_is_identical_across_device_placements(delta):
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    e_smx, e_y = _data(300, 4, 2)

    def run(smx_device, y_device):
        uq = _uq(smx_device)
        return uq.uncertainty_at(jax.device_put(jnp.asarray(e_smx), smx_device),
                                 jax.device_put(jnp.asarray(e_y), y_device), 0.2, delta=delta)

    ref, *others = [run(cpu, cpu), run(gpu, gpu), run(gpu, cpu), run(cpu, gpu)]
    for placement, r in zip(["gpu,gpu", "gpu,cpu", "cpu,gpu"], others):
        for name, expected, got in zip(ref._fields, ref, r):
            if expected is None or isinstance(expected, int):
                assert got == expected, f"{name} differs for {placement}: {got!r} != {expected!r}"
            else:
                # Reductions on different devices may differ by rounding (one ulp of EC on CUDA),
                # so floating-point fields get a tolerance; integer and None fields stay exact.
                np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15,
                                           err_msg=f"{name} differs for {placement}")

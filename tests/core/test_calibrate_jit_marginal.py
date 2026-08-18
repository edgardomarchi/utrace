"""Step D, marginal slice: pins the jitted write path (_calibrate_write_jit) used
by _calibrate_impl when classes=None, and the explicit branch that keeps the
class-conditional path off it entirely.

Invariants under test:
- The marginal (classes=None) path's numeric output matches a hand-mirrored
  reference of the pre-jit eager write (HEAD `8f4251e`, _calibrate_impl's write
  logic before the classes=None jit split), byte-wise, not with a tolerance.
- The class-conditional (classes=[...]) path is untouched by this change: its
  output matches the same pre-jit reference logic applied to the filtered
  batch, byte-wise -- it is not merely unchanged by inspection, it is pinned.
- _calibrate_write_jit compiles ONCE across a streaming sequence of many
  distinct N (buffer offset) values at a fixed batch shape -- the property the
  whole change is for. Same _cache_size() pattern as
  test_label_dtype_canonicalisation.py.
- The class-conditional path never calls _calibrate_write_jit: its cache stays
  untouched (0) after class-conditional-only calibration. This is the property
  the explicit branch (rather than a shared implementation) exists to
  preserve, and nothing else in the suite would catch it drifting.

No torch import: tests/core/ is torch-free by convention.
"""
import numpy as np
import jax.numpy as jnp

from utrace import UncertaintyQuantifier
from utrace.uncertaintyQuantifier import _calibrate_write_jit


def _make_softmax(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n, k))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def _make_labels(n: int, k: int, seed: int, force_class: int | None = None) -> np.ndarray:
    labels = np.random.default_rng(seed + 99).integers(0, k, n).astype(np.int32)
    if force_class is not None:
        # guarantee at least one sample of `force_class` so class-conditional
        # calibration never degenerates to an empty (N==0) buffer
        labels[0] = force_class
    return labels


def _reference_write(max_N, buffer, N, y, smx, score_fn, batched):
    """Hand-mirror of _calibrate_impl's write logic as it existed at HEAD
    `8f4251e`, before this commit split the marginal path onto
    _calibrate_write_jit -- eager `.at[].set()`, no jit. `y`/`smx` are assumed
    already class-filtered by the caller if applicable (mirrors the fact that,
    at HEAD, the class filter ran before this exact code, shared by both the
    marginal and class-conditional cases).
    """
    scores = score_fn(y, smx)
    num_scores = len(scores)
    if batched:
        new_buffer = buffer.at[N:N + num_scores].set(jnp.asarray(scores, dtype=jnp.float64))
    else:
        new_buffer = jnp.full((max_N,), jnp.inf, dtype=jnp.float64).at[:num_scores].set(
            jnp.asarray(scores, dtype=jnp.float64)
        )
    new_N = N + num_scores if batched else num_scores
    return new_buffer, new_N


def test_marginal_matches_pre_jit_reference():
    """classes=None: current jitted write path vs. a hand-mirrored copy of
    HEAD's pre-jit eager write, driven by an identical batch sequence."""
    K, max_N = 10, 2000
    batch_sizes = [50, 137, 60, 90]  # varying sizes, mirrors a real dataloader

    uq = UncertaintyQuantifier(N=max_N, classes=None)
    ref_buffer = jnp.full(max_N, jnp.inf, dtype=jnp.float64)
    ref_N = 0

    for i, bs in enumerate(batch_sizes):
        smx = _make_softmax(bs, K, i)
        y = _make_labels(bs, K, i)
        uq.calibrate(smx, y, batched=True)

        y_jax = jnp.asarray(y).astype(uq.label_dtype_)
        smx_jax = jnp.asarray(smx)
        ref_buffer, ref_N = _reference_write(
            max_N, ref_buffer, ref_N, y_jax, smx_jax, uq.cal_score_, batched=True
        )

    assert uq._state.N == ref_N, f"N mismatch: current={uq._state.N} reference={ref_N}"

    current_sorted = np.asarray(uq.conformity_scores_)
    reference_sorted = np.asarray(jnp.sort(ref_buffer))
    np.testing.assert_array_equal(
        current_sorted, reference_sorted,
        err_msg="marginal path's sorted conformity_scores_ must byte-match the pre-jit reference",
    )

    uq.alpha = 0.1
    q_level = min(np.ceil((ref_N + 1) * (1 - 0.1)) / ref_N, 1.0)
    from utrace.utils import _masked_quantile_higher
    ref_q_hat = float(_masked_quantile_higher(jnp.sort(ref_buffer), jnp.int32(ref_N), q_level))
    assert uq._state.q_hat == ref_q_hat, (
        f"q_hat mismatch: current={uq._state.q_hat} reference={ref_q_hat}"
    )


def test_class_conditional_matches_pre_jit_reference():
    """classes=[C]: untouched eager path vs. the same pre-jit reference logic,
    applied to the class-filtered batch by hand -- pins that this path is
    byte-identical to HEAD, not merely "probably unchanged"."""
    K, max_N, C = 5, 2000, 2
    batch_sizes = [80, 45, 110]

    uq = UncertaintyQuantifier(N=max_N, classes=[C])
    ref_buffer = jnp.full(max_N, jnp.inf, dtype=jnp.float64)
    ref_N = 0
    classes_jax = jnp.asarray([C])

    for i, bs in enumerate(batch_sizes):
        smx = _make_softmax(bs, K, i + 10)
        y = _make_labels(bs, K, i + 10, force_class=C)
        uq.calibrate(smx, y, batched=True)

        y_jax = jnp.asarray(y).astype(uq.label_dtype_)
        smx_jax = jnp.asarray(smx)
        mask = jnp.isin(y_jax, classes_jax)
        y_filtered = y_jax[mask]
        smx_filtered = smx_jax[mask]
        ref_buffer, ref_N = _reference_write(
            max_N, ref_buffer, ref_N, y_filtered, smx_filtered, uq.cal_score_, batched=True
        )

    assert uq._state.N == ref_N, f"N mismatch: current={uq._state.N} reference={ref_N}"

    current_sorted = np.asarray(uq.conformity_scores_)
    reference_sorted = np.asarray(jnp.sort(ref_buffer))
    np.testing.assert_array_equal(
        current_sorted, reference_sorted,
        err_msg="class-conditional path's sorted conformity_scores_ must byte-match the pre-jit reference",
    )


def test_jit_write_compiles_once_across_streaming_offsets():
    """_calibrate_write_jit must compile ONCE across a streaming sequence of
    many distinct buffer offsets (N values) at a fixed batch shape -- the
    property the whole change exists for. Same pattern as
    test_label_dtype_canonicalisation.py: clear_cache(), exercise, assert
    _cache_size().
    """
    _calibrate_write_jit.clear_cache()

    K, B, num_batches = 10, 50, 45  # at least 40, as a dataloader would produce
    max_N = B * num_batches
    uq = UncertaintyQuantifier(N=max_N, classes=None)

    for i in range(num_batches):
        smx = _make_softmax(B, K, i)
        y = _make_labels(B, K, i)
        uq.calibrate(smx, y, batched=True)

    assert _calibrate_write_jit._cache_size() == 1, (
        f"_calibrate_write_jit was traced {_calibrate_write_jit._cache_size()} times across "
        f"{num_batches} distinct offsets; expected exactly 1, meaning the offset was not "
        "treated as a traced (non-static) value."
    )
    assert uq._state.N == B * num_batches


def test_class_conditional_never_hits_jit_write():
    """The explicit branch's whole purpose: class-conditional calibration must
    NEVER call _calibrate_write_jit, regardless of how many distinct filtered
    batch sizes it sees. If this ever regresses to a shared implementation,
    this is the only test that would catch it."""
    _calibrate_write_jit.clear_cache()

    K, max_N, C = 6, 5000, 1
    uq = UncertaintyQuantifier(N=max_N, classes=[C])

    for i, bs in enumerate([70, 33, 91, 12]):  # deliberately irregular sizes
        smx = _make_softmax(bs, K, i + 50)
        y = _make_labels(bs, K, i + 50, force_class=C)
        uq.calibrate(smx, y, batched=True)

    assert _calibrate_write_jit._cache_size() == 0, (
        f"_calibrate_write_jit was traced {_calibrate_write_jit._cache_size()} times during "
        "class-conditional-only calibration; expected 0 -- the class-conditional path must "
        "never reach the marginal-only jitted write."
    )

    # Sanity: a subsequent marginal call DOES reach it, proving the assertion
    # above isn't vacuously true because the jit function is unreachable.
    uq2 = UncertaintyQuantifier(N=100, classes=None)
    uq2.calibrate(_make_softmax(20, K, 0), _make_labels(20, K, 0), batched=True)
    assert _calibrate_write_jit._cache_size() == 1, (
        "sanity check failed: a marginal (classes=None) call must reach "
        "_calibrate_write_jit, otherwise the cache_size==0 assertion above is vacuous"
    )

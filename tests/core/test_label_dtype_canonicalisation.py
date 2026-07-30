"""Verifies that calibrate_from_proba canonicalises labels to the score
family's declared dtype (jnp.int32 for 'lac'), ON DEVICE, regardless of the
caller's input dtype.

No torch import: tests/core/ is torch-free by convention.
"""
import numpy as np
import jax.numpy as jnp

from utrace import UncertaintyQuantifier
from utrace.scores import lac_cal


def _make_probas(n_samples: int, n_classes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n_samples, n_classes))
    return p / p.sum(axis=1, keepdims=True)


def test_label_dtype_is_canonicalised_to_int32():
    """
    This is verified indirectly, through lac_cal's own JIT cache: lac_cal is @jit-decorated and JAX retraces (and caches a new entry) per distinct (shape, dtype) combination of its arguments. If the caller's label dtype leaked through uncanonicalised, feeding numpy int32, numpy int64, and a    jnp array of labels would produce THREE distinct traces (one per incoming dtype) and the cache would grow to 3. If canonicalisation actually happens, every variant collapses to the same (shape, jnp.int32) trace and the cache stays at 1 — a test that cannot pass by accident if the implementation still preserved the caller's dtype instead of casting it.

    The probability array and all shapes are held IDENTICAL across the three variants on purpose: lac_cal's jit cache also keys on the shape and dtype of `smx` (the probabilities), not just `y`. Varying probabilities or shapes between the three calibrate_from_proba calls would each independently justify a new trace, and the cache-size assertion would then say nothing about label-dtype canonicalisation specifically.
    """
    lac_cal.clear_cache()

    n_samples, n_classes = 50, 5
    probas = _make_probas(n_samples, n_classes)
    label_values = np.random.default_rng(1).integers(0, n_classes, n_samples)

    variants = {
        "numpy_int32": label_values.astype(np.int32),
        "numpy_int64": label_values.astype(np.int64),
        "jnp_array": jnp.asarray(label_values.astype(np.int64)),
    }

    scores_by_variant = {}
    for name, y in variants.items():
        uq = UncertaintyQuantifier(N=200, classes=None)
        uq.calibrate_from_proba(probas, y, batched=True)
        scores_by_variant[name] = np.asarray(uq.conformity_scores_)

    assert lac_cal._cache_size() == 1, (
        f"lac_cal was traced {lac_cal._cache_size()} times across dtype variants;"
        "expected exactly 1, meaning labels were not canonicalised to a single dtype"
        "before reaching the jitted score function."
    )

    reference = scores_by_variant["numpy_int32"]
    for name, scores in scores_by_variant.items():
        np.testing.assert_array_equal(
            scores, reference,
            err_msg=f"conformity_scores_ differ for label dtype variant {name!r}"
        )

def test_labels_reach_score_fn_as_canonical_dtype():
    """Labels must reach the score function in the score family's canonical dtype.

    Fails against the previous implementation, which canonicalised to int64 via numpy: the int64 input below would arrive unchanged.
    """
    rng = np.random.default_rng(0)
    n, k = 50, 3
    probs = rng.random((n, k))
    probs /= probs.sum(axis=1, keepdims=True)
    labels_int64 = rng.integers(0, k, size=n).astype(np.int64)

    uq = UncertaintyQuantifier(N=100)
    seen = {}
    original = uq.cal_score_

    def spy(y, smx):
        seen['dtype'] = y.dtype
        return original(y, smx)

    uq.cal_score_ = spy
    uq.calibrate_from_proba(probs, labels_int64)

    assert seen['dtype'] == jnp.dtype(uq.label_dtype_)
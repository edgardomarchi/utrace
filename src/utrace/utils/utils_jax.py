import jax.numpy as jnp
from jax import jit


@jit
def _masked_quantile_higher(sorted_padded: jnp.ndarray,
                            n_valid: jnp.ndarray,
                            q: jnp.ndarray) -> jnp.ndarray:
    """Quantile with method='higher' over the first n_valid entries of `sorted_padded`, which *must be sorted* ascending with +inf padding after position n_valid.

    Shape-stable: operates on the full (max_N,) array, so JAX compiles it once regardless of n_valid. Replicates jnp.quantile(sorted_padded[:n_valid], q, method='higher') exactly (verified by test_masked_quantile_matches_jnp).
    """
    pos = q * (n_valid - 1)
    idx = jnp.ceil(pos).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_valid - 1)
    return sorted_padded[idx]
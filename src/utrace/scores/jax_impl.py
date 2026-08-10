import jax.numpy as jnp
from jax import jit

__all__ = ["lac", "lac_cal"]


@jit
def lac_cal(y: jnp.ndarray, smx: jnp.ndarray,
    ) -> jnp.ndarray:
    return 1 - smx[jnp.arange(len(y)), y]

@jit
def lac(smx:jnp.ndarray) -> jnp.ndarray:
    """LAC score.
    Args:
        smx (np.array): model output of the softmax function
    Returns:
        np.array: LAC score
    """
    return 1 - smx

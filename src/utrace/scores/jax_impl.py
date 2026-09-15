import jax.numpy as jnp
from jax import jit

__all__ = ["lac", "lac_cal", 'abs_error', 'abs_error_cal']

# =============================================================================
# Classification
# =============================================================================

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

# =============================================================================
# Regression
# =============================================================================

@jit
def abs_error_cal(y: jnp.ndarray, y_hat:jnp.ndarray) -> jnp.ndarray:
    """Calibration scores: |y-y_hat|."""
    return jnp.abs(y - y_hat)

@jit
def abs_error(y_hat: jnp.ndarray, q_hat: jnp.ndarray | float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calibration region:
        [y_hat - q_hat, y_hat + q_hat]
    """
    return y_hat - q_hat, y_hat + q_hat


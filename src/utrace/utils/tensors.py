import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)

def to_jax(array_like: ArrayLike) -> jnp.ndarray:
    """
    Converts any array-like object to a JAX array.
    Uses zero-copy if possible.
    """

    if isinstance(array_like, jnp.ndarray):
        return array_like

    # Zero-Copy with DLPack
    if hasattr(array_like, '__dlpack__'):
        try:
            return jax.dlpack.from_dlpack(array_like)
        except Exception as e:
            logger.debug("DLPack transfer failed. Using fallback. Error: %s", e)
    
    # Safeguard fallback
    if hasattr(array_like, 'cpu') and hasattr(array_like, 'numpy'):
        return jnp.asarray(array_like.cpu().numpy())
 
    # Final Fallback
    return jnp.asarray(array_like)


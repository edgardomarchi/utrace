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


def flatten_to_pixels(tensor_batch: jnp.ndarray, channel_axis: int = 1) -> jnp.ndarray:
    """
    Flattens a batch of images into a batch of pixels, preserving the channel dimension.

    Parameters
    ----------
    tensor_batch : jnp.ndarray
        A 4D tensor of images
    channel_axis : int
        The axis of the channel dimension in the input tensor.
        Default is 1.
    Returns
    -------
    jnp.ndarray
        A 2D tensor of flattened images
    """

    if tensor_batch.ndim == 1:
        return tensor_batch

    tensor_channel_last = jnp.moveaxis(tensor_batch, source=channel_axis, destination=-1)
    
    C = tensor_channel_last.shape[-1]
    
    return jnp.reshape(tensor_channel_last, (-1, C))

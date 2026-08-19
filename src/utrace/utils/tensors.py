import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

logger = logging.getLogger(__name__)

def to_jax(array_like: ArrayLike) -> jnp.ndarray:
    """
    Converts any array-like object to a JAX array.
    Uses zero-copy if possible.

    Device Placement Contract:
    - Genuine framework tensors (e.g., PyTorch CPU/CUDA tensors) that carry a real device:
      Preserves their original device via DLPack, performing zero-copy where possible (e.g., a torch CUDA tensor remains on the GPU).
    - Host-only arrays (e.g., NumPy ndarrays) with no meaningful device of their own:
      Places them on JAX's default (compute) device via `jnp.asarray`. Preserving the CPU origin for host arrays would incorrectly pin them to CPU and cause device mismatches during JIT, so normalizing to the default compute device is the correct choice.
    - Does NOT reconcile mismatches between two genuine tensors on different devices (e.g., one on CPU, another on GPU); that remains the caller's responsibility.
    """

    if isinstance(array_like, jnp.ndarray):
        return array_like

    # Check numpy.ndarray before __dlpack__ because NumPy arrays implement __dlpack__ but
    # lack a meaningful device. Passing them through DLPack pins them to CPU and triggers
    # unaligned-copy warnings. Instead, place them on the JAX default compute device.
    if isinstance(array_like, np.ndarray):
        return jnp.asarray(array_like)

    # Zero-Copy with DLPack for genuine framework tensors (e.g., PyTorch, MXNet)
    if hasattr(array_like, '__dlpack__'):
        try:
            return jax.dlpack.from_dlpack(array_like)
        except Exception as e:  # noqa: BLE001 - deliberate: this is the middle tier of a
            # three-tier fallback (DLPack, then cpu/numpy, then generic); swallowing whatever
            # DLPack raises and falling through to the next tier is the mechanism, not an
            # oversight.
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

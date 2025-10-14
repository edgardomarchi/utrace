import logging
from torch import Tensor
from torch.utils.dlpack import to_dlpack
from jax import devices
import jax.numpy as jnp
from jax.dlpack import from_dlpack

logger = logging.getLogger(__name__)

def transfer_to_jax(tensor_torch: Tensor, jax_device: str) -> jnp.ndarray:
    """
    Transfers a torch Tensor to a jax array, considering the devices they are in.
    """
    source_device_type = tensor_torch.device.type
    jax_device = devices()[0].platform  # Assuming single device for simplicity

    # Optimal case: tensor and jax are in the GPU.
    if source_device_type == 'cuda' and jax_device == 'gpu':
        logger.debug("GPU -> GPU transfer detected. Using DLPack (Zero-Copy).")
        #dlpack_capsule = to_dlpack(tensor_torch)
        return from_dlpack(tensor_torch)

    # Fallback for every other case (GPU->CPU, CPU->GPU, CPU->CPU)
    else:
        logger.debug("%s -> %s transfer detected. Using CPU copy", source_device_type.upper(), jax_device.upper())

        np_array = tensor_torch.cpu().numpy()
        return jnp.asarray(np_array)

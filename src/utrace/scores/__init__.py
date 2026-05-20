from ..config import USE_JAX

if USE_JAX:
    from .jax_impl import *
else:
    from .numpy_impl import *

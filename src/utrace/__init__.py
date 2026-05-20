from importlib.metadata import PackageNotFoundError, version
from .uncertaintyQuantifier import UncertaintyQuantifier

try:
    __version__ = version('utrace')
except PackageNotFoundError:
    __version__ = '(local)'

del PackageNotFoundError
del version

from .config import USE_JAX
if USE_JAX:
    import jax
    jax.config.update("jax_enable_x64", True)
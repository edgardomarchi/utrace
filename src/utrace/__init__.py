from importlib.metadata import PackageNotFoundError, version
from .uncertaintyQuantifier import UncertaintyQuantifier

try:
    __version__ = version('utrace')
except PackageNotFoundError:
    __version__ = '(local)'

del PackageNotFoundError
del version

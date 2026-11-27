from importlib.metadata import PackageNotFoundError, version

from .bounds import Betting, Bound, EmpiricalBernstein, Hoeffding, SampleStats, Tightest
from .budget import BudgetComponent, ComponentKind, UncertaintyBudget
from .scores.objects import LAC, AbsResidual, Score
from .uncertaintyQuantifier import (
    SearchStatus,
    SearchStatusWarning,
    TuningDataReuseWarning,
    UncertaintyEvaluation,
    UncertaintyQuantifier,
)

try:
    __version__ = version('utrace')
except PackageNotFoundError:
    __version__ = '(local)'

del PackageNotFoundError
del version

import jax

jax.config.update("jax_enable_x64", True)
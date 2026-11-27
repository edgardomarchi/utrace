"""Concentration bounds: one-sided lower confidence bounds on the mean of `[0, 1]`-valued values.

A `Bound` turns a `SampleStats` (the number of values, their mean, their centred sum of squares
and, only when the bound needs them, the values themselves) and a failure probability `level`
into a lower confidence bound on the population mean `E[X]`, assuming `X_1, ..., X_n` i.i.d. in
`[0, 1]`. Each bound uses the whole `level` it is given: splitting a level between several
bounds is the job of `Tightest`, or of the caller.

`UncertaintyQuantifier.uncertainty_at(..., delta=..., bound=...)` applies a bound to the `EC` term
of `U_delta` at level `delta/2` (the coverage term stays Clopper-Pearson at `delta/2`). The default,
`DEFAULT_BOUND = Tightest((Hoeffding(), EmpiricalBernstein()))`, reproduces the bound the package
has always used, bit for bit: each member at `delta/4`, the larger lower bound taken.

Bounds are frozen dataclasses: equal parameters give equal, hashable instances. They run on the
host, on numpy scalars and arrays, outside any jit. This module imports numpy only.
"""

import abc
import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

__all__ = [
    "DEFAULT_BOUND",
    "Bound",
    "EmpiricalBernstein",
    "Hoeffding",
    "SampleStats",
    "Tightest",
    "resolve_bound",
]


@dataclass(frozen=True, eq=False)
class SampleStats:
    """What a bound may read about the values it bounds.

    Attributes
    ----------
    n : int
        Number of values.
    mean : float
        Their mean.
    sum_sq_dev : float
        Their centred sum of squares, `sum((x - mean)^2)`. A consumer clips it at 0 (it can come
        out a rounding error below 0).
    values : np.ndarray or None
        The values themselves, in evaluation order, shape `(n,)`; given only when the bound's
        `needs_values` is True.
    """
    n: int
    mean: float
    sum_sq_dev: float = 0.0
    values: np.ndarray | None = None


@dataclass(frozen=True)
class Bound(abc.ABC):
    """Base class of the bound objects. See the module docstring.

    `name` is the string alias (`resolve_bound`); `needs_values` says whether `lower` reads
    `SampleStats.values` (False: the sufficient statistics are enough).
    """
    name: ClassVar[str]
    needs_values: ClassVar[bool] = False

    @abc.abstractmethod
    def lower(self, stats: SampleStats, level: float) -> float:
        """One-sided lower confidence bound on `E[X]` at failure probability `level`, for `stats.n`
        i.i.d. values in `[0, 1]`. Returns `-inf` when the bound is undefined for this `n`; it is
        not clipped (the caller clips at 0)."""

    def describe(self) -> str:
        """A short human-readable name, e.g. `'tightest(hoeffding, empirical_bernstein)'`."""
        return self.name


def _check_level(level) -> float:
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"a bound's level must lie in (0, 1), got {level!r}.")
    return level


@dataclass(frozen=True)
class Hoeffding(Bound):
    """Hoeffding's inequality, one-sided: `mean - sqrt(ln(1/level) / (2 n))`.

    Uses `n` and `mean` only. `-inf` for `n == 0`.
    """
    name: ClassVar[str] = "hoeffding"

    def lower(self, stats: SampleStats, level: float) -> float:
        level = _check_level(level)
        n = int(stats.n)
        if n < 1:
            return -math.inf
        margin = np.sqrt(np.log(1.0 / level) / (2.0 * n))
        return float(stats.mean) - float(margin)


@dataclass(frozen=True)
class EmpiricalBernstein(Bound):
    """Empirical Bernstein bound of Maurer and Pontil (2009), "Empirical Bernstein Bounds and
    Sample Variance Penalization", COLT, arXiv:0907.3740, Theorem 4, one-sided:
    `mean - sqrt(2 V ln(2/level) / n) - 7 ln(2/level) / (3 (n - 1))`, with
    `V = max(0, sum_sq_dev) / (n - 1)` the unbiased sample variance.

    Uses `n`, `mean` and `sum_sq_dev`. `-inf` for `n < 2` (no sample variance).
    """
    name: ClassVar[str] = "empirical_bernstein"

    def lower(self, stats: SampleStats, level: float) -> float:
        level = _check_level(level)
        n = int(stats.n)
        if n < 2:
            return -math.inf
        L = np.log(2.0 / level)
        V = max(0.0, float(stats.sum_sq_dev)) / (n - 1)
        margin = np.sqrt(2.0 * V * L / n) + 7.0 * L / (3.0 * (n - 1))
        return float(stats.mean) - float(margin)


@dataclass(frozen=True)
class Tightest(Bound):
    """The largest of its members' lower bounds, each at `level / len(members)`.

    The members' events are fixed in advance, so by a union bound the largest holds with
    probability at least `1 - level`; taking it is not a data-dependent choice of level. The
    split is even and does not depend on the data. A list of members is accepted and stored as a
    tuple.
    """
    members: tuple[Bound, ...] = field(default=())
    name: ClassVar[str] = "tightest"

    def __post_init__(self):
        members = tuple(self.members)
        if not members:
            raise ValueError("Tightest needs at least one member bound.")
        for i, m in enumerate(members):
            if not isinstance(m, Bound):
                raise TypeError(
                    f"Tightest members must be Bound instances, got {type(m).__name__} at position {i}."
                )
        object.__setattr__(self, "members", members)

    @property
    def needs_values(self) -> bool:  # type: ignore[override]
        return any(m.needs_values for m in self.members)

    def lower(self, stats: SampleStats, level: float) -> float:
        level = _check_level(level)
        share = level / len(self.members)
        return max(m.lower(stats, share) for m in self.members)

    def describe(self) -> str:
        return f"tightest({', '.join(m.describe() for m in self.members)})"


DEFAULT_BOUND = Tightest((Hoeffding(), EmpiricalBernstein()))

_ALIASES = {
    "hoeffding": Hoeffding(),
    "empirical_bernstein": EmpiricalBernstein(),
}
_SUPPORTED = "'hoeffding', 'empirical_bernstein'"


def resolve_bound(bound) -> Bound:
    """The bound object for `bound=`: `None` gives `DEFAULT_BOUND`, a `Bound` instance is returned
    as is, and a string alias maps to the corresponding bound with default parameters."""
    if bound is None:
        return DEFAULT_BOUND
    if isinstance(bound, Bound):
        return bound
    if isinstance(bound, str):
        if bound in _ALIASES:
            return _ALIASES[bound]
        if bound == "betting":
            raise ValueError(
                "bound='betting' is not yet available in this version. "
                f"Supported: {_SUPPORTED}, or a Bound instance."
            )
        raise ValueError(f"Unknown bound {bound!r}. Supported: {_SUPPORTED}, or a Bound instance.")
    raise TypeError(f"bound must be None, a string or a Bound instance, got {type(bound).__name__}.")

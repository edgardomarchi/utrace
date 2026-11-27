"""Concentration bounds: one-sided lower confidence bounds on the mean of `[0, 1]`-valued values.

A `Bound` turns a `SampleStats` (the number of values, their mean, their centred sum of squares
and, only when the bound needs them, the values themselves) and a failure probability `level`
into a lower confidence bound on the population mean `E[X]`, assuming `X_1, ..., X_n` i.i.d. in
`[0, 1]`. Each bound uses the whole `level` it is given: splitting a level between several
bounds is the job of `Tightest`, or of the caller.

`Hoeffding` and `EmpiricalBernstein` work from the sufficient statistics; `Betting` (Waudby-Smith
and Ramdas, 2024) reads the values themselves, in order (`needs_values` is True).

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
from typing import ClassVar, Literal

import numpy as np

__all__ = [
    "DEFAULT_BOUND",
    "Betting",
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


def _hedged_parts(x: np.ndarray, level: float) -> np.ndarray:
    """The predictable bets `lambda~_t` of the hedged capital process (before the cap `c/m`)."""
    n = len(x)
    t = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (t + 1)
    s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (t + 1)
    s2_prev = np.r_[0.25, s2[:-1]]
    return np.sqrt(2 * np.log(1 / level) / (n * s2_prev))


def _hedged_lower(x: np.ndarray, level: float, c: float, iters: int) -> float:
    """The betting lower bound on the values `x` in the order given; bisection on `m`, returning
    the rejected end, so the bound is never rounded upward."""
    lam_t = _hedged_parts(x, level)
    thr = np.log(1 / level)

    def rejected(m: float) -> bool:
        lam = np.minimum(lam_t, c / m)
        return bool(np.cumsum(np.log1p(lam * (x - m))).max() >= thr)

    lo, hi = 0.0, 1.0                     # hi = 1 is never rejected (every factor <= 1); lo is 0 or rejected
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rejected(mid):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class Betting(Bound):
    """Betting lower confidence bound of Waudby-Smith and Ramdas, "Estimating means of bounded
    random variables by betting", J. R. Stat. Soc. B 86(1):1-27 (2024), doi:10.1093/jrsssb/qkad009,
    Theorem 3 and Remark 3 (the hedged capital process), lower side only.

    For a candidate mean `m`, the capital is `K_t(m) = prod_{i<=t} (1 + lambda_i(m) (x_i - m))`, with
    `lambda_i(m) = min(lambda~_i, c/m)`, `lambda~_t = sqrt(2 ln(1/level) / (n sigma^2_{t-1}))`,
    and the running estimates `mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)` and
    `sigma^2_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2) / (t + 1)` (`sigma^2_0 = 1/4`). If the true mean
    is `<= m`, `K_t(m)` is a non-negative supermartingale, so by Ville's inequality
    `max_t K_t(m) >= 1/level` with probability at most `level`. The bound is the smallest `m` not
    rejected, found by `iters` bisection steps on `[0, 1]`, returning the rejected end. It adapts to
    the variance, and is valid for every `n` (a single value never rejects anything, so `n == 1`
    gives 0). `-inf` for `n == 0`.

    It reads the values (`needs_values` is True), and **depends on their order**. With
    `order="permute"` (default) the values are first permuted by `np.random.default_rng(seed)`,
    fixed before the data are seen, so the result is deterministic and an input that arrives sorted
    (by patient, slice, class or score) does not bias it; `order="given"` uses the order received
    (`uncertainty_at` passes the covered samples in evaluation order).

    Like every bound here it assumes **independent** values. It does not fix dependence between
    samples (the pixels of one image, for example): for such data, average within each
    independent unit first and bound the unit means, with the `units=` option of
    `uncertainty_budget`.

    Cost: `iters` passes of a cumulative sum over `n` values (`n = 10**6` is about a second); memory
    a few arrays of `n` float64.

    Parameters
    ----------
    c : float, default 0.5
        The cap on the bet, `lambda <= c/m`; in `(0, 1)`.
    iters : int, default 50
        Bisection steps; the bound is resolved to `2**-iters`.
    order : {"permute", "given"}, default "permute"
    seed : int, default 0
        Seed of the permutation (`order="permute"` only).
    """
    c: float = 0.5
    iters: int = 50
    order: Literal["permute", "given"] = "permute"
    seed: int = 0
    name: ClassVar[str] = "betting"
    needs_values: ClassVar[bool] = True

    def __post_init__(self):
        if not (isinstance(self.c, (int, float)) and 0.0 < float(self.c) < 1.0):
            raise ValueError(f"Betting: c must be in (0, 1), got {self.c!r}.")
        if isinstance(self.iters, bool) or not isinstance(self.iters, int) or self.iters < 1:
            raise ValueError(f"Betting: iters must be an integer >= 1, got {self.iters!r}.")
        if self.order not in ("permute", "given"):
            raise ValueError(f"Betting: order must be 'permute' or 'given', got {self.order!r}.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"Betting: seed must be a non-negative integer, got {self.seed!r}.")

    def lower(self, stats: SampleStats, level: float) -> float:
        level = _check_level(level)
        if stats.values is None:
            raise ValueError("Betting needs the values (SampleStats.values); none were given.")
        x = np.asarray(stats.values, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != int(stats.n):
            raise ValueError(f"Betting: values must have shape (n,) = ({int(stats.n)},), got {x.shape}.")
        if x.shape[0] == 0:
            return -math.inf
        if not np.all((x >= 0.0) & (x <= 1.0)):
            raise ValueError("Betting: values must lie in [0, 1].")
        if self.order == "permute":
            x = x[np.random.default_rng(self.seed).permutation(x.shape[0])]
        return float(_hedged_lower(x, level, float(self.c), self.iters))

    def describe(self) -> str:
        return "betting" if self.order == "given" else f"betting(permute, seed={self.seed})"


DEFAULT_BOUND = Tightest((Hoeffding(), EmpiricalBernstein()))

_ALIASES = {
    "hoeffding": Hoeffding(),
    "empirical_bernstein": EmpiricalBernstein(),
    "betting": Betting(),
}
_SUPPORTED = "'hoeffding', 'empirical_bernstein', 'betting'"


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
        raise ValueError(f"Unknown bound {bound!r}. Supported: {_SUPPORTED}, or a Bound instance.")
    raise TypeError(f"bound must be None, a string or a Bound instance, got {type(bound).__name__}.")

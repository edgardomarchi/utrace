"""The structured uncertainty budget returned by `UncertaintyQuantifier.uncertainty_budget`.

On one labelled evaluation set, at one alpha, `U_delta` (the upper confidence bound of
`uncertainty_at`) is split exactly into the model's error and what the method adds to it:

    U_delta = error + accepted_not_covered + selection + coverage + efficiency
              model    structural             structural  statistical statistical

with `B` the score's acceptance region (`{argmax}` for `LAC`, `y_hat ± tol` for `AbsResidual`), `C` the
prediction set or interval, `pi` the per-sample efficiency, and every mean taken over the valid
samples (those of `classes`, or all):

- `error = mean 1{y not in B}`: the model's error on the evaluation set (top-1 error for `LAC`);
- `accepted_not_covered = mean 1{y in B, y not in C}`: acceptable answers the set rejects (for `LAC`,
  an empty set on a correct answer);
- `selection = mean 1{y in C} (1{y in B} - pi)`: what the uniform-selection credit `pi` misses (for
  `LAC`, the multi-label term: `pi = 1/|C|` against how often the top-1 label is the true one);
- `coverage = EC (coverage - coverage_lower)` and `efficiency = coverage_lower (EC - EC_lower)`: the
  two terms of the confidence margin `U_delta - U_empirical`.

The first three sum to `U_empirical - error` exactly (in real arithmetic); the last two to
`U_delta - U_empirical`. This module holds only the result types; numpy-free.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["BudgetComponent", "ComponentKind", "UncertaintyBudget"]


class ComponentKind(str, Enum):
    """What a budget component measures. A `str` subclass (`kind == "structural"` works).

    - `MODEL`: the model's own error, which the budget explains `U_delta` against.
    - `STRUCTURAL`: what the construction of the method adds; it does not shrink with more data.
    - `STATISTICAL`: the price of stating a confidence level; it shrinks with the evaluation size.
    """
    MODEL = "model"
    STRUCTURAL = "structural"
    STATISTICAL = "statistical"


@dataclass(frozen=True)
class BudgetComponent:
    """One term of a budget.

    Attributes
    ----------
    name : str
        `'error'`, `'accepted_not_covered'`, `'selection'`, `'coverage'` or `'efficiency'`.
    value : float
        Its value (not clipped: `selection` can be slightly negative on a finite sample).
    kind : ComponentKind
    method : str
        How it was obtained, e.g. `'Clopper-Pearson at delta/2'` or the bound's `describe()`.
    """
    name: str
    value: float
    kind: ComponentKind
    method: str


@dataclass(frozen=True)
class UncertaintyBudget:
    """Result of `UncertaintyQuantifier.uncertainty_budget`.

    Attributes
    ----------
    evaluation : UncertaintyEvaluation
        Exactly what `uncertainty_at(softmax, y, alpha, tol=tol, delta=delta, bound=bound)` returns.
    components : tuple of BudgetComponent
        `error, accepted_not_covered, selection, coverage, efficiency`, summing to
        `evaluation.U_delta` (to rounding). Empty when nothing was evaluated (`alpha` `nan`, or no
        valid sample).
    error : float
        The model's error on the valid evaluation samples, `mean 1{y not in B}` (the `error`
        component); `nan` when nothing was evaluated.
    error_upper : float
        One-sided Clopper-Pearson upper bound on the error probability at level `1 - delta`: the
        direct ("binomial") bound on the error from the same labels, a comparator for `U_delta`, not
        a part of it. `nan` when nothing was evaluated.
    """
    evaluation: Any
    components: tuple[BudgetComponent, ...]
    error: float
    error_upper: float

    @property
    def U(self):
        """`evaluation.U`, the point estimate."""
        return self.evaluation.U

    @property
    def U_delta(self):
        """`evaluation.U_delta`, the confidence bound the `components` decompose."""
        return self.evaluation.U_delta

    @property
    def total(self) -> float:
        """The sum of `components` (equal to `U_delta` up to rounding; `nan` if there are none)."""
        return sum(c.value for c in self.components) if self.components else float("nan")

    def component(self, name: str) -> BudgetComponent:
        """The component called `name`; `KeyError` if there is none."""
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

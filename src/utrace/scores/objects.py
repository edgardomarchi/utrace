"""Score objects: everything task-specific that `UncertaintyQuantifier` needs.

A `Score` bundles the conformity score `s(output, y)`, the prediction set or interval it
builds from a calibrated `q_hat`, the per-sample efficiency `pi` over the acceptance region
`B`, the metric the alpha search steers by together with its feasibility rule, the label
dtype and validation, and what `predict` returns. `UncertaintyQuantifier` holds one score
object and never branches on the task.

Scores are frozen dataclasses with no fields (their description is in `ClassVar`s), so every
instance of a class is equal and hashes equally: they are passed to the jitted functions as
static arguments and a fresh instance per call reuses the compiled trace. Parameters that vary
per call (`tol`, `target_ratio`) are not part of the object; `resolve_params` turns them into a
pytree of `float64` scalars that the jitted functions receive as an ordinary (traced) argument,
so sweeping them on one calibrated instance compiles once.

Two scores ship:

- `LAC` (`score='lac'`, classification): the set `{k : 1 - p_k <= q_hat}`. The acceptance region
  is `B = {argmax}`, so `pi = 1/|C|`. The search steers by the unconditional mean set size
  (`<= 1`); see CONTRIBUTING.md, "Departures from the published paper". It takes no per-call
  parameters.
- `AbsResidual` (`score='abs_error'`, regression): the interval `y_hat ± q_hat` built from the
  absolute residual `|y - y_hat|`. The acceptance region is the fixed tolerance window
  `B = y_hat ± tol`, `pi = |C ∩ B| / |C|`, and the search steers by the mean of `pi` over covered
  samples (`>= target_ratio`). `tol` is required.
"""

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .jax_impl import abs_error, abs_error_cal, lac, lac_cal

__all__ = ["LAC", "AbsResidual", "Score", "TolParams", "resolve_score"]


class TolParams(NamedTuple):
    """Per-call parameters of a tolerance-window score: `B = y_hat ± tol`, and the search target.

    Both leaves are `float64` jax scalars (see `AbsResidual.resolve_params`): Python floats in
    the same position would be weakly typed and compile a second trace.
    """
    tol: jax.Array
    target: jax.Array


@dataclass(frozen=True)
class Score(abc.ABC):
    """Base class of the score objects. See the module docstring.

    Host-side members (`resolve_params`, `check_label_dtype`, `prepare_labels`,
    `check_label_range`, `describe_target`) run on numpy/jax arrays outside any trace and raise
    `ValueError` on bad input. Traced members (`cal_score`, `region`, `evaluate`, `per_sample`,
    `init_metric`, `feasible`, `step_up`, `predict`) are pure `jnp` code called inside the
    jitted functions of `uncertaintyQuantifier.py`, with `self` static and `params` traced.
    """

    name: ClassVar[str]
    """The string accepted by `UncertaintyQuantifier(score=...)`."""
    label_dtype: ClassVar[Any]
    """Dtype labels are cast to before scoring (`jnp.int32`, `jnp.float64`)."""
    uses_classes: ClassVar[bool]
    """Whether `UncertaintyQuantifier(classes=...)` is meaningful for this score."""

    # ---- host side ----------------------------------------------------------------------

    @abc.abstractmethod
    def resolve_params(self, *, tol, target_ratio, where: str):
        """Validate the per-call parameters and return them as a pytree (`()` if none)."""

    @abc.abstractmethod
    def check_label_dtype(self, y, where: str) -> None:
        """`ValueError` if the dtype or values of `y` (host numpy or jax) are not acceptable."""

    @abc.abstractmethod
    def check_output(self, output, y, where: str) -> None:
        """`ValueError` if the model output (host numpy or jax) does not have the shape this
        score expects, or (when the labels `y` are given) does not match them. Called by
        `calibrate` (with `y`) and `predict` (with `y=None`)."""

    @abc.abstractmethod
    def prepare_labels(self, y: np.ndarray, output: np.ndarray, where: str) -> np.ndarray:
        """Host labels in canonical form, after `check_label_dtype` (tuning/evaluation paths).
        Raises `ValueError` where `check_output` would."""

    @abc.abstractmethod
    def check_label_range(self, y, K: int | None, where: str) -> None:
        """`ValueError` if a label is outside the range this score accepts. `K` is the number of
        output columns when output rows are 1-D (`None` otherwise, or when unknown)."""

    @abc.abstractmethod
    def describe_target(self, params) -> tuple[str, str, str]:
        """`(metric name, operator, target)` for the `SearchStatusWarning` messages."""

    # ---- traced -------------------------------------------------------------------------

    @abc.abstractmethod
    def cal_score(self, y, output):
        """Conformity scores `s(output, y)`, shape `(n,)`. Called top-level, outside any trace."""

    @abc.abstractmethod
    def region(self, output, q_hat):
        """The prediction set or interval at `q_hat` (a pytree of arrays)."""

    @abc.abstractmethod
    def evaluate(self, y, output, q_hat, valid_mask, params):
        """`(search metric, EC)` at `q_hat`: what the alpha search steers by, and the mean of `pi`
        over valid covered samples that `U = 1 - EC * (1 - alpha)` is built from."""

    @abc.abstractmethod
    def per_sample(self, y, output, q_hat, valid_mask, params):
        """`(is_covered, succ, z)` per sample: coverage, the samples `EC` averages over, and
        their `pi` (0 elsewhere). Used by `uncertainty_at` for the counts and the variance."""

    @abc.abstractmethod
    def init_metric(self, params):
        """Metric value the search starts from, chosen so its first step goes from alpha 1.0
        down to 0.5 (`step_up(init_metric)` is False)."""

    @abc.abstractmethod
    def feasible(self, metric, params):
        """The target rule: True where `metric` meets the target. Used for the floor check and
        for every loop iterate. Exact comparison."""

    @abc.abstractmethod
    def step_up(self, metric, params):
        """True where the search must increase alpha. Equal to `~feasible(metric)` for every
        non-NaN metric; kept as a separate member so a NaN metric behaves as it always did."""

    @abc.abstractmethod
    def predict(self, output, region):
        """What `UncertaintyQuantifier.predict` returns, given `region(output, q_hat)`."""

    def accepted(self, y, output, params):
        """`1{y in B}` per sample, a boolean array of shape `(n,)`: whether the model's answer is
        acceptable, `B` being the score's acceptance region (the complement is the model's error).
        Traced. Needed only by `UncertaintyQuantifier.uncertainty_budget`; a score that does not
        implement it works everywhere else."""
        raise NotImplementedError(
            f"score {self.name!r} does not implement accepted(y, output, params), which "
            "uncertainty_budget needs (the acceptance region B of the score)."
        )


# ---- LAC --------------------------------------------------------------------------------


def _check_integral_label_dtype(y, where: str) -> None:
    """Raise `ValueError` unless `y` holds labels that cast to integers without loss.

    Integer and boolean dtypes pass. A floating dtype passes only if every value is finite
    and integral (`2.0`); the error names the first other value and how often it occurs.
    Any other dtype (complex, object, string) is rejected. `y` is host numpy or a jax
    array (on any device).

    Without it the integer cast truncates toward zero: `0.9` becomes class 0, and `-0.5`
    becomes 0 and passes the `[0, K)` range check.
    """
    if jnp.issubdtype(y.dtype, jnp.integer) or jnp.issubdtype(y.dtype, jnp.bool_):
        return
    if not jnp.issubdtype(y.dtype, jnp.floating):
        raise ValueError(
            f"{where}: labels must have an integer, boolean or floating dtype; got {y.dtype}."
        )
    xp = jnp if isinstance(y, jax.Array) else np
    flat = y.ravel()
    bad = ~(xp.isfinite(flat) & (flat == xp.trunc(flat)))
    n_bad = int(bad.sum())
    if n_bad == 0:
        return
    first = np.asarray(flat[int(xp.argmax(bad))])[()]
    n_first = int((xp.isnan(flat) if np.isnan(first) else flat == first).sum())
    raise ValueError(
        f"{where}: floating-point labels must be finite and integral (such as 2.0); "
        f"got {first!s} ({n_first} occurrence{'s' if n_first != 1 else ''}; "
        f"{n_bad} invalid label{'s' if n_bad != 1 else ''} in total)."
    )


def _check_label_range(y, K: int | None, where: str) -> None:
    """Raise `ValueError` if any label of the integer array `y` is outside `[0, K)`.

    `y` is host numpy on the tuning and evaluation paths, the device array on `calibrate`.

    Without it a label `-1` wraps to the last class and any label `>= K` clamps to it (JAX
    gather semantics), silently producing a wrong coverage and `EC`. Skipped when `K` is
    unknown or `y` is empty.
    """
    if K is None or y.size == 0:
        return
    lo, hi = int(y.min()), int(y.max())
    if lo < 0 or hi >= K:
        raise ValueError(
            f"{where}: labels must lie in [0, K) with K={K} (the number of softmax columns); "
            f"got min={lo}, max={hi}."
        )


@dataclass(frozen=True)
class LAC(Score):
    """Least-ambiguous set-valued classifier: `C = {k : 1 - p_k <= q_hat}`.

    Input: a softmax output of shape `(n, K)`; labels integers in `[0, K)`. `B = {argmax}`, so
    `pi = 1/|C|` for a covered sample (the argmax belongs to every non-empty LAC set). The search
    steers by the unconditional mean set size, feasible when `<= 1`. Takes no `tol` or
    `target_ratio`. `predict` returns `(argmax, boolean sets)`.

    `evaluate` and `per_sample` reproduce, expression for expression, the classification code
    they replaced, so classification results are bit-for-bit unchanged; do not "simplify" them.
    """

    name: ClassVar[str] = "lac"
    label_dtype: ClassVar[Any] = jnp.int32
    uses_classes: ClassVar[bool] = True

    def resolve_params(self, *, tol, target_ratio, where: str):
        if tol is not None or target_ratio is not None:
            raise ValueError(
                f"{where}: score 'lac' takes no tol or target_ratio (classification steers by "
                f"mean prediction-set size <= 1); got tol={tol!r}, target_ratio={target_ratio!r}."
            )
        return ()

    def check_label_dtype(self, y, where: str) -> None:
        _check_integral_label_dtype(y, where)

    def check_output(self, output, y, where: str) -> None:
        # Classification's shape checks (columns, rows) live in UncertaintyQuantifier and run
        # in their established order; nothing is added here.
        return None

    def prepare_labels(self, y, output, where: str):
        return y.flatten().astype(int)

    def check_label_range(self, y, K, where: str) -> None:
        _check_label_range(y, K, where)

    def describe_target(self, params):
        return "mean prediction-set size", "<=", "1"

    def cal_score(self, y, smx):
        return lac_cal(y, smx)

    def region(self, smx, q_hat):
        scores = lac(smx)
        return scores <= q_hat

    def evaluate(self, y, smx, q_hat, valid_mask, params):
        n_valid = jnp.maximum(valid_mask.sum(), 1)
        prediction_sets = self.region(smx, q_hat)

        set_sizes = prediction_sets.sum(axis=1)
        setsize = jnp.where(valid_mask, set_sizes, 0.0).sum() / n_valid

        is_covered = prediction_sets[jnp.arange(y.shape[0]), y]
        mask_succ = is_covered & (set_sizes > 0) & valid_mask
        safe_sizes = jnp.where(mask_succ, set_sizes, 1)
        inv_succ = jnp.where(mask_succ, 1.0 / safe_sizes, 0.0)
        n_succ = mask_succ.sum()
        EC = inv_succ.sum() / jnp.maximum(n_succ, 1)

        return setsize, EC

    def per_sample(self, y, smx, q_hat, valid_mask, params):
        prediction_sets = self.region(smx, q_hat)
        set_sizes = prediction_sets.sum(axis=1)
        is_covered = prediction_sets[jnp.arange(y.shape[0]), y]
        succ = is_covered & (set_sizes > 0) & valid_mask
        z = jnp.where(succ, 1.0 / jnp.maximum(set_sizes, 1), 0.0)
        return is_covered, succ, z

    def init_metric(self, params):
        return 0.0

    def feasible(self, metric, params):
        return metric <= 1.0

    def step_up(self, metric, params):
        return metric > 1.0

    def predict(self, smx, region):
        return jnp.argmax(smx, axis=1), region

    def accepted(self, y, smx, params):
        """`B = {argmax}`: the top-1 label is the true one."""
        return jnp.argmax(smx, axis=1) == y


# ---- AbsResidual ------------------------------------------------------------------------


def _intersection_length(lower_cp, upper_cp, lower_tol, upper_tol):
    """Length of `[lower_cp, upper_cp] ∩ [lower_tol, upper_tol]` (0 if they do not overlap)."""
    left = jnp.maximum(lower_cp, lower_tol)
    right = jnp.minimum(upper_cp, upper_tol)
    return jnp.maximum(0.0, right - left)


@dataclass(frozen=True)
class AbsResidual(Score):
    """Absolute residual `s = |y - y_hat|`, interval `C = [y_hat - q_hat, y_hat + q_hat]`.

    Input: point predictions `y_hat`, a 1-D array of shape `(n,)` (an `(n, 1)` column is
    rejected, not reshaped); targets finite numbers of the same shape.
    The acceptance region is the fixed window `B = y_hat ± tol` chosen by the user, and
    `pi = |C ∩ B| / |C|` (1 when `C` is the single point `y_hat`, since then `C ⊆ B`). The search
    steers by the mean of `pi` over valid covered samples, feasible when `>= target_ratio`.
    `tol` is required (finite, `> 0`, in the units of the target); `target_ratio` defaults to 1.0
    (`C ⊆ B`) and must lie in `(0, 1]`. `classes` is not supported. `predict` returns
    `(lower, upper)`.

    Because `C` and `B` share their centre, `pi = min(1, tol / q_hat)` is the same for every
    sample: the search metric depends on the tuning data only through whether some sample is
    covered. See CONTRIBUTING.md, "Departures from the published paper".
    """

    name: ClassVar[str] = "abs_error"
    label_dtype: ClassVar[Any] = jnp.float64
    uses_classes: ClassVar[bool] = False

    def resolve_params(self, *, tol, target_ratio, where: str):
        if tol is None:
            raise ValueError(
                f"{where}: score 'abs_error' requires tol, the half-width of the tolerance "
                f"window B = y_hat ± tol, in the units of the target."
            )
        try:
            tol_value = float(tol)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: tol must be a finite number > 0; got {tol!r}.") from None
        if not np.isfinite(tol_value) or tol_value <= 0.0:
            raise ValueError(f"{where}: tol must be a finite number > 0; got {tol!r}.")
        if target_ratio is None:
            ratio_value = 1.0
        else:
            try:
                ratio_value = float(target_ratio)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{where}: target_ratio must be a finite number in (0, 1]; got {target_ratio!r}."
                ) from None
            # 0 is excluded: every alpha would be feasible and the search always FLOOR_LIMITED.
            if not np.isfinite(ratio_value) or not (0.0 < ratio_value <= 1.0):
                raise ValueError(
                    f"{where}: target_ratio must be a finite number in (0, 1]; got {target_ratio!r}."
                )
        return TolParams(jnp.asarray(tol_value, dtype=jnp.float64),
                         jnp.asarray(ratio_value, dtype=jnp.float64))

    def check_label_dtype(self, y, where: str) -> None:
        if not (jnp.issubdtype(y.dtype, jnp.integer) or jnp.issubdtype(y.dtype, jnp.floating)):
            raise ValueError(
                f"{where}: regression targets must have an integer "
                f"or floating-point dtype; got {y.dtype}."
            )
        xp = jnp if isinstance(y, jax.Array) else np
        if not bool(xp.all(xp.isfinite(y))):
            raise ValueError(f"{where}: regression targets must contain only finite values.")

    def check_output(self, output, y, where: str) -> None:
        if y is None:
            if output.ndim != 1:
                raise ValueError(
                    f"{where}: score 'abs_error' expects the model output to be a 1-D array of "
                    f"point predictions, shape (n_samples,); got shape {tuple(output.shape)}."
                )
            return
        if output.ndim != 1 or tuple(y.shape) != tuple(output.shape):
            raise ValueError(
                f"{where}: score 'abs_error' expects the model output to be a 1-D array of "
                f"point predictions, shape (n_samples,), and targets of the same shape; got "
                f"predictions {tuple(output.shape)} and targets {tuple(y.shape)}."
            )

    def prepare_labels(self, y, output, where: str):
        self.check_output(output, y, where)
        return y.astype(np.float64)

    def check_label_range(self, y, K, where: str) -> None:
        return None

    def describe_target(self, params):
        return "overlap ratio", ">=", repr(float(params.target))

    def cal_score(self, y, y_hat):
        return abs_error_cal(y, y_hat)

    def region(self, y_hat, q_hat):
        return abs_error(y_hat, q_hat)

    def _efficiency(self, y_hat, region, params):
        lower_cp, upper_cp = region
        length = upper_cp - lower_cp
        inter = _intersection_length(lower_cp, upper_cp, y_hat - params.tol, y_hat + params.tol)
        positive = length > 0
        # length == 0 (q_hat == 0): C is the point y_hat, inside B, so pi = 1.
        return jnp.where(positive, jnp.minimum(1.0, inter / jnp.where(positive, length, 1.0)), 1.0)

    def _covered(self, y, region):
        lower_cp, upper_cp = region
        return (y >= lower_cp) & (y <= upper_cp)

    def evaluate(self, y, y_hat, q_hat, valid_mask, params):
        region = self.region(y_hat, q_hat)
        pi = self._efficiency(y_hat, region, params)
        valid_covered = self._covered(y, region) & valid_mask
        n_valid_covered = valid_covered.sum()
        ratio = jnp.where(
            n_valid_covered > 0,
            (pi * valid_covered).sum() / jnp.maximum(n_valid_covered, 1),
            0.0,
        )
        return ratio, ratio

    def per_sample(self, y, y_hat, q_hat, valid_mask, params):
        region = self.region(y_hat, q_hat)
        is_covered = self._covered(y, region)
        succ = is_covered & valid_mask
        z = jnp.where(succ, self._efficiency(y_hat, region, params), 0.0)
        return is_covered, succ, z

    def init_metric(self, params):
        return params.target + 1.0

    def feasible(self, metric, params):
        return metric >= params.target

    def step_up(self, metric, params):
        return metric < params.target

    def predict(self, y_hat, region):
        return region

    def accepted(self, y, y_hat, params):
        """`B = y_hat ± tol` (closed): `|y - y_hat| <= tol`."""
        return jnp.abs(y - y_hat) <= params.tol


# ---- resolver ---------------------------------------------------------------------------

_LAC = LAC()
_ABS_RESIDUAL = AbsResidual()


def resolve_score(score) -> Score:
    """The score object for `UncertaintyQuantifier(score=...)`: a `Score` instance is returned
    as is; `'lac'` and `'abs_error'` map to module-level singletons."""
    if isinstance(score, Score):
        return score
    match score:
        case 'lac':
            return _LAC
        case 'abs_error':
            return _ABS_RESIDUAL
        case 'aps':
            raise ValueError(
                "score='aps' is not implemented in the JAX backend. "
                "The only implementation of APS lived in the numpy "
                "backend, which is unreachable in the current "
                "configuration. This is a known gap, not a typo. "
                "'lac' is the supported value."
            )
        case _:
            raise ValueError(
                f"Unknown score {score!r}. The supported value is 'lac' (clf) and 'abs_error' (reg)."
            )

"""Score objects (`utrace.scores.objects`) driving `UncertaintyQuantifier`.

Covers: string/object equivalence, the deprecated `eps` alias, per-call parameter validation
(`tol` required and `> 0` for `AbsResidual`, rejected by `LAC`), `pi = 1` at `q_hat == 0`, the jit
cache under a `tol`/`target_ratio` sweep, classification values frozen before the refactor
(bit-for-bit), the single exact target rule, `predict`'s return per score, custom scores, and
`classes` with a score that does not support it.
"""
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

import utrace
import utrace.uncertaintyQuantifier as uqm
from utrace import (
    LAC,
    AbsResidual,
    Score,
    SearchStatus,
    SearchStatusWarning,
    UncertaintyQuantifier,
)
from utrace.scores import TolParams
from utrace.uncertaintyQuantifier import TuningDataReuseWarning


def _clf(seed, K, n):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0, size=(n, K))
    y = rng.integers(0, K, size=n)
    logits[np.arange(n), y] += 1.5
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), y


def _reg(seed, n, noise=0.3):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return y + rng.normal(scale=noise, size=n), y


def _calibrated_reg(N=200, seed=0):
    y_hat, y = _reg(seed, N)
    uq = UncertaintyQuantifier(N=N, score="abs_error")
    uq.calibrate(y_hat, y)
    return uq


def _calibrated_clf(N=200, K=4, seed=0):
    p, y = _clf(seed, K, N)
    uq = UncertaintyQuantifier(N=N)
    uq.calibrate(p, y)
    return uq


def _params(tol, target=1.0):
    return TolParams(jnp.asarray(tol, dtype=jnp.float64), jnp.asarray(target, dtype=jnp.float64))


# ---- 1. string and object are equivalent ---------------------------------------------------

def test_exports():
    assert utrace.LAC is LAC and utrace.AbsResidual is AbsResidual and utrace.Score is Score
    assert utrace.scores.LAC is LAC


def test_strings_map_to_singletons():
    assert UncertaintyQuantifier(score="lac").score_ is UncertaintyQuantifier().score_
    assert UncertaintyQuantifier(score="abs_error").score_ is UncertaintyQuantifier(score="abs_error").score_
    assert isinstance(UncertaintyQuantifier().score_, LAC)
    assert isinstance(UncertaintyQuantifier(score="abs_error").score_, AbsResidual)
    assert LAC() == LAC() and hash(LAC()) == hash(LAC())
    assert LAC() != AbsResidual()


def test_string_and_object_give_identical_classification_results():
    p, y = _clf(3, 5, 900)
    results = []
    for score in ("lac", LAC()):
        uq = UncertaintyQuantifier(N=300, score=score)
        uq.calibrate(p[:300], y[:300])
        alpha = uq.tune_alpha(p[300:600], y[300:600])
        results.append((alpha, uq.uncertainty_at(p[600:], y[600:], alpha, delta=0.05)))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_string_and_object_give_identical_regression_results():
    y_hat, y = _reg(4, 900)
    results = []
    for score in ("abs_error", AbsResidual()):
        uq = UncertaintyQuantifier(N=300, score=score)
        uq.calibrate(y_hat[:300], y[:300])
        alpha = uq.tune_alpha(y_hat[300:600], y[300:600], tol=0.3, target_ratio=0.9)
        results.append((alpha, uq.uncertainty_at(y_hat[600:], y[600:], alpha, tol=0.3, delta=0.05)))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


# ---- 2. eps, the deprecated alias of tol ---------------------------------------------------

def test_eps_alias_warns_and_matches_tol():
    uq = _calibrated_reg()
    y_hat, y = _reg(5, 300)
    a_tol = uq.tune_alpha(y_hat, y, tol=0.25, target_ratio=0.9)
    with pytest.warns(DeprecationWarning, match="eps is deprecated"):
        a_eps = uq.tune_alpha(y_hat, y, eps=0.25, target_ratio=0.9)
    assert a_tol == a_eps
    r_tol = uq.uncertainty_at(y_hat, y, a_tol, tol=0.25, delta=0.05)
    with pytest.warns(DeprecationWarning, match="0.5.0"):
        r_eps = uq.uncertainty_at(y_hat, y, a_tol, eps=0.25, delta=0.05)
    assert r_tol == r_eps


@pytest.mark.parametrize("call", ["get_uncertainty", "tune_alpha", "uncertainty_at"])
def test_eps_and_tol_together_raise_type_error(call):
    uq = _calibrated_reg()
    y_hat, y = _reg(5, 100)
    args = (y_hat, y, 0.3) if call == "uncertainty_at" else (y_hat, y)
    with pytest.raises(TypeError, match="tol only"):
        getattr(uq, call)(*args, tol=0.2, eps=0.2)


def test_get_uncertainty_with_eval_set_emits_one_deprecation_warning():
    uq = _calibrated_reg()
    y_hat, y = _reg(6, 400)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        uq.get_uncertainty(y_hat[:200], y[:200], eps=0.3,
                           eval_softmax=y_hat[200:], eval_y=y[200:])
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__      # attributed to the caller


# ---- 3. tol required (and > 0) for AbsResidual ---------------------------------------------

@pytest.mark.parametrize("call", ["get_uncertainty", "tune_alpha", "uncertainty_at"])
def test_tol_required_for_abs_residual_before_any_computation(call):
    uq = UncertaintyQuantifier(N=10, score="abs_error")      # not even calibrated
    y_hat, y = _reg(7, 20)
    args = (y_hat, y, 0.3) if call == "uncertainty_at" else (y_hat, y)
    with pytest.raises(ValueError, match="requires tol"):
        getattr(uq, call)(*args)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.inf, np.nan, "a"])
def test_tol_must_be_finite_and_positive(bad):
    uq = _calibrated_reg()
    y_hat, y = _reg(7, 20)
    with pytest.raises(ValueError, match="tol must be a finite number > 0"):
        uq.tune_alpha(y_hat, y, tol=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan])
def test_target_ratio_must_lie_in_unit_interval(bad):
    uq = _calibrated_reg()
    y_hat, y = _reg(7, 20)
    with pytest.raises(ValueError, match="target_ratio"):
        uq.tune_alpha(y_hat, y, tol=0.2, target_ratio=bad)


def test_abs_residual_target_ratio_defaults_to_one():
    uq = _calibrated_reg()
    y_hat, y = _reg(8, 200)
    assert uq.tune_alpha(y_hat, y, tol=0.3) == uq.tune_alpha(y_hat, y, tol=0.3, target_ratio=1.0)


# ---- 4. LAC rejects tol and target_ratio ---------------------------------------------------

@pytest.mark.parametrize("call, kwargs", [
    (call, kwargs)
    for call in ("get_uncertainty", "tune_alpha", "uncertainty_at")
    for kwargs in ({"tol": 0.1}, {"target_ratio": 0.9}, {"eps": 0.1})
    if not (call == "uncertainty_at" and "target_ratio" in kwargs)   # it has no target_ratio
])
def test_lac_rejects_per_call_parameters(call, kwargs):
    uq = _calibrated_clf()
    p, y = _clf(9, 4, 50)
    args = (p, y, 0.3) if call == "uncertainty_at" else (p, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="score 'lac' takes no tol or target_ratio"):
            getattr(uq, call)(*args, **kwargs)


def test_tune_alpha_parameters_are_keyword_only():
    uq = _calibrated_reg()
    y_hat, y = _reg(9, 20)
    with pytest.raises(TypeError):
        uq.tune_alpha(y_hat, y, 30, 0.2)


# ---- 5. q_hat == 0 gives pi = 1 ------------------------------------------------------------

def test_pi_is_one_at_zero_q_hat_in_evaluate_and_per_sample():
    score = AbsResidual()
    y_hat = jnp.array([1.0, 2.0, 3.0])
    y = y_hat
    mask = jnp.array([True, True, True])
    metric, EC = score.evaluate(y, y_hat, jnp.float64(0.0), mask, _params(0.1))
    assert float(metric) == 1.0 and float(EC) == 1.0
    covered, succ, z = score.per_sample(y, y_hat, jnp.float64(0.0), mask, _params(0.1))
    assert np.asarray(covered).all() and np.asarray(succ).all()
    np.testing.assert_array_equal(np.asarray(z), [1.0, 1.0, 1.0])


def test_zero_calibration_residuals_give_ec_one_through_public_api():
    y_cal = np.arange(50, dtype=float)
    uq = UncertaintyQuantifier(N=50, score="abs_error")
    uq.calibrate(y_cal, y_cal)                         # every residual is 0, so q_hat = 0
    r = uq.uncertainty_at(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.5, tol=0.1)
    assert r.EC == 1.0 and r.n_succ == 2
    assert r.U == 1.0 - 1.0 * (1.0 - 0.5)


# ---- 6. the jit cache does not grow per call -----------------------------------------------

def test_tol_and_target_sweep_compiles_once():
    uq = _calibrated_reg(N=300, seed=1)
    y_hat_t, y_t = _reg(10, 200)
    y_hat_e, y_e = _reg(11, 250)
    uqm._search_uncertainty.clear_cache()
    uqm._evaluate_at_alpha.clear_cache()
    tols = [0.05, 0.1, np.float32(0.15), np.float64(0.2), 0.25, 3, 0.35, 0.4, 0.45, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SearchStatusWarning)
        for tol in tols:
            for target in (0.8, 0.9, 1.0):
                alpha = uq.tune_alpha(y_hat_t, y_t, tol=tol, target_ratio=target)
                uq.uncertainty_at(y_hat_e, y_e, alpha, tol=tol)
    assert uqm._search_uncertainty._cache_size() == 1
    assert uqm._evaluate_at_alpha._cache_size() == 1


def test_fresh_score_instance_reuses_the_trace():
    p, y = _clf(12, 3, 600)
    uqm._search_uncertainty.clear_cache()
    for _ in range(3):
        uq = UncertaintyQuantifier(N=200, score=LAC())
        uq.calibrate(p[:200], y[:200])
        uq.tune_alpha(p[200:], y[200:])
    assert uqm._search_uncertainty._cache_size() == 1


# ---- 7. classification values frozen before the refactor (bit-for-bit) ---------------------

# Produced by the public API on the pre-refactor code (branch unify-clf-reg-utrace merged with
# main, before the score objects), printed with repr. Compared with ==, not approx.
_FROZEN = [
    (0, 3, 200, 150, 180, None, {
        'tune_alpha': 0.4825870655477047, 'status': 'converged',
        'gu_U': 0.48542999375898105, 'gu_alpha': 0.4825870655477047,
        'gu_eval_U': 0.48827292197025735, 'gu_eval_alpha': 0.4825870655477047,
        'eval': {'U': 0.48827292197025735, 'EC': 0.989010989010989, 'n_covered': 91, 'n_valid': 180, 'n_succ': 91, 'coverage': 0.5055555555555555, 'U_empirical': 0.5, 'delta': 0.05, 'coverage_lower': 0.43017901054286856, 'EC_lower': 0.8338431140455663, 'U_delta': 0.641298194251894},
    }),
    (1, 10, 500, 300, 257, None, {
        'tune_alpha': 0.7305389223620296, 'status': 'converged',
        'gu_U': 0.7516412959119911, 'gu_alpha': 0.7305389223620296,
        'gu_eval_U': 0.7574850301258266, 'gu_eval_alpha': 0.7305389223620296,
        'eval': {'U': 0.7574850301258266, 'EC': 0.9, 'n_covered': 50, 'n_valid': 257, 'n_succ': 50, 'coverage': 0.19455252918287938, 'U_empirical': 0.8249027237354085, 'delta': 0.05, 'coverage_lower': 0.14798534903756283, 'EC_lower': 0.6906670920597079, 'U_delta': 0.8977913893127856},
    }),
    (2, 5, 60, 40, 90, [1, 3], {
        'tune_alpha': 0.6071428572759032, 'status': 'converged',
        'gu_U': 0.6071428572759032, 'gu_alpha': 0.6071428572759032,
        'gu_eval_U': 0.650793650911914, 'gu_eval_alpha': 0.6071428572759032,
        'eval': {'U': 0.650793650911914, 'EC': 0.8888888888888888, 'n_covered': 18, 'n_valid': 36, 'n_succ': 18, 'coverage': 0.5, 'U_empirical': 0.5555555555555556, 'delta': 0.05, 'coverage_lower': 0.3292217994577103, 'EC_lower': 0.540000708988402, 'U_delta': 0.8222199948783989},
    }),
    (3, 2, 1000, 700, 600, None, {
        'tune_alpha': 0.3216783218085766, 'status': 'converged',
        'gu_U': 0.3216783218085766, 'gu_alpha': 0.3216783218085766,
        'gu_eval_U': 0.3216783218085766, 'gu_eval_alpha': 0.3216783218085766,
        'eval': {'U': 0.3216783218085766, 'EC': 1.0, 'n_covered': 424, 'n_valid': 600, 'n_succ': 424, 'coverage': 0.7066666666666667, 'U_empirical': 0.29333333333333333, 'delta': 0.05, 'coverage_lower': 0.6684482248519119, 'EC_lower': 0.9720045573627764, 'U_delta': 0.35026527908288374},
    }),
    (4, 7, 300, 256, 100, [0], {
        'tune_alpha': 0.6956521747633815, 'status': 'converged',
        'gu_U': 0.6956521747633815, 'gu_alpha': 0.6956521747633815,
        'gu_eval_U': 0.6956521747633815, 'gu_eval_alpha': 0.6956521747633815,
        'eval': {'U': 0.6956521747633815, 'EC': 1.0, 'n_covered': 4, 'n_valid': 13, 'n_succ': 4, 'coverage': 0.3076923076923077, 'U_empirical': 0.6923076923076923, 'delta': 0.05, 'coverage_lower': 0.0909203945720966, 'EC_lower': 0.2598964063496009, 'U_delta': 0.9763701161868243},
    }),
    (5, 4, 25, 30, 31, None, {
        'tune_alpha': 0.6538461539894342, 'status': 'converged',
        'gu_U': 0.6538461539894342, 'gu_alpha': 0.6538461539894342,
        'gu_eval_U': 0.6640271494603333, 'gu_eval_alpha': 0.6538461539894342,
        'eval': {'U': 0.6640271494603333, 'EC': 0.9705882352941176, 'n_covered': 17, 'n_valid': 31, 'n_succ': 17, 'coverage': 0.5483870967741935, 'U_empirical': 0.467741935483871, 'delta': 0.05, 'coverage_lower': 0.36034232716081716, 'EC_lower': 0.611585259938846, 'U_delta': 0.7796199441763829},
    }),
]


@pytest.mark.parametrize("seed, K, N, Bt, Be, classes, expected", _FROZEN)
def test_classification_values_bit_for_bit(seed, K, N, Bt, Be, classes, expected):
    p, y = _clf(seed, K, N + Bt + Be)
    cal, tune, ev = slice(0, N), slice(N, N + Bt), slice(N + Bt, None)
    uq = UncertaintyQuantifier(N=N, classes=classes)
    uq.calibrate(p[cal], y[cal])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TuningDataReuseWarning)
        alpha = uq.tune_alpha(p[tune], y[tune])
        status = uq.search_status_
        U1, a1 = uq.get_uncertainty(p[tune], y[tune])
    U2, a2 = uq.get_uncertainty(p[tune], y[tune], eval_softmax=p[ev], eval_y=y[ev])
    r = uq.uncertainty_at(p[ev], y[ev], alpha, delta=0.05)
    assert float(alpha) == expected['tune_alpha']
    assert status.value == expected['status']
    assert (float(U1), float(a1)) == (expected['gu_U'], expected['gu_alpha'])
    assert (float(U2), float(a2)) == (expected['gu_eval_U'], expected['gu_eval_alpha'])
    for field, value in expected['eval'].items():
        assert getattr(r, field) == value, field


# ---- 8. one exact target rule --------------------------------------------------------------

def test_metric_just_below_target_is_not_feasible():
    """Pre-refactor, `>= target - 1e-6` in the loop (exact elsewhere) returned CONVERGED at
    alpha = 0.25 here, where the overlap ratio is target - 5e-7."""
    N = 9
    uq = UncertaintyQuantifier(N=N, score="abs_error")
    uq.calibrate(np.array([0.1] * 8 + [1.0]), np.zeros(N))
    target = 0.9
    tol = target * 1.0 - 5e-7                      # pi = tol / q_hat at q_hat = 1.0
    alpha = uq.tune_alpha(np.zeros(20), np.zeros(20), tol=tol, target_ratio=target)
    q_hat = uqm._q_hat_from_alpha(uq.conformity_scores_, jnp.int32(N), jnp.float64(alpha))
    assert uq.search_status_ is SearchStatus.CONVERGED
    assert float(q_hat) == 0.1                     # pi = 1 >= target, not target - 5e-7
    assert alpha > 0.25


@dataclass(frozen=True)
class _MetricOfQ(AbsResidual):
    """`AbsResidual` whose search metric is a controlled function of `q_hat`."""
    metric_fn: Callable = None

    def evaluate(self, y, output, q_hat, valid_mask, params):
        return jnp.asarray(self.metric_fn(q_hat), dtype=jnp.float64), jnp.asarray(0.5, dtype=jnp.float64)


@pytest.mark.parametrize("gap", [5e-7, 1e-9, 0.0])
def test_never_returns_a_sub_floor_alpha(gap):
    """At and below the floor q_hat is the largest score; there the metric is target - gap."""
    n_cs = 9
    cs = jnp.array([0.1] * 8 + [1.0])
    score = _MetricOfQ(metric_fn=lambda q: jnp.where(q >= 1.0, 0.9 - gap, 1.0))
    alpha, _U, status = uqm._search_uncertainty(
        jnp.zeros(4), jnp.zeros(4), jnp.ones(4, bool), cs, jnp.int32(n_cs), 30, score,
        _params(0.5, 0.9))
    floor = 1.0 / (n_cs + 1)
    assert float(alpha) >= floor
    if gap == 0.0:
        assert int(status) == uqm._STATUS_FLOOR_LIMITED and float(alpha) == floor
    else:
        assert int(status) == uqm._STATUS_CONVERGED


@pytest.mark.parametrize("score, params", [(LAC(), ()), (AbsResidual(), _params(0.3, 0.9))])
def test_step_up_is_the_complement_of_feasible(score, params):
    metrics = jnp.concatenate([jnp.linspace(0.0, 2.0, 201), jnp.array([0.9, 1.0, 0.9 - 1e-12])])
    np.testing.assert_array_equal(np.asarray(score.step_up(metrics, params)),
                                  ~np.asarray(score.feasible(metrics, params)))
    assert not bool(score.step_up(jnp.asarray(score.init_metric(params)), params))


# ---- 9. predict per score ------------------------------------------------------------------

def test_predict_returns_lower_upper_for_abs_residual():
    uq = _calibrated_reg()
    uq.alpha = 0.2
    y_hat = np.array([0.0, 1.0, -2.0])
    lower, upper = uq.predict(y_hat)
    np.testing.assert_allclose(lower, y_hat - uq._state.q_hat)
    np.testing.assert_allclose(upper, y_hat + uq._state.q_hat)


def test_predict_returns_argmax_and_sets_for_lac():
    uq = _calibrated_clf(K=4)
    uq.alpha = 0.2
    p, _ = _clf(13, 4, 10)
    y_pred, y_sets = uq.predict(p)
    np.testing.assert_array_equal(y_pred, p.argmax(1))
    assert y_sets.shape == (10, 4) and y_sets.dtype == bool


# ---- 10. custom scores ---------------------------------------------------------------------

def test_custom_score_instance_is_accepted():
    @dataclass(frozen=True)
    class Halved(AbsResidual):
        name = "halved"

        def cal_score(self, y, y_hat):
            return 0.5 * super().cal_score(y, y_hat)

    uq = UncertaintyQuantifier(N=100, score=Halved())
    assert uq.score_name == "halved" and isinstance(uq.score_, Halved)
    y_hat, y = _reg(14, 300)
    uq.calibrate(y_hat[:100], y[:100])
    alpha = uq.tune_alpha(y_hat[100:], y[100:], tol=0.2, target_ratio=0.8)
    assert 1.0 / 101 <= alpha <= 1.0


@pytest.mark.parametrize("bad", [3, None, LAC, ["lac"]])
def test_non_score_non_string_is_rejected(bad):
    with pytest.raises(ValueError, match="Unknown score"):
        UncertaintyQuantifier(score=bad)


# ---- 11. classes with a score that does not support it -------------------------------------

def test_classes_with_abs_residual_raises_at_init():
    with pytest.raises(ValueError, match="classes=.*not supported by score 'abs_error'"):
        UncertaintyQuantifier(score="abs_error", classes=[0])
    UncertaintyQuantifier(score="lac", classes=[0])     # still fine for LAC

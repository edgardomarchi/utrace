# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How the method behaves on a trained MNIST model
#
# This notebook produces **evidence about the method**, for the author's decisions about it. It is not
# the user-facing tutorial. It uses the public API of `utrace` only (`UncertaintyQuantifier`,
# `tune_alpha`, `uncertainty_at`, `get_uncertainty`, `predict`, `UncertaintyEvaluation`); the model,
# data loading and the noise transform are the ones of `scripts/MNIST_example.py`.
#
# **Design.** A CNN is trained on the 10 000 MNIST `t10k` images. The 60 000 training images are never
# seen by it and form the *pool*. For each noise level the pool's softmax is computed once and cached.
# Each repetition draws a stratified random partition of the pool into four disjoint subsets:
# calibration, tuning, evaluation and *validation* (the remainder, about 54 000 samples, large on
# purpose: it stands in for the population). Everything the method estimates is then compared with what
# validation shows.
#
# Notation: `U_E` is the argmax error rate on validation (the quantity the method bounds);
# `U_reuse` is `get_uncertainty` on the tuning set alone (the legacy path); `U` and `U_empirical` and
# `U_delta` come from `uncertainty_at` on the evaluation set at the tuned `alpha`; `U_star = 1 - EC *
# coverage` measured on validation at the same `alpha` is the population value of the estimand that
# `U_delta` bounds.

# %% [markdown]
# ## Parameters

# %%
import os

# QUICK is read from the environment so the same file runs both ways without editing:
#   UTRACE_NB_QUICK=1 (default) -> a few minutes on CPU;  UTRACE_NB_QUICK=0 -> the full run.
QUICK = os.environ.get("UTRACE_NB_QUICK", "1") != "0"
MODE = "quick" if QUICK else "full"                 # every output file carries it, so the two modes never overwrite each other

SEED = 20261015
POOL_NOISES_FULL = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]   # AWGN std, in the units of MNIST_example.py (see below)
DELTAS = [0.05, 0.01]
CAL_SIZE, TUNE_SIZE, EVAL_SIZE = 2000, 2000, 2000
EVAL_SIZES = [300, 1000, 2000, 5000]                # evaluation-size experiment
DEGRADED = 1.5                                      # the "one degraded level" of the secondary experiments
MIN_COUNT_UNIFORM = 100                             # fewest DISTINCT samples with |C|=k for the uniform-selection check to use k

if QUICK:
    L, L_CC = 4, 3
    NOISES = [0.0, DEGRADED]
    SWEEP_POINTS = 40
else:
    L, L_CC = 50, 20
    NOISES = POOL_NOISES_FULL
    SWEEP_POINTS = 150

PARAMS = {"QUICK": QUICK, "MODE": MODE, "SEED": SEED, "L": L, "L_CC": L_CC, "NOISES": NOISES, "DELTAS": DELTAS,
              "CAL_SIZE": CAL_SIZE, "TUNE_SIZE": TUNE_SIZE, "EVAL_SIZE": EVAL_SIZE, "EVAL_SIZES": EVAL_SIZES,
              "DEGRADED": DEGRADED, "MIN_COUNT_UNIFORM": MIN_COUNT_UNIFORM, "SWEEP_POINTS": SWEEP_POINTS}
print(PARAMS)

# %% [markdown]
# ## Setup: paths, versions, output locations

# %%
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utrace
from utrace import SearchStatusWarning, TuningDataReuseWarning, UncertaintyQuantifier
from utrace.utils.pytorch.example_models import ImageClassifierCNN, train_and_save
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

T_START = time.time()


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "utrace").exists():
            return q
    raise RuntimeError("run this notebook from inside the utrace repository")


REPO = _find_repo_root()
BUILD = REPO / "notebooks" / "_build"
FIG = BUILD / "figures" / MODE
RESULTS_PATH = BUILD / f"results_{MODE}.json"
for d_ in (BUILD / "cache", FIG, BUILD / "model"):
    d_.mkdir(parents=True, exist_ok=True)
DATA_ROOT = REPO / "data"          # where scripts/MNIST_example.py (run from the repo root) keeps MNIST
assert (DATA_ROOT / "MNIST" / "raw").exists(), f"MNIST not found under {DATA_ROOT}; download it with the example script first"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()  # noqa: PLW1510


# Every location this notebook writes to must be git-ignored; fail loudly otherwise.
for p in (BUILD / "cache" / "x.npy", FIG / "x.png", BUILD / "model" / "x.pt", RESULTS_PATH):
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(p)]).returncode == 0, f"{p} is not git-ignored"  # noqa: PLW1510

COMMIT = _git("rev-parse", "HEAD")
DIRTY = bool(_git("status", "--porcelain", "--", "src"))
print("repo", REPO, "\ncommit", COMMIT, "(src dirty)" if DIRTY else "")
print("python", sys.version.split()[0], "| jax", jax.__version__, jax.default_backend(),
      "| torch", torch.__version__, "| utrace", utrace.__file__)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

# %% [markdown]
# ## Model and degradation
#
# **Model.** `ImageClassifierCNN`, trained with `train_and_save` exactly as `MNIST_example.py` does when
# `train_model=True`: `datasets.MNIST(train=False)` (the 10 000 `t10k` images), `ToTensor` then
# `Normalize((0.5,), (0.5,))`, batch 300, shuffled, Adam lr 1e-3, 20 epochs. The only addition is a seed.
# The checkpoint kept under `.model/` has no metadata saying which split it was trained on, so it is not
# used: the model is retrained here (once; cached under `notebooks/_build/model/`).
#
# **Degradation.** As in the script: `ToTensor`, `Normalize((0.5,), (0.5,))` (pixels in [-1, 1]), then
# `AddGaussianNoise(0., sigma)`: `x + sigma * randn`, **no clipping**, a fresh draw per image access.
# Here each pool image is accessed exactly once per noise level, with a fixed torch seed, so one noise
# realisation per level is shared by all repetitions (repetitions vary the partition, not the noise).

# %%
MODEL_PTH = BUILD / "model" / "ImageClassifierCNN_t10k_seed42.pt"
TRAIN_SEED, TRAIN_BATCH, TRAIN_EPOCHS = 42, 300, 20
base_tf = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

torch.manual_seed(TRAIN_SEED)
net = ImageClassifierCNN()
if not MODEL_PTH.exists():
    t0 = time.time()
    train_ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=False, transform=transforms.Compose(base_tf))
    assert len(train_ds) == 10_000
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True)
    train_and_save(classifier=net, train_dataloader=train_loader, model_pth=MODEL_PTH,
                   device="cpu", epochs=TRAIN_EPOCHS, seed=TRAIN_SEED)
    print(f"trained on t10k in {time.time() - t0:.0f}s")
net.load_state_dict(torch.load(MODEL_PTH, weights_only=True))
net.eval()
model = Pytorch_wrapper(net, classes=np.arange(10), device="cpu")
K = 10
print("model checkpoint:", MODEL_PTH.name, hashlib.sha256(MODEL_PTH.read_bytes()).hexdigest()[:16])

# %%
def pool_softmax(sigma: float):
    """Softmax (float64, shape (60000, 10)) and labels of the 60 000 training images at noise `sigma`."""
    tag = f"sigma{sigma:g}"
    cache = BUILD / "cache" / f"pool_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["smx"].astype(np.float64), z["y"].astype(np.int64)
    ds = datasets.MNIST(root=str(DATA_ROOT), train=True, download=False,
                        transform=transforms.Compose([*base_tf, AddGaussianNoise(0., float(sigma))]))
    assert len(ds) == 60_000
    torch.manual_seed(SEED + round(sigma * 100))
    loader = DataLoader(ds, batch_size=1000, shuffle=False)
    smx, ys = [], []
    for X, y in loader:
        smx.append(np.asarray(model.predict_proba(X).cpu()))
        ys.append(y.numpy())
    smx, y = np.concatenate(smx), np.concatenate(ys)
    np.savez_compressed(cache, smx=smx.astype(np.float32), y=y.astype(np.int8))
    return np.load(cache)["smx"].astype(np.float64), y.astype(np.int64)


POOL = {}
t0 = time.time()
for s in sorted(set(NOISES) | {DEGRADED}):
    POOL[s] = pool_softmax(s)
    smx_, y_ = POOL[s]
    assert smx_.shape == (60_000, K) and y_.min() == 0 and y_.max() == K - 1
    print(f"sigma={s:<4} accuracy on the pool: {(smx_.argmax(1) == y_).mean():.4f}")
print(f"softmax ready in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Helpers

# %%
def stratified_partition(y: np.ndarray, sizes: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    """Disjoint index blocks of the given sizes plus the remainder, each stratified by class.

    Within each class the samples are shuffled and given the relative rank (position + U[0,1)) / n_c;
    sorting every sample by that key and cutting consecutive blocks makes each block hold every class in
    proportion to its frequency (to within one sample), with sizes exact.
    """
    key = np.empty(len(y))
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(len(idx))
        key[idx[perm]] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    order = np.argsort(key, kind="stable")
    cuts = np.cumsum(sizes)
    blocks = np.split(order, cuts)
    return [rng.permutation(b) for b in blocks]


def fields(r) -> dict:
    """Every field of an UncertaintyEvaluation as plain Python numbers."""
    return {k: (None if v is None else float(v)) for k, v in r._asdict().items()}


def margin_terms(r):
    """Exact decomposition U_delta - U_empirical = EC*(coverage - coverage_lower) + coverage_lower*(EC - EC_lower)."""
    t_cov = r.EC * (r.coverage - r.coverage_lower)
    t_ec = r.coverage_lower * (r.EC - r.EC_lower)
    assert abs((r.U_delta - r.U_empirical) - (t_cov + t_ec)) < 1e-12
    return float(t_cov), float(t_ec)


def size_counts(uq, smx, y):
    """Per prediction-set size k: n_k = #{covered, |C| = k} and c_k = #{covered, |C| = k, argmax correct}.

    Uses the instance's current `alpha`. Each sample is counted once, so `n_k` is a count of the samples
    passed in; callers pass sets of distinct images.
    """
    _, S = uq.predict(smx)
    sz = S.sum(1)
    covered = S[np.arange(len(y)), y]
    ok = smx.argmax(1) == y
    return (np.bincount(sz[covered], minlength=K + 1).tolist(),
            np.bincount(sz[covered & ok], minlength=K + 1).tolist())


def cp_interval(k: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval for k successes in n trials."""
    from scipy.stats import beta
    return (float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0,
            float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0)


def run_quiet(fn, *a, **k):
    """Call fn, return (result, number of TuningDataReuseWarning, number of SearchStatusWarning)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return (out, sum(issubclass(x.category, TuningDataReuseWarning) for x in w),
            sum(issubclass(x.category, SearchStatusWarning) for x in w))


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng([SEED, *[round(k * 100) if isinstance(k, float) else int(k) for k in key]])

# %% [markdown]
# ## Experiment 1: marginal, legacy vs separate evaluation, bound and reference
#
# Per noise level and repetition: `get_uncertainty` on the tuning set alone (`U_reuse`; the
# `TuningDataReuseWarning` is expected and counted), then `tune_alpha` on the tuning set,
# `uncertainty_at` on the evaluation set for each `delta`, and the references on validation at the same
# `alpha`. Prediction sets at `alpha` (assigning `alpha`, then `predict`) give the set composition on
# the evaluation set and, on each repetition's validation set, the per-repetition counts that serve as the
# secondary uniform-selection estimate (Experiment 5 holds the primary one).

# %%
def marginal_rep(smx, y, sizes, rng):
    cal, tun, ev, val = stratified_partition(y, sizes, rng)
    uq = UncertaintyQuantifier(N=len(cal))
    uq.calibrate(smx[cal], y[cal])

    (res_reuse, n_reuse_w, _) = run_quiet(uq.get_uncertainty, smx[tun], y[tun])
    U_reuse, alpha_reuse = float(res_reuse[0]), float(res_reuse[1])
    (alpha, _, n_status_w) = run_quiet(uq.tune_alpha, smx[tun], y[tun])
    alpha = float(alpha)
    status = uq.search_status_.value
    assert alpha == alpha_reuse

    ev_res = {d: uq.uncertainty_at(smx[ev], y[ev], alpha, delta=d) for d in DELTAS}
    r_val = uq.uncertainty_at(smx[val], y[val], alpha)
    U_star = float(1.0 - r_val.EC * r_val.coverage)
    assert U_star == float(r_val.U_empirical)
    U_E = float((smx[val].argmax(1) != y[val]).mean())

    uq.alpha = alpha
    _, S_ev = uq.predict(smx[ev])
    sz = S_ev.sum(1)
    n_k, c_k = size_counts(uq, smx[val], y[val])      # one repetition's validation set: distinct images

    rec = {"alpha": alpha, "status": status, "U_reuse": U_reuse, "U_E": U_E, "U_star": U_star,
               "n_reuse_warn": n_reuse_w, "n_status_warn": n_status_w,
               "cov_val": float(r_val.coverage), "EC_val": float(r_val.EC),
               "frac_empty": float((sz == 0).mean()), "frac_single": float((sz == 1).mean()), "frac_multi": float((sz > 1).mean()),
               "n_k": n_k, "c_k": c_k, "per_delta": {}}
    for d, r in ev_res.items():
        t_cov, t_ec = margin_terms(r)
        rec["per_delta"][str(d)] = dict(fields(r), term_coverage=t_cov, term_EC=t_ec)
    return rec


EXP1 = {}
t0 = time.time()
for ni, sigma in enumerate(NOISES):
    smx, y = POOL[sigma]
    EXP1[str(sigma)] = [marginal_rep(smx, y, [CAL_SIZE, TUNE_SIZE, EVAL_SIZE], rng_for(1, sigma, rep)) for rep in range(L)]
    print(f"sigma={sigma:<4} {L} repetitions   ({time.time() - t0:.0f}s elapsed)")
T_EXP1 = time.time() - t0

# %% [markdown]
# ## Experiment 2: evaluation-set size
#
# Calibration and tuning fixed (2000 each, so `alpha` is the same for every size within a repetition);
# the evaluation set is the first `s` samples of a random stratified-then-shuffled pool of 5000.

# %%
def size_rep(smx, y, rng):
    cal, tun, ev, val = stratified_partition(y, [CAL_SIZE, TUNE_SIZE, max(EVAL_SIZES)], rng)
    uq = UncertaintyQuantifier(N=len(cal))
    uq.calibrate(smx[cal], y[cal])
    alpha = float(run_quiet(uq.tune_alpha, smx[tun], y[tun])[0])
    r_val = uq.uncertainty_at(smx[val], y[val], alpha)
    out = {"alpha": alpha, "U_star": float(r_val.U_empirical), "sizes": {}}
    for s in EVAL_SIZES:
        per = {}
        for d in DELTAS:
            r = uq.uncertainty_at(smx[ev[:s]], y[ev[:s]], alpha, delta=d)
            t_cov, t_ec = margin_terms(r)
            per[str(d)] = dict(fields(r), term_coverage=t_cov, term_EC=t_ec)
        out["sizes"][str(s)] = per
    return out


EXP2 = {}
t0 = time.time()
for sigma in sorted({0.0, DEGRADED}):
    smx, y = POOL[sigma]
    EXP2[str(sigma)] = [size_rep(smx, y, rng_for(2, sigma, rep)) for rep in range(L)]
print(f"done in {time.time() - t0:.0f}s")
T_EXP2 = time.time() - t0

# %% [markdown]
# ## Experiment 3: class-conditional
#
# One quantifier per class (`classes=[k]`), the same four-subset scheme. `N_cal` is the number of
# calibration samples of class `k` (counted from the labels, since the instance does not expose it);
# the search floor is `1/(N_cal + 1)`. `U_E` for class `k` is the argmax error rate on the validation
# samples whose true label is `k`.

# %%
def cc_rep(smx, y, rng):
    cal, tun, ev, val = stratified_partition(y, [CAL_SIZE, TUNE_SIZE, EVAL_SIZE], rng)
    rows = []
    for k in range(K):
        uq = UncertaintyQuantifier(N=len(cal), classes=[k])
        uq.calibrate(smx[cal], y[cal])
        n_cal = int((y[cal] == k).sum())
        alpha = float(run_quiet(uq.tune_alpha, smx[tun], y[tun])[0])
        row = {"k": k, "N_cal": n_cal, "floor": 1.0 / (n_cal + 1)}
        if np.isnan(alpha):
            row.update(status="EMPTY", alpha=None)
            rows.append(row)
            continue
        status = uq.search_status_.value
        r = uq.uncertainty_at(smx[ev], y[ev], alpha, delta=0.05)
        r_val = uq.uncertainty_at(smx[val], y[val], alpha)
        mk = y[val] == k
        row.update(status=status, alpha=alpha, n_valid_eval=r.n_valid,
                   U=float(r.U), U_empirical=float(r.U_empirical), U_delta=float(r.U_delta),
                   U_star=float(r_val.U_empirical), U_E=float((smx[val][mk].argmax(1) != k).mean()))
        rows.append(row)
    return rows


EXP3 = {}
t0 = time.time()
for sigma in sorted({0.0, DEGRADED}):
    smx, y = POOL[sigma]
    EXP3[str(sigma)] = [cc_rep(smx, y, rng_for(3, sigma, rep)) for rep in range(L_CC)]
print(f"done in {time.time() - t0:.0f}s")
T_EXP3 = time.time() - t0

# %% [markdown]
# ## Experiment 4: alpha sweep on one fixed partition
#
# `uncertainty_at` on the whole validation set over a grid of `alpha` from the floor `1/(N_cal + 1)`
# to 1, geometric spacing (the interesting range is near the floor).

# %%
EXP4 = {}
t0 = time.time()
for sigma in sorted({0.0, DEGRADED}):
    smx, y = POOL[sigma]
    cal, tun, ev, val = stratified_partition(y, [CAL_SIZE, TUNE_SIZE, EVAL_SIZE], rng_for(4, sigma))
    uq = UncertaintyQuantifier(N=len(cal))
    uq.calibrate(smx[cal], y[cal])
    alpha = float(run_quiet(uq.tune_alpha, smx[tun], y[tun])[0])
    floor = 1.0 / (len(cal) + 1)
    grid = np.unique(np.concatenate([np.geomspace(floor, 1.0, SWEEP_POINTS), [alpha]]))
    rows = [fields(uq.uncertainty_at(smx[val], y[val], float(a))) | {"alpha": float(a)} for a in grid]
    EXP4[str(sigma)] = {"alpha_tuned": alpha, "status": uq.search_status_.value, "floor": floor,
                            "U_E": float((smx[val].argmax(1) != y[val]).mean()), "rows": rows}
print(f"done in {time.time() - t0:.0f}s")
T_EXP4 = time.time() - t0

# %% [markdown]
# ## Experiment 5: uniform-selection check
#
# The paper's assumption: given that the set covers the true label and has `|C| = k`, the argmax
# prediction is the true label with probability `1/k` (the true label is "uniformly selected" among the
# `k` members). It is the step that turns `1 - EC*coverage` into an upper bound on the error rate.
#
# `k = 1` is trivially 1: a covered singleton contains only the true label, and the argmax lies in every
# non-empty set, so it is the true label. The check is informative for `k >= 2` only (asserted below for
# `k = 1`).
#
# **Primary estimate.** For each noise level one fixed (seeded) partition; on its validation set, which
# holds distinct images only, `P(argmax correct | covered, |C| = k)` with the count of distinct samples
# and a two-sided 95% Clopper-Pearson interval. A `k` is used only when it has at least
# `MIN_COUNT_UNIFORM` distinct samples. **Secondary estimate**: the same quantity per repetition of
# Experiment 1, summarised as mean and 5th-95th percentile over the repetitions that meet the same
# threshold on their own validation set. Counts are never pooled across repetitions: the validation sets
# of different repetitions share most of their images, so a pooled count would overstate the number of
# independent samples by about the number of repetitions.

# %%
EXP5 = {}
t0 = time.time()
for sigma in NOISES:
    smx, y = POOL[sigma]
    cal, tun, ev, val = stratified_partition(y, [CAL_SIZE, TUNE_SIZE, EVAL_SIZE], rng_for(5, sigma))
    assert len(np.unique(val)) == len(val)
    uq = UncertaintyQuantifier(N=len(cal))
    uq.calibrate(smx[cal], y[cal])
    uq.alpha = float(run_quiet(uq.tune_alpha, smx[tun], y[tun])[0])
    n_k, c_k = size_counts(uq, smx[val], y[val])
    EXP5[str(sigma)] = {"alpha": uq.alpha, "status": uq.search_status_.value, "n_val": len(val), "n_k": n_k, "c_k": c_k}
print(f"done in {time.time() - t0:.0f}s")
T_EXP5 = time.time() - t0

# %% [markdown]
# ## Aggregation

# %%
def arr(rows, key, delta=None):
    if delta is None:
        return np.array([r[key] for r in rows], dtype=float)
    return np.array([r["per_delta"][str(delta)][key] for r in rows], dtype=float)


def band(x):
    return float(np.mean(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))


def pm(x):
    m, lo, hi = band(x)
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def cp_upper(k: int, n: int) -> float:
    """Upper end of the two-sided 95% Clopper-Pearson interval for k violations in n repetitions
    (informative when the observed rate is 0, where the plain standard error is 0)."""
    from scipy.stats import beta
    return float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0


SUMMARY = {"noise": {}}
for s in NOISES:
    rows = EXP1[str(s)]
    U_E, U_reuse, U_star = arr(rows, "U_E"), arr(rows, "U_reuse"), arr(rows, "U_star")
    ent = {"U_E": band(U_E), "U_star": band(U_star), "U_reuse": band(U_reuse),
               "U": band(arr(rows, "U", 0.05)), "U_empirical": band(arr(rows, "U_empirical", 0.05)),
               "alpha": band(arr(rows, "alpha")), "cov_val": band(arr(rows, "cov_val")),
               "bias": {"U_reuse": band(U_reuse - U_E), "U": band(arr(rows, "U", 0.05) - U_E),
                         "U_empirical": band(arr(rows, "U_empirical", 0.05) - U_E)},
               "sets": {"empty": band(arr(rows, "frac_empty")), "single": band(arr(rows, "frac_single")),
                         "multi": band(arr(rows, "frac_multi"))},
               "status": {st: sum(r["status"] == st for r in rows) for st in ("converged", "floor_limited", "infeasible")},
               "reuse_warnings": int(sum(r["n_reuse_warn"] for r in rows)), "per_delta": {}}
    for d in DELTAS:
        Ud = arr(rows, "U_delta", d)
        v_star, v_E = float((Ud < U_star).mean()), float((Ud < U_E).mean())
        ent["per_delta"][str(d)] = {
            "U_delta": band(Ud), "bias": band(Ud - U_E),
            "term_coverage": band(arr(rows, "term_coverage", d)), "term_EC": band(arr(rows, "term_EC", d)),
            "viol_star": v_star, "viol_star_se": float(np.sqrt(v_star * (1 - v_star) / len(rows))),
            "viol_star_upper95": cp_upper(round(v_star * len(rows)), len(rows)),
            "viol_E": v_E, "viol_E_se": float(np.sqrt(v_E * (1 - v_E) / len(rows))),
            "viol_E_upper95": cp_upper(round(v_E * len(rows)), len(rows)),
            "nominal_se": float(np.sqrt(d * (1 - d) / len(rows))), "n": len(rows)}
    SUMMARY["noise"][str(s)] = ent

# uniform-selection check: primary = one fixed partition (distinct images); secondary = per repetition
UNIFORM = {}
for s in NOISES:
    e5 = EXP5[str(s)]
    rep_n = np.array([r["n_k"] for r in EXP1[str(s)]])
    rep_c = np.array([r["c_k"] for r in EXP1[str(s)]])
    assert e5["c_k"][1] == e5["n_k"][1] and (rep_c[:, 1] == rep_n[:, 1]).all()   # k = 1 is trivially 1
    rows = []
    for k in range(2, K + 1):
        n, c = e5["n_k"][k], e5["c_k"][k]
        if n == 0 and rep_n[:, k].sum() == 0:
            continue
        lo, hi = cp_interval(c, n) if n else (None, None)
        use = rep_n[:, k] >= MIN_COUNT_UNIFORM
        p_rep = rep_c[use, k] / rep_n[use, k]
        rows.append({"k": k, "n_distinct": n, "n_correct": c, "p": (c / n if n else None), "ci_lo": lo, "ci_hi": hi,
                         "meets_min": n >= MIN_COUNT_UNIFORM, "one_over_k": 1.0 / k,
                         "rep_n_used": int(use.sum()), "rep_median_n": float(np.median(rep_n[:, k])),
                         "rep_mean": float(p_rep.mean()) if use.any() else None,
                         "rep_p05": float(np.percentile(p_rep, 5)) if use.any() else None,
                         "rep_p95": float(np.percentile(p_rep, 95)) if use.any() else None})
    UNIFORM[str(s)] = rows
SUMMARY["uniform_selection"] = UNIFORM

uni_tab = pd.DataFrame([dict(sigma=float(s), **r) for s in NOISES for r in UNIFORM[str(s)]])
uni_tab.round(3)

viol = pd.DataFrame([dict(sigma=s, delta=d, n=e["n"], **{k: e[k] for k in ("viol_star", "viol_star_se", "viol_star_upper95", "viol_E", "viol_E_se", "viol_E_upper95", "nominal_se")})
                     for s in NOISES for d, e in SUMMARY["noise"][str(s)]["per_delta"].items()])
viol  # noqa: B018  (last expression of a notebook cell: displayed on purpose)

# %%
status_tab = pd.DataFrame({float(s): SUMMARY["noise"][str(s)]["status"] for s in NOISES}).T
status_tab.index.name = "sigma"
status_tab["TuningDataReuseWarning (legacy calls)"] = [SUMMARY["noise"][str(s)]["reuse_warnings"] for s in NOISES]
status_tab  # noqa: B018

# %%
size_rows = []
for s_ in sorted({0.0, DEGRADED}):
    for sz in EVAL_SIZES:
        for d in DELTAS:
            rs = [r["sizes"][str(sz)][str(d)] for r in EXP2[str(s_)]]
            size_rows.append({"sigma": s_, "eval_size": sz, "delta": d,
                                  "U_delta": np.mean([r["U_delta"] for r in rs]),
                                  "U_empirical": np.mean([r["U_empirical"] for r in rs]),
                                  "term_coverage": np.mean([r["term_coverage"] for r in rs]),
                                  "term_EC": np.mean([r["term_EC"] for r in rs]),
                                  "U_star": np.mean([r["U_star"] for r in EXP2[str(s_)]]),
                                  "viol_vs_U_star": np.mean([r["U_delta"] < e["U_star"] for r, e in zip(rs, EXP2[str(s_)], strict=True)])})
size_tab = pd.DataFrame(size_rows)
SUMMARY["eval_size"] = size_tab.round(6).to_dict(orient="list")
size_tab.round(4)

# %%
cc_tabs = {}
for s in sorted({0.0, DEGRADED}):
    rows = [r for rep in EXP3[str(s)] for r in rep]
    df = pd.DataFrame(rows)
    g = df.groupby("k")
    inf_ = df[df["status"] == "infeasible"].groupby("k")
    t = pd.DataFrame({
        "N_cal": g["N_cal"].mean(), "floor 1/(N_cal+1)": g["floor"].mean(),
        "converged": g["status"].apply(lambda x: int((x == "converged").sum())),
        "floor_limited": g["status"].apply(lambda x: int((x == "floor_limited").sum())),
        "infeasible": g["status"].apply(lambda x: int((x == "infeasible").sum())),
        "mean U": g["U"].mean(), "mean U_delta(0.05)": g["U_delta"].mean(), "mean U_E": g["U_E"].mean(),
        "U_E in infeasible reps": inf_["U_E"].mean(),
        "mean U_star": g["U_star"].mean(),
        "viol U_delta<U_star": g.apply(lambda x: float((x["U_delta"] < x["U_star"]).mean())),
    })
    cc_tabs[str(s)] = t
    print(f"sigma = {s}  ({len(EXP3[str(s)])} repetitions per class; mean U, U_delta and U_E are over all repetitions of the class, "
          "'U_E in infeasible reps' only over those that returned INFEASIBLE)")
    display_ = t.round(4)
    print(display_.to_string())
    print()
SUMMARY["class_conditional"] = {s: t.round(6).reset_index().to_dict(orient="list") for s, t in cc_tabs.items()}

# %% [markdown]
# ## Figures

# %%
C = {"U_E": "#222222", "U_star": "#7f7f7f", "U_reuse": "#d62728", "U": "#ff7f0e", "U_empirical": "#2ca02c",
         "d05": "#1f77b4", "d01": "#9467bd"}


def line_band(ax, xs, key_fn, label, color, ls="-"):
    m = np.array([band(key_fn(s))[0] for s in xs]); lo = np.array([band(key_fn(s))[1] for s in xs])
    hi = np.array([band(key_fn(s))[2] for s in xs])
    ax.plot(xs, m, ls, color=color, marker="o", ms=3, label=label)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, logy in zip(ax, (False, True)):
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U_E"), r"$U_E$ (validation error)", C["U_E"])
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U_star"), r"$U^*$ (validation)", C["U_star"], "--")
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U_reuse"), r"$U_{reuse}$ (legacy)", C["U_reuse"])
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U", 0.05), r"$U$", C["U"])
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U_empirical", 0.05), r"$U_{empirical}$", C["U_empirical"])
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U_delta", 0.05), r"$U_\delta$, $\delta=0.05$", C["d05"])
    line_band(a, NOISES, lambda s: arr(EXP1[str(s)], "U_delta", 0.01), r"$U_\delta$, $\delta=0.01$", C["d01"])
    a.set_xlabel(r"noise $\sigma$"); a.set_ylabel("uncertainty")
    if logy:
        a.set_yscale("log"); a.set_title("same, log scale")
ax[0].legend(fontsize=7, ncol=2)
fig.suptitle("Figure 1. What each quantity reports against the validation error (mean and 5-95% over repetitions)")
fig.tight_layout(); fig.savefig(FIG / "fig1_U_vs_noise.png"); plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, d in zip(ax, DELTAS):
    tc = np.array([band(arr(EXP1[str(s)], "term_coverage", d))[0] for s in NOISES])
    te = np.array([band(arr(EXP1[str(s)], "term_EC", d))[0] for s in NOISES])
    x = np.arange(len(NOISES))
    a.bar(x, tc, color="#1f77b4", label=r"coverage term $EC\,(cov-cov_{lower})$")
    a.bar(x, te, bottom=tc, color="#ff7f0e", label=r"EC term $cov_{lower}\,(EC-EC_{lower})$")
    a.set_xticks(x, [f"{s:g}" for s in NOISES]); a.set_xlabel(r"noise $\sigma$"); a.set_title(rf"$\delta={d}$")
ax[0].set_ylabel(r"mean $U_\delta - U_{empirical}$"); ax[0].legend(fontsize=7)
fig.suptitle("Figure 2. The two terms of the margin U_delta - U_empirical (exact decomposition), mean over repetitions")
fig.tight_layout(); fig.savefig(FIG / "fig2_margin_terms.png"); plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, sigma in zip(ax, sorted({0.0, DEGRADED})):
    rows = EXP2[str(sigma)]
    for d, col in zip(DELTAS, (C["d05"], C["d01"])):
        m = [np.mean([r["sizes"][str(s)][str(d)]["U_delta"] for r in rows]) for s in EVAL_SIZES]
        a.plot(EVAL_SIZES, m, "-o", color=col, ms=3, label=rf"$U_\delta$, $\delta={d}$")
        tcv = [np.mean([r["sizes"][str(s)][str(d)]["term_coverage"] for r in rows]) for s in EVAL_SIZES]
        tec = [np.mean([r["sizes"][str(s)][str(d)]["term_EC"] for r in rows]) for s in EVAL_SIZES]
        a.plot(EVAL_SIZES, tcv, "--", color=col, alpha=0.7, label=f"coverage term, {d}")
        a.plot(EVAL_SIZES, tec, ":", color=col, alpha=0.9, label=f"EC term, {d}")
    a.axhline(np.mean([r["U_star"] for r in rows]), color=C["U_star"], ls="-.", label=r"$U^*$")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("evaluation-set size"); a.set_title(rf"$\sigma={sigma}$")
ax[0].legend(fontsize=6, ncol=2)
fig.suptitle("Figure 3. U_delta and its margin terms against evaluation-set size (calibration and tuning fixed)")
fig.tight_layout(); fig.savefig(FIG / "fig3_eval_size.png"); plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
cm = plt.get_cmap("viridis")
kk = np.arange(2, K + 1)
for a in ax:
    a.plot(kk, 1 / kk, "k--", label=r"$1/k$ (uniform selection)")
for i, s in enumerate(NOISES):
    col, dx = cm(i / max(1, len(NOISES) - 1)), (i - (len(NOISES) - 1) / 2) * 0.05
    u = [r for r in UNIFORM[str(s)] if r["meets_min"]]
    if u:
        ax[0].errorbar([r["k"] + dx for r in u], [r["p"] for r in u],
                       yerr=[[r["p"] - r["ci_lo"] for r in u], [r["ci_hi"] - r["p"] for r in u]],
                       fmt="o", ms=4, capsize=2, color=col, label=rf"$\sigma={s:g}$")
    u = [r for r in UNIFORM[str(s)] if r["rep_n_used"] > 0]
    if u:
        ax[1].errorbar([r["k"] + dx for r in u], [r["rep_mean"] for r in u],
                       yerr=[[r["rep_mean"] - r["rep_p05"] for r in u], [r["rep_p95"] - r["rep_mean"] for r in u]],
                       fmt="o", ms=4, capsize=2, color=col, label=rf"$\sigma={s:g}$")
for a, t in zip(ax, ("primary: one fixed partition, 95% Clopper-Pearson", "secondary: per repetition, mean and 5-95%")):
    a.set_xlabel(r"prediction-set size $k=|C|$"); a.set_title(t, fontsize=9); a.set_xticks(kk)
ax[0].set_ylabel(r"$P(\mathrm{argmax\ correct}\mid\mathrm{covered},|C|=k)$"); ax[0].legend(fontsize=7)
fig.suptitle(f"Figure 4. Uniform-selection check, k >= 2 (k = 1 is trivially 1); only k with >= {MIN_COUNT_UNIFORM} distinct samples")
fig.tight_layout(); fig.savefig(FIG / "fig4_uniform_selection.png"); plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, sigma in zip(ax, sorted({0.0, DEGRADED})):
    e = EXP4[str(sigma)]
    al = np.array([r["alpha"] for r in e["rows"]])
    a.plot(al, [r["U"] for r in e["rows"]], color=C["U"], label=r"$U=1-EC(1-\alpha)$")
    a.plot(al, [r["U_empirical"] for r in e["rows"]], color=C["U_empirical"], label=r"$U_{empirical}=1-EC\cdot cov$")
    a.axhline(e["U_E"], color=C["U_E"], ls="--", label=r"$U_E$")
    a.axvline(e["alpha_tuned"], color="#d62728", ls=":", label=rf"tuned $\alpha$ ({e['status']})")
    a.set_xscale("log"); a.set_xlabel(r"$\alpha$"); a.set_title(rf"$\sigma={sigma}$")
ax[0].set_ylabel("uncertainty (validation set)"); ax[0].legend(fontsize=7)
fig.suptitle("Figure 5. Alpha sweep on the validation set (one fixed partition)")
fig.tight_layout(); fig.savefig(FIG / "fig5_alpha_sweep.png"); plt.show()

# %% [markdown]
# ## Persist the results
#
# `results_<mode>.json` holds every number plus the provenance (versions, backend, seed, parameters).
# Timestamps, run times and the repository commit are kept apart under `volatile` so two runs with the
# same seed can be compared on everything else (the commit changes whenever the notebook is committed,
# without changing any number).

# %%
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return "nan"
    return o


RESULTS = clean({
    "meta": {"python": sys.version.split()[0], "jax": jax.__version__,
              "jax_backend": jax.default_backend(), "torch": torch.__version__, "numpy": np.__version__, "seed": SEED,
              "params": PARAMS, "model_sha256": hashlib.sha256(MODEL_PTH.read_bytes()).hexdigest(),
              "model_training": {"split": "MNIST t10k (train=False), 10000 images", "seed": TRAIN_SEED,
                                  "batch": TRAIN_BATCH, "epochs": TRAIN_EPOCHS}},
    "experiment1_marginal": EXP1, "experiment2_eval_size": EXP2, "experiment3_class_conditional": EXP3,
    "experiment4_alpha_sweep": EXP4, "experiment5_uniform_selection": EXP5, "summary": SUMMARY})
RESULTS["volatile"] = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "utrace_commit": COMMIT, "src_dirty": DIRTY,
                           "seconds": {"total": time.time() - T_START, "exp1": T_EXP1, "exp2": T_EXP2, "exp3": T_EXP3, "exp4": T_EXP4, "exp5": T_EXP5}}
out = RESULTS_PATH
out.write_text(json.dumps(RESULTS, sort_keys=True, indent=1))
stable = json.dumps({k: v for k, v in RESULTS.items() if k != "volatile"}, sort_keys=True)
NONVOLATILE_SHA256 = hashlib.sha256(stable.encode()).hexdigest()
print(out, "| sha256 of the non-volatile part:", NONVOLATILE_SHA256)

# %% [markdown]
# ## Summary: facts
#
# Everything printed below is read off the results of this run (mean and [5th, 95th percentile] over
# the repetitions). A check is flagged `FAIL` when the violation rate exceeds `delta` by more than
# three binomial standard errors of the nominal rate.

# %%
print(f"QUICK={QUICK}, L={L}, pool=60000, sizes cal/tune/eval = {CAL_SIZE}/{TUNE_SIZE}/{EVAL_SIZE}\n")
fails = []
for s in NOISES:
    e = SUMMARY["noise"][str(s)]
    print(f"sigma={s:g}: status {e['status']}, mean alpha {e['alpha'][0]:.5f}; "
          f"sets at alpha on evaluation: empty {e['sets']['empty'][0]:.3f}, singleton {e['sets']['single'][0]:.3f}, multi {e['sets']['multi'][0]:.3f}")
    print(f"  U_E        {pm(arr(EXP1[str(s)], 'U_E'))}")
    print(f"  U* (val)   {pm(arr(EXP1[str(s)], 'U_star'))}")
    print(f"  U_reuse-U_E      {pm(arr(EXP1[str(s)], 'U_reuse') - arr(EXP1[str(s)], 'U_E'))}")
    print(f"  U - U_E          {pm(arr(EXP1[str(s)], 'U', 0.05) - arr(EXP1[str(s)], 'U_E'))}")
    print(f"  U_empirical-U_E  {pm(arr(EXP1[str(s)], 'U_empirical', 0.05) - arr(EXP1[str(s)], 'U_E'))}")
    for d in DELTAS:
        p = e["per_delta"][str(d)]
        dom = "coverage" if p["term_coverage"][0] > p["term_EC"][0] else "EC"
        flag = "FAIL" if p["viol_star"] > d + 3 * p["nominal_se"] else "ok"
        flagE = "FAIL" if p["viol_E"] > d + 3 * p["nominal_se"] else "ok"
        if flag == "FAIL" or flagE == "FAIL":
            fails.append((s, d, flag, flagE))
        print(f"  delta={d}: U_delta-U_E {pm(arr(EXP1[str(s)], 'U_delta', d) - arr(EXP1[str(s)], 'U_E'))}; "
              f"margin terms: coverage {p['term_coverage'][0]:.4f}, EC {p['term_EC'][0]:.4f} (dominant: {dom}); "
              f"P(U_delta<U*)={p['viol_star']:.3f}±{p['viol_star_se']:.3f} (95% upper {p['viol_star_upper95']:.3f}) [{flag}], "
              f"P(U_delta<U_E)={p['viol_E']:.3f}±{p['viol_E_se']:.3f} (95% upper {p['viol_E_upper95']:.3f}) [{flagE}]")
print("\nchecks failed:", fails if fails else "none")
print(f"\nuniform selection, P(argmax correct | covered, |C|=k) vs 1/k (k=1 is trivially 1; k>=2 shown; needs >= {MIN_COUNT_UNIFORM} distinct samples):")
for s in NOISES:
    u = UNIFORM[str(s)]
    ok = [r for r in u if r["meets_min"]]
    print(f"  sigma={s:g} primary (one partition, {EXP5[str(s)]['n_val']} distinct validation images): "
          + (", ".join(f"k={r['k']}: {r['p']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] vs {r['one_over_k']:.3f} (n={r['n_distinct']})" for r in ok) or "no k >= 2 meets the threshold"))
    small = [r for r in u if not r["meets_min"]]
    if small:
        print("      not tested (too few distinct samples): " + ", ".join(f"k={r['k']} (n={r['n_distinct']})" for r in small))
    print("      secondary (per repetition): "
          + (", ".join(f"k={r['k']}: {r['rep_mean']:.3f} [{r['rep_p05']:.3f}, {r['rep_p95']:.3f}] over {r['rep_n_used']}/{L} reps" for r in u if r["rep_n_used"]) or "no k >= 2 meets the threshold in any repetition"))

# %% [markdown]
# ## Summary: inferences
#
# *(Written after reading the full run, `QUICK=False`; see the report for the numbers. With
# `QUICK=True` the cell above has too few repetitions to support these statements.)*

# %%
# The inferences below are hand-written from one specific full run. They refer to it by the SHA-256 of the
# non-volatile part of its results file; if the results of THIS run differ (other mode, seed, parameters,
# package versions, or code), they may no longer follow from the numbers above.
INFERENCES_WRITTEN_FROM_SHA256 = "61bcdcf0a864e1de9f420c468e60aa51f3cef2d2c9058c089e914852b9f75c88"
if NONVOLATILE_SHA256 == INFERENCES_WRITTEN_FROM_SHA256:
    print("inferences match this run (non-volatile sha256", NONVOLATILE_SHA256[:16] + "...)")
else:
    print("*" * 100)
    print("NOTICE: the inferences below were written from results with sha256", INFERENCES_WRITTEN_FROM_SHA256)
    print("        this run's results have sha256                              ", NONVOLATILE_SHA256)
    print(f"        (QUICK={QUICK}, L={L}). Do not rely on the inferences below without re-reading the facts above.")
    print("*" * 100)

# %% [markdown]
# Written from the full run whose results file has non-volatile SHA-256
# `61bcdcf0a864e1de9f420c468e60aa51f3cef2d2c9058c089e914852b9f75c88`
# (`QUICK=False`, `L=50`, seed 20261015, CPU). Each statement is
# tagged with the evidence it rests on; they are inferences from one model, one dataset and one
# partition scheme, not properties of the method in general.
#
# 1. **No check failed.** `U_delta` was never below `U*` nor below `U_E` in 50 repetitions, for both
#    `delta`, at every `sigma` (95% Clopper-Pearson upper limit on the violation rate: 0.071). This is
#    consistent with the guarantee, but 50 repetitions cannot tell a bound that is right from one that is
#    very conservative: a nominal rate of 0.05 would produce about 2.5 violations, and zero were seen.
#    The combined bound is conservative by construction (a union bound with a distribution-free
#    Hoeffding term), as the stage-two Monte Carlo already showed.
# 2. **Which margin term dominates depends on the regime.** The margin `U_delta - U_empirical` at
#    `delta=0.05` is about 0.038 at `sigma=0`, almost all of it the EC term (0.0297 vs 0.0081). It is
#    ~`sqrt(ln(2/delta)/(2 n_succ)) * coverage_lower`, the Hoeffding width, because `EC_lower` is
#    worst-case for a variable in `(0, 1]`. Here 99.9% of the sets are singletons, so `1/|C|` is almost
#    constant and that worst case is far from the truth. As the error grows the EC term shrinks (it is
#    multiplied by `coverage_lower`, which falls with `sigma`) and the coverage term (a binomial width
#    that is largest near `p = 0.5`) takes over: they are about equal at `sigma=1` (0.0221 vs 0.0208) and
#    the coverage term dominates from `sigma=1.5` on (0.0197 vs 0.0151). A bound that adapts to the
#    variance of `1/|C|` would mainly help in the low-error regime; this is an inference from the sets'
#    composition, not something this notebook tested.
# 3. **The margin shrinks as 1/sqrt(n_eval).** At `sigma=0`, `delta=0.05`, `U_delta` goes from 0.129
#    (n = 300) to 0.052 (n = 5000) over a `U_empirical` of 0.028; the EC term falls from 0.0753 to 0.0188
#    (a factor 4.0 for a size ratio of 16.7, i.e. sqrt). At `sigma=1.5` the same sizes give 0.801 to 0.735
#    over 0.713 to 0.717.
# 4. **Evaluation-set size, not tuning reuse, is what limits `U_delta` here.** With 2000 samples the
#    tightest margin over `U_E` is about 0.03 (`sigma=2.5`) to 0.044 (`sigma=1`). For individual classes
#    (about 200 evaluation samples each) it is about 0.12 to 0.14 at `sigma=0`.
# 5. **The bias that stage one targeted is not visible in this setup.** The mean of `U_reuse - U_E` is
#    within +-0.0026 at every `sigma` (standard error 0.0005 to 0.0011), and `U - U_E` has a mean within
#    0.0002 of it at every level and the same standard deviation (e.g. 0.0074 for both at
#    `sigma=1.5`). Both are dominated by the error of the tuned `alpha` itself (`U ~ alpha` when
#    coverage ~ `1 - alpha`), which the separate evaluation set does not change. With 2000 tuning samples
#    the correlation between the errors in `alpha` and in `U` is not detectable by this comparison;
#    this does not say it is absent for smaller tuning sets or other models.
# 6. **Uniform selection is tested at one noise level only.** About 99% or more of the sets are
#    singletons, and `k = 1` is trivially 1, so almost nothing is left to test. With counts of distinct
#    images (one fixed partition per `sigma`, 54 000 distinct validation images), `k = 2` reaches the
#    100-sample threshold only at `sigma = 1`: `P(argmax correct | covered, |C| = 2) = 0.500`, 95%
#    Clopper-Pearson interval [0.418, 0.582], `n = 154`, against 0.5. At the other levels `k = 2` has
#    0, 87, 24, 33 and 13 distinct samples (`sigma` = 0, 0.5, 1.5, 2, 2.5) and no `k >= 3` meets the
#    threshold anywhere, so those cases are untested, not confirmed. The per-repetition estimates agree
#    where they exist (`sigma = 1`: mean 0.493, 5th-95th percentile [0.466, 0.520], 31 of 50
#    repetitions; `sigma = 0.5`: 0.516 [0.495, 0.545], 9 of 50). The interval at `sigma = 1` contains 0.5
#    and is about +-0.08 wide: it excludes large departures, not small ones. The `k = 2` figures quoted
#    in an earlier version (0.40 to 0.56) were pooled over repetitions that share images and are
#    withdrawn. The population `U*` exceeds `U_E` by 0.0003 to 0.0013 at every `sigma`, i.e. the
#    estimand is only very slightly conservative. Whether the assumption holds for models with larger
#    sets is open.
# 7. **Class-conditional runs hit the infeasible regime under noise.** At `sigma=1.5`, class 1 returns
#    `INFEASIBLE` in 20 of 20 repetitions (so `U = U_delta = 1`), class 9 in 8, class 6 in 5, class 4 in
#    2. In those repetitions the argmax error for the class is itself 1.000 (class 1), 0.990 (9), 0.989
#    (6) and 0.981 (4): the classifier is almost always wrong on them, so `U = 1` is not far from `U_E`. At `sigma=0` there are two `FLOOR_LIMITED` results in 200. The floor is 0.0044 to 0.0055
#    (`N_cal` about 180 to 225).
# 8. **The tuned alpha sits at the minimum of the validation curve in the two cases plotted** (Figure 5:
#    one partition, `sigma=0` and `1.5`). Nothing is claimed beyond that.
# 9. **Empty sets are 0.1% to 0.5% of the evaluation set** and count as uncovered.

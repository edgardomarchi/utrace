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
# # Periodic verification of a fixed model under progressive degradation
#
# A worked example of `utrace` for practitioners. A digit classifier is put into service
# (*commissioning*): we calibrate `utrace` once and choose its operating point `alpha0` once. Then the
# input degrades slowly, the way an ageing sensor would, and every so often we check the model again
# on fresh labelled data **with the same, frozen instrument**. The question is what each number tells
# us as the input gets worse, and whether an alarm can be raised before labels are available.

# %% [markdown]
# ### Notation
#
# `C` is a prediction set and `|C|` its size (written `size(C)` in the table, where `|` would break
# it); "covered" means the true label is in `C`. `EC` is the mean of `1/|C|` over covered samples,
# `coverage` the fraction covered.
#
# | symbol | formula | needs labels? | estimates or bounds |
# |---|---|---|---|
# | `alpha0` | chosen by `tune_alpha` at commissioning | yes (tuning set) | the operating point; `1 - alpha0` is the nominal coverage |
# | `delta` | 0.05 | - | error level of every bound and test |
# | `U` | `1 - EC (1 - alpha0)` | yes | point estimate of `U*` that assumes the nominal coverage |
# | `U_empirical` | `1 - EC * coverage` | yes | point estimate of `U*` with the measured coverage |
# | `U*` | `1 - e * p` (population `EC` and coverage) | - | the population quantity; under uniform selection it bounds the error probability |
# | `U_delta` | `1 - EC_lower * coverage_lower` | yes | upper confidence bound on `U*` at level `1 - delta` |
# | `U_E` | top-1 error rate on the large validation set | yes | the reference: the model's actual error probability |
# | `err_eval` | top-1 error rate on the labelled evaluation set | yes | estimate of the error probability |
# | binomial bound | Clopper-Pearson upper bound (`1 - delta`) on `err_eval` | yes | upper confidence bound on the error probability |
# | `U_inf` | `1 - mean(1{C non-empty} / size(C))` | no | population value is a lower bound on `U*` |
# | `U_inf_lower` | `1 - (mean + min(Hoeffding, emp. Bernstein))` | no | lower confidence bound on `U_inf`, hence on `U*` |

# %% [markdown]
# ## Parameters

# %%
import os

# UTRACE_NB_QUICK=1 (the default) -> a few minutes on CPU;  UTRACE_NB_QUICK=0 -> the full run.
QUICK = os.environ.get("UTRACE_NB_QUICK", "1") != "0"
MODE = "quick" if QUICK else "full"                 # every output carries it, so the two modes never overwrite each other
SEED = 20261020

CAL_SIZE, TUNE_SIZE, EVAL_SIZE = 2000, 2000, 2000   # commissioning: calibration, tuning; labelled evaluation per level
N_UNLABELED = 5000                                  # unlabelled batch per level (labels are expensive, images are not)
L = 3 if QUICK else 20                              # repetitions
DELTA = 0.05                                        # error level of every bound and test

# The degradation: contrast loss x -> m + c (x - m), m the image mean, pixels in [0, 1], then clipped.
# Chosen in a screening run of five candidates; see the report. AWGN is kept as a reference.
DEGRADATION = "contrast"
LEVELS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]   # contrast factor c; 1 = commissioning
AWGN_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]              # noise std, in normalised units (as mnist_method_evaluation.py)

PARAMS = {"QUICK": QUICK, "MODE": MODE, "SEED": SEED, "CAL_SIZE": CAL_SIZE, "TUNE_SIZE": TUNE_SIZE,
          "EVAL_SIZE": EVAL_SIZE, "N_UNLABELED": N_UNLABELED, "L": L, "DELTA": DELTA,
          "DEGRADATION": DEGRADATION, "LEVELS": LEVELS, "AWGN_LEVELS": AWGN_LEVELS}
print(PARAMS)

# %% [markdown]
# ## Model, data and degradations (helper code)
#
# Not about `utrace`: loads the classifier and MNIST, degrades images and caches the classifier's
# softmax outputs. The model is a small CNN trained on the 10 000 MNIST `t10k` images, exactly as in
# `notebooks/mnist_method_evaluation.py` (same checkpoint file, so it is trained only once). The 60 000
# MNIST training images, never seen by the model, form the *pool* everything below is drawn from.

# %% jupyter={"source_hidden": true}
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
from scipy.stats import beta, binomtest, fisher_exact, mannwhitneyu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utrace
from utrace import SearchStatusWarning, UncertaintyQuantifier
from utrace.utils.pytorch.example_models import ImageClassifierCNN, train_and_save
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper

T_START = time.time()


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "utrace").exists():
            return q
    raise RuntimeError("run this notebook from inside the utrace repository")


REPO = _find_repo_root()
BUILD = REPO / "notebooks" / "_build"
FIG = BUILD / "figures" / f"periodic_verification_{MODE}"
RESULTS_PATH = BUILD / f"periodic_verification_results_{MODE}.json"
CACHE = BUILD / "cache"
for d_ in (CACHE, FIG, BUILD / "model"):
    d_.mkdir(parents=True, exist_ok=True)
DATA_ROOT = REPO / "data"
assert (DATA_ROOT / "MNIST" / "raw").exists(), f"MNIST not found under {DATA_ROOT}"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()  # noqa: PLW1510


# Every location this notebook writes to must be git-ignored; fail loudly otherwise.
for p in (CACHE / "x.npz", FIG / "x.png", BUILD / "model" / "x.pt", RESULTS_PATH):
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(p)]).returncode == 0, f"{p} is not git-ignored"  # noqa: PLW1510
COMMIT = _git("rev-parse", "HEAD")
DIRTY = bool(_git("status", "--porcelain", "--", "src"))
print("commit", COMMIT, "(src dirty)" if DIRTY else "", "| python", sys.version.split()[0], "| jax", jax.__version__,
      jax.default_backend(), "| torch", torch.__version__, "| utrace", utrace.__version__)

# --- model: as in mnist_method_evaluation.py (same file name, so the checkpoint is shared) ---
MODEL_PTH = BUILD / "model" / "ImageClassifierCNN_t10k_seed42.pt"
TRAIN_SEED, TRAIN_BATCH, TRAIN_EPOCHS = 42, 300, 20
base_tf = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
torch.manual_seed(TRAIN_SEED)
net = ImageClassifierCNN()
if not MODEL_PTH.exists():
    train_ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=False, transform=transforms.Compose(base_tf))
    assert len(train_ds) == 10_000
    train_and_save(classifier=net, train_dataloader=DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True),
                   model_pth=MODEL_PTH, device="cpu", epochs=TRAIN_EPOCHS, seed=TRAIN_SEED)
net.load_state_dict(torch.load(MODEL_PTH, weights_only=True))
net.eval()
model = Pytorch_wrapper(net, classes=np.arange(10), device="cpu")
MODEL_SHA256 = hashlib.sha256(MODEL_PTH.read_bytes()).hexdigest()
K = 10

# --- data: the pool in raw pixel space [0, 1], i.e. what ToTensor gives, before Normalize ---
_raw = datasets.MNIST(root=str(DATA_ROOT), train=True, download=False)
X01 = _raw.data.float().div(255).unsqueeze(1)          # (60000, 1, 28, 28)
Y = _raw.targets.numpy().astype(np.int64)
POOL_N = len(Y)
_ref = datasets.MNIST(root=str(DATA_ROOT), train=True, download=False, transform=transforms.Compose(base_tf))
assert all(torch.equal(_ref[i][0], (X01[i] - 0.5) / 0.5) for i in range(16))   # same tensors as the torchvision pipeline

IDENTITY = {"contrast": 1.0, "awgn": 0.0}


def degrade(name: str, level: float) -> torch.Tensor:
    """The pool, degraded and normalised as the model expects.

    contrast: m + c (x - m) in [0, 1] (m the per-image mean), clipped to [0, 1], then Normalize.
    awgn: as in mnist_method_evaluation.py, Normalize then x + sigma * N(0, 1), no clipping; one noise
          realisation per level, from a seeded generator, so the cache is deterministic.
    """
    x = X01
    if name == "contrast" and level != IDENTITY[name]:
        m = X01.mean(dim=(1, 2, 3), keepdim=True)
        x = (m + level * (X01 - m)).clamp(0.0, 1.0)
    x = (x - 0.5) / 0.5
    if name == "awgn" and level != IDENTITY[name]:
        g = torch.Generator().manual_seed(SEED + round(level * 1000))
        x = x + torch.randn(x.shape, generator=g) * level
    return x


def pool_softmax(name: str, level: float) -> np.ndarray:
    """Softmax of the whole pool at one degradation level (float64, (60000, 10)); cached in the build directory."""
    tag = "clean" if level == IDENTITY[name] else f"{name}_{level:g}"
    f = CACHE / f"periodic_verification_pool_{tag}.npz"
    if not f.exists():
        x = degrade(name, level)
        smx = np.concatenate([np.asarray(model.predict_proba(x[i:i + 1000]).cpu()) for i in range(0, POOL_N, 1000)])
        np.savez_compressed(f, smx=smx.astype(np.float32))
    return np.load(f)["smx"].astype(np.float64)


t0 = time.time()
SMX = {(DEGRADATION, lv): pool_softmax(DEGRADATION, lv) for lv in LEVELS}
SMX |= {("awgn", lv): pool_softmax("awgn", lv) for lv in AWGN_LEVELS}
CLEAN = SMX[(DEGRADATION, IDENTITY[DEGRADATION])]
assert np.array_equal(CLEAN, SMX[("awgn", 0.0)])
T_SOFTMAX = time.time() - t0
print(f"softmax of the pool ready for {len(SMX)} levels in {T_SOFTMAX:.0f}s; clean accuracy {(CLEAN.argmax(1) == Y).mean():.4f}")

# slide-sized figures: 16:9, large fonts, colour-blind-safe palette (validated), no titles inside the images
plt.rcParams.update({"figure.figsize": (12.8, 7.2), "figure.dpi": 80, "font.size": 16, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2.5, "lines.markersize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"U": "#2a78d6", "U_delta": "#eb6834", "binom": "#1baf7a", "U_E": "#222222", "recal": "#e87ba4",
       "empty": "#eda100", "single": "#2a78d6", "multi": "#6250d6", "U_inf": "#008300", "maxsm": "#6250d6",
       "ref": "#7f7f7f", "U_emp": "#e34948",
       "struct": "#eda100", "stat": "#eb6834"}   # shading: structural part (empty sets, uniform selection), statistical margin


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.show()


def cp_upper(k: int, n: int, d: float = DELTA) -> float:
    """One-sided (1 - d) Clopper-Pearson upper bound on a binomial proportion, k successes in n."""
    return float(beta.ppf(1 - d, k + 1, n - k)) if k < n else 1.0


def cp_lower(k: int, n: int, d: float = DELTA) -> float:
    """One-sided (1 - d) Clopper-Pearson lower bound on a binomial proportion."""
    return float(beta.ppf(d, k, n - k + 1)) if k > 0 else 0.0


def band(x):
    """Mean and 5th-95th percentile over repetitions."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))


# %% [markdown]
# ## 1. What utrace measures
#
# `utrace` turns a classifier's softmax outputs into prediction *sets* (conformal prediction) and from
# them computes `U`, an upper bound on the probability that the model's top-1 answer is wrong.
# `U_delta` is the same bound stated with a confidence level: with probability at least `1 - delta`
# the true value is below it, **provided** the evaluation data are independent, representative of
# use, and satisfy the paper's uniform-selection assumption (which the code cannot check).
# Method: Marchi and Liebl, *U-TraCE: a conformal prediction approach to uncertainty quantification in
# black-box models*, Mach. Learn.: Sci. Technol. (2026),
# [doi:10.1088/2632-2153/ae35ce](https://doi.org/10.1088/2632-2153/ae35ce). Usage and caveats: the
# [README](https://github.com/edgardomarchi/utrace#readme).
#
# `utrace` works on softmax outputs only, so any model and framework can be used: here a PyTorch
# CNN, whose outputs were computed in the helper cell above.

# %% [markdown]
# ## 2. Commissioning, in four calls
#
# On clean data. Each repetition draws its own disjoint subsets of the pool; this section shows
# repetition 0. Calibration builds the reference (2000 labelled images); `tune_alpha` picks the
# operating point `alpha0` on another 2000; `uncertainty_at` measures `U` on a third, independent
# set of 2000.

# %%
def commissioning_split(rep: int):
    """Disjoint calibration, tuning, evaluation and unlabelled-reference blocks, and what is left of the pool."""
    perm = np.random.default_rng([SEED, rep]).permutation(POOL_N)
    cuts = np.cumsum([CAL_SIZE, TUNE_SIZE, EVAL_SIZE, N_UNLABELED])
    cal, tun, ev, unl, rest = np.split(perm, cuts)
    return cal, tun, ev, unl, rest


cal, tun, ev0, unl0, rest = commissioning_split(0)

uq = UncertaintyQuantifier(N=CAL_SIZE)                       # 1. create; N = how many calibration scores to keep
uq.calibrate(CLEAN[cal], Y[cal])                             # 2. calibrate on clean labelled data
alpha0 = uq.tune_alpha(CLEAN[tun], Y[tun])                   # 3. choose the operating point on other data
print(f"alpha0 = {alpha0:.5f}, search status: {uq.search_status_}")
comm = uq.uncertainty_at(CLEAN[ev0], Y[ev0], alpha0, delta=DELTA)   # 4. measure U on independent data

EXPLAIN = {
    "U": "point estimate of the error bound, using the nominal coverage 1 - alpha0 (no confidence level)",
    "EC": "mean of 1/|C| over the covered evaluation samples (1 when every covered set is a singleton)",
    "n_covered": "evaluation samples whose true label is inside the prediction set",
    "n_valid": "evaluation samples counted (all of them here: marginal calibration)",
    "n_succ": "covered samples with a non-empty set; the denominator of EC",
    "coverage": "measured coverage n_covered / n_valid, to compare with 1 - alpha0",
    "U_empirical": "1 - EC * coverage: U with the measured coverage instead of the nominal one",
    "delta": "the error level passed in",
    "coverage_lower": "Clopper-Pearson lower bound on the coverage, at level delta/2",
    "EC_lower": "lower bound on EC at level delta/2 (smaller of a Hoeffding and an empirical Bernstein margin)",
    "U_delta": "1 - EC_lower * coverage_lower: upper bound on the error probability at confidence 1 - delta",
}
for k, v in comm._asdict().items():
    print(f"{k:>15} = {float(v) if v is not None else None!s:<22} {EXPLAIN[k]}")
print(f"\n1 - alpha0 = {1 - alpha0:.4f}  vs  measured coverage {comm.coverage:.4f}")

# %% [markdown]
# `predict` returns the top-1 class and the prediction set, at the instance's current `alpha`
# (`predict` takes no alpha argument, so we assign `alpha0` first). A set with one label means
# "confident"; an **empty** set means "no class is plausible enough" (and counts as a miss); a set with
# several labels means "ambiguous between these".

# %%
uq.alpha = alpha0
pred0, sets0 = uq.predict(CLEAN[ev0])
size0 = sets0.sum(1)
covered0 = sets0[np.arange(len(ev0)), Y[ev0]]
assert np.isclose(covered0.mean(), comm.coverage)           # predict and uncertainty_at agree on the sets
print(f"on the commissioning evaluation set: {np.sum(size0 == 0)} empty, {np.sum(size0 == 1)} singleton, "
      f"{np.sum(size0 > 1)} multi-label sets")
_, sets_pool = uq.predict(CLEAN)
n_multi_pool = int((sets_pool.sum(1) > 1).sum())
print(f"multi-label sets at alpha0 in the whole clean pool of {POOL_N}: {n_multi_pool}")

# pick examples: correct singletons, then wrong singletons, empty sets and multi-label sets when they exist
ok_single = np.flatnonzero((size0 == 1) & covered0)
bad_single = np.flatnonzero((size0 == 1) & ~covered0)
empty = np.flatnonzero(size0 == 0)
multi = np.flatnonzero(size0 > 1)
pick = [*multi[:2], *empty[:2], *bad_single[:1]]
pick = [*ok_single[:8 - len(pick)], *pick]

fig, axs = plt.subplots(2, 4, figsize=(12.8, 7.2), gridspec_kw={"hspace": 0.45})
for a, i in zip(axs.flat, pick, strict=True):
    a.imshow(X01[ev0[i], 0], cmap="gray_r", vmin=0, vmax=1)
    s = "{" + ", ".join(str(c) for c in np.flatnonzero(sets0[i])) + "}"
    kind = "empty" if size0[i] == 0 else ("covered" if covered0[i] else "missed")
    a.set_title(f"set {s}   true {Y[ev0[i]]}\n({kind})", fontsize=17)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color(COL["ref"])
save(fig, "commissioning_sets")

# %% [markdown]
# **Figure `commissioning_sets`.** Clean evaluation images with their prediction set at `alpha0` and
# their true label. "Missed" means the true label is outside the set. Multi-label sets are shown only
# if the counts printed above are non-zero: with this model they are rare (none at all at this
# repetition's `alpha0`), so almost every set is a singleton or empty.

# %% [markdown]
# ## 3. Periodic verification with a frozen instrument
#
# At each degradation level we draw **fresh** images from the pool that were not used at
# commissioning: 2000 labelled images for the check, an unlabelled batch of `N_UNLABELED` for
# section 6, and the rest (about 42 000) as a large *validation* set that stands in for the truth:
# `U_E`, the model's actual top-1 error rate there.
#
# Nothing is recalibrated: the same quantifier and the same `alpha0` are used at every level. The
# commissioning calibration is the reference against which drift is measured; recalibrating on
# degraded data would move the reference with the drift and hide it (section 7 shows that case).
#
# Alongside `utrace` we compute the classic labelled check: a one-sided Clopper-Pearson upper bound on
# the top-1 error rate from the same 2000 labelled images. It sees errors only.

# %%
def u_inf_parts(sets: np.ndarray):
    """Label-free summary of a batch of prediction sets: z_i = 1{C_i non-empty} / |C_i|."""
    sz = sets.sum(1)
    z = np.where(sz > 0, 1.0 / np.maximum(sz, 1), 0.0)
    return z, sz


def hoeffding_margin(n: int, d: float) -> float:
    """One-sided Hoeffding margin at level d for the mean of n variables in [0, 1]: sqrt(ln(1/d) / (2n))."""
    return float(np.sqrt(np.log(1 / d) / (2 * n)))


def eb_margin(z: np.ndarray, d: float) -> float:
    """One-sided empirical Bernstein margin at level d, Maurer and Pontil (2009), arXiv:0907.3740, Theorem 4:
    sqrt(2 V ln(2/d) / n) + 7 ln(2/d) / (3 (n - 1)), V the unbiased sample variance, z in [0, 1]."""
    n = len(z)
    v = float(np.var(z, ddof=1))
    return float(np.sqrt(2 * v * np.log(2 / d) / n) + 7 * np.log(2 / d) / (3 * (n - 1)))


def u_inf_lower(z: np.ndarray, delta: float = DELTA) -> float:
    """Lower confidence bound (level 1 - delta) on U_inf = 1 - E[z]: the smaller of the two margins, each at delta/2."""
    margin = min(hoeffding_margin(len(z), delta / 2), eb_margin(z, delta / 2))
    return 1.0 - (float(z.mean()) + margin)


def label_free(uq_, smx):
    """Everything section 6 computes on an unlabelled batch, at the instance's current alpha."""
    _, S = uq_.predict(smx)
    z, sz = u_inf_parts(S)
    maxsm = smx.max(1)
    return {"n": len(z), "n_empty": int((sz == 0).sum()), "empty": float((sz == 0).mean()),
            "multi": float((sz > 1).mean()), "U_inf": float(1 - z.mean()), "U_inf_lower": u_inf_lower(z),
            "maxsm_mean": float(maxsm.mean())}, maxsm


def commission(rep: int) -> dict:
    """The four calls of section 2, for repetition `rep`, plus the commissioning references of section 6."""
    cal, tun, ev, unl, rest = commissioning_split(rep)
    q = UncertaintyQuantifier(N=CAL_SIZE)
    q.calibrate(CLEAN[cal], Y[cal])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a0 = float(q.tune_alpha(CLEAN[tun], Y[tun]))
    r = q.uncertainty_at(CLEAN[ev], Y[ev], a0, delta=DELTA)
    q.alpha = a0
    lf, maxsm = label_free(q, CLEAN[unl])
    k_err = int((CLEAN[ev].argmax(1) != Y[ev]).sum())
    return {"rep": rep, "uq": q, "alpha0": a0, "status": q.search_status_.value,
            "n_status_warnings": sum(issubclass(x.category, SearchStatusWarning) for x in w),
            "eval": {k: (None if v is None else float(v)) for k, v in r._asdict().items()},
            "err_eval": k_err / len(ev), "binom_upper": cp_upper(k_err, len(ev)),
            "label_free": lf, "_maxsm": maxsm, "_split": (cal, tun, ev, unl, rest)}


def level_split(rest: np.ndarray, rep: int, deg_id: int, li: int):
    """Fresh labelled evaluation set, unlabelled batch and validation remainder, for one level."""
    sub = np.random.default_rng([SEED, rep, deg_id, li]).permutation(rest)
    return sub[:EVAL_SIZE], sub[EVAL_SIZE:EVAL_SIZE + N_UNLABELED], sub[EVAL_SIZE + N_UNLABELED:]


def decompose(r, S: np.ndarray, smx: np.ndarray, y: np.ndarray) -> dict:
    """Exact split of U_delta - err_eval on one evaluation set (section 3b).

    Structural part, U_empirical - err_eval = P(empty and argmax correct) + sum_{k>=2} P(covered, |C|=k) (p_k - 1/k),
    with p_k = P(argmax correct | covered, |C| = k). It is exact for the LAC score: the argmax lies in every
    non-empty set, so a correct argmax is either covered or in an empty set, and a covered singleton is correct.
    Statistical margin, U_delta - U_empirical = EC (coverage - coverage_lower) + coverage_lower (EC - EC_lower).
    """
    n = len(y)
    sz = S.sum(1)
    covered = S[np.arange(n), y]
    correct = smx.argmax(1) == y
    assert int(covered.sum()) == int(r.n_covered)
    err = float((~correct).sum() / n)
    empty_correct = float(((sz == 0) & correct).sum() / n)
    multi = float(sum((covered & (sz == k) & correct).sum() - (covered & (sz == k)).sum() / k for k in range(2, K + 1)) / n)
    cov_term = float(r.EC * (r.coverage - r.coverage_lower))
    ec_term = float(r.coverage_lower * (r.EC - r.EC_lower))
    assert abs((r.U_empirical - err) - (empty_correct + multi)) < 1e-12, (float(r.U_empirical - err), empty_correct + multi)
    assert abs((r.U_delta - r.U_empirical) - (cov_term + ec_term)) < 1e-12, (float(r.U_delta - r.U_empirical), cov_term + ec_term)
    return {"dec_empty_correct": empty_correct, "dec_multi": multi, "dec_coverage": cov_term, "dec_EC": ec_term}


def verify(c: dict, name: str, deg_id: int, li: int, level: float) -> dict:
    """One periodic check with the frozen commissioning instrument (no recalibration)."""
    q, a0 = c["uq"], c["alpha0"]
    smx = SMX[(name, level)]
    ev, unl, val = level_split(c["_split"][4], c["rep"], deg_id, li)
    r = q.uncertainty_at(smx[ev], Y[ev], a0, delta=DELTA)
    _, S = q.predict(smx[ev])
    sz = S.sum(1)
    k_err = int((smx[ev].argmax(1) != Y[ev]).sum())
    dec = decompose(r, S, smx[ev], Y[ev])
    lf, maxsm = label_free(q, smx[unl])
    lf0 = c["label_free"]
    rec = {"level": level, **{k: (None if v is None else float(v)) for k, v in r._asdict().items()},
           "U_E": float((smx[val].argmax(1) != Y[val]).mean()), "n_val": len(val),
           "err_eval": k_err / len(ev), "binom_upper": cp_upper(k_err, len(ev)), "binom_lower": cp_lower(k_err, len(ev)),
           "eval_empty": float((sz == 0).mean()), "eval_single": float((sz == 1).mean()), "eval_multi": float((sz > 1).mean()),
           "p_drift": float(binomtest(int(r.n_covered), int(r.n_valid), 1 - a0, alternative="less").pvalue),
           "p_L1b": float(fisher_exact([[int(r.n_covered), int(r.n_valid - r.n_covered)],
                                        [int(c["eval"]["n_covered"]), int(c["eval"]["n_valid"] - c["eval"]["n_covered"])]],
                                       alternative="less").pvalue),
           **dec, "label_free": lf,
           "p_A1": float(fisher_exact([[lf["n_empty"], lf["n"] - lf["n_empty"]],
                                       [lf0["n_empty"], lf0["n"] - lf0["n_empty"]]], alternative="greater").pvalue),
           "p_A3": float(mannwhitneyu(maxsm, c["_maxsm"], alternative="less").pvalue)}
    rec["alarm"] = {"A1": rec["p_A1"] < 0.05, "A2": lf["U_inf_lower"] > c["eval"]["U_delta"], "A3": rec["p_A3"] < 0.05,
                    "L1": rec["p_drift"] < 0.05, "L1b": rec["p_L1b"] < 0.05, "L2": rec["binom_upper"] > c["binom_upper"],
                    "L2_formal": rec["binom_lower"] > c["binom_upper"]}
    return rec


t0 = time.time()
COMM = [commission(rep) for rep in range(L)]
assert COMM[0]["alpha0"] == float(alpha0) and COMM[0]["eval"]["U_delta"] == float(comm.U_delta)   # same as the four calls above
FROZEN = {DEGRADATION: [[verify(c, DEGRADATION, 0, li, lv) for li, lv in enumerate(LEVELS)] for c in COMM],
          # AWGN (section 8) is computed here too, so that section 3b can compare the two degradations
          "awgn": [[verify(c, "awgn", 1, li, lv) for li, lv in enumerate(AWGN_LEVELS)] for c in COMM]}
T_FROZEN = time.time() - t0
print(f"{L} repetitions x {len(LEVELS)} levels in {T_FROZEN:.0f}s; alpha0 over repetitions:",
      np.round([c["alpha0"] for c in COMM], 5), "| status:", sorted({c["status"] for c in COMM}))


def col(recs, key, sub=None):
    """(L, n_levels) array of one field across repetitions and levels."""
    return np.array([[(r[sub][key] if sub else r[key]) for r in row] for row in recs], dtype=float)


def table(recs, levels, cols):
    rows = []
    for li, lv in enumerate(levels):
        row = {"level": lv}
        for label, (key, sub) in cols.items():
            row[label] = col(recs, key, sub)[:, li].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("level")


F = FROZEN[DEGRADATION]
table(F, LEVELS, {"U_E": ("U_E", None), "err_eval": ("err_eval", None), "binomial bound": ("binom_upper", None),
                  "U": ("U", None), "U_empirical": ("U_empirical", None), "U_delta": ("U_delta", None)}).round(4)

# %% [markdown]
# Means over the `L` repetitions. Next, the figure: the slide version of this table, with bands.
# The recalibrated `U_delta` (thin dashed) is computed in section 7.

# %%
def recalibrated(c: dict, name: str, deg_id: int, li: int, level: float) -> dict:
    """Section 7: calibrate and tune on DEGRADED data at this level, then evaluate on the same fresh set."""
    smx = SMX[(name, level)]
    cal, tun = c["_split"][0], c["_split"][1]
    ev, _, _ = level_split(c["_split"][4], c["rep"], deg_id, li)
    q = UncertaintyQuantifier(N=CAL_SIZE)
    q.calibrate(smx[cal], Y[cal])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        a = float(q.tune_alpha(smx[tun], Y[tun]))
    r = q.uncertainty_at(smx[ev], Y[ev], a, delta=DELTA)
    return {"level": level, "alpha": a, "status": q.search_status_.value, "U": float(r.U), "U_delta": float(r.U_delta),
            "coverage": float(r.coverage)}


t0 = time.time()
RECAL = {DEGRADATION: [[recalibrated(c, DEGRADATION, 0, li, lv) for li, lv in enumerate(LEVELS)] for c in COMM],
         "awgn": [[recalibrated(c, "awgn", 1, li, lv) for li, lv in enumerate(AWGN_LEVELS)] for c in COMM]}
T_RECAL = time.time() - t0


def line_band(ax, x, Y_, label, color, ls="-", marker="o", lw=2.5, fill=True):
    m, lo, hi = (np.array(v) for v in zip(*[band(Y_[:, j]) for j in range(Y_.shape[1])], strict=True))
    ax.plot(x, m, ls, color=color, marker=marker, lw=lw, label=label)
    if fill:
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


def thin_band(ax, x, Y_, color):
    """5th and 95th percentiles over repetitions as thin outlines (so shading underneath stays readable)."""
    for q_ in (5, 95):
        ax.plot(x, np.percentile(Y_, q_, axis=0), "-", color=color, lw=0.8, alpha=0.5)


def frozen_figure(recs, recal, levels, xlabel, invert, name):
    """U_E, U_empirical, U_delta and the binomial bound; shaded: structural part and statistical margin."""
    ue, uemp, ud, bu = (col(recs, k) for k in ("U_E", "U_empirical", "U_delta", "binom_upper"))
    fig, ax = plt.subplots()
    ax.fill_between(levels, ue.mean(0), uemp.mean(0), color=COL["struct"], alpha=0.6, lw=0,
                    label="structural (conservative construction)")
    ax.fill_between(levels, uemp.mean(0), ud.mean(0), color=COL["stat"], alpha=0.45, lw=0,
                    label=r"statistical margin ($\delta$ = 0.05)")
    for a, label, c_, ls, mk in ((ue, r"$U_E$: actual error rate (validation)", COL["U_E"], "-", "s"),
                                 (uemp, r"$U_{empirical}$ (point estimate)", COL["U_emp"], "-", "o"),
                                 (ud, r"$U_\delta$, frozen instrument", COL["U_delta"], "-", "D"),
                                 (bu, "binomial bound on the error", COL["binom"], "--", "^")):
        ax.plot(levels, a.mean(0), ls, color=c_, marker=mk, label=label)
        thin_band(ax, levels, a, c_)
    ax.plot(levels, col(recal, "U_delta").mean(0), ":", color=COL["recal"], lw=1.8,
            label=r"$U_\delta$, recalibrated at each level")
    ax.set_xlabel(xlabel); ax.set_ylabel("error probability or its bound")
    ax.set_ylim(0, None)
    if invert:
        ax.invert_xaxis()
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, name)


CONTRAST_X = "contrast factor $c$  (1 = commissioning; degradation increases to the right)"
AWGN_X = r"noise standard deviation $\sigma$ (normalised pixel units)"
frozen_figure(F, RECAL[DEGRADATION], LEVELS, CONTRAST_X, True, "frozen_verification")

# %% [markdown]
# **Figure `frozen_verification`.** Against the contrast factor, with the instrument frozen at
# commissioning: the actual error `U_E` on the large validation set, the point estimate `U_empirical`,
# the bound `U_delta`, and the binomial Clopper-Pearson bound on the top-1 error from the same labelled
# images. Lines are means over repetitions, thin outlines the 5th-95th percentiles. Yellow: the
# structural part, between `U_E` and `U_empirical`. Orange: the statistical margin, between
# `U_empirical` and `U_delta`. The thin dotted line is `U_delta` when the quantifier is recalibrated at
# each level (section 7).

# %% [markdown]
# ## 3b. Where `U_delta`'s excess over the error comes from
#
# `U_delta` is above the actual error for two reasons, and they behave differently:
#
# ```
# U_delta = err_eval + (U_empirical - err_eval) + (U_delta - U_empirical)
#                       structural part          statistical margin
# ```
#
# - The **structural part** comes from how the method is built. It does not shrink with more data.
#   On the same evaluation set it splits exactly into two pieces:
#   - correct answers whose set is **empty**: the model was right, but its confidence fell below the
#     frozen threshold, and an empty set counts as a miss;
#   - for sets with `k >= 2` labels, the gap between how often the top-1 answer is right and the `1/k`
#     that the uniform-selection assumption credits.
# - The **statistical margin** is the price of stating a confidence level, `1 - delta`. It shrinks as
#   `1/sqrt(n)` and splits exactly into a coverage term and an `EC` term.
#
# Both identities are asserted, to `1e-12`, for every repetition and level, in `decompose` above.

# %%
LAYERS = {"dec_empty_correct": ("empty set, answer correct", COL["struct"], None),
          "dec_multi": ("multi-label (uniform selection)", COL["multi"], None),
          "dec_coverage": ("coverage term", COL["stat"], None),
          "dec_EC": ("EC term", COL["stat"], "//")}

from matplotlib.colors import to_rgba

# a layer goes in the legend unless it is exactly zero in every repetition and level of both degradations
NONZERO = {k: any(np.any(col(FROZEN[n], k) != 0) for n in (DEGRADATION, "awgn")) for k in LAYERS}
fig, axs = plt.subplots(1, 2, sharey=True)
for ax, (name, levels, xlabel, invert) in zip(axs, ((DEGRADATION, LEVELS, "contrast factor $c$", True),
                                                    ("awgn", AWGN_LEVELS, r"noise std $\sigma$", False)), strict=True):
    bottom = np.zeros(len(levels))
    for k, (label, c_, hatch) in LAYERS.items():
        top = bottom + col(FROZEN[name], k).mean(0)
        ax.fill_between(levels, bottom, top, facecolor=to_rgba(c_, 0.3 if hatch else 0.85),
                        edgecolor=c_ if hatch else "none", hatch=hatch, lw=0,
                        label=label if NONZERO[k] and ax is axs[0] else None)
        bottom = top
    ax.plot(levels, bottom, "-", color=COL["U_E"], lw=1.5, label=r"total $U_\delta - err_{eval}$" if ax is axs[0] else None)
    ax.set_xlabel(xlabel)
    if invert:
        ax.invert_xaxis()
axs[0].set_ylabel(r"$U_\delta - err_{eval}$ (mean over repetitions)")
axs[0].legend(loc="upper left", frameon=False)
fig.tight_layout()
save(fig, "decomposition")
print("layers that are zero in every repetition and level (left out of the legend):",
      [LAYERS[k][0] for k, v in NONZERO.items() if not v] or "none")


def decomposition_table(recs, levels):
    out = {"level": levels}
    for k, (label, _, _) in LAYERS.items():
        out[label] = col(recs, k).mean(0)
    out["U_delta - err_eval"] = (col(recs, "U_delta") - col(recs, "err_eval")).mean(0)
    out["U_empirical - err_eval"] = (col(recs, "U_empirical") - col(recs, "err_eval")).mean(0)
    out["U_empirical - U_E"] = (col(recs, "U_empirical") - col(recs, "U_E")).mean(0)
    out["coverage"] = col(recs, "coverage").mean(0)
    return pd.DataFrame(out).set_index("level")


DECOMP = {name: decomposition_table(FROZEN[name], lv) for name, lv in ((DEGRADATION, LEVELS), ("awgn", AWGN_LEVELS))}
print(DEGRADATION); print(DECOMP[DEGRADATION].round(4).to_string())
print("\nawgn"); print(DECOMP["awgn"].round(4).to_string())

# %% [markdown]
# **Figure `decomposition`.** Left: contrast loss; right: AWGN; same vertical scale. The mean of
# `U_delta - err_eval` is stacked into its four exact parts: correct answers in empty sets and the
# multi-label term (the structural part), then the coverage and `EC` terms of the statistical margin.
# A layer that is zero in every repetition and level is left out of the legend (the cell above prints
# which, if any). A layer can be slightly negative where `P(top-1 correct | covered, |C| = k)` is below
# `1/k`; the black line is the total. In the table, `U_empirical - U_E` differs from
# `U_empirical - err_eval` only by the sampling error of `err_eval` against the validation error.

# %% [markdown]
# ## 4. Why `U` and `U_delta` separate
#
# `U = 1 - EC (1 - alpha0)` takes the coverage for granted: it plugs in the **nominal** value
# `1 - alpha0` that the calibration promised. When the input degrades, the promise no longer holds,
# but `U` cannot know. `U_delta` instead **measures** the coverage on the evaluation set (and takes a
# conservative bound on it). So when coverage falls, `U_delta` follows and `U` does not.
#
# With this model almost every non-empty set holds one label (multi-label sets are at most a few per
# thousand, and absent in some repetitions), so `EC` is 1 or within a few thousandths of it and
# `U` stays at about `alpha0` at every level: flat.

# %%
fig, ax = plt.subplots()
for a, label, c_, mk in ((col(F, "U_E"), r"$U_E$: actual error rate (validation)", COL["U_E"], "s"),
                         (col(F, "U_empirical"), r"$U_{empirical}$ (measured coverage)", COL["U_emp"], "o"),
                         (col(F, "U"), r"$U$ (nominal coverage $1-\alpha_0$)", COL["U"], "D")):
    ax.plot(LEVELS, a.mean(0), "-", color=c_, marker=mk, label=label)
    thin_band(ax, LEVELS, a, c_)
ax.set_xlabel(CONTRAST_X); ax.set_ylabel("error probability or its estimate")
ax.set_ylim(0, None); ax.invert_xaxis(); ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
save(fig, "nominal_U_blind")

# %% [markdown]
# **Figure `nominal_U_blind`.** What a user would see without re-measuring coverage: `U`, computed with
# the nominal coverage `1 - alpha0`, stays flat while the actual error `U_E` rises more than twenty-fold.
# `U_empirical`, the same formula with the coverage measured on the evaluation set, follows the error.
# Mean lines, with the 5th-95th percentiles as thin outlines.

# %%
cov, cov_lo = col(F, "coverage"), col(F, "coverage_lower")
nominal = np.array([1 - c["alpha0"] for c in COMM])
reject = col(F, "p_drift") < 0.05

fig, ax = plt.subplots()
line_band(ax, LEVELS, cov, "measured coverage", COL["U_delta"], marker="D")
line_band(ax, LEVELS, cov_lo, "its Clopper-Pearson lower bound (one-sided, level $\\delta/2$)", COL["U_delta"], ls="--", marker="")
ax.axhline(nominal.mean(), color=COL["U"], lw=2.5, label=r"nominal coverage $1-\alpha_0$ (mean over repetitions)")
rate = reject.mean(0)
for x, y, r_ in zip(LEVELS, cov.mean(0), rate, strict=True):
    if r_ >= 0.5:
        ax.plot(x, y, "o", ms=20, mfc="none", mec=COL["U_E"], mew=2)
ax.plot([], [], "o", ms=20, mfc="none", mec=COL["U_E"], mew=2, label="drift test rejects at 5 % (in at least half of the repetitions)")
ax.set_xlabel(CONTRAST_X); ax.set_ylabel("coverage")
ax.invert_xaxis(); ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
save(fig, "coverage_drift")

# %% [markdown]
# **Figure `coverage_drift`.** Measured coverage on the fresh labelled set and its Clopper-Pearson lower
# bound (the one `U_delta` uses), against the nominal `1 - alpha0` that `U` assumes. Circled points: the
# drift test of section 5 rejects at 5 % in at least half of the repetitions.

# %%
comp = np.stack([col(F, k).mean(0) for k in ("eval_empty", "eval_single", "eval_multi")])
fig, ax = plt.subplots()
xs = np.arange(len(LEVELS))
bottom = np.zeros(len(LEVELS))
for frac, label, c_ in zip(comp, ("empty", "singleton", "multi-label"), (COL["empty"], COL["single"], COL["multi"]), strict=True):
    ax.bar(xs, frac, bottom=bottom, color=c_, label=label, edgecolor="white", linewidth=2, width=0.8)
    bottom += frac
ax.set_xticks(xs, [f"{lv:g}" for lv in LEVELS])
ax.set_xlabel(CONTRAST_X); ax.set_ylabel("fraction of prediction sets at $\\alpha_0$")
ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
fig.tight_layout()
save(fig, "set_composition")
pd.DataFrame(comp.T, index=pd.Index(LEVELS, name="contrast c"), columns=["empty", "singleton", "multi-label"]).round(4)

# %% [markdown]
# **Figure `set_composition`.** Mean fractions of empty, singleton and multi-label sets at `alpha0` on
# the labelled evaluation set, by contrast level. As contrast drops the model becomes less sure: the
# true class's softmax falls below the calibrated threshold and sets go **empty** rather than growing.
# An empty set always counts as uncovered, so coverage falls; whether the top-1 answer was right or not.

# %% [markdown]
# ## 5. A formal drift test
#
# Calibration promised coverage of at least `1 - alpha0`. A one-sided binomial test of
# `H0: coverage >= 1 - alpha0` on the labelled evaluation set (`scipy.stats.binomtest`,
# `alternative="less"`) asks whether the data contradict that promise. A small p-value says the
# instrument no longer behaves as calibrated. Call it **L1**.
#
# L1 has a catch: the conformal guarantee `coverage >= 1 - alpha0` holds on average over calibration
# sets, not for the one calibration we froze, whose own coverage can sit slightly below nominal. So
# L1 can reject on clean data more often than 5 %. **L1b** instead compares the current coverage with
# the coverage measured at commissioning on the commissioning evaluation set (one-sided Fisher exact
# test, `alternative="less"`). For a frozen instrument this is the right reference: with no drift,
# both sets are independent draws scored by the same calibration, so H0 is exactly "nothing changed
# since commissioning" whatever that calibration's own coverage happens to be.

# %%
pdrift, pl1b = col(F, "p_drift"), col(F, "p_L1b")
pd.DataFrame({"L1 median p-value": np.median(pdrift, 0), "L1 rejected at 5 %": reject.mean(0),
              "L1b median p-value": np.median(pl1b, 0), "L1b rejected at 5 %": (pl1b < 0.05).mean(0),
              "mean coverage": cov.mean(0), "mean 1 - alpha0": nominal.mean()},
             index=pd.Index(LEVELS, name="contrast c")).round(4)

# %% [markdown]
# ## 6. Early warning without labels
#
# Labels are expensive, so can we raise an alarm *before* labelled data arrive? At `alpha0`, on an
# unlabelled batch of `N_UNLABELED` degraded images per level (disjoint from the labelled set), and
# **without the labels**, we compute:
#
# - the fractions of empty and multi-label sets;
# - `U_inf = 1 - mean(1{C non-empty} / |C|)`. Since a set can only cover the true label if it is
#   non-empty, `U_inf` is a lower bound on the population `U*` that `U_delta` bounds from above, with no
#   assumption at all;
# - `U_inf_lower`, a lower confidence bound on `U_inf` at level `1 - delta`: `1 - (mean + margin)`,
#   with the smaller of a Hoeffding margin and an empirical Bernstein margin (Maurer and Pontil, 2009,
#   arXiv:0907.3740, Theorem 4), each at `delta/2`. This is the same construction `utrace` uses for its
#   bound on `EC`;
# - the mean max-softmax, the standard label-free baseline (Hendrycks and Gimpel, 2017,
#   arXiv:1610.02136), for a fair comparison.
#
# First, the margins against hand-computed values, and against the package's own `EC_lower`.

# %%
# Hand values (computed with plain `math`, independently of the functions above), level d = 0.025:
#  (a) n = 100, half ones, half zeros: V = 0.25 * 100/99; Hoeffding sqrt(ln 40 / 200); EB sqrt(2 V ln 80 / 100) + 7 ln 80 / 297
#  (b) n = 5000, 4950 ones and 50 zeros: V = 0.99 * 0.01 * 5000/4999
za = np.r_[np.ones(50), np.zeros(50)]
zb = np.r_[np.ones(4950), np.zeros(50)]
assert np.isclose(np.var(za, ddof=1), 0.25252525252525254, rtol=0, atol=1e-15)
assert np.isclose(hoeffding_margin(100, 0.025), 0.13581015157406195, rtol=0, atol=1e-14)
assert np.isclose(eb_margin(za, 0.025), 0.2520465087982955, rtol=0, atol=1e-14)
assert np.isclose(np.var(zb, ddof=1), 0.009901980396079216, rtol=0, atol=1e-15)
assert np.isclose(hoeffding_margin(5000, 0.025), 0.019206455826398416, rtol=0, atol=1e-14)
assert np.isclose(eb_margin(zb, 0.025), 0.006211443735008967, rtol=0, atol=1e-14)
assert np.isclose(u_inf_lower(zb, 0.05), 1 - (0.99 + 0.006211443735008967), rtol=0, atol=1e-14)   # (b): EB is the smaller

# Cross-check with the package: its EC_lower is EC - min(Hoeffding, EB) at delta/4 each, over the covered samples.
z_cov = 1.0 / size0[covered0 & (size0 > 0)]
m_pkg = comm.EC - comm.EC_lower
m_nb = min(hoeffding_margin(len(z_cov), DELTA / 4), eb_margin(z_cov, DELTA / 4))
print(f"package EC - EC_lower = {m_pkg:.10f}   notebook margin = {m_nb:.10f}   (n_succ = {comm.n_succ} = {len(z_cov)})")
assert len(z_cov) == comm.n_succ and np.isclose(m_pkg, m_nb, rtol=1e-6, atol=1e-9)
print("margin checks passed")

# %% [markdown]
# **Alarm rules**, each fixed from commissioning data only, before any degraded level is looked at:
#
# - **A1 (empty fraction):** one-sided Fisher exact test, current batch against the unlabelled clean
#   batch drawn at commissioning, at 5 %.
# - **A2 (`U_inf_lower`):** alarm when `U_inf_lower` exceeds the commissioning `U_delta`. With
#   probability at least `1 - delta`, `U*` now is above `U_inf_lower`; with probability at least
#   `1 - delta`, `U*` at commissioning was below the commissioning `U_delta`. So an alarm states, at
#   confidence `1 - 2 delta`, that the instrument's `U*` has got worse.
# - **A3 (max-softmax baseline):** one-sided Mann-Whitney U test (current values stochastically
#   smaller than at commissioning), at 5 %. Mann-Whitney rather than Welch because max-softmax is
#   heavily skewed, piled up near 1.
# - **Labelled references:** **L1** and **L1b**, the drift tests of section 5; **L2**, the binomial upper bound on
#   the error rising above its commissioning value. L2 as stated compares two noisy upper bounds, so at
#   level 0 it fires about half the time by construction. We therefore also show **L2 formal**: the
#   binomial *lower* bound now above the commissioning *upper* bound.
#
# The asymmetry matters. Without labels one can show that the instrument **got worse**, but never
# certify that it **is still fine**: a model can be confidently wrong without producing a single empty
# set. A label-free alarm says "verify with labels now", not "the model is wrong".

# %%
RULES = {"A1": "A1  empty fraction", "A2": r"A2  $U_{inf,lower}$ > comm. $U_\delta$", "A3": "A3  max-softmax",
         "L1": "L1  drift test vs nominal (labels)", "L1b": "L1b  coverage vs commissioning (labels)",
         "L2": "L2  binomial bound up (labels)", "L2_formal": "L2 formal (labels)"}


def alarm_summary(recs, levels):
    out = {}
    for k in RULES:
        a = np.array([[r["alarm"][k] for r in row] for row in recs], dtype=float)
        rate = a.mean(0)
        first = {t: next((levels[j] for j in range(1, len(levels)) if rate[j] >= t), None) for t in (0.5, 0.9)}
        n_fa = int(a[:, 0].sum())
        out[k] = {"rate": rate.tolist(), "false_alarm": float(rate[0]), "false_alarm_upper95": cp_upper(n_fa, len(recs)),
                  "first_50": first[0.5], "first_90": first[0.9]}
    return out


ALARMS = {DEGRADATION: alarm_summary(F, LEVELS)}
lf_tab = table(F, LEVELS, {"empty (unlab.)": ("empty", "label_free"), "multi (unlab.)": ("multi", "label_free"),
                           "U_inf": ("U_inf", "label_free"), "U_inf_lower": ("U_inf_lower", "label_free"),
                           "max-softmax": ("maxsm_mean", "label_free"), "U_delta (frozen)": ("U_delta", None),
                           "U_E": ("U_E", None)})
print("commissioning U_delta (A2 threshold), mean over repetitions:", round(np.mean([c["eval"]["U_delta"] for c in COMM]), 4))
lf_tab.round(4)

# %%
pd.DataFrame({RULES[k]: {"false alarm (level 0)": v["false_alarm"], "its 95% upper limit": v["false_alarm_upper95"],
                         "first level >= 50%": v["first_50"], "first level >= 90%": v["first_90"],
                         **{f"rate at c={lv:g}": v["rate"][j] for j, lv in enumerate(LEVELS)}}
              for k, v in ALARMS[DEGRADATION].items()}).T.round(3)

# %%
fig, ax = plt.subplots()
line_band(ax, LEVELS, col(F, "U_inf_lower", "label_free"), r"$U_{inf,lower}$ (no labels)", COL["U_inf"], marker="o")
line_band(ax, LEVELS, col(F, "empty", "label_free"), "empty-set fraction (no labels)", COL["empty"], ls="--", marker="s")
line_band(ax, LEVELS, col(F, "U_delta"), r"$U_\delta$, frozen (labelled)", COL["U_delta"], marker="D")
ax.axhline(np.mean([c["eval"]["U_delta"] for c in COMM]), color=COL["ref"], ls="-.", lw=2,
           label=r"commissioning $U_\delta$ (alarm A2 threshold)")
ax.set_xlabel(CONTRAST_X); ax.set_ylabel("probability")
ax.invert_xaxis(); ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
save(fig, "label_free")

# %% [markdown]
# **Figure `label_free`.** Computed on unlabelled batches of `N_UNLABELED` images per level:
# `U_inf_lower`, a lower confidence bound on a quantity that can only be smaller than `U*`, and the
# empty-set fraction; the frozen `U_delta` from the labelled set for reference. Alarm A2 fires when the
# green line crosses the dash-dotted commissioning `U_delta`.

# %%
ALARM_STYLE = {"A1": (COL["empty"], "-", "s"), "A2": (COL["U_inf"], "-", "o"), "A3": (COL["maxsm"], "-", "v"),
               "L1": (COL["U_delta"], "--", "D"), "L1b": (COL["U_delta"], ":", "P"),
               "L2": (COL["binom"], "--", "^"), "L2_formal": (COL["binom"], ":", "x")}


def alarm_figure(alarms, levels, xlabel, invert, name):
    fig, ax = plt.subplots()
    for k, (c_, ls, mk) in ALARM_STYLE.items():
        ax.plot(levels, alarms[k]["rate"], ls, color=c_, marker=mk, label=RULES[k])
    ax.axhline(0.05, color=COL["ref"], lw=1.8, ls="-.", label="5 % false-alarm level")
    ax.set_xlabel(xlabel); ax.set_ylabel(f"alarm rate over {L} repetitions")
    ax.set_ylim(-0.03, 1.03)
    if invert:
        ax.invert_xaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, name)


alarm_figure(ALARMS[DEGRADATION], LEVELS, "contrast factor $c$  (1 = commissioning)", True, "alarm_rates")

# %% [markdown]
# **Figure `alarm_rates`.** Fraction of repetitions in which each rule raises an alarm, by level.
# Solid lines: label-free rules (A1, A2, A3) on the unlabelled batch; dashed and dotted: the labelled
# references. At `c = 1` (fresh clean data) every alarm is a false alarm; the dash-dotted line marks
# 5 %. With `L` repetitions a rate is resolved only to `1/L`.

# %% [markdown]
# ## 7. For comparison: recalibrating at every level
#
# Here `calibrate` and `tune_alpha` are run again on degraded data at each level, then `U_delta` is
# measured on the same fresh labelled set. This answers "how good a bound can `utrace` give for the
# model *as it is now*, if we re-commission it?". It is not drift detection: the reference moves with
# the drift. The measured coverage again matches the new nominal `1 - alpha`, so nothing looks wrong;
# the degradation shows up only as a different operating point `alpha`, chosen afresh at each level.

# %%
R = RECAL[DEGRADATION]
pd.DataFrame({"alpha (recal.)": col(R, "alpha").mean(0), "coverage (recal.)": col(R, "coverage").mean(0),
              "U_delta (recal.)": col(R, "U_delta").mean(0), "U_delta (frozen)": col(F, "U_delta").mean(0),
              "U_E": col(F, "U_E").mean(0),
              "status": [", ".join(sorted({row[j]["status"] for row in R})) for j in range(len(LEVELS))]},
             index=pd.Index(LEVELS, name="contrast c")).round(4)

# %% [markdown]
# ## 8. AWGN as a reference degradation
#
# Additive Gaussian noise (as in `notebooks/mnist_method_evaluation.py`, applied after normalisation,
# no clipping), with the same frozen commissioning instruments.

# %%
ALARMS["awgn"] = alarm_summary(FROZEN["awgn"], AWGN_LEVELS)
frozen_figure(FROZEN["awgn"], RECAL["awgn"], AWGN_LEVELS, AWGN_X, False, "awgn_reference")
table(FROZEN["awgn"], AWGN_LEVELS, {"U_E": ("U_E", None), "err_eval": ("err_eval", None), "binomial bound": ("binom_upper", None),
                                    "U": ("U", None), "U_empirical": ("U_empirical", None),
                                    "U_delta": ("U_delta", None), "coverage": ("coverage", None),
                                    "empty (unlab.)": ("empty", "label_free"),
                                    "max-softmax": ("maxsm_mean", "label_free")}).round(4)

# %% [markdown]
# **Figure `awgn_reference`.** As `frozen_verification`, for additive Gaussian noise. The structural
# band is almost invisible: under noise the model's errors stay confident, so very few correct answers
# fall into empty sets, and `U_delta`'s excess over the error is almost all statistical margin.

# %%
pd.DataFrame({RULES[k]: {"false alarm (level 0)": v["false_alarm"], "first level >= 50%": v["first_50"],
                         "first level >= 90%": v["first_90"],
                         **{f"rate at sigma={lv:g}": v["rate"][j] for j, lv in enumerate(AWGN_LEVELS)}}
              for k, v in ALARMS["awgn"].items()}).T.round(3)

# %% [markdown]
# ## Persist the results
#
# `periodic_verification_results_<mode>.json` holds every number and the provenance. Timestamps, run
# times and the commit are kept apart under `volatile`, so two runs with the same seed can be compared
# on everything else.

# %%
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_") and k != "uq"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return "nan"
    return o


def summarise(recs, recal, levels):
    keys = {"U_E": ("U_E", None), "U": ("U", None), "U_empirical": ("U_empirical", None), "U_delta": ("U_delta", None),
            "binom_upper": ("binom_upper", None), "coverage": ("coverage", None), "coverage_lower": ("coverage_lower", None),
            "eval_empty": ("eval_empty", None), "eval_multi": ("eval_multi", None),
            "unl_empty": ("empty", "label_free"), "unl_multi": ("multi", "label_free"), "U_inf": ("U_inf", "label_free"),
            "U_inf_lower": ("U_inf_lower", "label_free"), "maxsm_mean": ("maxsm_mean", "label_free"),
            "err_eval": ("err_eval", None), **{k: (k, None) for k in LAYERS}}
    out = {}
    for j, lv in enumerate(levels):
        e = {k: band(col(recs, *v)[:, j]) for k, v in keys.items()}
        e["drift_reject_rate"] = float((col(recs, "p_drift")[:, j] < 0.05).mean())
        e["L1b_reject_rate"] = float((col(recs, "p_L1b")[:, j] < 0.05).mean())
        e["U_emp_minus_err_eval"] = band((col(recs, "U_empirical") - col(recs, "err_eval"))[:, j])
        e["U_emp_minus_U_E"] = band((col(recs, "U_empirical") - col(recs, "U_E"))[:, j])
        e["U_delta_recal"] = band(col(recal, "U_delta")[:, j])
        e["alpha_recal"] = band(col(recal, "alpha")[:, j])
        out[f"{lv:g}"] = e
    return out


SUMMARY = {"commissioning": {"alpha0": band([c["alpha0"] for c in COMM]), "U_delta": band([c["eval"]["U_delta"] for c in COMM]),
                             "U": band([c["eval"]["U"] for c in COMM]), "coverage": band([c["eval"]["coverage"] for c in COMM]),
                             "binom_upper": band([c["binom_upper"] for c in COMM]),
                             "unl_empty": band([c["label_free"]["empty"] for c in COMM]),
                             "status": sorted({c["status"] for c in COMM}), "n_multi_pool_rep0": n_multi_pool},
           DEGRADATION: summarise(F, RECAL[DEGRADATION], LEVELS), "awgn": summarise(FROZEN["awgn"], RECAL["awgn"], AWGN_LEVELS),
           "alarms": ALARMS}
RESULTS = clean({"meta": {"python": sys.version.split()[0], "jax": jax.__version__, "jax_backend": jax.default_backend(),
                          "torch": torch.__version__, "numpy": np.__version__, "utrace": utrace.__version__,
                          "params": PARAMS, "model_sha256": MODEL_SHA256,
                          "model_training": {"split": "MNIST t10k (train=False), 10000 images", "seed": TRAIN_SEED,
                                             "batch": TRAIN_BATCH, "epochs": TRAIN_EPOCHS}},
                 "commissioning": COMM, "frozen": FROZEN, "recalibrated": RECAL, "summary": SUMMARY})
RESULTS["volatile"] = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "utrace_commit": COMMIT, "src_dirty": DIRTY,
                       "seconds": {"total": time.time() - T_START, "softmax": T_SOFTMAX, "frozen": T_FROZEN,
                                   "recalibrated": T_RECAL}}
RESULTS_PATH.write_text(json.dumps(RESULTS, sort_keys=True, indent=1))
NONVOLATILE_SHA256 = hashlib.sha256(json.dumps({k: v for k, v in RESULTS.items() if k != "volatile"}, sort_keys=True).encode()).hexdigest()
print(RESULTS_PATH, "| sha256 of the non-volatile part:", NONVOLATILE_SHA256)
print("figures:", *sorted(str(p) for p in FIG.iterdir()), sep="\n  ")

# %% [markdown]
# ## 9. Summary
#
# ### Facts
#
# Read off the results of this run: mean and [5th, 95th percentile] over the `L` repetitions.

# %%
S = SUMMARY


def pm(b):
    return f"{b[0]:.4f} [{b[1]:.4f}, {b[2]:.4f}]"


c0 = S["commissioning"]
print(f"QUICK={QUICK}, L={L}; commissioning: alpha0 {pm(c0['alpha0'])} ({', '.join(c0['status'])}), "
      f"U_delta {pm(c0['U_delta'])}, binomial bound {pm(c0['binom_upper'])}, coverage {pm(c0['coverage'])}")
for name, levels in ((DEGRADATION, LEVELS), ("awgn", AWGN_LEVELS)):
    print(f"\n{name}:")
    print(f"{'level':>6} {'U_E':>8} {'err_eval':>8} {'U':>8} {'U_emp':>8} {'U_delta':>8} {'binom':>8} {'coverage':>8} "
          f"{'L1 rej':>6} {'L1b rej':>7} {'unl empty':>9} {'U_inf_lo':>9} {'maxsm':>7} {'U_d recal':>9}")
    for lv in levels:
        e = S[name][f"{lv:g}"]
        print(f"{lv:>6g} {e['U_E'][0]:8.4f} {e['err_eval'][0]:8.4f} {e['U'][0]:8.4f} {e['U_empirical'][0]:8.4f} "
              f"{e['U_delta'][0]:8.4f} {e['binom_upper'][0]:8.4f} {e['coverage'][0]:8.4f} {e['drift_reject_rate']:6.2f} "
              f"{e['L1b_reject_rate']:7.2f} {e['unl_empty'][0]:9.4f} {e['U_inf_lower'][0]:9.4f} "
              f"{e['maxsm_mean'][0]:7.4f} {e['U_delta_recal'][0]:9.4f}")
    print("  decomposition of U_delta - err_eval (means): empty&correct, multi-label | coverage term, EC term"
          " ; U_emp - err_eval vs U_emp - U_E:")
    for lv in levels:
        e = S[name][f"{lv:g}"]
        print(f"  {lv:>6g}  {e['dec_empty_correct'][0]:8.4f} {e['dec_multi'][0]:9.5f} | {e['dec_coverage'][0]:8.4f} "
              f"{e['dec_EC'][0]:8.4f} ;  {e['U_emp_minus_err_eval'][0]:8.4f} vs {e['U_emp_minus_U_E'][0]:8.4f}")
    print("  alarms (false-alarm rate at level 0; first level with rate >= 50 % / >= 90 %):")
    for k, v in S["alarms"][name].items():
        print(f"    {k:<10} {v['false_alarm']:.2f} (95% upper {v['false_alarm_upper95']:.2f});  {v['first_50']} / {v['first_90']}")

# pattern check: the levels at which U_E has risen by < 1 pp while U_delta - U has at least tripled
e0 = S[DEGRADATION][f"{LEVELS[0]:g}"]
g0 = e0["U_delta"][0] - e0["U"][0]
PATTERN_LEVELS = [lv for lv in LEVELS[1:] if S[DEGRADATION][f"{lv:g}"]["U_E"][0] - e0["U_E"][0] < 0.01
                  and S[DEGRADATION][f"{lv:g}"]["U_delta"][0] - S[DEGRADATION][f"{lv:g}"]["U"][0] >= 3 * g0]
print(f"\n'confidence drops before errors rise' (U_E +<1 pp and U_delta - U >= 3x its level-0 value {g0:.4f}): "
      f"{PATTERN_LEVELS if PATTERN_LEVELS else 'at no level'}")

# %% [markdown]
# ### Inferences

# %%
# The inferences below are hand-written from one specific full run, identified by the SHA-256 of the
# non-volatile part of its results file. If THIS run differs (mode, seed, parameters, versions, code),
# they may not follow from the numbers above.
INFERENCES_WRITTEN_FROM_SHA256 = "70680757497e1a22f5b656bb23b4d910a14b6b03938d67f8643dbfcd2a62d020"
if NONVOLATILE_SHA256 == INFERENCES_WRITTEN_FROM_SHA256:
    print("inferences match this run (non-volatile sha256", NONVOLATILE_SHA256[:16] + "...)")
else:
    print("*" * 100)
    print("NOTICE: the inferences below were written from results with sha256", INFERENCES_WRITTEN_FROM_SHA256)
    print("        this run's results have sha256                              ", NONVOLATILE_SHA256)
    print(f"        (QUICK={QUICK}, L={L}). Do not rely on the inferences below without re-reading the facts above.")
    print("*" * 100)

# %% [markdown]
# Written from the full run with non-volatile SHA-256
# `70680757497e1a22f5b656bb23b4d910a14b6b03938d67f8643dbfcd2a62d020` (`QUICK=False`, `L=20`, seed
# 20261020, CPU). One model, one dataset, two degradations: these are observations about this setup,
# not properties of the method in general. Numbers are means over the 20 repetitions.
#
# 1. **The "confidence drops before errors rise" pattern, as defined here, was not observed.** The
#    test was that `U_E` rises by less than 1 percentage point while `U_delta - U` at least triples
#    from its level-0 value (0.0145). `U_E` stays within 1 pp of level 0 down to `c = 0.6` (0.0280 ->
#    0.0344), where `U_delta - U` has grown only to 0.0235 (x1.6). At `c = 0.5`, `U_E` is already
#    +1.35 pp. None of the five screened degradations showed the pattern either (see the report).
# 2. **Which part of `U_delta - err_eval` dominates depends on the regime.**
#    - On clean data and under mild degradation the statistical margin dominates. At level 0,
#      `U_delta - err_eval` = 0.0149, of which 0.0144 is statistical (coverage term 0.0081, `EC`
#      term 0.0063) and 0.0005 structural. This holds for both degradations.
#    - Under contrast loss the structural part grows: 0.0005 -> 0.0066 (`c = 0.4`) -> 0.0216
#      (`c = 0.3`, now about equal to the statistical 0.0229) -> 0.0818 (`c = 0.15`, three quarters
#      of the total 0.1083). It is almost entirely correct answers in empty sets: low-contrast images
#      the model still gets right but whose confidence fell below the frozen threshold.
#    - Under AWGN the structural part stays small (at most 0.0031, at `sigma = 0.75`), because the
#      model's errors stay confident and few correct answers lose their set. The excess is
#      statistical at every noise level.
# 3. **The statistical margin rises toward coverage of about 0.5 and then falls, as expected for a
#    binomial margin.** Under AWGN it goes 0.0145 -> 0.0285 at `sigma = 1` (coverage 0.54) -> 0.0234 at
#    `sigma = 2` (coverage 0.21). Under contrast loss it goes 0.0144 -> 0.0281 at `c = 0.2` (coverage
#    0.57) -> 0.0265 at `c = 0.15` (coverage 0.34). The fall is visible over one level only. The
#    coverage term carries the change; the `EC` term stays between 0.0057 and 0.0065 throughout,
#    because `EC` is about 1.
# 4. **Against the binomial bound, `U_delta`'s extra width is a constant statistical part plus the
#    structural part.** `U_delta`'s statistical margin exceeds the binomial bound's own margin by an
#    almost constant 0.008 to 0.010 at every level of both degradations: `U_delta` bounds two
#    quantities, each at `delta/2`, where the binomial bound bounds one at `delta`. On top of that comes
#    the structural part. So the gap `U_delta - binomial bound` grows from 0.008 to 0.090 under contrast
#    loss, almost all of the growth structural, while under AWGN it stays at 0.008 to 0.013. The
#    frozen `U_delta` was never below `U_E` in any repetition, level or degradation. `U`, which assumes
#    the nominal coverage, stays at about 0.0275 while `U_E` rises from 0.028 to 0.579
#    (figure `nominal_U_blind`).
# 5. **The uniform-selection term is not measurable here.** Multi-label sets are a few per thousand at
#    most. Their term is within +-0.0001 in the mean at every level (per repetition between -0.0013 and
#    +0.0008), so this model and data say nothing about the cost of the uniform-selection assumption.
# 6. **Estimating the structural part against `err_eval` or `U_E` differs only by sampling.** The mean
#    of `U_empirical - err_eval` and that of `U_empirical - U_E` differ by at most 0.0023 (contrast)
#    and 0.0043 (AWGN). Per repetition, `err_eval - U_E` has a standard deviation of 0.003 to 0.012,
#    matching the binomial sampling error of 2000 images (for example 0.0037 expected at an error of
#    0.028, 0.0110 at 0.58).
# 7. **The early label-free signal is the confidence baseline, not `U_inf`.** At `c = 0.8`, where `U_E`
#    has moved by +0.1 pp, the max-softmax test (A3) alarms in 20 of 20 repetitions. The empty-fraction
#    test (A1) reaches 50 % at `c = 0.6` and 100 % at `c = 0.5`. The formal `U_inf_lower` rule (A2) first
#    fires at `c = 0.35` (10 %) and reaches 100 % only at `c = 0.25` (`U_E` 0.21). Under AWGN, A2 never
#    fires and the empty fraction stays at about 1 % or below while `U_E` reaches 0.79: label-free rules
#    do not see confident errors.
# 8. **L1b is the drift test to use with a frozen instrument; L1 is not.** On clean data L1 (against
#    the nominal `1 - alpha0`) fired in 3 of 20 (contrast) and 1 of 20 (AWGN) repetitions. L1b (against
#    the commissioning coverage) fired in 1 of 20 and 0 of 20. A separate 1000-draw check (see the report)
#    measured 9.8 % for L1 and 4.6 % for L1b at a nominal 5 %. L1b pays for this in power, because it
#    compares two samples: under contrast loss it reaches 50 % of repetitions at `c = 0.5` (L1 at
#    `c = 0.6`), and 40 % against 60 % at `c = 0.6`. The literal L2 fired in 10 of 20 on clean data in both
#    runs, as expected for two noisy bounds compared head to head.
# 9. **Recalibrating hides the drift but gives a slightly tighter bound.** Recalibrated `U_delta` is
#    below the frozen one by up to 0.058 (`c = 0.15`: 0.627 against 0.685) and was never below `U_E`.
#    The tuned `alpha` moves from 0.027 to 0.571, and the measured coverage follows `1 - alpha`.

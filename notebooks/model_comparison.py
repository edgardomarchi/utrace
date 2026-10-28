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
# # One instrument, two models per dataset
#
# `notebooks/periodic_verification_demo.py` (MNIST) and `notebooks/acdc_periodic_verification.py`
# (ACDC cardiac segmentation) verify one frozen `utrace` instrument on one model each. Each also runs on a
# second model (`UTRACE_NB_MODEL`): the fully connected classifier `fc` next to the CNN `cnn` on MNIST,
# and the Bai et al. network `bm` next to the Kerfoot network `km` on ACDC. Same seeds, splits, sizes,
# repetitions and degradation grids, so each pair is compared on the same images, patients and draws.
#
# This notebook **recomputes nothing**. It reads the four full-mode results files from
# `notebooks/_build/` and shows them side by side: where does the same instrument report different things
# for different models? Comparisons are within a dataset. The levels of MNIST (contrast loss, AWGN) and
# of ACDC (AWGN, Gaussian blur) are not comparable across datasets.
#
# ### Notation (as in the two notebooks)
#
# | quantity | meaning |
# |---|---|
# | `err` | the model's error (MNIST: top-1 error on the labelled evaluation set; ACDC: patient-averaged error of the class on the evaluation patients) |
# | `U_empirical` | `1 - EC * coverage`: the point estimate of `U*` with the measured coverage |
# | `U_delta` | MNIST: the package's confidence bound; ACDC: the patient-level betting bound `U_delta_patient` |
# | **structural part** | `U_empirical - err`: what the prediction sets add to the error (empty sets on correct answers, multi-label sets) |
# | **statistical margin** | `U_delta - U_empirical`: what the finite evaluation set adds (MNIST: coverage and `EC` terms; ACDC: the patient-level margin) |
# | confident / flagged / covered-wrong | shares of the errors: a non-empty set without the true label; an empty set; a non-empty set holding the true label with another label first |
# | AUROC, ECE | resolution and calibration of the max-softmax against the error |
# | `1 - q_hat` | the frozen threshold on the softmax of the class (ACDC) |
#
# The **near-degenerate regime** is the one in which the frozen sets are essentially singletons holding the
# top-1 label or empty, rarely: `U_empirical` is almost the error, the structural part is almost 0 and, for
# a single-class quantifier, the threshold is almost 0.5.

# %%
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "utrace").exists():
            return q
    raise RuntimeError("run this notebook from inside the utrace repository")


REPO = _find_repo_root()
BUILD = REPO / "notebooks" / "_build"
FIG = BUILD / "figures" / "model_comparison"
FIG.mkdir(parents=True, exist_ok=True)
assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(FIG / "x.png")]).returncode == 0, "FIG is not git-ignored"  # noqa: PLW1510

FILES = {"cnn": BUILD / "periodic_verification_results_full.json",
         "fc": BUILD / "periodic_verification_results_fc_full.json",
         "km": BUILD / "acdc_periodic_verification_results_full.json",
         "bm": BUILD / "acdc_periodic_verification_results_bm_full.json"}
NAME = {"cnn": "CNN", "fc": "FC", "km": "KM", "bm": "BM"}
PAIRS = {"MNIST": ("cnn", "fc"), "ACDC": ("km", "bm")}
R = {m: json.loads(f.read_text()) for m, f in FILES.items() if f.exists()}
missing = [m for m in FILES if m not in R]
print("missing results files:", {m: str(FILES[m]) for m in missing} if missing else "none")
for m, r in R.items():
    assert not r["meta"]["params"]["QUICK"], f"{m}: not a full run"


def nonvolatile_sha256(r: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in r.items() if k != "volatile"}, sort_keys=True).encode()).hexdigest()


HASH = {m: nonvolatile_sha256(r) for m, r in R.items()}
CKPT = {m: r["meta"]["model_sha256"] for m, r in R.items()}
print({m: (HASH[m][:16] + "...", CKPT[m][:16] + "...") for m in R})
AVAILABLE = {d: [m for m in p if m in R] for d, p in PAIRS.items()}

# slide-sized figures: 16:9, large fonts, the notebooks' colour-blind-safe palette, no titles inside the images
plt.rcParams.update({"figure.figsize": (12.8, 7.2), "figure.dpi": 80, "font.size": 16, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2.5, "lines.markersize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
MCOL = {"cnn": "#2a78d6", "fc": "#eb6834", "km": "#2a78d6", "bm": "#eb6834"}   # the first model of a pair blue, the second orange
MMARK = {"cnn": "o", "fc": "s", "km": "o", "bm": "s"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.show()


# --- accessors: one record per (model, degradation, level) with the same fields for both datasets ---
CLASSES = ("LV", "MYO", "RV")
MN_DEGS = {"contrast": "contrast loss (c)", "awgn": "AWGN (sigma)"}
AC_DEGS = {"awgn": "AWGN (sigma)", "blur": "Gaussian blur (sigma, px)", "bias": "bias field (c)"}


def mnist_levels(m: str, deg: str) -> list[float]:
    p = R[m]["meta"]["params"]
    return list(p["LEVELS"] if deg == "contrast" else p["AWGN_LEVELS"])


def mnist_rec(m: str, deg: str, lv: float) -> dict:
    e = R[m]["summary"][deg][f"{lv:g}"]
    return {"err": e["err_eval"][0], "U_E": e["U_E"][0], "U_emp": e["U_empirical"][0], "U_delta": e["U_delta"][0],
            "struct": e["dec_empty_correct"][0] + e["dec_multi"][0], "stat": e["dec_coverage"][0] + e["dec_EC"][0],
            "conf": e["share_confident_err"][0], "flag": e["share_flagged"][0], "covw": e["share_cov_wrong"][0],
            "auroc": e["auroc"][0], "ece": e["ece"][0], "coverage": e["coverage"][0]}


def acdc_levels(m: str, deg: str) -> list[float]:
    return list(R[m]["meta"]["params"]["LEVELS"][deg])


def acdc_rec(m: str, deg: str, lv: float, classes=CLASSES) -> dict:
    """Class-averaged record (the three classes weigh the same)."""
    e = R[m]["summary"]["levels"][deg][f"{lv:g}"]
    g = lambda k, f, i=0: float(np.mean([(e[c][f][i] if isinstance(e[c][f], list) else e[c][f]) for c in classes]))
    err = float(np.mean([e[c]["err_eval_patient"][0] for c in classes]))
    return {"err": err, "U_E": float(np.mean([e[c]["U_E_patient"] for c in classes])), "U_emp": g(0, "U_emp_patient"),
            "U_delta": g(0, "U_delta_patient"), "struct": g(0, "U_emp_patient") - err,
            "stat": g(0, "U_delta_patient") - g(0, "U_emp_patient"), "conf": g(0, "share_conf_err"),
            "flag": g(0, "share_flagged"), "covw": g(0, "share_cov_wrong"), "auroc": g(0, "auroc"),
            "ece": e["ece"][0], "coverage": g(0, "coverage_pixel")}


def series(m: str, deg: str, field: str) -> tuple[list[float], np.ndarray]:
    if m in ("cnn", "fc"):
        lv = mnist_levels(m, deg)
        v = np.array([mnist_rec(m, deg, x)[field] for x in lv])
    else:
        lv = acdc_levels(m, deg)
        v = np.array([acdc_rec(m, deg, x)[field] for x in lv])
    if field == "auroc":
        v = np.where(v < 0, np.nan, v)      # the notebooks store -1 where there is no correct (or no wrong) pixel to rank
    return lv, v


def degs(dataset: str) -> dict:
    return MN_DEGS if dataset == "MNIST" else AC_DEGS


# %% [markdown]
# ## 1. Clean snapshot
#
# Each model before any degradation: the error, `U_empirical` and `U_delta`, with the frozen operating point
# (`alpha0`, and for ACDC the threshold `1 - q_hat`). MNIST: the evaluation set of 2000 images, mean over the 20
# repetitions. ACDC: per class, the patient-level quantities over each repetition's 100 evaluation patients.

# %%
rows = []
for m in AVAILABLE["MNIST"]:
    c = R[m]["summary"]["commissioning"]
    e = mnist_rec(m, "contrast", 1.0)
    rows.append({"dataset": "MNIST", "model": NAME[m], "class": "all", "error U_E": e["U_E"], "err_eval": e["err"],
                 "U_empirical": e["U_emp"], "U_delta": e["U_delta"], "U_emp - err": e["U_emp"] - e["err"],
                 "U_delta - U_emp": e["U_delta"] - e["U_emp"], "alpha0": c["alpha0"][0], "threshold 1-q_hat": np.nan})
for m in AVAILABLE["ACDC"]:
    for cname in CLASSES:
        s = R[m]["summary"]["clean_snapshot"][cname]
        rows.append({"dataset": "ACDC", "model": NAME[m], "class": cname, "error U_E": s["U_E"], "err_eval": s["err_eval_patient"][0],
                     "U_empirical": s["U_emp_patient"][0], "U_delta": s["U_delta_patient"][0],
                     "U_emp - err": s["U_emp_patient"][0] - s["err_eval_patient"][0],
                     "U_delta - U_emp": s["U_delta_patient"][0] - s["U_emp_patient"][0], "alpha0": s["alpha0"][0],
                     "threshold 1-q_hat": s["threshold"][0]})
CLEAN = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(CLEAN.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [1, 3]}, sharey=False)
for ax, dataset in zip(axes, ("MNIST", "ACDC")):
    cats = ["all"] if dataset == "MNIST" else list(CLASSES)
    for mi, m in enumerate(AVAILABLE[dataset]):
        d = CLEAN[(CLEAN["model"] == NAME[m])]
        for ci, cat in enumerate(cats):
            r_ = d[d["class"] == cat].iloc[0]
            x0 = ci + (mi - 0.5) * 0.3
            ax.plot([x0] * 2, [r_["err_eval"], r_["U_delta"]], color=MCOL[m], lw=1.5, alpha=0.5)
            ax.plot(x0, r_["err_eval"], MMARK[m], color="white", mec=MCOL[m], mew=2.5, label=f"{NAME[m]} error" if ci == 0 else None)
            ax.plot(x0, r_["U_empirical"], MMARK[m], color=MCOL[m], ms=6, label=f"{NAME[m]} $U_{{emp}}$" if ci == 0 else None)
            ax.plot(x0, r_["U_delta"], "^", color=MCOL[m], label=f"{NAME[m]} $U_\\delta$" if ci == 0 else None)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_xlabel(dataset)
axes[0].set_ylabel("error and bounds")
for ax in axes:
    ax.legend(ncol=1 if ax is axes[0] else 2, loc="upper left", fontsize=11 if ax is axes[0] else 14)
save(fig, "clean_snapshot")

# %% [markdown]
# **Figure `clean_snapshot`.** Per model (colour) and class: open marker, the error; filled marker, `U_empirical`; triangle,
# `U_delta`; the vertical line joins the error to `U_delta`. MNIST on the left, ACDC on the right.

# %% [markdown]
# ## 2. `U_delta - error`: structural part against statistical margin
#
# `U_delta - err = (U_empirical - err) + (U_delta - U_empirical)`: the structural part plus the statistical margin.
# For ACDC the quantities are averaged over the three classes.

# %%
DECOMP = {}
for dataset in ("MNIST", "ACDC"):
    ds = degs(dataset)
    ds = {k: v for k, v in ds.items() if k != "bias"}
    if not AVAILABLE[dataset]:
        continue
    fig, axes = plt.subplots(1, len(ds), figsize=(12.8, 7.2), sharey=False)
    for ax, (deg, label) in zip(np.atleast_1d(axes), ds.items()):
        for m in AVAILABLE[dataset]:
            lv, st = series(m, deg, "struct")
            _, sa = series(m, deg, "stat")
            DECOMP[(m, deg)] = {"level": lv, "struct": st, "stat": sa}
            ax.plot(lv, st, MMARK[m] + "-", color=MCOL[m], label=f"{NAME[m]} structural")
            ax.plot(lv, sa, MMARK[m] + "--", color=MCOL[m], mfc="white", label=f"{NAME[m]} statistical")
        ax.set_xlabel(label)
        if deg == "contrast":
            ax.invert_xaxis()
    np.atleast_1d(axes)[0].set_ylabel(r"part of $U_\delta - $ error")
    np.atleast_1d(axes)[0].legend()
    save(fig, f"decomposition_{dataset.lower()}")

# %% [markdown]
# **Figures `decomposition_mnist`, `decomposition_acdc`.** Solid: the structural part `U_empirical - err`; dashed
# with open markers: the statistical margin `U_delta - U_empirical`. Colour: the model. Left to right: the
# degradations, with the level on the horizontal axis (the clean level at the left, except for contrast loss,
# where the axis is reversed so that the degradation grows to the right).

# %% [markdown]
# ## 3. Error types
#
# The share of the errors that are confident (non-empty set without the true label), flagged (empty set) and covered
# but wrong, by level. The label-free bound sees only the flagged ones.

# %%
for dataset in ("MNIST", "ACDC"):
    ds = {k: v for k, v in degs(dataset).items() if k != "bias"}
    if not AVAILABLE[dataset]:
        continue
    fig, axes = plt.subplots(1, len(ds), figsize=(12.8, 7.2), sharey=True)
    for ax, (deg, label) in zip(np.atleast_1d(axes), ds.items()):
        for m in AVAILABLE[dataset]:
            lv, fl = series(m, deg, "flag")
            _, cf = series(m, deg, "conf")
            _, cw = series(m, deg, "covw")
            ax.plot(lv, fl, MMARK[m] + "-", color=MCOL[m], label=f"{NAME[m]} flagged")
            ax.plot(lv, cf, MMARK[m] + "--", color=MCOL[m], mfc="white", label=f"{NAME[m]} confident")
            ax.plot(lv, cw, MMARK[m] + ":", color=MCOL[m], ms=5, label=f"{NAME[m]} covered but wrong")
        ax.set_xlabel(label)
        if deg == "contrast":
            ax.invert_xaxis()
    np.atleast_1d(axes)[0].set_ylabel("share of the errors")
    np.atleast_1d(axes)[0].legend(fontsize=12)
    save(fig, f"error_types_{dataset.lower()}")

# %% [markdown]
# **Figures `error_types_mnist`, `error_types_acdc`.** The three shares sum to 1 at every level: solid, flagged errors
# (empty set); dashed, confident errors; dotted, covered but wrong (multi-label sets). Colour: the model.

# %% [markdown]
# ## 4. Resolution and calibration against the error
#
# AUROC (the max-softmax ranks correct above wrong answers) and top-label ECE, at each level, against the error `U_E`
# the level produces: at equal error, which model ranks and calibrates better?

# %%
for dataset in ("MNIST", "ACDC"):
    ds = {k: v for k, v in degs(dataset).items() if k != "bias"}
    if not AVAILABLE[dataset]:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    for m in AVAILABLE[dataset]:
        for deg, ls in zip(ds, ("-", "--")):
            _, ue = series(m, deg, "U_E")
            _, au = series(m, deg, "auroc")
            _, ec = series(m, deg, "ece")
            axes[0].plot(ue, au, MMARK[m] + ls, color=MCOL[m], mfc="white" if ls == "--" else MCOL[m], label=f"{NAME[m]}, {deg}")
            axes[1].plot(ue, ec, MMARK[m] + ls, color=MCOL[m], mfc="white" if ls == "--" else MCOL[m])
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("top-label ECE")
    for ax in axes:
        ax.set_xlabel("error $U_E$ at the level")
    axes[0].legend(fontsize=12)
    save(fig, f"resolution_calibration_{dataset.lower()}")

# %% [markdown]
# **Figures `resolution_calibration_mnist`, `resolution_calibration_acdc`.** Each line is one degradation (solid and
# dashed with open markers), each marker one level, placed at the error the level produces. Left, AUROC; right, ECE.
# ACDC values are averaged over the classes (AUROC) or pooled (ECE), as the notebook reports them.

# %% [markdown]
# ## 5. Alarm levels
#
# The first degradation level at which each rule alarms in at least 50 % of the 20 repetitions (`-` if never).
# A1-A4 use no labels; L1, L1b and L2 need labels (see the two notebooks).

# %%
alarm_rows = []
for m in AVAILABLE["MNIST"]:
    for deg in MN_DEGS:
        for rule, v in R[m]["summary"]["alarms"][deg].items():
            alarm_rows.append({"dataset": "MNIST", "degradation": deg, "rule": rule, "model": NAME[m],
                               "first level at 50 %": v["first_50"], "false alarm at level 0": v["false_alarm"]})
for m in AVAILABLE["ACDC"]:
    ew = R[m]["summary"]["early_warning"]
    for deg, rates in ew["rates"].items():
        lv = acdc_levels(m, deg)
        for rule, rr in rates.items():
            first = next((lv[i] for i in range(1, len(lv)) if rr[i] >= 0.5), None)
            alarm_rows.append({"dataset": "ACDC", "degradation": deg, "rule": rule, "model": NAME[m],
                               "first level at 50 %": first, "false alarm at level 0": rr[0]})
ALARMS = pd.DataFrame(alarm_rows)
ALARMS["first level at 50 %"] = ALARMS["first level at 50 %"].map(lambda v: "-" if v is None or pd.isna(v) else f"{v:g}")
ALARM_TABLE = ALARMS.pivot(index=["dataset", "degradation", "rule"], columns="model", values="first level at 50 %").fillna("-")
print(ALARM_TABLE.to_string())
print("\nfalse-alarm rate at level 0 (20 repetitions):")
print(ALARMS.pivot(index=["dataset", "degradation", "rule"], columns="model", values="false alarm at level 0").round(2).to_string())

# %% [markdown]
# ## 6. ACDC: the patient-level margin and the threshold
#
# The betting margin at `n = 100` evaluation patients on clean data, against the package's pixel-level margin, and the
# frozen threshold `1 - q_hat` of each class (0.5 is the degenerate value: see section 3 of the ACDC notebook).

# %%
rows = []
for m in AVAILABLE["ACDC"]:
    for r_ in R[m]["summary"]["margin"]:
        if r_["n"] == 100:
            s = R[m]["summary"]["clean_snapshot"][r_["class"]]
            rows.append({"model": NAME[m], "class": r_["class"], "betting margin, n=100": r_["betting"],
                         "Hoeffding": r_["Hoeffding"], "pixel-level margin": r_["pixel-level"],
                         "threshold 1-q_hat": s["threshold"][0], "|threshold - 0.5|": abs(s["threshold"][0] - 0.5)})
ACDC_MARGIN = pd.DataFrame(rows)
print(ACDC_MARGIN.round(5).to_string(index=False))

if AVAILABLE["ACDC"]:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    w = 0.35
    for mi, m in enumerate(AVAILABLE["ACDC"]):
        d = ACDC_MARGIN[ACDC_MARGIN["model"] == NAME[m]]
        axes[0].bar(np.arange(3) + (mi - 0.5) * w, d["betting margin, n=100"], w, color=MCOL[m], label=NAME[m])
        axes[1].bar(np.arange(3) + (mi - 0.5) * w, d["threshold 1-q_hat"], w, color=MCOL[m], label=NAME[m])
    axes[1].axhline(0.5, color="#7f7f7f", ls="--")
    for ax, yl in zip(axes, ("betting margin at $n = 100$", r"threshold $1 - \hat q$")):
        ax.set_xticks(range(3), CLASSES)
        ax.set_ylabel(yl)
    axes[1].set_ylim(0, 0.8)
    axes[0].legend()
    save(fig, "acdc_margin_threshold")

# %% [markdown]
# **Figure `acdc_margin_threshold`.** Left: the patient-level betting margin at `n = 100`, per class and model. Right: the frozen
# threshold `1 - q_hat` (dashed: 0.5).

# %% [markdown]
# ## 7. Is a model in the near-degenerate regime?
#
# Three numbers per model (per class for ACDC) on clean data and over all levels: the structural part, the flagged
# share of the errors, and (ACDC) the distance of the threshold from 0.5.

# %%
rows = []
for m in AVAILABLE["MNIST"]:
    allst = np.concatenate([series(m, deg, "struct")[1] for deg in MN_DEGS])
    allfl = np.concatenate([series(m, deg, "flag")[1] for deg in MN_DEGS])
    rows.append({"model": NAME[m], "class": "all", "clean structural": mnist_rec(m, "contrast", 1.0)["struct"],
                 "max structural": allst.max(), "clean flagged share": mnist_rec(m, "contrast", 1.0)["flag"],
                 "max flagged share": allfl.max(), "|threshold-0.5|": np.nan})
for m in AVAILABLE["ACDC"]:
    for c in CLASSES:
        allst = np.concatenate([[acdc_rec(m, deg, x, (c,))["struct"] for x in acdc_levels(m, deg)] for deg in ("awgn", "blur")])
        allfl = np.concatenate([[acdc_rec(m, deg, x, (c,))["flag"] for x in acdc_levels(m, deg)] for deg in ("awgn", "blur")])
        rows.append({"model": NAME[m], "class": c, "clean structural": acdc_rec(m, "awgn", 0.0, (c,))["struct"],
                     "max structural": allst.max(), "clean flagged share": acdc_rec(m, "awgn", 0.0, (c,))["flag"],
                     "max flagged share": allfl.max(),
                     "|threshold-0.5|": abs(R[m]["summary"]["clean_snapshot"][c]["threshold"][0] - 0.5)})
REGIME = pd.DataFrame(rows)
print(REGIME.round(4).to_string(index=False))

# %% [markdown]
# ## 8. Summary
#
# ### Facts
#
# Read off the four results files: see the tables above (sections 1, 5, 6 and 7) and the figures. The identity of the
# inputs:

# %%
for m in R:
    print(f"{NAME[m]:>4}: results non-volatile sha256 {HASH[m]}  checkpoint sha256 {CKPT[m][:16]}...")
print("\nstructural and statistical parts at selected levels (class-averaged for ACDC):")
for (m, deg), d in DECOMP.items():
    lv = np.array(d["level"])
    idx = sorted({0, len(lv) // 2, int(np.argmax(d["struct"])), len(lv) - 1})
    print(f"  {NAME[m]:>4} {deg:>8}: " + "; ".join(f"{lv[i]:g}: struct {d['struct'][i]:.4f} stat {d['stat'][i]:.4f}" for i in idx))

# %% [markdown]
# ### Inferences

# %%
# The inferences below are hand-written from one specific set of four full runs, identified by the SHA-256 of the
# non-volatile part of each results file. If THESE runs differ, they may not follow from the numbers above.
INFERENCES_WRITTEN_FROM_SHA256 = {
    "cnn": "5cc2a69cc7ede9a45ff74aa55c54f6f8043a3438f7ff18f1db2a32c51340fbbe",
    "fc": "dba5bed8c153d02098fe841a59785bc9d8926c52985dc364f10570617bf83928",
    "km": "335cea1ed5da50ecdf9e908d5e4f6bd1e353e62f28dd10d493170a340d95899e",
    "bm": "d8a1b73b35a6ea22d82e396be134bc23905f2aef19f3f51b43ed58d2130e4fd4",
}
if {m: HASH.get(m) for m in INFERENCES_WRITTEN_FROM_SHA256} == INFERENCES_WRITTEN_FROM_SHA256:
    print("inferences match these runs")
else:
    print("*" * 100)
    print("NOTICE: the inferences below were written from other results than these (hashes differ or files are missing).")
    print("        Do not rely on them without re-reading the facts above.")
    print("*" * 100)

# %% [markdown]
# Written from the four full runs with the non-volatile SHA-256 values listed in the code cell above (CNN checkpoint
# `abedad0c...`, FC `2b116354...`, KM `464ca796...`, BM `7a9c4275...`; `L = 20`; CPU; one machine). These are
# observations about these four models, two datasets and these degradations, not properties of the method in general.
# Numbers are means over the 20 repetitions; ACDC values are averaged over the three classes unless a class is named.
#
# 1. **On clean data the same instrument reports `U_empirical` equal to the error for all four models, whatever the
#    error.** `U_empirical - err` is 0.0005 (CNN), 0.0034 (FC), 0.0000 (KM, all classes) and 0.0015 to 0.0059 (BM).
#    The errors differ by an order of magnitude: CNN 0.028 and FC 0.076; KM 0.041, 0.107, 0.069 and BM 0.153, 0.401,
#    0.307 (LV, MYO, RV). `U_delta` follows: 0.042 and 0.101 on MNIST; 0.096, 0.159, 0.123 (KM) against 0.212, 0.453, 0.379
#    (BM). The statistical margin `U_delta - U_empirical` is of the same size within a dataset: 0.014 and 0.021 on
#    MNIST, 0.052 to 0.055 (KM) and 0.045 to 0.074 (BM) on ACDC.
# 2. **Neither second model leaves the near-degenerate regime in the way that matters on its own data, but FC does leave
#    it under contrast loss, as the CNN does.**
#    - **BM stays near it.** The structural part is 0.0015 to 0.0059 on clean data and at most 0.0071 at any level
#      before BM collapses (KM: at most 0.0002); the flagged share of the errors is at most 0.021 (KM: 0.0008); the
#      structural part is at most about 16 % of BM's statistical margin of 0.045 to 0.074. What BM does differently is
#      small but measurable: the frozen thresholds `1 - q_hat` are 0.474 (LV), 0.480 (MYO) and 0.493 (RV), up to 0.026
#      below 0.5, against 0.4999, 0.4972 and 0.4976 for KM, so BM's sets include two-label pixels, up to a fraction of
#      0.006 to 0.018 depending on the class (KM: at most 0.0004).
#    - **FC leaves it only under contrast loss.** On clean data and under AWGN its structural part is 0.0034 and 0.0040 to
#      0.0066, below its statistical margin of 0.021 to 0.032. Under contrast loss it is 0.0183 at `c = 0.5`, 0.0390 at
#      0.4 (above the statistical 0.0267), 0.1208 at 0.2, with 0.40 of the errors flagged. The CNN does the same, later:
#      0.0026 at `c = 0.5`, 0.0066 at 0.4, 0.0707 at 0.2 (statistical 0.0281), 0.49 of the errors flagged at 0.15.
#      At a similar error the FC has the larger structural part: 0.0390 at `U_E` = 0.159 (FC, `c = 0.4`) against 0.0216
#      at `U_E` = 0.132 (CNN, `c = 0.3`). At the same contrast the FC flags more: 0.246 against 0.100 at `c = 0.5`.
# 3. **What the instrument can see depends on the model's resolution, and the models differ most there.** The AUROC of
#    the max-softmax on clean data is 0.942 (CNN) and 0.914 (FC); 0.765, 0.787, 0.755 (KM) and 0.496, 0.464, 0.368 (BM),
#    i.e. no resolution at all for BM, and for RV below 0.5. The top-label ECE is 0.020 (CNN), 0.013 (FC), 0.067 (KM) and 0.202
#    (BM). The bound `U_delta` does not use the confidence, only the coverage, so it is as valid for BM as for KM; but
#    the label-free rules, which do use it, have little to work with on BM: 0.964 to 0.988 of BM's errors are confident
#    (KM: 0.999), only 0.007 to 0.021 flagged.
# 4. **The ranking of two models can reverse with the degradation.** The FC is less accurate than the CNN on clean data
#    (0.076 against 0.028) and under contrast loss (0.116 against 0.041 at `c = 0.5`), yet more robust to AWGN: at `sigma = 0.75`
#    its error is 0.135 against 0.213, at `sigma = 2` 0.431 against 0.786, with a better calibration (ECE 0.280 against 0.753).
#    On ACDC the same happens in a smaller way: BM is worse than KM everywhere, and collapses under AWGN at levels that
#    leave KM almost unchanged: at `sigma = 0.05` BM's errors are 0.887, 0.959, 0.930 and KM's 0.052, 0.120, 0.074; from
#    `sigma = 0.1` BM predicts no foreground at all. Under blur 4 BM's errors are 0.957 to 0.991 and KM's 0.108, 0.408, 0.300.
# 5. **The decomposition of `U_delta - error` differs where a model collapses.** For BM at `sigma >= 0.1` the
#    coverage is 0, `U_empirical = U_delta = 1` and both parts are 0; at blur 4 the statistical margin has shrunk to
#    0.014 (class average) from 0.059, with the coverage at 0.01 to 0.04. KM keeps a margin of 0.053 at blur 4.
# 6. **Alarm levels differ by the model, and the label-free rules that alarm first on BM are the ones that go blind.**
#    - **MNIST, contrast loss:** the first level at which a rule reaches 50 % is the same for CNN and FC for A3 (0.8),
#      L1 (0.6) and the formal L2 (0.5), and later for FC for A1 (0.8 against 0.6), A2 (0.35 against 0.3), L1b (0.6
#      against 0.5) and the literal L2 (0.8 against 0.6). Under AWGN the two models have the same levels for every
#      rule (A3 and the literal L2 0.25, L1, L1b and the formal L2 0.5, A1 0.75) and A2 never fires.
#    - **ACDC, AWGN:** KM reaches 50 % at 0.05 (A3, L1b), 0.10 (A1), 0.30 (the formal L2) and 0.40 (A4); BM at
#      0.02 (A1, A3, L1b) and 0.03 (A4, the formal L2), i.e. at noise 2.5 to 13 times smaller. At `sigma >= 0.1` BM's A1 and
#      A3 alarm in 0 % of repetitions, because they test the predicted-foreground pixels and there are none; A4 and the
#      labelled rules alarm in 100 %. Under blur BM reaches 50 % at 1.5 for A1, A3, A4 and L1b and 2 for the formal L2; KM
#      at 1.0 (L1b), 1.5 (A3), 2.5 (A1), 3 (A4 and the formal L2).
#    - Under the bias field no rule reaches 50 % for either ACDC model.
# 7. **ACDC: the patient-level margin is of one size for both models, and 47 to 170 times the pixel-level margin.** At `n = 100`
#    the betting margin is 0.055, 0.051, 0.054 (KM) and 0.062, 0.047, 0.078 (BM, LV, MYO, RV), against 3e-4 to 6e-4 (KM) and
#    8e-4 to 1.0e-3 (BM) for the package's pixel-level margin. BM's RV margin is the largest, 0.078. The thresholds are in
#    point 2.
# 8. **What this says about the instrument.** It reports the model's error through `U_empirical` and bounds it through
#    `U_delta` for all four models; it does not, by itself, say whether a model is suitable (BM's clean `U_delta` of 0.21
#    to 0.45 says the error is large, but only a reader who compares it with a requirement concludes that BM does not
#    apply to ACDC). What changes with the model is the part of its report that depends on the confidence: the structural
#    part, the flagged errors, the label-free alarms.

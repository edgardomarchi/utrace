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
# # Periodic verification of a cardiac segmentation model: what the patient-level bound says
#
# The companion of `notebooks/periodic_verification_demo.py` (MNIST) for a real black box: KM, a
# pre-trained residual U-Net that segments the left-ventricle pool (LV), the myocardium (MYO) and the
# right-ventricle pool (RV) in short-axis cardiac MRI (Kerfoot et al. 2019), on the ACDC data (Bernard
# et al. 2018), which the model never saw. As there, the instrument is commissioned once on clean data,
# one `utrace` quantifier per class, and frozen; then the input degrades and the frozen instrument is
# checked again on fresh labelled patients.
#
# Segmentation changes one thing that matters more than anything else here: **the unit of
# independence is the patient, not the pixel.** The pixels of one patient share anatomy, scanner and
# the model's systematic mistakes on that heart. `utrace`'s `U_delta` treats every pixel as an
# independent sample, so on this data it is far too narrow. This notebook keeps `utrace` for what it
# does at pixel level (calibration, the frozen threshold, the prediction sets) and states the bound at
# patient level, in the notebook, from public outputs only.
#
# Sections 1 to 4 set up the data, the bound, the commissioning and the frozen checks. From section 5 on
# the order is the talk's: the model before any degradation (5), the frozen instrument under degradation
# (6), one slice pixel by pixel (7), pixels against patients (8), early warning without labels (9),
# confident errors and resolution (10), and the summary (11).

# %% [markdown]
# ### Notation
#
# Per class `k` (LV, MYO, RV), only the pixels whose ground truth is `k` are used (`classes=[k]`).
# `C` is a pixel's prediction set and `|C|` its size (written `size(C)` in the table); "covered" means
# `k` is in `C`. Patients are indexed by `i`; ED and ES of one patient are one unit.
#
# | symbol | formula | needs labels? | estimates or bounds |
# |---|---|---|---|
# | `alpha0_k` | chosen by `tune_alpha` at commissioning | yes (tuning patients) | the operating point of class `k` |
# | `1 - q_hat_k` | the frozen softmax threshold: `k` is in `C` iff `smx[k] >= 1 - q_hat_k` | - | read for display only (no public attribute) |
# | `delta` | 0.05 | - | error level of every bound |
# | `n_ik` | number of class-`k` pixels of patient `i` | yes | - |
# | `s_ik` | `sum over covered class-k pixels of 1/size(C)` | yes | - |
# | `v_ik` | `s_ik / n_ik` | yes | `U_empirical` computed inside one patient, as `1 - v_ik` |
# | `err_ik` | fraction of patient `i`'s class-`k` pixels whose top-1 label is not `k` | yes | the patient's per-class error |
# | `U*_k` | `1 - E_i[v_ik]` | - | the patient-level target; under uniform selection it bounds `E_i[err_ik]` |
# | `U_delta_patient` | `1 - LB_bet(mean_i v_ik)` | yes | upper confidence bound on `U*_k` at level `1 - delta` (betting, patients i.i.d.) |
# | `errbound_patient` | `1 - LB_bet(mean_i (1 - err_ik))` | yes | upper confidence bound on `E_i[err_ik]` (the MNIST binomial bound, at patient level) |
# | Hoeffding versions | `1 - (mean - sqrt(ln(1/delta) / (2 n)))` | yes | the same two bounds by Hoeffding, for reference |
# | `U_delta` (pixel-level, naive) | `uncertainty_at(pooled class-k pixels, alpha0_k, delta)` | yes | the package's bound, treating every pixel as independent |
# | `U_empirical` (pixel-level) | `1 - sum_i s_ik / sum_i n_ik` | yes | its point estimate |
# | `U_E,k` (patient-averaged) | `mean_i err_ik` over all 150 patients | yes | the reference error: a finite-population value |
# | `U_E,k` (pixel-weighted) | `sum_i wrong_ik / sum_i n_ik` over all 150 patients | yes | the error the paper and the script report |
# | hesitant success | `C` empty, top-1 label `k` | yes | correct answers the frozen threshold rejects |
# | flagged error | `C` empty, top-1 label not `k` | yes | errors the set marks as doubtful |
# | confident error | `C` non-empty, `k` not in `C` (so the top-1 label is wrong) | yes | errors made with a confident set |
# | AUROC | `P(m_correct > m_wrong) + P(tie) / 2`, `m` the max-softmax | yes | resolution of confidence, per class |
# | top-label ECE | `sum_b (n_b / n) abs(acc_b - conf_b)`, 15 equal-mass bins | yes | calibration of the top-label confidence, on all foreground pixels |
# | `u` | `1 - max-softmax`, computed as the sum of the three other softmax entries (float64) | no | the model's own doubt about a pixel; resolves values that `max-softmax` rounds to 1 in float32 |
# | reference / monitoring patients | a split of each repetition's 100 evaluation patients, 10 + 10 per group | - | early warning (section 9): reference scored clean at commissioning, monitoring at each level |
# | predicted-foreground pixels | pixels whose top-1 label is LV, MYO or RV, whatever their ground truth | no | the population of the label-free rules; each pixel is judged by the frozen quantifier of its predicted class |
# | A1 | per-patient empty-set fraction of the predicted-foreground pixels; one-sided Mann-Whitney (monitoring larger) | no | alarm at 5 % |
# | A3 | per-patient mean max-softmax of the predicted-foreground pixels; one-sided Mann-Whitney (monitoring smaller) | no | alarm at 5 % |
# | A4 | per-patient predicted-foreground fraction of the image (the segmented area); two-sided Mann-Whitney | no | alarm at 5 % |
# | L1b | per-patient coverage `cov_ik / n_ik`, per class; one-sided Mann-Whitney (monitoring smaller) | yes | alarm when any class rejects at `0.05 / 3` |
# | L2 formal | monitoring betting lower bound on `E_i[err_ik]` above the reference betting upper bound, per class, each at `0.05 / 3` | yes | alarm when any class fires |

# %% [markdown]
# ## Parameters

# %%
import os

# UTRACE_NB_QUICK=1 (the default) -> a reduced run;  UTRACE_NB_QUICK=0 -> the full run.
QUICK = os.environ.get("UTRACE_NB_QUICK", "1") != "0"
MODE = "quick" if QUICK else "full"                 # every output carries it, so the two modes never overwrite each other
SEED = 20261112

SPLIT_PER_GROUP = (6, 4, 20)                        # calibration, tuning, evaluation patients per pathology group (x 5 groups)
L = 2 if QUICK else 20                              # repetitions (each redraws the patient split)
DELTA = 0.05                                        # error level of every bound
CLASSES = {1: "LV", 2: "MYO", 3: "RV"}              # model channels after the package's `relabel`; background is not used

# Degradations, in the order of the diagnostic. Level 0 is the shared clean level. The quick grids are
# subsets of the full ones, so both modes share the cache.
LEVELS = {"awgn": [0.0, 0.1, 0.3, 0.5, 0.75] if QUICK else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
          "blur": [0.0, 1.5, 3.0, 4.0] if QUICK else [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
          "bias": [0.0, 0.5] if QUICK else [0.0, 0.25, 0.5]}

MISS_DRAWS = 100 if QUICK else 500                  # section 8: draws of evaluation subsets per class and level
MISS_N = (20, 40)                                   # their sizes (patients)
MARGIN_N = (10, 20, 30, 50, 75, 100)                # section 8: subsample sizes for the margin against n
MARGIN_DRAWS = 40 if QUICK else 200
BOOT_N = (150, 200, 300, 500, 800)                  # section 8: resampled (with replacement) beyond the 100 patients, a check of the extrapolation
SIM_DRAWS = 300 if QUICK else 1000                  # section 2: simulated miss rate of the betting bound
EW_REF_PER_GROUP = 10                               # section 9: reference patients per group (the other 10 monitor)
EW_ALPHA = 0.05                                     # section 9: level of every alarm; the per-class labelled rules use EW_ALPHA / 3
NULL_SPLITS = 100 if QUICK else 500                 # section 9: random reference/monitoring splits on clean data

PARAMS = {"QUICK": QUICK, "MODE": MODE, "SEED": SEED, "SPLIT_PER_GROUP": SPLIT_PER_GROUP, "L": L, "DELTA": DELTA,
          "CLASSES": CLASSES, "LEVELS": LEVELS, "MISS_DRAWS": MISS_DRAWS, "MISS_N": MISS_N, "MARGIN_N": MARGIN_N,
          "MARGIN_DRAWS": MARGIN_DRAWS, "BOOT_N": BOOT_N, "SIM_DRAWS": SIM_DRAWS, "EW_REF_PER_GROUP": EW_REF_PER_GROUP,
          "EW_ALPHA": EW_ALPHA, "NULL_SPLITS": NULL_SPLITS}
print(PARAMS)

# %% [markdown]
# ## Data, model, degradations and cache (helper code)
#
# Not about `utrace`. A notebook-local loader reads the 150 unique ACDC patients from `training/` and
# `testing/` and skips `validation/`, which on this data is a byte-identical copy of
# `testing/patient126-150` (the package's `ACDCDataset` counts those 25 patients twice; it is not
# changed here). Preprocessing is the package's: crop/pad to 256 x 256 (`ACDCDataset.resize_and_pad`),
# per-slice min-max, and `relabel`. KM is loaded exactly as `scripts/ACDC_example.py` loads it, on CPU,
# in eval mode. For every degradation level the softmax of all **foreground-labelled** pixels (LV, MYO,
# RV) of all 150 patients is computed once and cached in float32 (float16 would quantise both the
# thresholds, near 0.5, and the softmax values, mostly near 1, and the ties would change the sets). A
# second, smaller file per level keeps the background-labelled pixels that the model labels LV, MYO or
# RV: the label-free rules of section 9 need every pixel the model calls foreground, whatever its truth.

# %% jupyter={"source_hidden": true}
os.environ.setdefault("JAX_PLATFORMS", "cpu")       # CPU only, so that every number is bitwise reproducible
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
from monai.bundle.utils import load_bundle_config
from monai.transforms import GaussianSmooth, RandBiasField
from scipy.special import betaincinv
from scipy.stats import beta as beta_dist
from scipy.stats import mannwhitneyu

import utrace
from utrace import SearchStatusWarning, UncertaintyQuantifier
from utrace.utils.pytorch.dataset_wrapper import ACDCDataset
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.utils import relabel

T_START = time.time()
assert jax.default_backend() == "cpu", jax.default_backend()
DEVICE = torch.device("cpu")


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "utrace").exists():
            return q
    raise RuntimeError("run this notebook from inside the utrace repository")


REPO = _find_repo_root()
BUILD = REPO / "notebooks" / "_build"
FIG = BUILD / "figures" / f"acdc_periodic_verification_{MODE}"
RESULTS_PATH = BUILD / f"acdc_periodic_verification_results_{MODE}.json"
CACHE = BUILD / "cache"
for d_ in (CACHE, FIG):
    d_.mkdir(parents=True, exist_ok=True)

# The data and the model are third-party and live outside the repository, at the paths
# scripts/ACDC_example.py uses relative to scripts/ (see its docstring for where to get them).
DATA_DIR = Path(os.environ.get("UTRACE_ACDC_DATA", REPO / "scripts" / "../../data/ACDC")).resolve()
MODEL_DIR = Path(os.environ.get("UTRACE_ACDC_MODEL", REPO / "scripts" / "../../models/ventricular_short_axis")).resolve()
assert (DATA_DIR / "training").is_dir() and (DATA_DIR / "testing").is_dir(), f"ACDC not found under {DATA_DIR}"
assert (MODEL_DIR / "models" / "model.pt").exists(), f"KM bundle not found under {MODEL_DIR}"


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()  # noqa: PLW1510


# Every location this notebook writes to must be git-ignored; fail loudly otherwise.
for p in (CACHE / "x.npz", FIG / "x.png", RESULTS_PATH):
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(p)]).returncode == 0, f"{p} is not git-ignored"  # noqa: PLW1510
COMMIT = _git("rev-parse", "HEAD")
DIRTY = bool(_git("status", "--porcelain", "--", "src"))
print("commit", COMMIT, "(src dirty)" if DIRTY else "", "| python", sys.version.split()[0], "| jax", jax.__version__,
      jax.default_backend(), "| torch", torch.__version__, torch.get_num_threads(), "threads | monai", monai.__version__,
      "| utrace", utrace.__version__)
print("data", DATA_DIR, "| model", MODEL_DIR)

# --- patients: training/ and testing/ only ---
def read_info(path: Path) -> dict:
    return dict(line.split(": ", 1) for line in path.read_text().strip().splitlines())


PATIENTS = []
for subset in ("training", "testing"):
    for pdir in sorted((DATA_DIR / subset).glob("patient*")):
        info = read_info(pdir / "Info.cfg")
        vols = []
        for gt in sorted(pdir.glob("*_gt.nii.gz")):
            frame = int(gt.name.split("frame")[1][:2])
            assert frame in (int(info["ED"]), int(info["ES"])), (pdir.name, frame)
            vols.append({"img": str(gt).replace("_gt.nii.gz", ".nii.gz"), "gt": str(gt),
                         "phase": "ED" if frame == int(info["ED"]) else "ES"})
        assert sorted(v["phase"] for v in vols) == ["ED", "ES"], pdir.name
        PATIENTS.append({"id": pdir.name, "subset": subset, "group": info["Group"], "vols": vols})
N_PAT = len(PATIENTS)
PGROUP = np.array([p["group"] for p in PATIENTS])
GROUPS = sorted(set(PGROUP))
assert N_PAT == 150 and len({p["id"] for p in PATIENTS}) == 150 and all((PGROUP == g).sum() == 30 for g in GROUPS)

# --- the package's crop/pad: a method of ACDCDataset that only reads self.target_size ---
_PAD = SimpleNamespace(target_size=(256, 256))


def resize_and_pad(t: torch.Tensor) -> torch.Tensor:
    return ACDCDataset.resize_and_pad(_PAD, t)


def minmax(x: torch.Tensor) -> torch.Tensor:
    """Per-slice min-max on (S, 1, H, W), the same arithmetic as scripts/ACDC_example.py's MinMaxNormalize."""
    lo = x.amin(dim=(1, 2, 3), keepdim=True)
    hi = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - lo) / (hi - lo)


t0 = time.time()
_xs, _ys, PAT_RANGE, SLICE_INFO = [], [], [], []
for pi, p in enumerate(PATIENTS):
    a = len(_xs)
    for v in p["vols"]:
        img = nib.load(v["img"]).get_fdata()
        lab = nib.load(v["gt"]).get_fdata()
        for s in range(lab.shape[2]):
            _xs.append(resize_and_pad(torch.tensor(img[:, :, s], dtype=torch.float32).unsqueeze(0)))
            _ys.append(relabel(resize_and_pad(torch.tensor(lab[:, :, s], dtype=torch.int).unsqueeze(0))))
            SLICE_INFO.append((v["img"], s))
    PAT_RANGE.append((a, len(_xs)))
X_RAW = torch.stack(_xs)                                   # (2978, 1, 256, 256) float32, cropped/padded, NOT normalised
Y_ALL = torch.stack(_ys)[:, 0].numpy().astype(np.int8)     # (2978, 256, 256), labels 0 BG, 1 LV, 2 MYO, 3 RV
del _xs, _ys
N_SLICES = len(X_RAW)
HW = 256 * 256
SLICE_PAT = np.concatenate([np.full(b - a, i) for i, (a, b) in enumerate(PAT_RANGE)])
assert np.all(np.isfinite(minmax(X_RAW).numpy()))         # no constant slice
T_LOAD = time.time() - t0
print(f"{N_PAT} patients, {N_SLICES} slices loaded in {T_LOAD:.1f}s")

# foreground pixels, in slice order then row-major (h, w): the row order of flatten_batch
FG_FLAT = Y_ALL.reshape(-1) > 0
FG_Y = Y_ALL.reshape(-1)[FG_FLAT].astype(np.int8)
FG_P = np.repeat(SLICE_PAT, HW)[FG_FLAT].astype(np.int16)          # patient index 0..149
FG_S = np.repeat(np.arange(N_SLICES), HW)[FG_FLAT].astype(np.int16)  # global slice index
IDX = {k: np.flatnonzero(FG_Y == k) for k in CLASSES}                 # class-k rows, patients contiguous and ordered
PK = {k: FG_P[IDX[k]] for k in CLASSES}
NPIX = {k: np.bincount(PK[k], minlength=N_PAT) for k in CLASSES}
assert all(NPIX[k].min() > 0 for k in CLASSES)

# --- KM: loaded as scripts/ACDC_example.py loads it, on CPU, in eval mode ---
parser = load_bundle_config(str(MODEL_DIR), "train.json")
net = parser.get_parsed_content("network_def")
net.load_state_dict(torch.load(MODEL_DIR / "models" / "model.pt", map_location=DEVICE))
net.eval()
model = Pytorch_wrapper(net.to(DEVICE), device=DEVICE, classes=np.arange(4))
MODEL_SHA256 = hashlib.sha256((MODEL_DIR / "models" / "model.pt").read_bytes()).hexdigest()
MODEL_VERSION = json.loads((MODEL_DIR / "configs" / "metadata.json").read_text())["version"]
print("KM bundle version", MODEL_VERSION, "| model.pt sha256", MODEL_SHA256[:16] + "...")

# --- degradations, in the diagnostic's order of operations; seeds fixed per (degradation, level, patient, slice) ---
DEG_ID = {"clean": 0, "awgn": 1, "blur": 2, "bias": 3}


def slice_seed(deg: str, level: float, pi: int, si: int) -> int:
    return int(np.random.SeedSequence([SEED, DEG_ID[deg], round(level * 1000), pi, si]).generate_state(1)[0])


def _plain(y) -> torch.Tensor:
    y = y.as_tensor() if hasattr(y, "as_tensor") else torch.as_tensor(y)
    return y.to(torch.float32)


def degrade(x_raw: torch.Tensor, deg: str, level: float, pi: int) -> torch.Tensor:
    """Model input for one patient's raw slices (S, 1, 256, 256).

    awgn: min-max, then + sigma N(0, 1), no clip (as scripts/ACDC_example.py).
    blur: min-max, then MONAI GaussianSmooth(sigma) in pixels, then min-max.
    bias: min-max, then MONAI RandBiasField(degree 3, coefficients ~ U(-c, c)), then min-max.
    """
    x = minmax(x_raw)
    if level == 0.0:
        return x
    out = []
    for si in range(len(x)):
        seed = slice_seed(deg, level, pi, si)
        if deg == "awgn":
            g = torch.Generator().manual_seed(seed)
            out.append(x[si] + torch.randn(x[si].shape, generator=g) * level)
        elif deg == "blur":
            out.append(_plain(GaussianSmooth(sigma=level)(x[si].clone())))
        elif deg == "bias":
            t = RandBiasField(degree=3, coeff_range=(-level, level), prob=1.0)
            t.set_random_state(seed=seed)
            out.append(_plain(t(x[si].clone())))
        else:
            raise ValueError(deg)
    xs = torch.stack(out)
    return xs if deg == "awgn" else minmax(xs)


def level_key(deg: str, level: float) -> tuple[str, float]:
    return ("clean", 0.0) if level == 0.0 else (deg, float(level))


def cache_file(key) -> Path:
    return CACHE / ("acdc_km_fg_clean.npz" if key[0] == "clean" else f"acdc_km_fg_{key[0]}_{key[1]:g}.npz")


def fp_cache_file(key) -> Path:
    return cache_file(key).with_name(cache_file(key).name.replace("_fg_", "_fp_"))


BATCH = 32
INFER_SECONDS = {}


def patient_softmax(key, pi: int) -> np.ndarray:
    """Softmax of every pixel of one patient's slices at one level, (S * 65536, 4) float32, in fixed batches of
    BATCH slices inside the patient: the batching of the cache, so the values are the cached ones bitwise."""
    a, b = PAT_RANGE[pi]
    x = degrade(X_RAW[a:b], key[0], key[1], pi)
    return np.concatenate([model.predict_proba(x[s0:min(b - a, s0 + BATCH)]).numpy() for s0 in range(0, b - a, BATCH)])


def _write_npz(f: Path, **arrays):
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.rename(f)


def level_softmax(key) -> tuple[np.ndarray, dict]:
    """At one level, for the 150 patients: the softmax of every foreground-labelled pixel, float32 (n_fg, 4), and
    of every background-labelled pixel the model labels LV, MYO or RV (a false positive), with its patient and
    slice. Both cached. If only the first file exists, it is re-inferred and checked bitwise against the file."""
    f, f_fp = cache_file(key), fp_cache_file(key)
    if not (f.exists() and f_fp.exists()):
        parts, fp, fp_pat, fp_sl, t_all = [], [], [], [], time.time()
        for pi, (a, b) in enumerate(PAT_RANGE):
            pr = patient_softmax(key, pi)
            fg = FG_FLAT[a * HW:b * HW]
            parts.append(pr[fg])
            m = ~fg & (pr.argmax(1) > 0)
            fp.append(pr[m]); fp_pat.append(np.full(m.sum(), pi, np.int16))
            fp_sl.append(np.repeat(np.arange(a, b, dtype=np.int16), HW)[m])
        smx = np.concatenate(parts)
        if f.exists():
            assert np.array_equal(np.load(f)["smx"], smx), f"re-inference differs from the cached {f.name}"
        else:
            _write_npz(f, smx=smx, y=FG_Y, patient=FG_P, slice=FG_S)
        _write_npz(f_fp, smx=np.concatenate(fp), patient=np.concatenate(fp_pat), slice=np.concatenate(fp_sl))
        INFER_SECONDS[f"{key[0]}_{key[1]:g}"] = time.time() - t_all
    z = np.load(f)
    assert np.array_equal(z["y"], FG_Y) and np.array_equal(z["patient"], FG_P) and np.array_equal(z["slice"], FG_S)
    smx = z["smx"]
    assert smx.dtype == np.float32 and smx.shape == (len(FG_Y), 4)
    zf = np.load(f_fp)
    fpd = {"smx": zf["smx"], "patient": zf["patient"], "slice": zf["slice"]}
    assert fpd["smx"].dtype == np.float32 and np.all(fpd["smx"].argmax(1) > 0)
    return smx, fpd


t0 = time.time()
KEYS = sorted({level_key(d, lv) for d, lvs in LEVELS.items() for lv in lvs}, key=lambda k: (DEG_ID[k[0]], k[1]))
SMX, FP = {}, {}
for key in KEYS:
    SMX[key], FP[key] = level_softmax(key)
CLEAN = SMX[("clean", 0.0)]
T_CACHE = time.time() - t0
CACHE_BYTES = {g(k).name: g(k).stat().st_size for k in KEYS for g in (cache_file, fp_cache_file)}
print(f"{len(KEYS)} levels ready in {T_CACHE:.0f}s ({len(INFER_SECONDS)} inferred now, "
      f"{sum(INFER_SECONDS.values()):.0f}s); {len(FG_Y):,} foreground pixels; cache file "
      f"{np.mean([CACHE_BYTES[cache_file(k).name] for k in KEYS]) / 1e6:.1f} MB per level, false positives "
      f"{min(len(FP[k]['smx']) for k in KEYS):,} to {max(len(FP[k]['smx']) for k in KEYS):,} pixels per level")

# slide-sized figures: 16:9, large fonts, the MNIST notebook's colour-blind-safe palette, no titles inside the images
plt.rcParams.update({"figure.figsize": (12.8, 7.2), "figure.dpi": 80, "font.size": 16, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2.5, "lines.markersize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"U": "#2a78d6", "U_delta": "#eb6834", "binom": "#1baf7a", "U_E": "#222222", "recal": "#e87ba4",
       "empty": "#eda100", "single": "#2a78d6", "multi": "#6250d6", "U_inf": "#008300", "maxsm": "#6250d6",
       "ref": "#7f7f7f", "U_emp": "#e34948"}
CLASS_COL = {1: COL["U"], 2: COL["U_delta"], 3: COL["binom"]}        # LV, MYO, RV (validated as a categorical set)
# outcome of a foreground pixel under its true class's frozen set (section 7): the four error-related outcomes are
# validated as a categorical set; a correct, covered pixel is near-white, so that the errors stand out
OUTCOME_COL = {"cov_correct": "#f2f2f2", "conf_err": COL["U_emp"], "flagged": COL["empty"], "hesitant": COL["multi"],
               "cov_wrong": COL["binom"]}
DEG_COL = {"awgn": COL["U_delta"], "blur": COL["U"], "bias": COL["ref"]}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.show()


def doubt(smx: np.ndarray) -> np.ndarray:
    """u = 1 - max-softmax, as the sum of the other three entries in float64 (float32 rounds most max-softmax
    values of this model to exactly 1). It is 0 only where those entries underflow in float32."""
    return np.sort(smx, axis=1)[:, :-1].astype(np.float64).sum(1)       # no cancellation against the max


def band(x):
    """Mean and 5th-95th percentile over repetitions."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))


# %% [markdown]
# ## 1. The loader against the package, and the cache
#
# The notebook's loader must give the same arrays as the package's `ACDCDataset` (with no transform),
# bitwise, on slices of patients that are not duplicated (001-125). `ACDCDataset` walks every
# subdirectory, so it also lists the 25 copies in `validation/`.

# %%
ds = ACDCDataset(str(DATA_DIR), target_size=(256, 256))
where = {(img, s): g for g, (img, s) in enumerate(SLICE_INFO)}
n_val_items = sum(f"{os.sep}validation{os.sep}" in img for img, _, _ in ds.file_slices)
candidates = [i for i, (img, _, _) in enumerate(ds.file_slices)
              if f"{os.sep}validation{os.sep}" not in img and int(Path(img).parent.name[7:]) <= 125]
sample = np.random.default_rng([SEED, 0xACDC]).choice(candidates, 60, replace=False)
for i in sample:
    img_path, _, s = ds.file_slices[i]
    g = where[(img_path, s)]
    x_pkg, y_pkg = ds[int(i)]
    assert torch.equal(x_pkg, X_RAW[g]), i                                  # crop/pad, float32
    assert np.array_equal(y_pkg[0].numpy().astype(np.int8), Y_ALL[g]), i    # crop/pad and relabel
    ref = (x_pkg - x_pkg.min()) / (x_pkg.max() - x_pkg.min())               # scripts/ACDC_example.py MinMaxNormalize
    assert torch.equal(ref, minmax(X_RAW[g:g + 1])[0]), i
LOADER_CHECK = {"package_items": len(ds), "package_items_in_validation": n_val_items, "notebook_slices": N_SLICES,
                "slices_compared": len(sample), "bitwise_equal": True}
print(LOADER_CHECK)
del ds

pd.DataFrame({"pixels": {CLASSES[k]: int(NPIX[k].sum()) for k in CLASSES},
              "per patient, median": {CLASSES[k]: int(np.median(NPIX[k])) for k in CLASSES},
              "min": {CLASSES[k]: int(NPIX[k].min()) for k in CLASSES},
              "max": {CLASSES[k]: int(NPIX[k].max()) for k in CLASSES}})

# %% [markdown]
# `ACDCDataset` holds 3542 slices, 564 of them the `validation/` copies; the notebook's loader holds
# the 2978 unique ones. On the sampled slices the image after crop/pad, the label after `relabel` and the
# min-max-normalised image are bitwise equal to the package's.
#
# **`eval()` against train mode.** This notebook calls `net.eval()`; `scripts/ACDC_example.py` does
# not, so the script runs KM in train mode. On 20 seeded clean slices, the softmax in both modes:

# %%
_net_train = copy.deepcopy(net).train()            # a copy: train mode must not touch the instrument's model
_layers = {type(m_).__name__ for m_ in net.modules()}
_norms = {(m_.affine, m_.track_running_stats) for m_ in net.modules() if isinstance(m_, torch.nn.InstanceNorm2d)}
_drop = {m_.p for m_ in net.modules() if isinstance(m_, torch.nn.Dropout)}
_g = np.random.default_rng([SEED, 0xE7A1]).choice(N_SLICES, 20, replace=False)
with torch.no_grad():
    _x = minmax(X_RAW[np.sort(_g)])
    _d = (torch.softmax(net(_x), 1) - torch.softmax(_net_train(_x), 1)).abs().max().item()
EVAL_CHECK = {"slices": 20, "max_abs_softmax_diff": _d,
              "batchnorm_layers": sum(isinstance(m_, torch.nn.modules.batchnorm._BatchNorm) for m_ in net.modules()),
              "instance_norm_layers (affine, track_running_stats)": sorted(_norms), "dropout_p": sorted(_drop)}
print(EVAL_CHECK)
assert _d == 0.0, "train and eval mode differ; the text below no longer holds"
del _net_train

# %% [markdown]
# The two modes give the same softmax, bitwise: KM normalises with `InstanceNorm2d` without running
# statistics, which normalises each image by its own statistics in either mode, has no batch
# normalisation, and its dropout has `p = 0`. The missing `eval()` in the script changes nothing.

# %% [markdown]
# ## 2. A patient-level bound: betting confidence intervals
#
# With patients as the unit, the bound is on a mean of `n` independent values in `[0, 1]`, `v_ik`
# (or `1 - err_ik`), with `n` around 100: small. Hoeffding's margin, `sqrt(ln(1/delta) / (2n))`, ignores
# the variance, and KM's per-patient values vary little (sd 0.02 to 0.04 on clean data). The betting
# confidence interval of Waudby-Smith and Ramdas, *Estimating means of bounded random variables by
# betting*, J. R. Stat. Soc. B 86(1):1-27 (2024), doi:10.1093/jrsssb/qkad009 (Theorem 3 and Remark 3,
# the hedged capital process), adapts to the variance and is valid at every `n`. One-sided, at level
# `1 - delta`:
#
# - for a candidate mean `m`, the capital `K_t(m) = prod_{i<=t} (1 + lambda_i(m) (x_i - m))`, with
#   `lambda_i(m) = min(lambda~_i, c/m)`, `c = 1/2`;
# - `lambda~_t = sqrt(2 ln(1/delta) / (n sigma^2_{t-1}))`, with the running estimates
#   `mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)` and `sigma^2_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2) / (t + 1)`;
# - if the true mean is `<= m`, `K_t(m)` is a non-negative supermartingale, so by Ville's inequality
#   `max_t K_t(m) >= 1/delta` with probability at most `delta`. The lower bound is the smallest `m`
#   that is not rejected.
#
# Each factor is positive and non-increasing in `m`, so the rejected `m` form an interval `[0, LB)`.
# The code finds `LB` by bisection and returns the rejected end, so the bound is never rounded upward.
# It depends on the order of the `x_i`: the order is a fixed, seeded permutation, chosen before the
# data are looked at. Ported from the diagnostic's `bounds.py` (which used a grid of 20 000 points; it is
# kept below as `hedged_lower_grid`, for comparison only).

# %%
def hoeffding_margin(n: int, d: float = DELTA) -> float:
    """One-sided Hoeffding margin at level d for the mean of n variables in [0, 1]."""
    return float(np.sqrt(np.log(1 / d) / (2 * n)))


def _hedged_parts(x: np.ndarray, d: float):
    n = len(x)
    t = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (t + 1)
    s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (t + 1)
    s2_prev = np.r_[0.25, s2[:-1]]
    return np.sqrt(2 * np.log(1 / d) / (n * s2_prev))


def hedged_lower(x, d: float = DELTA, c: float = 0.5, iters: int = 50) -> float:
    """One-sided (1 - d) betting lower bound on the mean of [0, 1]-valued i.i.d. x, in the order given
    (Waudby-Smith and Ramdas 2024, Thm 3 and Remark 3, lower-side capital only)."""
    x = np.asarray(x, dtype=float)
    lam_t = _hedged_parts(x, d)
    thr = np.log(1 / d)

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


def hedged_lower_grid(x, d: float = DELTA, c: float = 0.5, grid: int = 20001) -> float:
    """The diagnostic's grid version (bounds.py), verbatim in substance: the smallest non-rejected grid point."""
    x = np.asarray(x, dtype=float)
    lam_t = _hedged_parts(x, d)
    m = np.linspace(0, 1, grid)[1:]
    lam = np.minimum(lam_t[None, :], c / m[:, None])
    logK = np.cumsum(np.log1p(lam * (x[None, :] - m[:, None])), axis=1)
    keep = logK.max(1) < np.log(1 / d)
    return float(m[np.argmax(keep)]) if keep.any() else 1.0


# Hand-checkable case: x = (1, 1, 1), delta = 0.05. For every m near the answer the cap c/m is below
# lambda~_t (lambda~_1 = sqrt(2 ln 20 / (3 * 0.25)) = 2.83, lambda~_2 = 3.57, lambda~_3 = 4.20, against
# c/m = 2.21), so each factor is 1 + (0.5/m)(1 - m) = 0.5 + 0.5/m, the capital grows with t, and m is
# rejected iff (0.5 + 0.5/m)^3 >= 20, i.e. m <= 0.5 / (20^(1/3) - 0.5) = 0.225791...
lb_hand = 0.5 / (20 ** (1 / 3) - 0.5)
assert abs(hedged_lower([1.0, 1.0, 1.0]) - lb_hand) < 1e-12, (hedged_lower([1.0, 1.0, 1.0]), lb_hand)
assert np.allclose(_hedged_parts(np.ones(3), 0.05), [np.sqrt(2 * np.log(20) / 0.75), np.sqrt(2 * np.log(20) / (3 * 0.15625)),
                                                     np.sqrt(2 * np.log(20) / (3 * (0.25 + 0.0625 + (1 / 6) ** 2) / 3))])
# a single observation never rejects anything: every factor is at most 1 + 4.9 (1 - m) or 0.5 + 0.5/m, both < 20
assert hedged_lower([1.0]) < 1e-12
assert abs(hoeffding_margin(100, 0.05) - np.sqrt(np.log(20) / 200)) < 1e-15
# against the diagnostic's grid version: below it by less than one grid step
_rng = np.random.default_rng([SEED, 2])
for x_ in [_rng.beta(30, 3, 75), _rng.beta(4, 1, 25), _rng.binomial(1, 0.5, 40).astype(float), np.full(10, 0.9)]:
    diff = hedged_lower_grid(x_) - hedged_lower(x_)
    assert 0 <= diff <= 1 / 20000 + 1e-12, diff
print(f"hand case: LB = {hedged_lower([1.0, 1.0, 1.0]):.12f}, closed form {lb_hand:.12f}; grid comparison passed")

# %% [markdown]
# A short simulation of the miss rate (the fraction of draws whose lower bound is above the true mean)
# at `delta = 0.05`, against Hoeffding, on a low-variance skewed law like KM's `v_ik`, a wide one, and a
# Bernoulli.

# %%
t0 = time.time()
_rng = np.random.default_rng([SEED, 3])
SIM = []
for name, draw, mu in (("Beta(30, 3)", lambda n: _rng.beta(30, 3, n), 30 / 33),
                       ("Beta(4, 1)", lambda n: _rng.beta(4, 1, n), 0.8),
                       ("Bernoulli(0.5)", lambda n: _rng.binomial(1, 0.5, n).astype(float), 0.5)):
    for n in (25, 100):
        miss_b = miss_h = 0
        marg_b = []
        for _ in range(SIM_DRAWS):
            x_ = draw(n)
            lb = hedged_lower(x_)
            miss_b += lb > mu
            miss_h += x_.mean() - hoeffding_margin(n) > mu
            marg_b.append(x_.mean() - lb)
        SIM.append({"law": name, "n": n, "draws": SIM_DRAWS, "miss betting": miss_b / SIM_DRAWS,
                    "miss Hoeffding": miss_h / SIM_DRAWS, "margin betting": float(np.mean(marg_b)),
                    "margin Hoeffding": hoeffding_margin(n)})
T_SIM = time.time() - t0
pd.DataFrame(SIM).set_index(["law", "n"]).round(4)

# %% [markdown]
# Both stay below 5 %; the betting bound is conservative because Ville's inequality covers every `t`,
# not only `t = n`. On the low-variance law its margin is far below Hoeffding's; on the Bernoulli it
# costs little.

# %% [markdown]
# ## 3. Commissioning: one quantifier per class
#
# Each repetition draws its own split **by patient**, stratified by pathology group (DCM, HCM, MINF, NOR,
# RV; 30 patients each): 6 calibration, 4 tuning and 20 evaluation patients per group, so 30 / 20 / 100.
# Calibration uses every class-`k` pixel of the calibration patients (pixel pooling inside calibration
# is legitimate: the bound conditions on the threshold). `tune_alpha` picks `alpha0_k` on the tuning
# patients' class-`k` pixels. The instrument is then frozen.

# %%
def split(rep: int):
    """Calibration, tuning and evaluation patients (sorted index arrays), stratified by group."""
    rng = np.random.default_rng([SEED, rep])
    nc, nt, _ = SPLIT_PER_GROUP
    parts = ([], [], [])
    for g in GROUPS:
        perm = rng.permutation(np.flatnonzero(PGROUP == g))
        for part, sl in zip(parts, (perm[:nc], perm[nc:nc + nt], perm[nc + nt:]), strict=True):
            part.extend(sl.tolist())
    return tuple(np.array(sorted(x)) for x in parts)


def rows(k: int, patients: np.ndarray) -> np.ndarray:
    """Foreground rows of class k belonging to the given patients."""
    return IDX[k][np.isin(PK[k], patients)]


def commission(rep: int) -> dict:
    cal, tun, ev = split(rep)
    out = {"rep": rep, "cal": cal, "tun": tun, "ev": ev,
           "order": np.random.default_rng([SEED, rep, 1]).permutation(len(ev)),   # fixed betting order, per repetition
           "classes": {}}
    for k in CLASSES:
        rc, rt = rows(k, cal), rows(k, tun)
        q = UncertaintyQuantifier(N=len(rc), classes=[k])
        q.calibrate(CLEAN[rc].astype(np.float64), FG_Y[rc].astype(np.int64))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a0 = float(q.tune_alpha(CLEAN[rt].astype(np.float64), FG_Y[rt].astype(np.int64)))
        q.alpha = a0
        # FRICTION: no public accessor for the frozen q_hat. Read from the private state, for display only;
        # nothing below depends on it (the sets come from the public predict).
        q_hat = float(q._state.q_hat)
        out["classes"][k] = {"uq": q, "alpha0": a0, "status": q.search_status_.value,
                             "n_status_warnings": sum(issubclass(x.category, SearchStatusWarning) for x in w),
                             "threshold": 1.0 - q_hat, "n_cal_pixels": len(rc), "n_tune_pixels": len(rt)}
    return out


t0 = time.time()
COMM = [commission(rep) for rep in range(L)]
T_COMMISSION = time.time() - t0
pd.DataFrame([{"rep": c["rep"], "class": CLASSES[k], **{f: c["classes"][k][f] for f in
               ("alpha0", "threshold", "status", "n_status_warnings", "n_cal_pixels", "n_tune_pixels")}}
              for c in COMM[:3] for k in CLASSES]).set_index(["rep", "class"])

# %% [markdown]
# **Why the frozen threshold lands at about 0.5.** `tune_alpha` returns the smallest `alpha` at which
# the mean set size over the tuning pixels is at most 1. With a threshold above 0.5 no pixel can have two
# labels (the softmax sums to 1), so the mean size is `1 - empty fraction <= 1` and the search keeps
# lowering `alpha`, i.e. the threshold; below 0.5 two-label sets appear, and the mean size passes 1 as
# soon as they outnumber the empty sets, which are just as rare, so the search stops at or just below 0.5.
#
# So KM's frozen sets are, essentially, singletons holding the top-1 label or empty, and a covered pixel
# is, almost always, a correctly labelled pixel: `U_empirical` is almost exactly the error, and U-TraCE
# here is close to its degenerate case, a direct bound on the error with almost no structural part
# (section 6 measures how close).

# %% [markdown]
# ## 4. Periodic verification with the frozen instrument
#
# At every level the frozen quantifiers are applied to the repetition's 100 evaluation patients, through
# `predict` at `alpha0_k`. Per patient and class: `n_ik`, the outcome counts and `s_ik`, hence `v_ik` and
# `err_ik`. The outcome categories (covered and correct, covered and wrong, hesitant success, flagged
# error, confident error) are asserted to partition each patient's class-`k` pixels.
#
# The bounds per class, level and repetition: `U_delta_patient` and `errbound_patient` (betting, at
# `delta = 0.05`), their Hoeffding versions, and the naive pixel-level `U_delta` from `uncertainty_at` on
# the pooled class-`k` evaluation pixels.
#
# **References.** `U_E,k` is the per-class error over **all 150 patients** at that level, both
# patient-averaged (the estimand of the patient bound) and pixel-weighted. Using all 150 is legitimate,
# because calibration does not change the model's errors, but these are finite-population values: the
# error of these 150 patients, not of a wider population of patients.

# %%
def pixel_bound(n_valid: int, n_cov: int, S: float, S2: float, d: float = DELTA) -> tuple[float, float]:
    """The package's (U_delta, U_empirical) from pooled counts (_confidence_bound in uncertaintyQuantifier.py),
    for the resampling of section 8, where calling uncertainty_at 9000 times is not needed. With LAC a covered
    pixel has a non-empty set, so n_succ = n_covered. Checked against uncertainty_at at every level below."""
    EC = S / n_cov
    cov_lo = float(betaincinv(float(n_cov), float(n_valid - n_cov + 1), d / 2))
    m = np.sqrt(np.log(4 / d) / (2 * n_cov))
    if n_cov >= 2:
        Lg = np.log(8 / d)
        V = max(0.0, S2 - S * S / n_cov) / (n_cov - 1)
        m = min(m, np.sqrt(2 * V * Lg / n_cov) + 7 * Lg / (3 * (n_cov - 1)))
    return 1.0 - max(0.0, EC - m) * cov_lo, 1.0 - EC * n_cov / n_valid


def references(smx: np.ndarray) -> dict:
    """Per-class error over all 150 patients: patient-averaged and pixel-weighted (instrument-free)."""
    pred = smx.argmax(1)
    out = {}
    for k in CLASSES:
        wrong = np.bincount(PK[k], weights=pred[IDX[k]] != k, minlength=N_PAT)
        out[k] = {"err_patient": float(np.mean(wrong / NPIX[k])), "err_pixel": float(wrong.sum() / NPIX[k].sum())}
    return out


REF = {key: references(SMX[key]) for key in KEYS}

N_BINS = 15


def auroc(m_correct: np.ndarray, m_wrong: np.ndarray) -> float:
    """P(m_correct > m_wrong) + P(tie) / 2, from the Mann-Whitney statistic (midranks, so a tie counts 1/2)."""
    if len(m_correct) == 0 or len(m_wrong) == 0:
        return float("nan")
    return float(mannwhitneyu(m_correct, m_wrong, alternative="two-sided").statistic / (len(m_correct) * len(m_wrong)))


def equal_mass_bins(conf: np.ndarray) -> np.ndarray:
    """Bin index per sample: edges at the 1/15, ..., 14/15 quantiles; tied values are never split across bins."""
    inner = np.unique(np.quantile(conf, np.arange(1, N_BINS) / N_BINS))
    return np.searchsorted(inner, conf, side="left")


def top_label_ece(conf: np.ndarray, correct: np.ndarray) -> tuple[float, int]:
    """Top-label ECE over the equal-mass bins, and the number of non-empty bins."""
    b = equal_mass_bins(conf)
    ece = 0.0
    for k in np.unique(b):
        m = b == k
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece), len(np.unique(b))


COUNT_FIELDS = ("n", "cov", "s", "s2", "correct", "empty", "multi", "cov_correct", "cov_wrong", "hesitant",
                "flagged", "conf_err")


def evaluate(c: dict, key) -> dict:
    """One periodic check of repetition c's frozen instruments at one level, on its evaluation patients."""
    smx_all = SMX[key]
    ev, order = c["ev"], c["order"]
    rec = {"classes": {}}
    conf_fg, correct_fg = [], []
    for k in CLASSES:
        ck = c["classes"][k]
        q, a0 = ck["uq"], ck["alpha0"]
        r = rows(k, ev)
        smx = smx_all[r].astype(np.float64)
        pos = np.searchsorted(ev, PK[k][np.isin(PK[k], ev)])          # 0..99, the patient within the evaluation set
        y_pred, S = q.predict(smx)
        sz = S.sum(1)
        cov = S[:, k]
        correct = y_pred == k
        empty = sz == 0
        kinds = {"cov_correct": cov & correct, "cov_wrong": cov & ~correct, "hesitant": empty & correct,
                 "flagged": empty & ~correct, "conf_err": ~empty & ~cov}
        assert np.all(sum(v.astype(int) for v in kinds.values()) == 1), "the outcomes partition the pixels"
        assert not np.any(kinds["conf_err"] & correct), "LAC: a non-empty set always holds the top-1 label"
        inv = np.where(cov, 1.0 / np.maximum(sz, 1), 0.0)
        per = {"n": np.bincount(pos, minlength=len(ev)), "cov": np.bincount(pos, cov, len(ev)),
               "s": np.bincount(pos, inv, len(ev)), "s2": np.bincount(pos, inv ** 2, len(ev)),
               "correct": np.bincount(pos, correct, len(ev)), "empty": np.bincount(pos, empty, len(ev)),
               "multi": np.bincount(pos, sz > 1, len(ev)),
               **{name: np.bincount(pos, v, len(ev)) for name, v in kinds.items()}}
        assert np.array_equal(per["n"], NPIX[k][ev])
        assert np.array_equal(sum(per[name] for name in kinds), per["n"]), "per patient, the outcomes partition the pixels"
        v = per["s"] / per["n"]
        acc = per["correct"] / per["n"]
        n_ev = len(ev)
        # the package's pixel-level bound, every pixel an independent sample (naive here)
        ua = q.uncertainty_at(smx, FG_Y[r].astype(np.int64), a0, delta=DELTA)
        assert ua.n_valid == per["n"].sum() and ua.n_covered == per["cov"].sum() == ua.n_succ
        ud_pix, uemp_pix = pixel_bound(int(per["n"].sum()), int(per["cov"].sum()), float(per["s"].sum()), float(per["s2"].sum()))
        assert abs(ud_pix - float(ua.U_delta)) < 1e-9 and abs(uemp_pix - float(ua.U_empirical)) < 1e-9, (ud_pix, ua)
        m = smx.max(1)
        n_wrong = int((~correct).sum())
        rec["classes"][k] = {
            "_per": per,
            "U_emp_patient": float(1 - v.mean()), "err_eval_patient": float(1 - acc.mean()),
            "U_delta_patient": 1 - hedged_lower(v[order]), "errbound_patient": 1 - hedged_lower(acc[order]),
            "U_delta_patient_hoeffding": float(min(1.0, 1 - (v.mean() - hoeffding_margin(n_ev)))),   # a bound above 1 is trivial
            "errbound_patient_hoeffding": float(min(1.0, 1 - (acc.mean() - hoeffding_margin(n_ev)))),
            "U_delta_pixel": float(ua.U_delta), "U_emp_pixel": float(ua.U_empirical),
            "err_eval_pixel": float(n_wrong / len(r)), "coverage_pixel": float(ua.coverage),
            "empty_frac": float(empty.mean()), "multi_frac": float((sz > 1).mean()),
            **{f"share_{name}": (float(kinds[name].sum() / n_wrong) if n_wrong else float("nan"))
               for name in ("conf_err", "flagged", "cov_wrong")},
            "hesitant_frac": float(kinds["hesitant"].mean()),
            "auroc": auroc(m[correct], m[~correct]),
            "maxsm_correct": float(m[correct].mean()) if correct.any() else float("nan"),
            "maxsm_wrong": float(m[~correct].mean()) if n_wrong else float("nan")}
        conf_fg.append(m)
        correct_fg.append(smx.argmax(1) == k)
    ece, nb = top_label_ece(np.concatenate(conf_fg), np.concatenate(correct_fg))
    rec["ece"], rec["ece_bins"] = ece, nb
    return rec


t0 = time.time()
FROZEN = [{key: evaluate(c, key) for key in KEYS} for c in COMM]
T_FROZEN = time.time() - t0
print(f"{L} repetitions x {len(KEYS)} levels x {len(CLASSES)} classes in {T_FROZEN:.0f}s; every partition and the "
      "pixel-bound identity against uncertainty_at asserted")


def col(deg: str, k: int, field: str) -> np.ndarray:
    """(L, n_levels) array of one per-class field across repetitions and the levels of one degradation."""
    return np.array([[FROZEN[rep][level_key(deg, lv)]["classes"][k][field] for lv in LEVELS[deg]] for rep in range(L)],
                    dtype=float)


def ref(deg: str, k: int, field: str = "err_patient") -> np.ndarray:
    return np.array([REF[level_key(deg, lv)][k][field] for lv in LEVELS[deg]])


def level_table(deg: str) -> pd.DataFrame:
    """Every bound, per class and level (means over repetitions)."""
    out = []
    for k, cname in CLASSES.items():
        for j, lv in enumerate(LEVELS[deg]):
            row = {"class": cname, "level": lv, "U_E patient-avg": ref(deg, k)[j],
                   "U_E pixel-wtd": ref(deg, k, "err_pixel")[j]}
            for label, f in (("U_emp (patient)", "U_emp_patient"), ("U_delta_patient", "U_delta_patient"),
                             ("errbound_patient", "errbound_patient"), ("U_delta_pat Hoeff.", "U_delta_patient_hoeffding"),
                             ("errbound Hoeff.", "errbound_patient_hoeffding"), ("U_emp (pixel)", "U_emp_pixel"),
                             ("U_delta (pixel)", "U_delta_pixel")):
                row[label] = col(deg, k, f)[:, j].mean()
            out.append(row)
    return pd.DataFrame(out).set_index(["class", "level"])


def estimates_table(deg: str) -> pd.DataFrame:
    """As in the MNIST notebook: the explicit error next to U-TraCE's estimate and bounds, per class and level.
    U_E over all 150 patients; the others on the evaluation patients, means over repetitions."""
    out = []
    for k, cname in CLASSES.items():
        for j, lv in enumerate(LEVELS[deg]):
            out.append({"class": cname, "level": lv, "U_E (150 patients)": ref(deg, k)[j],
                        **{label: col(deg, k, f)[:, j].mean() for label, f in
                           (("err (evaluation)", "err_eval_patient"), ("U_empirical", "U_emp_patient"),
                            ("U_delta_patient", "U_delta_patient"), ("U_delta (pixel, naive)", "U_delta_pixel"))}})
    return pd.DataFrame(out).set_index(["class", "level"])


def line_band(ax, x, Y_, label, color, ls="-", marker="o", lw=2.5, band_style="fill", ms=None):
    """Mean over repetitions, and its 5th-95th percentile band: 'fill' (shaded, with outlines in the series
    colour), 'outline' (the two percentile lines only) or None. At most one filled band per panel, so that
    overlapping fills never mix into a colour no series has."""
    m, lo, hi = (np.array(v) for v in zip(*[band(Y_[:, j]) for j in range(Y_.shape[1])], strict=True))
    ax.plot(x, m, ls, color=color, marker=marker, lw=lw, label=label, ms=ms)
    if band_style == "fill":
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
    if band_style in ("fill", "outline"):
        for y_ in (lo, hi):
            ax.plot(x, y_, "-", color=color, lw=0.9, alpha=0.85)


DEG_X = {"awgn": r"AWGN $\sigma$", "blur": r"blur $\sigma$ (px)", "bias": "bias field $c$"}
CK = ("clean", 0.0)

# %% [markdown]
# ## 5. Before any degradation: the clean snapshot
#
# The model as commissioned, on clean data. Per class: the reference error `U_E` over all 150 patients
# (patient-averaged), and, from each repetition's frozen instrument on its own 100 evaluation patients, the
# measured error, `U_empirical` and the patient-level bound `U_delta_patient`, with the package's naive
# pixel-level `U_delta` for comparison. The operating point of each class, `alpha0` and the frozen
# threshold `1 - q_hat`, is listed with them.


# %%
def clean_band(k: int, f: str) -> tuple[float, float, float]:
    return band([FROZEN[r][CK]["classes"][k][f] for r in range(L)])


def fmt_band(b, f=".4f") -> str:
    return f"{b[0]:{f}} [{b[1]:{f}}, {b[2]:{f}}]"


SNAP = {CLASSES[k]: {"U_E": REF[CK][k]["err_patient"], "U_E_pixel": REF[CK][k]["err_pixel"],
                     **{f: clean_band(k, f) for f in ("err_eval_patient", "U_emp_patient", "U_delta_patient",
                                                      "errbound_patient", "U_delta_pixel")},
                     "alpha0": band([c["classes"][k]["alpha0"] for c in COMM]),
                     "threshold": band([c["classes"][k]["threshold"] for c in COMM])} for k in CLASSES}
print(f"clean data, mean [5th, 95th percentile] over {L} repetitions (U_E: all 150 patients, no spread)")
pd.DataFrame({cname: {"U_E, patient-averaged (150)": f"{s['U_E']:.4f}", "U_E, pixel-weighted (150)": f"{s['U_E_pixel']:.4f}",
                      "err, evaluation patients": fmt_band(s["err_eval_patient"]),
                      "U_empirical (patient level)": fmt_band(s["U_emp_patient"]),
                      "U_delta_patient": fmt_band(s["U_delta_patient"]),
                      "U_delta, pixel-level (naive)": fmt_band(s["U_delta_pixel"]),
                      "alpha0": fmt_band(s["alpha0"], ".5f"), "threshold 1 - q_hat": fmt_band(s["threshold"])}
              for cname, s in SNAP.items()})

# %%
SNAP_SERIES = (("U_E", r"$U_E$, all 150 patients", COL["U_E"], "*", 17, None),
               ("U_emp_patient", r"$U_{empirical}$, evaluation patients", COL["U_emp"], "o", 10, None),
               ("U_delta_patient", r"$U_{\delta,patient}$ (betting)", COL["U_delta"], "D", 10, None),
               ("U_delta_pixel", r"$U_\delta$, pixel-level (naive)", COL["multi"], "o", 10, "white"))
fig, ax = plt.subplots(figsize=(12.8, 7.2))
for j, (k, cname) in enumerate(CLASSES.items()):
    for si, (f, label, c_, mk, ms, mfc) in enumerate(SNAP_SERIES):
        x_ = j + (si - 1.5) * 0.17
        s = SNAP[cname][f]
        if f == "U_E":
            ax.plot(x_, s, mk, color=c_, ms=ms, label=label if j == 0 else None, zorder=3)
            txt = s
        else:
            ax.errorbar(x_, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt=mk, color=c_, ms=ms, mfc=mfc or c_, mew=2,
                        capsize=6, lw=2, label=label if j == 0 else None, zorder=3)
            txt = s[0]
        if f in ("U_E", "U_delta_patient"):
            ax.text(x_ - 0.05, txt, f"{txt:.3f}", va="center", ha="right", fontsize=14)
ax.set_xticks(range(len(CLASSES)), list(CLASSES.values()), fontsize=18)
ax.set_xlim(-0.75, len(CLASSES) - 0.4)
ax.set_ylim(0, 1.3 * max(SNAP[c_]["U_delta_patient"][2] for c_ in SNAP))
ax.set_ylabel("error probability (clean data)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, ncol=2)
fig.tight_layout()
save(fig, "acdc_clean_snapshot")

# %% [markdown]
# **Figure `acdc_clean_snapshot`.** KM on clean data, per class. Star: the patient-averaged error over all
# 150 patients. Points with whiskers: the mean and the 5th-95th percentiles over the repetitions of
# `U_empirical` on each repetition's 100 evaluation patients, of the patient-level bound `U_delta_patient`,
# and of the package's pixel-level `U_delta` (open). Markers are not connected: the classes are not a
# series. `U_empirical` sits on the reference error because with KM's frozen sets a covered pixel is
# almost always a correctly labelled one (section 3), and the evaluation patients are drawn from the same
# 150; the pixel-level `U_delta` sits on it too, which is the problem of section 8, not a virtue.

# %% [markdown]
# ## 6. The frozen instrument under degradation
#
# The same four quantities at every level of AWGN and blur, in absolute terms: the reference error of the
# 150 patients at that level, `U_empirical` and `U_delta_patient` on each repetition's evaluation patients,
# and the naive pixel-level `U_delta`.


# %%
fig, axs_v = plt.subplots(2, 3, sharey="row", figsize=(12.8, 7.2), gridspec_kw={"hspace": 0.42, "wspace": 0.1})
for ri, deg in enumerate(("awgn", "blur")):
    for ci, (k, cname) in enumerate(CLASSES.items()):
        ax = axs_v[ri, ci]
        x_ = LEVELS[deg]
        first = (ri, ci) == (0, 0)
        line_band(ax, x_, col(deg, k, "U_delta_patient"), r"$U_{\delta,patient}$ (betting), 5-95 % band" if first else None,
                  COL["U_delta"], marker="D", lw=2.2, ms=7)
        line_band(ax, x_, col(deg, k, "U_emp_patient"), r"$U_{empirical}$ (patient level), 5-95 % band" if first else None,
                  COL["U_emp"], marker="o", lw=2.2, ms=6, band_style="outline")
        ax.plot(x_, ref(deg, k), "-", color=COL["U_E"], marker="*", ms=10, lw=1.6,
                label=r"$U_E$, all 150 patients" if first else None, zorder=4)
        line_band(ax, x_, col(deg, k, "U_delta_pixel"), r"$U_\delta$, pixel-level (naive)" if first else None,
                  COL["multi"], ls="--", marker="", lw=1.4, band_style=None)
        ax.set_xlabel(DEG_X[deg], fontsize=15, labelpad=1)
        if ri == 0:
            ax.text(0.5, 1.04, cname, transform=ax.transAxes, ha="center", fontsize=17)
    axs_v[ri, 0].set_ylim(bottom=0)                      # once per row, after every panel of the row is drawn
fig.supylabel("error probability", fontsize=18, x=0.05)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=14)
save(fig, "acdc_frozen_verification")

BIAS_FLAT = {CLASSES[k]: {"max |U_E change|": float(np.abs(ref("bias", k) - ref("bias", k)[0]).max()),
                          "max |U_delta_patient change| (mean)": float(np.abs(col("bias", k, "U_delta_patient").mean(0)
                                                                              - col("bias", k, "U_delta_patient").mean(0)[0]).max())}
             for k in CLASSES}
print("bias field, change from clean over its levels", LEVELS["bias"], ":", BIAS_FLAT)

# %% [markdown]
# **Figure `acdc_frozen_verification`.** Rows: AWGN and Gaussian blur; columns: LV, MYO, RV; the level on
# the horizontal axis, from clean (0). Black stars: the patient-averaged error of all 150 patients at that
# level, the reference. Red: `U_empirical` of the frozen instrument on the evaluation patients (mean, and its
# 5th-95th percentiles over repetitions as thin lines); orange: `U_delta_patient`, shaded between its 5th and
# 95th percentiles; violet dashed: the package's pixel-level `U_delta`. `U_empirical` and `U_E` almost
# coincide: KM's frozen sets are, nearly always, the top-1 label or empty (section 3), so `U_empirical`
# is nearly the error itself, and the mean over repetitions of a subsample's error is the error of the 150.
# The bound sits a roughly constant margin above them, narrowing at the strongest AWGN levels. The
# pixel-level `U_delta` sits on the error: it carries no margin worth the name (section 8). The bias field,
# the negative control, is flat; the change from clean is printed above.
#
# The same numbers as tables, per degradation:

# %%
for deg in LEVELS:
    print(f"\n{deg}: U_E over all 150 patients; the rest on the evaluation patients, means over {L} repetitions")
    print(estimates_table(deg).round(4).to_string())

# %% [markdown]
# Every bound, with the Hoeffding versions and the pixel-weighted error:

# %%
for deg in LEVELS:
    print(f"\n{deg} (means over {L} repetitions; U_E over all 150 patients)")
    print(level_table(deg).round(4).to_string())

# %% [markdown]
# The same three bounds as the excess over the patient-averaged reference error. Below zero a bound
# underestimates the error of these 150 patients.

# %%
EXCESS = {"U_delta_patient": (r"$U_{\delta,patient}$ (betting)", COL["U_delta"], "-", "D", "fill"),
          "errbound_patient": (r"errbound$_{patient}$ (betting, on the error)", COL["binom"], "--", "^", None),
          "U_delta_pixel": (r"$U_\delta$, pixel-level (naive)", COL["multi"], "-", "o", "outline")}

fig = plt.figure(figsize=(12.8, 7.2))
gs = fig.add_gridspec(3, 3, height_ratios=[3, 3, 1.4], hspace=0.55, wspace=0.12)
axs = {}
for ri, deg in enumerate(LEVELS):
    for ci, k in enumerate(CLASSES):
        ax = fig.add_subplot(gs[ri, ci], sharey=axs.get((ri, 0)))
        axs[(ri, ci)] = ax
        for f, (label, c_, ls, mk, bs) in EXCESS.items():
            line_band(ax, LEVELS[deg], col(deg, k, f) - ref(deg, k), label if (ri, ci) == (0, 0) else None, c_, ls=ls,
                      marker=mk, lw=2.0, ms=7, band_style=bs)
        ax.axhline(0, color=COL["ref"], lw=1.5)
        ax.set_xlabel(DEG_X[deg], fontsize=15, labelpad=1)
        if ri == 0:
            ax.text(0.5, 1.04, CLASSES[k], transform=ax.transAxes, ha="center", fontsize=17)
        if ci:
            ax.tick_params(labelleft=False)
fig.supylabel("excess over the patient-averaged error $U_E$", fontsize=17, x=0.04)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False, fontsize=14)
save(fig, "acdc_frozen_excess")

# %% [markdown]
# **Figure `acdc_frozen_excess`** (secondary). Rows: AWGN, Gaussian blur, and (thin) the bias field, the
# negative control; columns: LV, MYO, RV. The excess of the two patient-level betting bounds and of the
# naive pixel-level `U_delta` over the patient-averaged error of all 150 patients at that level: mean over
# repetitions; the 5th-95th percentiles shaded for `U_delta_patient` and drawn as thin lines for the
# pixel-level `U_delta`. `errbound_patient` lies on `U_delta_patient` (they differ by less than 2e-4,
# below), so its band, the same, is not drawn. Below the grey zero line a bound underestimates the error.

# %% [markdown]
# **How close U-TraCE is to a bound on the error here.** `U_delta_patient - errbound_patient` is the whole
# structural part of the patient bound: it would be zero if the sets were exactly the top-1 label when
# right and wrong otherwise.

# %%
THRESH = np.array([[c["classes"][k]["threshold"] for k in CLASSES] for c in COMM])
STRUCT = []
for deg in LEVELS:
    for k, cname in CLASSES.items():
        d_ = col(deg, k, "U_delta_patient") - col(deg, k, "errbound_patient")
        STRUCT.append({"degradation": deg, "class": cname, "max |U_delta_pat - errbound_pat|": float(np.abs(d_).max()),
                       "mean U_emp - err (patient)": float((col(deg, k, "U_emp_patient") - col(deg, k, "err_eval_patient")).mean()),
                       "max multi-label fraction": float(col(deg, k, "multi_frac").max()),
                       "max empty fraction": float(col(deg, k, "empty_frac").max())})
print("frozen threshold 1 - q_hat over repetitions, per class:",
      {CLASSES[k]: [round(THRESH[:, j].min(), 4), round(THRESH[:, j].max(), 4)] for j, k in enumerate(CLASSES)},
      f"| above 0.5 in {int((THRESH > 0.5).sum())} of {THRESH.size}")
pd.DataFrame(STRUCT).set_index(["degradation", "class"])

# %% [markdown]
# ## 7. One slice, pixel by pixel
#
# U-TraCE gives per-class rates, not a per-pixel error, and nothing below is a per-pixel estimate of the
# error. What can be shown per pixel is the model's output, its prediction set at the frozen threshold
# and its confidence, against the ground truth.
#
# **The slice, by rule, fixed before any output is looked at.** Among repetition 0's 100 evaluation
# patients, the one whose clean mean per-class error (the mean over LV, MYO and RV of `err_ik`) is the
# median (with 100 patients, the lower of the two middle ones); of its ED volume, the slices that contain
# all three classes; of those, the one nearest the middle of their range (the lower one on a tie). It is
# run clean and at the middle AWGN and middle blur levels, with the cache's degradation seeds, so its
# softmax is the cached one (asserted). The middle level of a degradation is the one whose reference error
# (patient-averaged, mean over the three classes) is closest to the midpoint between the clean and the
# most severe level's; section 8 uses the same levels.
#
# **The outcome map** uses repetition 0's frozen per-class quantifiers: each foreground pixel is judged by
# the quantifier of its *true* class, with the outcome definitions of section 4 (covered and correct;
# covered but wrong, a multi-label set; hesitant success, an empty set on a correct top-1 label; flagged
# error, an empty set on a wrong one; confident error, a non-empty set without the true class).
# Background pixels have no quantifier and are left uncoloured.

# %%
def middle_level(deg: str) -> float:
    e = np.array([np.mean([REF[level_key(deg, lv)][k]["err_patient"] for k in CLASSES]) for lv in LEVELS[deg]])
    return float(LEVELS[deg][int(np.argmin(np.abs(e - (e[0] + e[-1]) / 2)))])


MIDDLE = {deg: middle_level(deg) for deg in ("awgn", "blur")}
MISS_KEYS = {"clean": CK, "awgn": level_key("awgn", MIDDLE["awgn"]), "blur": level_key("blur", MIDDLE["blur"])}
print("middle levels:", MIDDLE)

ev0 = COMM[0]["ev"]
_pat_err = np.mean([1 - FROZEN[0][CK]["classes"][k]["_per"]["correct"] / FROZEN[0][CK]["classes"][k]["_per"]["n"]
                    for k in CLASSES], axis=0)                              # clean mean per-class error, per patient
OV_PI = int(ev0[np.argsort(_pat_err, kind="stable")[(len(ev0) - 1) // 2]])
_a, _b = PAT_RANGE[OV_PI]
_ed = next(v["img"] for v in PATIENTS[OV_PI]["vols"] if v["phase"] == "ED")
_full = [g for g in range(_a, _b) if SLICE_INFO[g][0] == _ed and set(np.unique(Y_ALL[g])) >= set(CLASSES)]
OV_G = min(_full, key=lambda g: (abs(g - (_full[0] + _full[-1]) / 2), g))   # global slice index
OVERLAY = {"patient": PATIENTS[OV_PI]["id"], "group": PATIENTS[OV_PI]["group"], "patient_index": OV_PI,
           "clean_mean_error": float(_pat_err[list(ev0).index(OV_PI)]), "ed_slice": SLICE_INFO[OV_G][1],
           "ed_slices_with_all_classes": [SLICE_INFO[g][1] for g in _full], "global_slice": OV_G,
           "conditions": {name: f"{key[0]} {key[1]:g}" for name, key in MISS_KEYS.items()}}
print(OVERLAY)

OUTCOMES = ("cov_correct", "cov_wrong", "hesitant", "flagged", "conf_err")
OUTCOME_LABEL = {"cov_correct": "covered and correct", "cov_wrong": "covered but wrong", "hesitant": "hesitant success",
                 "flagged": "flagged error (empty set)", "conf_err": "confident error"}
OV = {}
for name, key in MISS_KEYS.items():
    pr = patient_softmax(key, OV_PI)
    fg_p = FG_FLAT[_a * HW:_b * HW]
    assert np.array_equal(pr[fg_p], SMX[key][FG_P == OV_PI]), "the overlay's softmax is the cached one"
    s_ = pr[(OV_G - _a) * HW:(OV_G - _a + 1) * HW]                          # (65536, 4), row-major (h, w)
    y_ = Y_ALL[OV_G].reshape(-1)
    outcome = np.full(HW, -1)                                               # -1: background, no quantifier
    for k in CLASSES:
        r_ = np.flatnonzero(y_ == k)
        y_pred, S = COMM[0]["classes"][k]["uq"].predict(s_[r_].astype(np.float64))
        sz, cov, ok = S.sum(1), S[:, k], y_pred == k
        kinds = {"cov_correct": cov & ok, "cov_wrong": cov & ~ok, "hesitant": (sz == 0) & ok,
                 "flagged": (sz == 0) & ~ok, "conf_err": (sz > 0) & ~cov}
        for oi, o in enumerate(OUTCOMES):
            outcome[r_[kinds[o]]] = oi
    assert np.all(outcome[y_ > 0] >= 0)
    OV[name] = {"img": degrade(X_RAW[_a:_b], key[0], key[1], OV_PI)[OV_G - _a, 0].numpy(), "gt": Y_ALL[OV_G],
                "pred": s_.argmax(1).reshape(256, 256), "outcome": outcome.reshape(256, 256),
                "u": doubt(s_).reshape(256, 256)}
    e_ = OV[name]["pred"] != OV[name]["gt"]
    OVERLAY[name] = {
        "slice_error": {CLASSES[k]: float((OV[name]["pred"][OV[name]["gt"] == k] != k).mean()) for k in CLASSES},
        "outcome_counts": {o: int((outcome == oi).sum()) for oi, o in enumerate(OUTCOMES)},
        "median_u_wrong_fg": float(np.median(OV[name]["u"][e_ & (OV[name]["gt"] > 0)])),
        "median_u_correct_fg": float(np.median(OV[name]["u"][~e_ & (OV[name]["gt"] > 0)])),
        "U_delta_patient_rep0": {CLASSES[k]: FROZEN[0][key]["classes"][k]["U_delta_patient"] for k in CLASSES}}
pd.DataFrame({name: {**{f"error {c}": v for c, v in d["slice_error"].items()},
                     **{OUTCOME_LABEL[o]: v for o, v in d["outcome_counts"].items()},
                     "median u, wrong fg pixels": d["median_u_wrong_fg"],
                     "median u, correct fg pixels": d["median_u_correct_fg"]}
              for name, d in OVERLAY.items() if name in MISS_KEYS})

# %%
_ys, _xs = np.nonzero(Y_ALL[OV_G] > 0)
_c, _h = ((_ys.min() + _ys.max()) // 2, (_xs.min() + _xs.max()) // 2), max(np.ptp(_ys), np.ptp(_xs)) // 2 + 24
CROP = (slice(max(0, _c[0] - _h), _c[0] + _h), slice(max(0, _c[1] - _h), _c[1] + _h))
U_NORM = LogNorm(vmin=1e-12, vmax=1.0)
CLASS_CMAP = ListedColormap([CLASS_COL[k] for k in CLASSES])
OUT_CMAP = ListedColormap([OUTCOME_COL[o] for o in OUTCOMES])
ROW_LABEL = {"clean": "clean", "awgn": rf"AWGN $\sigma$ = {MIDDLE['awgn']:g}", "blur": rf"blur $\sigma$ = {MIDDLE['blur']:g} px"}
COL_TITLE = ("input", "ground truth", "prediction", "frozen-set outcome", "1 - max-softmax")

fig, axs_o = plt.subplots(3, 5, figsize=(12.8, 8.4), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
for ri, name in enumerate(MISS_KEYS):
    d = OV[name]
    img = d["img"][CROP]
    for ci, ax in enumerate(axs_o[ri]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if ci in (1, 2):
            lab = (d["gt"] if ci == 1 else d["pred"])[CROP]
            ax.imshow(np.ma.masked_equal(lab, 0) - 1, cmap=CLASS_CMAP, vmin=-0.5, vmax=2.5, alpha=0.6, interpolation="nearest")
        elif ci == 3:
            out = d["outcome"][CROP]
            ax.imshow(img, cmap="gray", vmin=0, vmax=2, interpolation="nearest")       # dimmed under the outcomes
            ax.imshow(np.ma.masked_less(out, 0), cmap=OUT_CMAP, vmin=-0.5, vmax=len(OUTCOMES) - 0.5, interpolation="nearest")
        elif ci == 4:
            hm = ax.imshow(np.clip(d["u"][CROP], 1e-12, 1), cmap="Oranges", norm=U_NORM, alpha=0.75, interpolation="nearest")
            ax.contour((d["pred"] != d["gt"])[CROP].astype(float), levels=[0.5], colors="black", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ri == 0:
            ax.set_title(COL_TITLE[ci], fontsize=15)
    axs_o[ri, 0].set_ylabel(ROW_LABEL[name], fontsize=15)
cax = fig.add_axes([0.915, 0.3, 0.013, 0.45])
cb = fig.colorbar(hm, cax=cax, ticks=[1e-12, 1e-8, 1e-4, 1])
thr_ = float(np.mean(THRESH))                              # a pixel's set is empty only if max-softmax < 1 - q_hat
cax.axhline(1 - thr_, color="black", lw=2.5)
cb.set_label("1 - max-softmax (log); black bar: the set is empty above it\nblack outline in the panels: wrong top-1 label",
             fontsize=12)
handles = [Patch(color=CLASS_COL[k], alpha=0.7, label=c) for k, c in CLASSES.items()]
handles += [Patch(facecolor=OUTCOME_COL[o], edgecolor="#666666", lw=0.8, label=OUTCOME_LABEL[o]) for o in OUTCOMES]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, 0.08), ncol=4, frameon=False, fontsize=13)
save(fig, "acdc_overlay")

# %% [markdown]
# **Figure `acdc_overlay`.** The slice chosen by rule: patient095 (group RV), ED slice 6 of 14 (all 14
# contain the three classes), cropped around the heart. Rows: clean, the middle AWGN level and the middle
# blur level (printed above). Columns: the input; the ground truth and the prediction over it (LV blue, MYO
# orange, RV green); the outcome of each foreground pixel under the frozen quantifier of its true class
# (background uncoloured); and `1 - max-softmax` on a log scale over the image, with the pixels whose top-1
# label is wrong outlined in black. The black bar on the colour bar is where a set becomes empty: a pixel
# is flagged only if `1 - max-softmax` exceeds about 0.5.
#
# What the figure shows, read with the table above: on this slice **every error is a confident error**,
# in all three conditions. Not one pixel has an empty or multi-label set, so the outcome map has only two
# colours, white and red. The wrong pixels are less confident than the right ones (median
# `1 - max-softmax` about `1e-6` against `1e-12` to `1e-9`), but still five to six orders of magnitude
# below the 0.5 at which the set would flag them. Under AWGN the red grows along the borders, LV against
# MYO and the RV free wall (slice errors LV 0.04 to 0.19, MYO 0.06 to 0.15, RV 0.10 to 0.19); under blur a
# block of the RV is lost at its left end (RV 0.10 to 0.27) while LV and MYO change less (0.04 to 0.06,
# 0.06 to 0.09). This is one
# slice of one patient, chosen by rule; the rates are in sections 6 and 10.

# %% [markdown]
# ## 8. Pixels are not the unit of independence
#
# The main result. One frozen instrument (repetition 0's commissioning) and its 100 evaluation patients
# as a finite population, at clean data and at the middle AWGN and middle blur levels of section 7. From
# that population, 500 draws
# without replacement of `n_eval` = 20 and 40 patients. For each draw:
#
# - the naive pixel-level `U_delta` on the pooled pixels of the drawn patients, against the population's
#   pixel-weighted `U_empirical`;
# - `U_delta_patient` on the drawn patients, against the population's `1 - mean v`.
#
# A bound "misses" when it falls below its target. Sampling without replacement from a finite population
# makes i.i.d. bounds conservative (the draws vary less than i.i.d. draws would), so a miss rate below
# `delta` is expected for the patient bound; any excess for the pixel bound is the point.

# %%
t0 = time.time()
MISS = []
for name, key in MISS_KEYS.items():
    for k, cname in CLASSES.items():
        per = FROZEN[0][key]["classes"][k]["_per"]
        n_, cov_, s_, s2_ = per["n"], per["cov"], per["s"], per["s2"]
        v = s_ / n_
        target_pix, target_pat = 1 - s_.sum() / n_.sum(), 1 - v.mean()
        for n_eval in MISS_N:
            rng = np.random.default_rng([SEED, 5, DEG_ID[key[0]], k, n_eval])
            miss_pix = miss_pat = 0
            for _ in range(MISS_DRAWS):
                i = rng.choice(len(v), n_eval, replace=False)            # the draw order is the betting order
                ud_pix, _ = pixel_bound(int(n_[i].sum()), int(cov_[i].sum()), float(s_[i].sum()), float(s2_[i].sum()))
                miss_pix += ud_pix < target_pix
                miss_pat += 1 - hedged_lower(v[i]) < target_pat
            for bound, miss in (("pixel-level U_delta", miss_pix), ("U_delta_patient", miss_pat)):
                p_ = miss / MISS_DRAWS
                MISS.append({"level": f"{key[0]} {key[1]:g}", "class": cname, "n_eval": n_eval, "bound": bound,
                             "miss rate": p_, "se": float(np.sqrt(p_ * (1 - p_) / MISS_DRAWS)), "draws": MISS_DRAWS})
T_MISS = time.time() - t0
MISS_DF = pd.DataFrame(MISS)
MISS_DF.pivot_table(index=["level", "n_eval", "class"], columns="bound", values=["miss rate", "se"]).round(3)

# %% [markdown]
# Then the width. The margin of each bound over its own point estimate, on clean data, against the
# number of evaluation patients `n`, by subsampling the same 100 patients (at `n = 100`, all of them in
# random orders): the betting margin `mean v - LB`, and the pixel-level margin `U_delta - U_empirical`.
# How the betting margin scales is fitted, not assumed: a power law `A n^b` over all `n` and over
# `n >= 50`, the local exponents between neighbouring `n`, and the two-term form `a / sqrt(n) + b / n`.
#
# Beyond `n = 100` nothing can be measured. As a check of the extrapolation, and only that, the margin
# is also computed on samples of 150 to 800 drawn **with replacement** from the 100 clean `v` values: that
# is the margin the bound would have if the population of patients were these 100, so it is a
# resampling, not a measurement.

# %%
t0 = time.time()
MARGIN, BOOT = [], []
for k, cname in CLASSES.items():
    per = FROZEN[0][("clean", 0.0)]["classes"][k]["_per"]
    n_, cov_, s_, s2_ = per["n"], per["cov"], per["s"], per["s2"]
    v = s_ / n_
    rng = np.random.default_rng([SEED, 6, k])
    for n_eval in MARGIN_N:
        mb, mp = [], []
        for _ in range(MARGIN_DRAWS):
            i = rng.choice(len(v), n_eval, replace=False)
            mb.append(v[i].mean() - hedged_lower(v[i]))
            ud, ue = pixel_bound(int(n_[i].sum()), int(cov_[i].sum()), float(s_[i].sum()), float(s2_[i].sum()))
            mp.append(ud - ue)
        MARGIN.append({"class": cname, "k": k, "n": n_eval, "betting": float(np.mean(mb)),
                       "betting sd over draws": float(np.std(mb)), "Hoeffding": hoeffding_margin(n_eval),
                       "pixel-level": float(np.mean(mp)), "ratio betting / pixel": float(np.mean(mb) / np.mean(mp))})
    for n_eval in BOOT_N:                                         # resampled, with replacement: a check, not a measurement
        mb = []
        for _ in range(MARGIN_DRAWS):
            i = rng.choice(len(v), n_eval, replace=True)
            mb.append(v[i].mean() - hedged_lower(v[i]))
        BOOT.append({"class": cname, "k": k, "n": n_eval, "betting (resampled)": float(np.mean(mb)),
                     "sd over draws": float(np.std(mb)), "Hoeffding": hoeffding_margin(n_eval)})
T_MARGIN = time.time() - t0
MARGIN_DF, BOOT_DF = pd.DataFrame(MARGIN), pd.DataFrame(BOOT)


def fit_margin(n: np.ndarray, m: np.ndarray) -> dict:
    b, logA = np.polyfit(np.log(n), np.log(m), 1)
    b_hi, logA_hi = np.polyfit(np.log(n[n >= 50]), np.log(m[n >= 50]), 1)
    (a2, b2), *_ = np.linalg.lstsq(np.c_[1 / np.sqrt(n), 1 / n], m, rcond=None)
    local = np.diff(np.log(m)) / np.diff(np.log(n))

    def rel(pred):
        return float(np.max(np.abs(pred / m - 1)))

    return {"power_A": float(np.exp(logA)), "power_b": float(b), "power_maxrelerr": rel(np.exp(logA) * n ** b),
            "power_A_n>=50": float(np.exp(logA_hi)), "power_b_n>=50": float(b_hi),
            "two_term_a": float(a2), "two_term_b": float(b2), "two_term_maxrelerr": rel(a2 / np.sqrt(n) + b2 / n),
            "local_exponents": local.tolist()}


def n_needed(f: dict, target: float, m100: float, boot: pd.DataFrame) -> dict:
    """Patients needed for a betting margin of `target`, under each form. Beyond n = 100 every one is an extrapolation."""
    a, b = f["two_term_a"], f["two_term_b"]
    u = (-a + np.sqrt(a * a + 4 * b * target)) / (2 * b) if b > 0 else target / a   # b u^2 + a u = target, u = 1/sqrt(n)
    out = {"power law, all n": float((target / f["power_A"]) ** (1 / f["power_b"])),
           "power law, n >= 50": float((target / f["power_A_n>=50"]) ** (1 / f["power_b_n>=50"])),
           "a/sqrt(n) + b/n": float(1 / u ** 2),
           "1/sqrt(n) from n = 100": float(100 * (m100 / target) ** 2)}
    bn, bm = boot.n.to_numpy(float), boot["betting (resampled)"].to_numpy()
    inside = bm.min() <= target <= bm.max()                      # log-log interpolation between resampled points
    out["resampled"] = float(np.exp(np.interp(np.log(target), np.log(bm[::-1]), np.log(bn[::-1])))) if inside else float("nan")
    return out


FITS, NEEDED = {}, []
for k, cname in CLASSES.items():
    d_ = MARGIN_DF[MARGIN_DF.k == k]
    FITS[k] = fit_margin(d_.n.to_numpy(float), d_.betting.to_numpy())
    m100 = float(d_[d_.n == 100].betting.iloc[0])
    for target in (0.05, 0.03, 0.02):
        for form, n_req in n_needed(FITS[k], target, m100, BOOT_DF[BOOT_DF.k == k]).items():
            NEEDED.append({"class": cname, "target margin": target, "form": form, "n needed": n_req,
                           "extrapolated": bool(n_req > 100)})
print(MARGIN_DF.drop(columns="k").set_index(["class", "n"]).round(4).to_string())
print("\nresampled beyond the 100 patients (with replacement; not a measurement):")
print(BOOT_DF.drop(columns="k").set_index(["class", "n"]).round(4).to_string())
print()
print(pd.DataFrame({CLASSES[k]: {kk: (np.round(vv, 3) if isinstance(vv, list) else round(vv, 4)) for kk, vv in f.items()}
                    for k, f in FITS.items()}).to_string())
NEEDED_DF = pd.DataFrame(NEEDED)
NEEDED_DF.pivot_table(index=["class", "target margin"], columns="form", values="n needed", sort=False).round(0)

# %% [markdown]
# The `n needed` table is an **extrapolation** wherever it exceeds 100, the largest `n` measured. The
# forms disagree there. The power law fitted on `n >= 50` follows the local exponent at the top of the
# measured range. `1/sqrt(n) from n = 100` scales the measured margin at `n = 100` by `1/sqrt(n)`, the
# asymptotic rate of the betting margin; it is the conservative column, as long as the margin keeps
# shrinking at least that fast, which the measured local exponents (all steeper than -1/2) support. `resampled` reads the
# margin off the with-replacement samples above.

# %%
fig, (axl, axr) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
for k, cname in CLASSES.items():
    d_ = MARGIN_DF[MARGIN_DF.k == k]
    axl.plot(d_.n, d_.betting, "-D", color=CLASS_COL[k], label=f"{cname}, patient level (betting)")
    axl.plot(d_.n, d_["pixel-level"], "--o", color=CLASS_COL[k], mfc="white", mew=2, label=f"{cname}, pixel level")
axl.set_xscale("log"); axl.set_yscale("log")
axl.set_xticks(MARGIN_N, [str(n) for n in MARGIN_N]); axl.minorticks_off()
axl.set_xlabel("evaluation patients $n$"); axl.set_ylabel("margin over the point estimate (clean)")
axl.legend(loc="center right", frameon=False, fontsize=12)

conds = [(name, n) for name in MISS_KEYS for n in MISS_N]
for ci, (name, n_eval) in enumerate(conds):
    lab = f"{MISS_KEYS[name][0]} {MISS_KEYS[name][1]:g}"
    for j, k in enumerate(CLASSES):
        for bound, dx, mk, fill in (("pixel-level U_delta", -0.18, "o", "white"), ("U_delta_patient", 0.18, "D", None)):
            r_ = MISS_DF[(MISS_DF.level == lab) & (MISS_DF["class"] == CLASSES[k]) & (MISS_DF.n_eval == n_eval)
                         & (MISS_DF.bound == bound)].iloc[0]
            axr.errorbar(ci + dx + (j - 1) * 0.08, r_["miss rate"], yerr=r_["se"], fmt=mk, color=CLASS_COL[k],
                         mfc=fill or CLASS_COL[k], mew=2, ms=9, capsize=3, lw=1.5)
axr.axhline(DELTA, color=COL["ref"], ls="-.", lw=2)
axr.text(len(conds) - 0.5, DELTA + 0.012, r"$\delta$ = 0.05", ha="right", fontsize=14, color=COL["ref"])
axr.set_xticks(range(len(conds)), [f"$n$={n}" for _, n in conds], fontsize=13)
for gi, nm in enumerate(MISS_KEYS):
    axr.text(2 * gi + 0.5, -0.075, "clean" if nm == "clean" else f"{nm.upper() if nm == 'awgn' else nm} {MISS_KEYS[nm][1]:g}",
             transform=axr.get_xaxis_transform(), ha="center", va="top", fontsize=14)
axr.set_ylim(-0.02, 1.0); axr.set_ylabel(f"miss rate over {MISS_DRAWS} draws")
axr.plot([], [], "o", color=COL["ref"], mfc="white", mew=2, label=r"pixel-level $U_\delta$")
axr.plot([], [], "D", color=COL["ref"], label=r"$U_{\delta,patient}$")
axr.legend(loc="upper right", frameon=False, fontsize=13)
axr.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "acdc_margin_scale")

# %% [markdown]
# **Figure `acdc_margin_scale`.** Left: the margin of each bound over its own point estimate on clean
# data, against the number of evaluation patients (log axes), per class: patient-level betting (filled,
# solid) and the package's pixel-level `U_delta` (open, dashed). Right: how often each bound falls below
# its population target, when 20 or 40 patients are drawn from repetition 0's 100 evaluation patients
# (clean, middle AWGN and middle blur level); bars are binomial standard errors, colours the classes
# (LV blue, MYO orange, RV green). The miss rates are markers only: the conditions are not a series. The
# dash-dotted line is `delta`.

# %%
fig, ax = plt.subplots()
n_fit, n_ext = np.geomspace(50, 100, 50), np.geomspace(100, 1000, 200)
for k, cname in CLASSES.items():
    d_, b_ = MARGIN_DF[MARGIN_DF.k == k], BOOT_DF[BOOT_DF.k == k]
    A_, b50 = FITS[k]["power_A_n>=50"], FITS[k]["power_b_n>=50"]
    m100 = float(d_[d_.n == 100].betting.iloc[0])
    ax.plot(d_.n, d_.betting, "-D", color=CLASS_COL[k], lw=1.3, ms=8, zorder=3, label=f"{cname}  (fit $n^{{{b50:.2f}}}$)")
    ax.plot(b_.n, b_["betting (resampled)"], "D", color=CLASS_COL[k], mfc="white", mew=2, ms=8, zorder=3)
    ax.plot(n_fit, A_ * n_fit ** b50, "--", color=CLASS_COL[k], lw=2.4)
    ax.plot(n_ext, A_ * n_ext ** b50, ":", color=CLASS_COL[k], lw=2.4)
    ax.plot(n_ext, m100 * np.sqrt(100 / n_ext), "-.", color=CLASS_COL[k], lw=1.4, alpha=0.75)
for target in (0.05, 0.03, 0.02):
    ax.axhline(target, color=COL["ref"], ls="-", lw=0.8)
    ax.text(1000, target * 1.03, f"{target:g}", ha="right", va="bottom", fontsize=13, color=COL["ref"])
    for k in CLASSES:
        n_req = n_needed(FITS[k], target, 1.0, BOOT_DF[BOOT_DF.k == k])["power law, n >= 50"]
        if n_req <= n_ext[-1]:
            ax.plot(n_req, target, "x", color=CLASS_COL[k], ms=12, mew=2.5, zorder=4)
ax.axvspan(100, 1000, color=COL["ref"], alpha=0.10, lw=0)
ax.text(108, 0.33, "extrapolation:\nbeyond the 100 patients measured", fontsize=14, va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8, 1000)
ax.set_xlabel("evaluation patients $n$"); ax.set_ylabel("betting margin of the clean $v$ term")
leg_classes = ax.legend(loc="lower left", frameon=False, fontsize=13, title="class", title_fontsize=13)
ax.add_artist(leg_classes)
g_ = COL["ref"]
style = [plt.Line2D([], [], color=g_, ls="-", marker="D", lw=1.3, label="measured (subsamples of the 100)"),
         plt.Line2D([], [], color=g_, ls="", marker="D", mfc="white", mew=2, label="resampled with replacement (markers only)"),
         plt.Line2D([], [], color=g_, ls="--", lw=2.4, label="power law fitted on $n$ = 50 to 100"),
         plt.Line2D([], [], color=g_, ls=":", lw=2.4, label="the same fit, extrapolated"),
         plt.Line2D([], [], color=g_, ls="-.", lw=1.4, label=r"$1/\sqrt{n}$ from $n$ = 100 (conservative)"),
         plt.Line2D([], [], color=g_, ls="", marker="x", ms=11, mew=2.5, label="fit reaches 0.05, 0.03, 0.02")]
ax.legend(handles=style, loc="lower left", bbox_to_anchor=(0.2, 0.0), frameon=False, fontsize=12)
fig.tight_layout()
save(fig, "acdc_sample_size")

# %% [markdown]
# **Figure `acdc_sample_size`.** The betting margin of the clean `v` term against `n`, per class (colour).
# Diamonds joined by a thin line: measured, by subsampling repetition 0's 100 evaluation patients (an
# ordered series in `n`). Open diamonds, not joined: resampled with replacement from the same 100 values
# (a check, not a measurement; each point is its own resampling). Dashed: the power law fitted on
# `n = 50` to 100; dotted: the same law beyond `n = 100`; dash-dotted: the conservative `1/sqrt(n)`
# continuation of the measured margin at `n = 100`. Crosses mark where the fitted law reaches a margin of
# 0.05, 0.03 and 0.02. The shaded region is an extrapolation.

# %% [markdown]
# ## 9. Early warning without labels
#
# As in section 6 of the MNIST notebook: can an alarm be raised before labelled data arrive? Three
# adaptations to segmentation.
#
# **The unit is the patient.** In each repetition the 100 evaluation patients are split by group into 50
# **reference** patients (10 per group), scored clean at commissioning, and 50 **monitoring** patients,
# scored at each level. (The bounds of sections 5 and 6 keep using all 100.) Every test compares
# per-patient statistics, reference against monitoring, so at level 0 both sides are clean, different
# patients and every alarm is a false alarm.
#
# **Without labels, the class of a pixel is unknown.** The per-class quantifiers are calibrated on, and
# apply to, the pixels whose *true* class is `k`. The formal label-free rule of the MNIST notebook, A2
# (`U_inf_lower` above the commissioning `U_delta`), is a lower bound on the population that `U_delta`
# bounds from above, here the true-class-`k` pixels, and which pixels those are is exactly what the
# labels say. So in segmentation even the label-free lower bound needs labels to define its population,
# and **A2 has no rigorous counterpart here**; none is computed. What a label-free rule can use is the
# pixels the model **predicts** as LV, MYO or RV, each judged by the frozen quantifier of its predicted
# class:
#
# - **A1**, the per-patient empty-set fraction of the predicted-foreground pixels; one-sided Mann-Whitney
#   (monitoring larger), at 5 %;
# - **A3**, the per-patient mean max-softmax of the predicted-foreground pixels; one-sided Mann-Whitney
#   (monitoring smaller), at 5 %;
# - **A4**, the per-patient predicted-foreground fraction of the image, the segmented area: a quantity a
#   clinical user would see drift; two-sided Mann-Whitney, at 5 %.
#
# A patient with no predicted-foreground pixel has no A1 or A3 value and is left out of those two tests.
#
# **Labelled references.** **L1b**, the per-patient coverage `cov_ik / n_ik`, one-sided Mann-Whitney
# (monitoring smaller); **L2 formal**, the monitoring patients' betting lower bound on the patient-averaged
# error above the reference patients' betting upper bound. Both per class at `0.05 / 3` (Bonferroni over
# the three classes); the rule alarms when any class rejects. A Mann-Whitney p-value that is undefined
# (every value tied on both sides) counts as no rejection.

# %%
N_SLICE_PAT = np.array([b - a for a, b in PAT_RANGE])
PRED = {key: SMX[key].argmax(1).astype(np.int8) for key in KEYS}


def label_free(c: dict, key) -> dict:
    """Per evaluation patient, from the model's output only: A1, A3 and A4 (section 9)."""
    ev = c["ev"]
    r = np.flatnonzero(np.isin(FG_P, ev) & (PRED[key] > 0))
    fp = FP[key]
    f_ = np.isin(fp["patient"], ev)
    smx = np.concatenate([SMX[key][r], fp["smx"][f_]])
    pos = np.searchsorted(ev, np.concatenate([FG_P[r], fp["patient"][f_]]))
    j = smx.argmax(1)
    empty = np.zeros(len(smx), bool)
    for k in CLASSES:
        m = j == k
        y_pred, S = c["classes"][k]["uq"].predict(smx[m].astype(np.float64))
        assert np.all(y_pred == k)
        empty[m] = S.sum(1) == 0
    n = np.bincount(pos, minlength=len(ev))
    with np.errstate(invalid="ignore", divide="ignore"):
        a1 = np.bincount(pos, empty, len(ev)) / n
        a3 = np.bincount(pos, smx.max(1), len(ev)) / n
    return {"A1": a1, "A3": a3, "A4": n / (N_SLICE_PAT[ev] * HW), "n_pred_fg": n}


def labelled(rep: int, key) -> dict:
    """Per evaluation patient and class: coverage and error, from section 4's counts."""
    out = {}
    for k in CLASSES:
        per = FROZEN[rep][key]["classes"][k]["_per"]
        out[k] = {"cov": per["cov"] / per["n"], "err": 1 - per["correct"] / per["n"]}
    return out


def ew_split(ev: np.ndarray, seed) -> tuple[np.ndarray, np.ndarray]:
    """Positions (0..99, within ev) of the reference and the monitoring patients, stratified by group, each in a
    random order fixed by the seed (the betting order of L2 formal)."""
    rng = np.random.default_rng(seed)
    ref_, mon_ = [], []
    for g in GROUPS:
        perm = rng.permutation(np.flatnonzero(PGROUP[ev] == g))
        ref_.extend(perm[:EW_REF_PER_GROUP]); mon_.extend(perm[EW_REF_PER_GROUP:])
    return rng.permutation(ref_), rng.permutation(mon_)


def _mw(x, y, alternative) -> float:
    p = mannwhitneyu(x, y, alternative=alternative, nan_policy="omit").pvalue
    return 1.0 if np.isnan(p) else float(p)


EW_RULES = {"A1": "A1  empty fraction", "A3": "A3  mean max-softmax", "A4": "A4  segmented area",
            "L1b": "L1b  coverage (labels)", "L2_formal": "L2 formal (labels)"}


def alarms(lf_ref, lf_mon, lab_ref, lab_mon, ref_pos, mon_pos) -> dict:
    """Every rule for one comparison: reference patients (clean) against monitoring patients (current level)."""
    a_k = EW_ALPHA / len(CLASSES)
    p = {"A1": _mw(lf_mon["A1"][mon_pos], lf_ref["A1"][ref_pos], "greater"),
         "A3": _mw(lf_mon["A3"][mon_pos], lf_ref["A3"][ref_pos], "less"),
         "A4": _mw(lf_mon["A4"][mon_pos], lf_ref["A4"][ref_pos], "two-sided")}
    p_l1b = {k: _mw(lab_mon[k]["cov"][mon_pos], lab_ref[k]["cov"][ref_pos], "less") for k in CLASSES}
    l2 = {k: (hedged_lower(lab_mon[k]["err"][mon_pos], a_k), 1 - hedged_lower(1 - lab_ref[k]["err"][ref_pos], a_k))
          for k in CLASSES}
    return {**{r: p[r] < EW_ALPHA for r in ("A1", "A3", "A4")},
            "L1b": any(v < a_k for v in p_l1b.values()), "L2_formal": any(lo > hi for lo, hi in l2.values()),
            "_p": p, "_p_l1b": p_l1b, "_l2": l2}


t0 = time.time()
LF = [{key: label_free(c, key) for key in KEYS} for c in COMM]
LAB = [{key: labelled(rep, key) for key in KEYS} for rep in range(L)]
EW_SPLITS = [ew_split(c["ev"], [SEED, c["rep"], 9]) for c in COMM]
EW = {deg: [[alarms(LF[rep][CK], LF[rep][level_key(deg, lv)], LAB[rep][CK], LAB[rep][level_key(deg, lv)], *EW_SPLITS[rep])
             for lv in LEVELS[deg]] for rep in range(L)] for deg in LEVELS}
T_EW = time.time() - t0


def cp_interval(x: int, n: int, a: float = 0.05) -> tuple[float, float]:
    """Two-sided (1 - a) Clopper-Pearson interval for x successes in n."""
    lo = 0.0 if x == 0 else float(beta_dist.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta_dist.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


EW_RATE = {deg: {r: np.array([[EW[deg][rep][j][r] for j in range(len(LEVELS[deg]))] for rep in range(L)], float).mean(0)
                 for r in EW_RULES} for deg in LEVELS}
# level 0 is the same comparison for every degradation (clean against clean)
assert all(np.array_equal([EW[d][rep][0][r] for rep in range(L)], [EW["awgn"][rep][0][r] for rep in range(L)])
           for d in LEVELS for r in EW_RULES)
FALSE_ALARM = {r: {"alarms": int(sum(EW["awgn"][rep][0][r] for rep in range(L))), "of": L,
                   "CP95": cp_interval(int(sum(EW["awgn"][rep][0][r] for rep in range(L))), L)} for r in EW_RULES}

t0 = time.time()
NULL = {r: 0 for r in EW_RULES}
for i in range(NULL_SPLITS):                       # random reference/monitoring splits on clean data
    rep = i % L
    a_ = alarms(LF[rep][CK], LF[rep][CK], LAB[rep][CK], LAB[rep][CK], *ew_split(COMM[rep]["ev"], [SEED, 10, i]))
    for r in EW_RULES:
        NULL[r] += a_[r]
NULL_CHECK = {r: {"alarms": int(v), "of": NULL_SPLITS, "rate": v / NULL_SPLITS, "CP95": cp_interval(int(v), NULL_SPLITS)}
              for r, v in NULL.items()}
T_NULL = time.time() - t0
print(f"early warning: {L} repetitions in {T_EW:.0f}s, {NULL_SPLITS} null splits in {T_NULL:.0f}s")
pd.DataFrame.from_dict(orient="index", data={EW_RULES[r]: {f"false alarms, level 0 ({L} repetitions)": f"{FALSE_ALARM[r]['alarms']} / {L}",
                            "its 95 % CP interval": "[{:.3f}, {:.3f}]".format(*FALSE_ALARM[r]["CP95"]),
                            f"null check ({NULL_SPLITS} random splits)": f"{NULL_CHECK[r]['alarms']} / {NULL_SPLITS}",
                            "rate": round(NULL_CHECK[r]["rate"], 4),
                            "its 95 % CP interval ": "[{:.3f}, {:.3f}]".format(*NULL_CHECK[r]["CP95"])}
                                                        for r in EW_RULES})

# %% [markdown]
# The label-free statistics themselves, on the monitoring patients (mean over patients, then over
# repetitions), and the alarm rate of every rule by level:


# %%
def lf_mean(deg: str, stat: str) -> np.ndarray:
    return np.array([[np.nanmean(LF[rep][level_key(deg, lv)][stat][EW_SPLITS[rep][1]]) for lv in LEVELS[deg]]
                     for rep in range(L)]).mean(0)


for deg, deg_levels in LEVELS.items():
    t_ = pd.DataFrame({"level": deg_levels, "A1 empty frac": lf_mean(deg, "A1"), "A3 max-softmax": lf_mean(deg, "A3"),
                       "A4 area": lf_mean(deg, "A4"),
                       "patients without pred. fg": [np.mean([(LF[rep][level_key(deg, lv)]["n_pred_fg"] == 0).sum()
                                                              for rep in range(L)]) for lv in deg_levels],
                       **{f"rate {r}": EW_RATE[deg][r] for r in EW_RULES}}).set_index("level")
    print(f"\n{deg}:"); print(t_.to_string(float_format=lambda v: f"{v:.4g}"))

# %%
# the reference palette's first five categorical slots, in an order validated for colour-blind separation
EW_STYLE = {"A1": (COL["empty"], "-", "s"), "A3": (COL["U"], "-", "v"), "A4": (COL["recal"], "-", "o"),
            "L1b": (COL["U_delta"], ":", "P"), "L2_formal": (COL["binom"], ":", "x")}
fig, axs_a = plt.subplots(1, 3, sharey=True, figsize=(12.8, 6.2), gridspec_kw={"width_ratios": [3, 2.3, 1.2], "wspace": 0.08})
for ax, deg in zip(axs_a, LEVELS, strict=True):
    for ri_, (r, (c_, ls, mk)) in enumerate(EW_STYLE.items()):
        off = (ri_ - 2) * 0.012                          # a small vertical offset, so that coinciding rules stay visible
        ax.plot(LEVELS[deg], EW_RATE[deg][r] + off, ls, color=c_, marker=mk, ms=9, mew=2, label=EW_RULES[r] if deg == "awgn" else None)
    ax.axhline(EW_ALPHA, color=COL["ref"], lw=1.8, ls="-.", label="5 %" if deg == "awgn" else None)
    ax.set_xlabel(DEG_X[deg]); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(LEVELS[deg][::2] if len(LEVELS[deg]) > 7 else LEVELS[deg])
axs_a[0].set_ylabel(f"alarm rate over {L} repetitions")
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=14)
save(fig, "acdc_alarm_rates")

# %% [markdown]
# **Figure `acdc_alarm_rates`.** Fraction of repetitions in which each rule alarms, by level, for AWGN,
# Gaussian blur and the bias field. Solid: label-free rules on the predicted-foreground pixels (A1, A3,
# A4); dotted: the labelled references (L1b, L2 formal), per class at `0.05 / 3`. At level 0 every alarm is
# a false alarm; the dash-dotted line is 5 %. With `L` repetitions a rate is resolved only to `1/L`. The
# lines are offset vertically by -0.024 to +0.024, so that rules with equal rates do not hide each other;
# the exact rates are in the table above.
#
# On clean data no label-free rule fired in the 20 repetitions; on 500 random reference/monitoring splits of
# clean data the rates are 3.6 % (A1, A3), 2.0 % (A4), 3.8 % (L1b) and 0 % (L2 formal), all compatible with
# the nominal 5 % or below it (table above). Against the expectation, **A1 does react**: empty sets are
# rare (about 5e-5 of the predicted-foreground pixels on clean data), but their per-patient fraction rises
# consistently enough to be detected: 70 % of repetitions at AWGN 0.1, 45 % at blur 2.0, 100 % beyond. **A3 reacts first** among the label-free
# rules, as early as L1b under AWGN. **A4**, the segmented area, needs AWGN 0.3 or blur 2.5. **L2 formal**,
# the only rule that states that the error itself got worse, needs AWGN 0.3 (where the three errors have
# risen by 0.25 to 0.29) or blur 3 (MYO up by 0.19, in 60 % of repetitions). Under the bias field the label-free rules stay at or below 5 %, and L1b fires in
# 15 % and 25 % of repetitions at `c` = 0.25 and 0.5, where the error does rise slightly (by up to 0.0025
# for LV and RV). A label-free alarm says "the output changed; verify with labels", never "the model is
# still fine".

# %% [markdown]
# ## 10. Confident errors, resolution and calibration
#
# The error types, from the same frozen sets: the share of each class's top-1 errors that are confident
# (non-empty set without `k`), flagged (empty set) or covered but wrong (multi-label set). They are
# about 100 % confident, so a table rather than a figure.

# %%
ET = []
for deg, deg_levels in LEVELS.items():
    for j, lv in enumerate(deg_levels):
        row = {"degradation": deg, "level": lv}
        for k, cname in CLASSES.items():
            for f, lab in (("share_conf_err", "confident"), ("share_flagged", "flagged"), ("share_cov_wrong", "cov-wrong")):
                row[f"{cname} {lab}"] = float(np.nanmean(col(deg, k, f)[:, j]))
            row[f"{cname} empty frac"] = float(col(deg, k, "empty_frac")[:, j].mean())
        ET.append(row)
ET_DF = pd.DataFrame(ET).set_index(["degradation", "level"])
ET_DF.round(4)

# %% [markdown]
# Resolution, per class: the AUROC of max-softmax for telling correctly from wrongly labelled pixels
# (ties count one half), on each repetition's pooled evaluation pixels. Calibration: the top-label ECE
# on all foreground evaluation pixels, 15 equal-mass bins with tied values never split, as in the MNIST
# notebook. Both are descriptive, not guarantees.

# %%
# hand checks of the AUROC tie convention and of the binning (as in the MNIST notebook)
assert auroc(np.array([1.0, 1.0, 0.9]), np.array([1.0, 0.5])) == (0.5 + 1 + 0.5 + 1 + 0 + 1) / 6
assert top_label_ece(np.array([1.0] * 10 + [0.6] * 5), np.array([True] * 9 + [False] + [True] * 3 + [False] * 2))[1] == 2


def ece_col(deg: str) -> np.ndarray:
    return np.array([[FROZEN[rep][level_key(deg, lv)]["ece"] for lv in LEVELS[deg]] for rep in range(L)])


RES = []
for deg, deg_levels in LEVELS.items():
    for j, lv in enumerate(deg_levels):
        row = {"degradation": deg, "level": lv, "ECE": float(ece_col(deg)[:, j].mean())}
        for k, cname in CLASSES.items():
            row[f"{cname} AUROC"] = float(col(deg, k, "auroc")[:, j].mean())
            row[f"{cname} maxsm correct / wrong"] = (f"{col(deg, k, 'maxsm_correct')[:, j].mean():.4f} / "
                                                          f"{col(deg, k, 'maxsm_wrong')[:, j].mean():.4f}")
        RES.append(row)
RES_DF = pd.DataFrame(RES).set_index(["degradation", "level"])
RES_DF.round(4)

# %%
fig, axs_ = plt.subplots(1, 3, sharey=True)
for ax, k in zip(axs_, CLASSES, strict=True):
    for deg, mk, bs in (("awgn", "o", "fill"), ("blur", "s", "outline")):
        line_band(ax, ref(deg, k), col(deg, k, "auroc"), "AWGN" if deg == "awgn" else "Gaussian blur", DEG_COL[deg],
                  marker=mk, band_style=bs)
    xb, yb = ref("blur", k), col("blur", k, "auroc").mean(0)
    for j in range(len(xb) - 1):                           # blur first lowers LV's error: arrow every step of the path
        ax.annotate("", xy=(xb[j + 1], yb[j + 1]), xytext=(xb[j], yb[j]),
                    arrowprops={"arrowstyle": "-|>", "color": DEG_COL["blur"], "lw": 1.4, "mutation_scale": 15,
                                "shrinkA": 5, "shrinkB": 5})
    for j, lv in ((0, LEVELS["blur"][0]), (len(xb) - 1, LEVELS["blur"][-1])):   # and label its two ends
        ax.annotate(f"blur {lv:g}", (xb[j], yb[j]), xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=12, color=DEG_COL["blur"])
    ax.set_xscale("log")                                    # spreads the low-error levels, where blur doubles back
    ax.set_xticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"]); ax.minorticks_off()
    ax.axhline(0.5, color=COL["ref"], lw=1.5, ls="-.", label="no resolution (0.5)")
    ax.text(0.5, 1.02, CLASSES[k], transform=ax.transAxes, ha="center", fontsize=17)
    ax.set_xlabel("$U_E$ (patient-averaged)", fontsize=16)
axs_[0].set_ylabel("AUROC of max-softmax, correct vs wrong")
axs_[0].legend(loc="lower left", frameon=False)
fig.tight_layout()
save(fig, "acdc_resolution")

# %% [markdown]
# **Figure `acdc_resolution`.** Per class, the AUROC of max-softmax for telling correctly from wrongly
# labelled pixels, against the patient-averaged reference error at each level (a common severity scale
# for the two degradations): mean over repetitions, with the 5th-95th percentiles shaded for AWGN and
# drawn as thin lines for blur, with the error on a log axis. Points of one degradation are joined in the
# order of increasing level; blur's steps carry arrows in that direction and its two ends their `sigma`
# (px), because blur first lowers LV's and RV's error and its path doubles back. Below the dash-dotted 0.5 line, wrong pixels are *more* confident than
# correct ones.


# %%
def severity_picks(deg: str) -> list[float]:
    return [0.0, MIDDLE[deg], LEVELS[deg][-1]]


def doubt_bins(u: np.ndarray, wrong: np.ndarray):
    """Equal-mass bins on u = 1 - max-softmax (ties never split): per bin the mean u, the error rate and the count."""
    b = equal_mass_bins(u)
    ks = np.unique(b)
    return (np.array([u[b == q_].mean() for q_ in ks]), np.array([wrong[b == q_].mean() for q_ in ks]),
            np.array([(b == q_).sum() for q_ in ks]))


RELIABILITY_LEVELS, REL = {}, {}
fg_ev0 = np.isin(FG_P, ev0)
for deg in ("awgn", "blur"):
    RELIABILITY_LEVELS[deg] = severity_picks(deg)
    for lv in RELIABILITY_LEVELS[deg]:
        smx = SMX[level_key(deg, lv)][fg_ev0]
        u = doubt(smx)
        REL[(deg, lv)] = (*doubt_bins(u, smx.argmax(1) != FG_Y[fg_ev0]), float((u == 0).mean()))
u_lo = min(r_[0][r_[0] > 0].min() for r_ in REL.values()) / 3            # the axis reaches the most confident bin
fig = plt.figure(figsize=(12.8, 7.6))
gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.12, wspace=0.1)
for ri, (deg, sym) in enumerate((("awgn", r"\sigma"), ("blur", r"\sigma_{blur}"))):
    for ci, lv in enumerate(RELIABILITY_LEVELS[deg]):
        ub, eb, cnt, z0 = REL[(deg, lv)]
        ubd = np.maximum(ub, u_lo)                                       # a bin of u = 0 only (underflow) sits at the edge
        ax = fig.add_subplot(gs[2 * ri, ci]); axc = fig.add_subplot(gs[2 * ri + 1, ci], sharex=ax)
        ax.plot([1e-3, 1], [1e-3, 1], color=COL["ref"], lw=1.5, ls="-.")
        ax.plot(ubd, eb, "-o", color=COL["U"], ms=7, lw=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1.0, u_lo); ax.set_ylim(3e-3, 1.5)                  # reversed: confidence grows to the right
        ue = np.mean([REF[level_key(deg, lv)][k]["err_patient"] for k in CLASSES])
        top = eb[0] < 0.06                       # bin 0 is the most confident (rightmost): the corner the curve leaves free
        ax.text(0.97, 0.95 if top else 0.05, f"${sym}$ = {lv:g}, mean $U_E$ = {ue:.2f}\n{len(ub)} bins"
                + (f"; {z0:.1%} at u = 0" if z0 >= 5e-4 else ""), transform=ax.transAxes, ha="right",
                va="top" if top else "bottom", fontsize=12.5)
        axc.bar(ubd, cnt, width=ubd * 0.6, color=COL["U"])
        axc.set_yscale("log"); axc.set_ylim(1, cnt.sum() * 1.5)
        ax.tick_params(labelbottom=False)
        if ci:
            ax.tick_params(labelleft=False); axc.tick_params(labelleft=False)
        else:
            ax.set_ylabel("observed error", fontsize=15); axc.set_ylabel("count", fontsize=14)
        if ri == 0:
            axc.tick_params(labelbottom=False)
fig.supxlabel(r"$u$ = 1 - max-softmax of the bin (log, reversed: more confident to the right)", fontsize=17, y=0.015)
save(fig, "acdc_reliability")
print("reliability levels (clean, middle, most severe):", RELIABILITY_LEVELS)

# %% [markdown]
# **Figure `acdc_reliability`.** Top row: AWGN; bottom row: Gaussian blur. Columns: clean, the middle
# level of section 7 and the most severe level. On all foreground pixels of repetition 0's 100 evaluation
# patients, the pixels are put in 15 equal-mass bins of `u = 1 - max-softmax` (computed from the other
# three softmax entries, so the values that float32 rounds to a max-softmax of exactly 1 are still
# ordered); per bin, the observed top-1 error against the bin's mean `u`, both on log axes, `u` reversed so
# that confidence grows to the right. The dash-dotted diagonal is perfect calibration (error = `u`). The
# strip under each panel gives the pixel count per bin (equal by construction, log scale). Every bin lies
# far above the diagonal: on clean data the most confident bins, at `u` below `1e-20`, are still wrong
# about 1 % of the time. Under strong AWGN the curve is flat: confidence no longer orders the errors.
#
# The earlier version of this figure, with the max-softmax itself on a linear axis, piled every bin at
# a confidence of about 1; on the `u` axis the bins spread out, so it is kept rather than replaced by the
# simpler distribution plot.

# %% [markdown]
# ## Persist the results
#
# `acdc_periodic_verification_results_<mode>.json` holds every number and the provenance. Timestamps,
# run times, machine paths and the commit are kept apart under `volatile`, so two runs with the same seed
# can be compared on everything else.

# %%
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_") and k != "uq"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return clean(o.item())
    if isinstance(o, float) and np.isnan(o):
        return "nan"
    return o


FIELDS = ("U_emp_patient", "err_eval_patient", "U_delta_patient", "errbound_patient", "U_delta_patient_hoeffding",
          "errbound_patient_hoeffding", "U_delta_pixel", "U_emp_pixel", "err_eval_pixel", "coverage_pixel",
          "empty_frac", "multi_frac", "share_conf_err", "share_flagged", "share_cov_wrong", "hesitant_frac", "auroc",
          "maxsm_correct", "maxsm_wrong")
SUMMARY = {"commissioning": {CLASSES[k]: {"alpha0": band([c["classes"][k]["alpha0"] for c in COMM]),
                                          "threshold": band([c["classes"][k]["threshold"] for c in COMM]),
                                          "status": sorted({c["classes"][k]["status"] for c in COMM}),
                                          "status_counts": {s: sum(c["classes"][k]["status"] == s for c in COMM)
                                                            for s in sorted({c["classes"][k]["status"] for c in COMM})}}
                                 for k in CLASSES},
           "levels": {deg: {f"{lv:g}": {"ece": band(ece_col(deg)[:, j]),
                                        **{CLASSES[k]: {"U_E_patient": REF[level_key(deg, lv)][k]["err_patient"],
                                                        "U_E_pixel": REF[level_key(deg, lv)][k]["err_pixel"],
                                                        **{f: band(np.nan_to_num(col(deg, k, f)[:, j], nan=-1.0))
                                                           for f in FIELDS}}
                                           for k in CLASSES}}
                            for j, lv in enumerate(LEVELS[deg])} for deg in LEVELS},
           "middle_levels": MIDDLE, "miss": MISS, "margin": MARGIN_DF.drop(columns="k").to_dict("records"),
           "margin_fits": {CLASSES[k]: f for k, f in FITS.items()}, "n_needed": NEEDED,
           "structural": STRUCT, "simulation": SIM, "loader_check": LOADER_CHECK, "cache_bytes": CACHE_BYTES,
           "reliability_levels": RELIABILITY_LEVELS, "eval_check": EVAL_CHECK, "clean_snapshot": SNAP,
           "bias_flat": BIAS_FLAT, "overlay": OVERLAY,
           "early_warning": {"rates": {deg: {r: v.tolist() for r, v in rr.items()} for deg, rr in EW_RATE.items()},
                             "false_alarm_level0": FALSE_ALARM, "null_check": NULL_CHECK,
                             "label_free_means_monitoring": {deg: {st: lf_mean(deg, st).tolist() for st in ("A1", "A3", "A4")}
                                                             for deg in LEVELS},
                             "p_values": {deg: [[{**a_["_p"], **{f"L1b_{CLASSES[k]}": v for k, v in a_["_p_l1b"].items()},
                                                  **{f"L2_{CLASSES[k]}": list(v) for k, v in a_["_l2"].items()}}
                                                 for a_ in row] for row in EW[deg]] for deg in LEVELS}}}
RESULTS = clean({"meta": {"python": sys.version.split()[0], "jax": jax.__version__, "jax_backend": jax.default_backend(),
                          "torch": torch.__version__, "monai": monai.__version__, "nibabel": nib.__version__,
                          "numpy": np.__version__, "utrace": utrace.__version__, "params": PARAMS,
                          "model_sha256": MODEL_SHA256, "model_version": MODEL_VERSION,
                          "n_patients": N_PAT, "n_slices": N_SLICES, "n_foreground_pixels": len(FG_Y)},
                 "commissioning": [{"rep": c["rep"], "cal": c["cal"], "tun": c["tun"], "ev": c["ev"],
                                    "classes": {CLASSES[k]: v for k, v in c["classes"].items()}} for c in COMM],
                 "frozen": [{f"{key[0]}_{key[1]:g}": {"ece": r["ece"], "ece_bins": r["ece_bins"],
                                                      **{CLASSES[k]: v for k, v in r["classes"].items()}}
                             for key, r in fr.items()} for fr in FROZEN],
                 "references": {f"{key[0]}_{key[1]:g}": {CLASSES[k]: v for k, v in r.items()} for key, r in REF.items()},
                 "summary": SUMMARY})
RESULTS["volatile"] = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "utrace_commit": COMMIT, "src_dirty": DIRTY,
                       "host": platform.node(), "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
                       "data_dir": str(DATA_DIR), "model_dir": str(MODEL_DIR), "inferred_now": INFER_SECONDS,
                       "seconds": {"total": time.time() - T_START, "load": T_LOAD, "cache": T_CACHE, "simulation": T_SIM,
                                   "commissioning": T_COMMISSION, "frozen": T_FROZEN, "miss": T_MISS, "margin": T_MARGIN,
                                   "early_warning": T_EW, "null_check": T_NULL}}
RESULTS_PATH.write_text(json.dumps(RESULTS, sort_keys=True, indent=1))
NONVOLATILE_SHA256 = hashlib.sha256(json.dumps({k: v for k, v in RESULTS.items() if k != "volatile"}, sort_keys=True).encode()).hexdigest()
print(RESULTS_PATH, "| sha256 of the non-volatile part:", NONVOLATILE_SHA256)
print("figures:", *sorted(str(p) for p in FIG.iterdir()), sep="\n  ")

# %% [markdown]
# ## 11. Summary
#
# ### Facts
#
# Read off the results of this run: mean and [5th, 95th percentile] over the `L` repetitions.

# %%
def pm(b, f=".4f"):
    return f"{b[0]:{f}} [{b[1]:{f}}, {b[2]:{f}}]"


S = SUMMARY
print(f"QUICK={QUICK}, L={L}, {N_PAT} patients, split {SPLIT_PER_GROUP} per group")
for k, cname in CLASSES.items():
    c_ = S["commissioning"][cname]
    print(f"  {cname:>3}: alpha0 {pm(c_['alpha0'])}, threshold 1 - q_hat {pm(c_['threshold'])}, status {c_['status_counts']}")
for deg, deg_levels in LEVELS.items():
    print(f"\n{deg}:  per class: U_E patient | U_E pixel | U_delta_patient | errbound_patient | pixel U_delta | "
          "confident share | AUROC ; ECE")
    for lv in deg_levels:
        e = S["levels"][deg][f"{lv:g}"]
        cells = [f"{e[CLASSES[k]]['U_E_patient']:.3f} {e[CLASSES[k]]['U_E_pixel']:.3f} {e[CLASSES[k]]['U_delta_patient'][0]:.3f} "
                 f"{e[CLASSES[k]]['errbound_patient'][0]:.3f} {e[CLASSES[k]]['U_delta_pixel'][0]:.3f} "
                 f"{e[CLASSES[k]]['share_conf_err'][0]:.3f} {e[CLASSES[k]]['auroc'][0]:.3f}" for k in CLASSES]
        print(f"  {lv:>5g}  " + "  |  ".join(cells) + f"  ;  {e['ece'][0]:.4f}")
print("\nmiss rates (section 8):")
print(MISS_DF.pivot_table(index=["level", "n_eval"], columns=["bound", "class"], values="miss rate").round(3).to_string())
print("\nbetting margin at n = 100 (clean):", {r["class"]: round(r["betting"], 4) for r in MARGIN if r["n"] == 100},
      "| pixel-level:", {r["class"]: f"{r['pixel-level']:.2e}" for r in MARGIN if r["n"] == 100})
print("fitted power-law exponent:", {CLASSES[k]: round(f["power_b"], 3) for k, f in FITS.items()},
      "| on n >= 50:", {CLASSES[k]: round(f["power_b_n>=50"], 3) for k, f in FITS.items()})
neg = {deg: {CLASSES[k]: float((col(deg, k, "U_delta_pixel") < ref(deg, k)).mean()) for k in CLASSES} for deg in LEVELS}
print("fraction of (repetition, level) cells with the pixel-level U_delta below the patient-averaged U_E:", neg)
negx = {deg: {CLASSES[k]: float((col(deg, k, "U_delta_pixel") < ref(deg, k, "err_pixel")).mean()) for k in CLASSES}
        for deg in LEVELS}
print("... with the pixel-level U_delta below the pixel-weighted U_E (its own estimand):", negx)
negp = {deg: {CLASSES[k]: float((col(deg, k, "U_delta_patient") < ref(deg, k)).mean()) for k in CLASSES} for deg in LEVELS}
print("... and with U_delta_patient below it:", negp)
print("\neval() against train mode:", EVAL_CHECK)
print("clean snapshot (U_E | err eval | U_emp | U_delta_patient | pixel U_delta):",
      {c_: " | ".join([f"{v['U_E']:.4f}"] + [pm(v[f]) for f in ("err_eval_patient", "U_emp_patient", "U_delta_patient",
                                                                 "U_delta_pixel")]) for c_, v in SNAP.items()})
print("bias field, change from clean:", BIAS_FLAT)
print("overlay slice:", {k_: OVERLAY[k_] for k_ in ("patient", "group", "ed_slice", "clean_mean_error")})
for name in MISS_KEYS:
    print(f"  {name}: slice error {OVERLAY[name]['slice_error']}, outcomes {OVERLAY[name]['outcome_counts']}, "
          f"median u wrong / correct {OVERLAY[name]['median_u_wrong_fg']:.2e} / {OVERLAY[name]['median_u_correct_fg']:.2e}")
print("\nearly warning, false alarms at level 0 and null check:")
for r in EW_RULES:
    print(f"  {r:>9}: {FALSE_ALARM[r]['alarms']}/{L} ; null {NULL_CHECK[r]['alarms']}/{NULL_SPLITS} = "
          f"{NULL_CHECK[r]['rate']:.3f}, CP95 [{NULL_CHECK[r]['CP95'][0]:.3f}, {NULL_CHECK[r]['CP95'][1]:.3f}]")
for deg, deg_levels in LEVELS.items():
    print(f"  {deg} alarm rates by level {deg_levels}:", {r: np.round(EW_RATE[deg][r], 2).tolist() for r in EW_RULES})

# %% [markdown]
# ### Inferences

# %%
# The inferences below are hand-written from one specific full run, identified by the SHA-256 of the
# non-volatile part of its results file. If THIS run differs (mode, seed, parameters, versions, code),
# they may not follow from the numbers above.
INFERENCES_WRITTEN_FROM_SHA256 = "db2153862962e4f4f1cfae3b54a6645a899a7ec615a36ffd42f4e62a05ddb144"
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
# `db2153862962e4f4f1cfae3b54a6645a899a7ec615a36ffd42f4e62a05ddb144` (`QUICK=False`, `L=20`, seed
# 20261112, CPU). One model (KM), one dataset (ACDC, 150 patients from one centre, balanced by design
# across five pathology groups), three degradations: these are observations about this setup, not
# properties of the method in general. The reference errors are finite-population values over these 150
# patients.
#
# 1. **Before any degradation, the honest bound costs about 0.05 per class, as much as the errors
#    themselves.** The patient-averaged clean errors are 0.041 (LV), 0.107 (MYO) and 0.069 (RV);
#    `U_empirical` on the evaluation patients matches them to 0.0003, and `U_delta_patient` is 0.096,
#    0.159 and 0.123 (5th-95th percentiles over repetitions within 0.004 of these). The naive pixel-level
#    `U_delta`, 0.037, 0.109 and 0.074, sits on the error, and for LV below it. The operating points are
#    `alpha0` 0.037, 0.108 and 0.072, with the frozen thresholds `1 - q_hat` at 0.500, 0.497 and 0.498.
# 2. **Frozen, the patient-level bound stays above the error at every level; the excess shrinks only at
#    the strongest AWGN.** `U_empirical` follows the reference error at every level (within 0.003). `U_delta_patient`
#    was below the patient-averaged error in 1 of the 200 AWGN cells of LV and in no other (repetition,
#    level) cell; its excess falls from about 0.05 to about 0.02 at AWGN `sigma = 0.75`, where the
#    per-patient `v_ik` crowd towards 0. Hoeffding's bound is 0.06 to 0.10 wider (0.164 against 0.096 for
#    clean LV) and reaches the trivial value 1 for RV at `sigma = 0.75`. The bias field is flat: the
#    errors and the bound move by at most 0.003.
# 3. **The unit of independence is the patient; the pixel-level `U_delta` is far too narrow and misses
#    about half the time.** At `n = 100` its margin is 3.3e-4 (LV), 5.7e-4 (MYO) and 4.8e-4 (RV), against a
#    betting margin of 0.055, 0.052 and 0.054: 90 to 166 times narrower. Drawing 20 or 40 patients from one
#    instrument's 100 evaluation patients, it fell below its population target in 39 % to 56 % of 500 draws
#    at every level tested (clean, AWGN 0.4, blur 3) and for every class, against a nominal 5 %;
#    `U_delta_patient` missed in 1 of 9000 draws (RV, AWGN 0.4, `n = 40`), as expected from sampling a
#    finite population without replacement. In the 20 repetitions the pixel-level `U_delta` was below
#    its own estimand, the pixel-weighted error of the 150 patients, in 33 % to 55 % of the cells of each
#    class and degradation.
# 4. **The betting margin scales almost as `1/n` here, and a margin of 0.05, 0.03 or 0.02 needs about 110,
#    190 or 290 patients.** The local exponent steepens from -0.75 (`n` 10 to 20) to -0.94 (75 to 100); a
#    power law on `n >= 50` has exponent -0.93 and gives 103 to 111, 179 to 192 and 276 to 296 patients;
#    resampling from the 100 gives 176 to 188 and 268 to 285 for the last two. All are
#    **extrapolations**: if the margin slowed to `1/sqrt(n)` right after `n = 100`, the conservative
#    column gives 106 to 122, 295 to 338 and 663 to 760. The two-term form misfits by up to 16 % and is not
#    used.
# 5. **Pixel by pixel, the errors are confident.** On the slice chosen by rule (patient095, ED slice 6),
#    all 183 (clean), 477 (AWGN 0.4) and 454 (blur 3) wrong foreground pixels are confident errors; no
#    pixel of the slice has an empty or a multi-label set. Their median `1 - max-softmax` is 1e-6 to 4e-6,
#    higher than the correct pixels' (1e-12 to 1e-9), but five to six orders of magnitude below the
#    roughly 0.5 at which the frozen set would be empty. Whatever the degradation, nothing in the sets
#    marks these errors.
# 6. **Without labels, three rules see the degradation early, but none can bound the error; in
#    segmentation even the label-free lower bound needs labels.** The MNIST rule A2 has no counterpart
#    here, because the population it bounds (the true-class pixels) is defined by the labels. Comparing 50
#    reference with 50 monitoring patients, the per-patient mean max-softmax (A3) alarmed in 85 % of
#    repetitions at AWGN 0.05 and in all from 0.1, and in all from blur 1.5; the per-patient empty
#    fraction (A1), though empty sets are rare, in 70 % at AWGN 0.1 and in all from 0.15, and from blur
#    2.5; the segmented area (A4) only later, in 35 % at AWGN 0.3 and blur 2.5 and in all at AWGN 0.4 and
#    blur 4. The labelled coverage test L1b
#    alarmed as early as A3 under AWGN and earlier under blur (95 % at 1.0); L2 formal, the only rule that
#    certifies a worse error, only from AWGN 0.3 and blur 3 (60 %). On clean data the false-alarm rates
#    were 0 of 20 for every label-free rule and 1 of 20 for L1b, and on 500 random clean splits 3.6 %
#    (A1, A3), 2.0 % (A4), 3.8 % (L1b) and 0 % (L2 formal). Under the bias field the label-free rules
#    stayed at or below 5 %. These alarms say that the output moved, not by how much the error did.
# 7. **Essentially every error is a confident error, at every level.** The confident share of top-1
#    errors is at least 0.998 in every class and level (mean over repetitions; lowest 0.9987, RV at bias
#    `c = 0.5`; flagged and covered-but-wrong errors at most 0.0009 each), and the empty-set fraction never
#    exceeds 5.3e-4 per repetition while the error goes from 0.04 to 0.92.
# 8. **The degradations differ in which structure they hurt, and in what they do to confidence.** AWGN
#    hurts all three (patient-averaged errors 0.86, 0.84 and 0.92 at `sigma = 0.75`). Blur is
#    class-selective: LV and RV first improve (LV 0.041 to 0.029 at 1.5 px) and reach 0.108 and 0.300 at
#    4 px, while MYO goes to 0.408. The clean AUROC of max-softmax is only 0.75 to 0.79; under AWGN it
#    crosses 0.5 for RV at `sigma = 0.4` (0.492; 0.324 at 0.75) and for MYO at 0.75 (0.477), and LV ends at
#    0.525; under blur it stays at 0.62 or above, but blur never reaches the errors at which AWGN inverts,
#    and for MYO the two curves coincide. Calibration is poor even on clean data: in the reliability
#    diagram on `1 - max-softmax`, the most confident bins are wrong about 1 % of the time at a stated
#    doubt below `1e-20`; top-label ECE is 0.067 clean, 0.871 at AWGN 0.75 and 0.262 at blur 4.
# 9. **KM's frozen instrument is close to the degenerate case of U-TraCE, a bound on the error.** The
#    thresholds `1 - q_hat` lie between 0.4936 and 0.5007 (above 0.5 in 13 of 60 class-repetitions);
#    multi-label sets are at most 4.3e-4 and empty sets at most 5.3e-4 of a class's pixels;
#    `U_delta_patient` and `errbound_patient` differ by at most 1.8e-4. With this model U-TraCE adds
#    nothing to a direct bound on the error; what the experiment adds is the patient as the unit.
# 10. **The paper's script runs KM without `eval()`, and that changes nothing.** KM uses `InstanceNorm2d`
#     without running statistics and dropout with `p = 0`, so the softmax is bitwise the same in both modes.

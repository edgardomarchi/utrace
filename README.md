# U-TraCE
[![Tests](https://github.com/edgardomarchi/utrace/actions/workflows/tests.yml/badge.svg)](https://github.com/edgardomarchi/utrace/actions/workflows/tests.yml)

![Project Status: WIP - Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)

**U**ncertainty **Tra**cking for **C**omplex **E**stimators.

U-TraCE is a Python package for computing estimates and upper confidence bounds on the uncertainty of black-box models: the marginal or class-conditional uncertainty of a classification model with a softmax output, and the uncertainty of a regression model's point predictions with respect to a tolerance chosen by the user.

The library is heavily based on Conformal Prediction, which allows its uncertainty estimates to be distribution-free. A key feature of this approach is that the resulting uncertainty measure relies on the model's actual accuracy (how often it is correct) rather than its confidence (e.g., raw softmax outputs).

U-TraCE provides an evaluation method that reports an estimate of, and an upper confidence bound on, the model's uncertainty, defined as the probability of error: $1-P(\hat{y}​=y_t​)$, where $\hat{y}$ is the model's prediction and $y_t$​ is the true label. For regression, a prediction is correct when it lies within the tolerance $\tau$ (`tol`) of the truth, so the uncertainty is $P(|y_t - \hat{y}| > \tau)$.

The point estimate `U` carries no confidence level. The bound `U_delta`, returned by `uncertainty_at(..., delta=...)`, holds with probability at least `1 - delta` only if (1) the evaluation samples are independent and representative of the data the model is used on, and (2) the uniform-selection assumption of the paper holds. The package cannot check either; in particular, pixels of one image are not independent samples.


## Setup

### Requirements

* Python 3.11-3.14

### Installation

Which install you want depends on what you're doing with it. The core install and its two GPU
extras are the same three commands regardless of scenario:

```bash
pip install utrace@git+https://github.com/edgardomarchi/utrace.git@main                # CPU only
pip install "utrace[cuda13]@git+https://github.com/edgardomarchi/utrace.git@main"      # + NVIDIA GPU for JAX
pip install "utrace[rocm7-local]@git+https://github.com/edgardomarchi/utrace.git@main" # + AMD GPU for JAX
```

**Using a black-box model, or a pure JAX model.** Pick whichever of the three commands above
matches your hardware. That's the whole install.

**Already have PyTorch working for your hardware, and want the `utrace.utils.pytorch`
helpers.** This is the primary use case the packaging is built around: run the same command
above, picking the JAX backend extra that matches your hardware, straight into your existing
environment. `utrace` never installs, upgrades, or replaces your PyTorch — none of these three
commands declare torch as a dependency at all, so there is nothing for them to touch. The
helpers under `utrace.utils.pytorch` require torch to already be present in the environment;
`utrace` does not provide it. Import one without torch installed and you'll get an `ImportError`
from that submodule, not from `utrace` itself.

**Want to run the example scripts** in `scripts/`. These need `torch`/`torchvision` (to train
and run the example models) plus `viz` (to plot the results); `scripts/ACDC_example.py` also
needs the `acdc` extra (MONAI, for loading the cardiac model bundle, and nibabel, for reading
NIfTI images). Because routing `torch` to the CPU-only PyTorch wheel index is a `uv`-specific
mechanism (see below), install this combination with `uv` rather than `pip`:

```bash
uv pip install "utrace[examples,acdc,viz]@git+https://github.com/edgardomarchi/utrace.git@main"
```

| extra | adds | needed for |
|---|---|---|
| `viz` | matplotlib, pandas | the plotting and reporting helpers in `utrace.utils` |
| `cuda13` | jax[cuda13] | GPU acceleration on NVIDIA, for the JAX backend only |
| `rocm7-local` | jax[rocm7-local] | GPU acceleration on AMD, for the JAX backend only, against an existing host ROCm install |
| `examples` | torch, torchvision (CPU build) | running the scripts in `scripts/` |
| `acdc` | monai, nibabel | `scripts/ACDC_example.py` specifically |

or add it to your `pyproject.toml`:

```toml
[dependencies]
...
utrace = { git = "https://github.com/edgardomarchi/utrace.git", branch = "main" }
```

#### GPU extras

**`rocm7-local`** needs ROCm 7.x already installed on the host (or inside the container) before
installing it — per JAX's own installation docs, "ROCm must already be present on the host
system or inside the container." JAX ships no extra that installs ROCm itself; `rocm7-local`
only adds the plugin and PJRT packages on top of it. The `-local` suffix is deliberate and
mirrors JAX's own extra name, so that the thing you have to supply yourself stays visible in the
command you type. The plugin is built against a specific ROCm release, and your installed ROCm
must match it — check the ROCm version your resolved `jax-rocm7-plugin` targets against AMD's
JAX compatibility matrix before relying on it.

**`cuda13`** is more self-contained — CUDA itself arrives via pip wheels, no separate host
install needed — but does need a driver recent enough for CUDA 13. Per JAX's installation docs,
that means NVIDIA driver version >= 580 on Linux.

**On `pip` vs `uv`.** Under this shape, no extra a normal user installs carries torch — `viz`,
`cuda13` and `rocm7-local` are all torch-free, so `[tool.uv.sources]` index routing (the
`uv`-specific mechanism that picks the right PyTorch wheel index) is irrelevant to them, and a
plain `pip install` works fine for all three. Routing only matters for the `examples` extra,
which does carry torch, and for the `dev`/`dev-cuda13`/`dev-rocm7` contributor groups documented
in `CLAUDE.md` — both are `uv` workflows already, so this is not a new constraint. A plain
`pip install utrace[examples]` still works, but installs PyPI's default `torch` build, which
bundles CUDA regardless of the extra — not what `examples` intends. Use `uv pip install` (as
above) to get the CPU build instead.

## Usage

After installation, the package can be imported:

```bash
$ python
>>> import utrace
>>> utrace.__version__
```

The evaluation uses three data sets: one to calibrate the quantifier, one to tune `alpha`, and a
separate one to measure `U` and its bound. They must be disjoint, and the library cannot check
that. The example below runs on synthetic softmax outputs (numpy arrays only, no torch):

```python
import numpy as np
from utrace import UncertaintyQuantifier

rng = np.random.default_rng(0)
K = 5


def synthetic(n):
    """Softmax outputs of a noisy 5-class classifier, and the true labels."""
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), y] += 2.5
    softmax = np.exp(logits)
    return softmax / softmax.sum(axis=1, keepdims=True), y


cal_x, cal_y = synthetic(1000)    # calibration
tune_x, tune_y = synthetic(1000)  # tuning
eval_x, eval_y = synthetic(5000)  # evaluation

uq = UncertaintyQuantifier(N=1000)
uq.calibrate(cal_x, cal_y)
alpha = uq.tune_alpha(tune_x, tune_y)
result = uq.uncertainty_at(eval_x, eval_y, alpha, delta=0.05)

print(f"alpha       = {alpha:.4f}")
print(f"U           = {result.U:.4f}")
print(f"U_empirical = {result.U_empirical:.4f}")
print(f"U_delta     = {result.U_delta:.4f}")
print(f"true error  = {np.mean(eval_x.argmax(axis=1) != eval_y):.4f}")
```

`U` is the point estimate, `U_empirical` is the same expression with the coverage measured on the
evaluation set, and `U_delta` is the upper confidence bound at level `1 - delta` (see the two
assumptions above). `U_delta` combines a Clopper-Pearson bound on the coverage with a lower bound on
the efficiency term `EC`; by default the latter is the tighter of a Hoeffding and an empirical
Bernstein bound, and `bound=` selects another (`utrace.bounds`: `'hoeffding'`,
`'empirical_bernstein'`, or `'betting'`, the betting bound of Waudby-Smith and Ramdas, which adapts to
the variance).

### Where `U_delta` comes from

`uncertainty_budget` returns the same evaluation as `uncertainty_at` (with `delta` required) and
splits its `U_delta` exactly into the model's error on the evaluation set and what the method adds
to it. Continuing the classification example:

```python
budget = uq.uncertainty_budget(eval_x, eval_y, alpha, delta=0.05)
for c in budget.components:
    print(f"{c.name:<22} {c.kind.value:<12} {c.value:.4f}")
print(f"{'total = U_delta':<35} {budget.total:.4f}")
print(f"{'error_upper':<35} {budget.error_upper:.4f}")
```

The components are `error` (the top-1 error, kind `model`); `accepted_not_covered` (correct
answers whose set is empty) and `selection` (multi-label sets, where `1/|C|` is credited against how
often the top-1 label is actually right), both `structural`; and `coverage` and `efficiency`, the
two `statistical` terms of the confidence margin `U_delta - U_empirical`. `selection` is not clipped
and can be slightly negative on a finite sample. `error_upper` is the direct Clopper-Pearson upper
bound on the error from the same labels, for comparison. For regression the same terms are defined
with the tolerance window `y_hat ± tol` in place of the top-1 label.

`budget.U_delta_V` and `budget.U_delta_W` are two single bounds beside it, each one lower confidence
bound at level `1 - delta` on a mean of per-sample values in `[0, 1]`: `V = pi 1{y in C}`, whose
`1 - E[V]` is the same `U*` that `U_delta` bounds as a product of two bounds, and `W`, which adds the
correct answers the set leaves out (`budget.V.components`, `budget.W.components` give their splits).
When the evaluation samples are not independent (the pixels of one image), pass `units=` with the id
of the independent unit of each sample (the image or the patient): `V` and `W` are then averaged per
unit and bounded over the units, a unit-averaged estimand. `units=` does not select samples; that is
`classes=`, and the two can be combined.

For regression, pass point predictions instead of softmax outputs, construct the quantifier with
`score='abs_error'`, and give the tolerance `tol` (in the units of the target) on each call; it
belongs to the call, so one calibrated quantifier can be evaluated for several tolerances:

```python
import numpy as np
from utrace import UncertaintyQuantifier

rng = np.random.default_rng(0)


def synthetic(n):
    """Point predictions of a regression model, and the true targets."""
    x = rng.uniform(-3, 3, size=n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    return np.sin(x), y


cal_x, cal_y = synthetic(1000)    # calibration
tune_x, tune_y = synthetic(1000)  # tuning
eval_x, eval_y = synthetic(5000)  # evaluation

tol = 0.4                         # an error is |y - y_hat| > tol, in the units of y
uq = UncertaintyQuantifier(N=1000, score="abs_error")
uq.calibrate(cal_x, cal_y)
alpha = uq.tune_alpha(tune_x, tune_y, tol=tol)
result = uq.uncertainty_at(eval_x, eval_y, alpha, tol=tol, delta=0.05)

print(f"alpha       = {alpha:.4f}")
print(f"U           = {result.U:.4f}")
print(f"U_empirical = {result.U_empirical:.4f}")
print(f"U_delta     = {result.U_delta:.4f}")
print(f"true error  = {np.mean(np.abs(eval_y - eval_x) > tol):.4f}")
```

Here `U_delta` (0.1898) comes out below `U` (0.1978): `U_delta` uses the coverage measured on the
evaluation set, not the nominal `1 - alpha`, so it is not guaranteed to be above `U`; it is always
at least `U_empirical` (0.1766), the same expression as `U` with the measured coverage.

`tune_alpha` also takes `target_ratio` (default 1.0, in `(0, 1]`): the search stops at the
smallest alpha for which the prediction intervals overlap the tolerance window by at least that
fraction on average. With the default the intervals lie inside the window, so `U` equals `alpha`.

Example scripts can be found in the `scripts` folder, and the jupytext notebooks in `notebooks/`
(how to run them is described in `CONTRIBUTING.md`, "Notebooks").

## Authors

*   **Edgardo Marchi**
*   **Maik Liebl**

## Acknowledgements

This project was developed in collaboration between the following institutions:

*   **Physikalisch-Technische Bundesanstalt (PTB)** - Germany's national metrology institute.
*   **Instituto Nacional de Tecnología Industrial (INTI)** - Argentina's national metrology institute.

This work was funded in part by PTB, which supported a three-month research stay at its Berlin facilities. Open-access publication of the associated article was covered under PTB's institutional agreement with the publisher. It was carried out by researchers at INTI and PTB as part of their institutional research activity.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
  
   <a href="https://www.ptb.de/cms/en.html" target="_blank" rel="noopener noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b4/Physikalisch-Technische_Bundesanstalt_2013_logo.png" alt="Logo PTB" width="200" style="margin-right: 20px;">
  </a>
  <a href="https://www.inti.gob.ar" target="_blank" rel="noopener noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/Avatar_2023_new_%281%29.png" alt="Logo INTI" width="150">
  </a>

</div>


## License

This project is licensed under the MIT License.

## Cite
```bibtex
@article{10.1088/2632-2153/ae35ce,
	author={Marchi, Edgardo José and Liebl, Maik},
	title={U-TraCE: A conformal prediction approach to uncertainty quantification in black-box models},
	journal={Machine Learning: Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/ae35ce},
	year={2026}
}
```

```bibtex
@software{utrace,
title = {U-TraCE},
author = {Edgardo Marchi and Maik Liebl},
year = {2025},
version = {0.3.0},
howpublished = {\url{https://github.com/edgardomarchi/utrace}}
}
```

# U-TraCE
[![Tests](https://github.com/edgardomarchi/utrace/actions/workflows/tests.yml/badge.svg)](https://github.com/edgardomarchi/utrace/actions/workflows/tests.yml)

![Project Status: WIP - Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)

**U**ncertainty **Tra**cking for **C**omplex **E**stimators.

U-TraCE is a Python package for computing upper bounds on the marginal or conditional uncertainty of any classification model with a softmax output.

The library is heavily based on Conformal Prediction, which allows its uncertainty estimates to be distribution-free. A key feature of this approach is that the resulting uncertainty measure relies on the model's actual accuracy (how often it is correct) rather than its confidence (e.g., raw softmax outputs).

U-TraCE provides an evaluation method that reports a guaranteed upper bound for the model's uncertainty, defined as the probability of error: $1-P(\hat{y}​=y_t​)$, where $\hat{y}$ is the model's prediction and $y_t$​ is the true label.


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

Example scripts can be found in the `scripts` folder.

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
version = {0.1.0},
howpublished = {\url{https://github.com/edgardomarchi/utrace}}
}
```

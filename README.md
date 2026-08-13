# U-TraCE

![Project Status: WIP - Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)

**U**ncertainty **Tra**cking for **C**omplex **E**stimators.

U-TraCE is a Python package for computing upper bounds on the marginal or conditional uncertainty of any classification model with a softmax output.

The library is heavily based on Conformal Prediction, which allows its uncertainty estimates to be distribution-free. A key feature of this approach is that the resulting uncertainty measure relies on the model's actual accuracy (how often it is correct) rather than its confidence (e.g., raw softmax outputs).

U-TraCE provides an evaluation method that reports a guaranteed upper bound for the model's uncertainty, defined as the probability of error: $1−P(\hat{y}​=y_t​)$, where $\hat{y}$​ is the model's prediction and $y_t$​ is the true label.


## Setup

### Requirements

* Python 3.12+

### Installation

A plain install gives a working CPU-only core — just JAX and NumPy, nothing else:

```bash
pip install utrace@git+ssh://git@github.com/edgardomarchi/utrace.git@main
```

Optional extras add capability on top of the core:

| extra | adds | needed for |
|---|---|---|
| `viz` | matplotlib, pandas | the plotting and reporting helpers in `utrace.utils` |
| `torch` | torch, torchvision (CPU build) | the `utrace.utils.pytorch` helpers |
| `cuda13` | jax[cuda13], torch/torchvision (CUDA 13 build) | GPU acceleration on NVIDIA |
| `rocm7-local` | jax[rocm7-local], torch/torchvision (ROCm 7.2 build), triton-rocm | GPU acceleration on AMD, against an existing host ROCm install |

```bash
pip install utrace\[viz\]@https://github.com/edgardomarchi/utrace.git@main
```

or add it to your `pyproject.toml`:

```toml
[dependencies]
...
utrace = { git = "https://github.com/edgardomarchi/utrace.git", branch = "main" }
```

`utrace.utils.pytorch` (the PyTorch-specific helpers) requires the `torch` extra.

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

**The pip limitation.** The `torch`, `cuda13` and `rocm7-local` extras all rely on
`[tool.uv.sources]` to route `torch`/`torchvision` to the matching PyTorch wheel index (CPU,
cu130, or rocm7.2) — a `uv`-specific mechanism invisible to `pip`. A plain
`pip install utrace[torch]` (or `[cuda13]`, or `[rocm7-local]`) installs PyPI's default `torch`
build instead, which bundles CUDA regardless of which extra was requested — not what any of the
three extras intend. `uv` is the supported, tested path for all three; if you must use `pip`,
point it at the matching index explicitly with `--extra-index-url` (not `--index-url`, which
excludes PyPI entirely and breaks resolution of `jax`, `numpy`, and everything else utrace
needs):

```bash
pip install utrace\[torch\]@https://github.com/edgardomarchi/utrace.git@main \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install utrace\[cuda13\]@https://github.com/edgardomarchi/utrace.git@main \
    --extra-index-url https://download.pytorch.org/whl/cu130
pip install utrace\[rocm7-local\]@https://github.com/edgardomarchi/utrace.git@main \
    --extra-index-url https://download.pytorch.org/whl/rocm7.2
```

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
*   **Instituto Nacional de Tecnología Industrial (INTI)** - The National Institute of Industrial Technology of Argentina.

The project was funded by PTB.

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
howpublished = {\url{https://github.com/edgardomarchi/utrace}}
}
```

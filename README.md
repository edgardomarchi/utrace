# U-TraCE

![Project Status: WIP - Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)

**U**ncertainty **Tra**cking for **C**omplex **E**stimators.

U-TraCE is a Python package for computing upper bounds on the marginal or conditional uncertainty of any classification model that outputs class probabilities.

The library is heavily based on Conformal Prediction, which allows its uncertainty estimates to be distribution-free. A key feature of this approach is that the resulting uncertainty measure relies on the model's actual accuracy (how often it is correct) rather than its confidence (e.g., raw softmax outputs).

U-TraCE provides an evaluation method that reports a guaranteed upper bound for the model's uncertainty, defined as the probability of error: $1−P(\hat{y}​=y_t​)$, where $\hat{y}$​ is the model's prediction and $y_t$​ is the true label.


## Setup

### Requirements

* Python 3.12+

### Installation

Install it directly into an activated virtual environment (e.g. with `rocm` backend):

```bash
pip install utrace\[rocm\]@git+ssh://git@github.com/edgardomarchi/utrace.git@main
```

or add it to your `pyproject.toml`:

```toml
[dependencies]
...
utrace = { git = "https://github.com/edgardomarchi/utrace.git", branch = "main" }
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

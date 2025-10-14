# U-TraCE

Uncertainty Tracking for Complex Estimators


[![Linux Build](https://img.shields.io/github/actions/workflow/status/edgardomarchi/utrace/main.yml?branch=main&label=linux)](https://github.com/edgardomarchi/utrace/actions)
[![Windows Build](https://img.shields.io/appveyor/ci/edgardomarchi/utrace/main.svg?label=windows)](https://ci.appveyor.com/project/edgardomarchi/utrace)
[![Code Coverage](https://img.shields.io/codecov/c/github/edgardomarchi/utrace)
](https://codecov.io/gh/edgardomarchi/utrace)
[![Code Quality](https://img.shields.io/scrutinizer/g/edgardomarchi/utrace.svg?label=quality)](https://scrutinizer-ci.com/g/edgardomarchi/utrace/?branch=main)
[![PyPI License](https://img.shields.io/pypi/l/utrace.svg)](https://pypi.org/project/utrace)
[![PyPI Version](https://img.shields.io/pypi/v/utrace.svg?label=version)](https://pypi.org/project/utrace)
[![PyPI Downloads](https://img.shields.io/pypi/dm/utrace.svg?color=orange)](https://pypistats.org/packages/utrace)

## Setup

### Requirements

* Python 3.12+

### Installation

Install it directly into an activated virtual environment:

```bash
$ pip install git+https://github.com/edgardomarchi/utrace.git@main
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

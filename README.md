<!-- <h1 align='center'>GPJax</h1>
<h2 align='center'>Gaussian processes in Jax.</h2> -->
<p align="center">
<img width="700" height="300" src="https://raw.githubusercontent.com/QuantClimate/GPJax/main/docs/static/gpjax_logo.svg" alt="GPJax's logo">
</p>

[![codecov](https://codecov.io/gh/QuantClimate/GPJax/graph/badge.svg)](https://codecov.io/gh/QuantClimate/GPJax)
[![CodeFactor](https://www.codefactor.io/repository/github/QuantClimate/GPJax/badge)](https://www.codefactor.io/repository/github/QuantClimate/GPJax)
[![PyPI version](https://badge.fury.io/py/GPJax.svg)](https://badge.fury.io/py/GPJax)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gpjax.svg)](https://anaconda.org/conda-forge/gpjax)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04455/status.svg)](https://doi.org/10.21105/joss.04455)
[![Downloads](https://pepy.tech/badge/gpjax)](https://pepy.tech/project/gpjax)
[![Slack Invite](https://img.shields.io/badge/Slack_Invite--blue?style=social&logo=slack)](https://join.slack.com/t/gpjax/shared_invite/zt-3cesiykcx-nzajjRdnV3ohw7~~eMlCYA)

[**Quickstart**](#simple-example)
| [**Install guide**](#installation)
| [**Documentation**](https://gpjax.quantclimate.com/)
| [**Slack Community**](https://join.slack.com/t/gpjax/shared_invite/zt-3cesiykcx-nzajjRdnV3ohw7~~eMlCYA)

GPJax aims to provide a low-level interface to Gaussian process (GP) models in
[Jax](https://github.com/google/jax), structured to give researchers maximum
flexibility in extending the code to suit their own needs. The idea is that the
code should be as close as possible to the maths we write on paper when working
with GP models.

# Package organisation

## Contributions

We would be delighted to receive contributions from interested individuals and
groups. To learn how you can get involved, please read our [guide for
contributing](https://github.com/QuantClimate/GPJax/blob/main/docs/contributing.md).
If you have any questions, we encourage you to [open an
issue](https://github.com/QuantClimate/GPJax/issues/new/choose). For
broader conversations, such as best GP fitting practices or questions about the
mathematics of GPs, we invite you to [open a
discussion](https://github.com/QuantClimate/GPJax/discussions).

Another way you can contribute to GPJax is through issue triaging. This can include
reproducing bug reports, asking for vital information such as version numbers and
reproduction instructions, or identifying stale issues. To get started, browse the
[open issues](https://github.com/QuantClimate/GPJax/issues).

As a contributor to GPJax, you are expected to abide by our [code of
conduct](docs/CODE_OF_CONDUCT.md). If you feel that you have either experienced or
witnessed behaviour that violates this standard, then we ask that you report any such
behaviours through [this form](https://jaxgaussianprocesses.com/contact/) or reach out to
one of the project's [_gardeners_](https://gpjax.quantclimate.com/GOVERNANCE.html#roles).

Feel free to join our [Slack
Channel](https://join.slack.com/t/gpjax/shared_invite/zt-3cesiykcx-nzajjRdnV3ohw7~~eMlCYA),
where we can discuss the development of GPJax and broader support for Gaussian
process modelling.

We appreciate all [the contributors to
GPJax](https://github.com/QuantClimate/GPJax/graphs/contributors) who have helped to shape
GPJax into the package it is today.

# Supported methods and interfaces

## Notebook examples

### Getting started

> - [**New to Gaussian Processes?**](https://gpjax.quantclimate.com/examples/intro_to_gps.html)
> - [**Introduction to Kernels**](https://gpjax.quantclimate.com/examples/intro_to_kernels.html)
> - [**Regression**](https://gpjax.quantclimate.com/examples/regression.html)
> - [**Classification**](https://gpjax.quantclimate.com/examples/classification.html)
> - [**Count Data Regression**](https://gpjax.quantclimate.com/examples/poisson.html)
> - [**Natural Gradients**](https://gpjax.quantclimate.com/examples/natural_gradients.html)

### Accelerating Gaussian processes

> - [**Sparse Gaussian Process Regression**](https://gpjax.quantclimate.com/examples/collapsed_vi.html)
> - [**Sparse Stochastic Variational Inference**](https://gpjax.quantclimate.com/examples/uncollapsed_vi.html)
> - [**Natural Gradients in Practice**](https://gpjax.quantclimate.com/examples/natgrads.html)
> - [**Dual Parameterisation of Sparse GPs (t-SVGP)**](https://gpjax.quantclimate.com/examples/dual_svgp.html)
> - [**State-Space (Markovian) Gaussian Processes**](https://gpjax.quantclimate.com/examples/state_space_gps.html)
> - [**Scalable Multi-Output GPs with OILMM**](https://gpjax.quantclimate.com/examples/oilmm.html)

### Applied modelling

> - [**Gaussian Processes Barycentres**](https://gpjax.quantclimate.com/examples/barycentres.html)
> - [**Graph Kernels**](https://gpjax.quantclimate.com/examples/graph_kernels.html)
> - [**Heteroscedastic Inference**](https://gpjax.quantclimate.com/examples/heteroscedastic_inference.html)
> - [**Multi-Output Gaussian Processes**](https://gpjax.quantclimate.com/examples/multioutput.html)
> - [**Orthogonal Additive Kernels**](https://gpjax.quantclimate.com/examples/oak.html)
> - [**Gaussian Processes for Vector Fields and Ocean Current Modelling**](https://gpjax.quantclimate.com/examples/oceanmodelling.html)
> - [**Spatial Modelling with Composable Gaussian Processes**](https://gpjax.quantclimate.com/examples/spatial_linear_gp.html)
> - [**Gridded Data with xarray**](https://gpjax.quantclimate.com/examples/xarray_workflow.html)
> - [**UCI Data Benchmarking**](https://gpjax.quantclimate.com/examples/yacht.html)

### Guides for customisation

> - [**Kernel Guide**](https://gpjax.quantclimate.com/examples/constructing_new_kernels.html)
> - [**Likelihood Guide**](https://gpjax.quantclimate.com/examples/likelihoods_guide.html)
> - [**Deep Kernel Learning**](https://gpjax.quantclimate.com/examples/deep_kernels.html)
> - [**Joint Inference with Numpyro**](https://gpjax.quantclimate.com/examples/numpyro_integration.html)
> - [**Backend Module Design**](https://gpjax.quantclimate.com/examples/backend.html)

## Conversion between `.ipynb` and `.py`
Above examples are stored in [examples](docs/examples) directory in the double
percent (`py:percent`) format. Checkout [jupytext
using-cli](https://jupytext.readthedocs.io/en/latest/using-cli.html) for more
info.

* To convert `example.py` to `example.ipynb`, run:

```bash
jupytext --to notebook example.py
```

* To convert `example.ipynb` to `example.py`, run:

```bash
jupytext --to py:percent example.ipynb
```

# Installation

## Stable version

The latest stable version of GPJax can be installed from [PyPI](https://pypi.org/project/gpjax/):

```bash
pip install gpjax
```

or from [conda-forge](https://github.com/conda-forge/gpjax-feedstock):

```bash
# with Pixi
pixi add gpjax
# or with conda
conda install --channel conda-forge gpjax
```

> **Note**
>
> We recommend you check your installation version:
> ```python
> python -c 'import gpjax; print(gpjax.__version__)'
> ```



## Development version
> **Warning**
>
> This version is possibly unstable and may contain bugs.

> **Note**
>
> We advise you create virtual environment before installing:
> ```
> conda create -n gpjax_experimental python=3.12
> conda activate gpjax_experimental
>  ```


Clone a copy of the repository to your local machine and run the setup
configuration in development mode.
```bash
git clone https://github.com/QuantClimate/GPJax.git
cd GPJax
uv venv
uv sync
```

> We recommend you check your installation passes the supplied unit tests:
>
> ```python
> uv run poe all-tests
> ```

# Citing GPJax

If you use GPJax in your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04455#).

```
@article{Pinder2022,
  doi = {10.21105/joss.04455},
  url = {https://doi.org/10.21105/joss.04455},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {75},
  pages = {4455},
  author = {Thomas Pinder and Daniel Dodd},
  title = {GPJax: A Gaussian Process Framework in JAX},
  journal = {Journal of Open Source Software}
}
```

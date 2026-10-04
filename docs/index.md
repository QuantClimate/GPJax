# GPJax

**Gaussian processes in JAX.**

GPJax is a didactic Gaussian process (GP) library in JAX, supporting GPU
acceleration and just-in-time compilation. We seek to provide a flexible API to
enable researchers to rapidly prototype and develop new ideas.

![Gaussian process posterior.](static/GP.svg)

## "Hello, GP!"

Typing GP models is as simple as the maths we would write on paper.

```python
import gpjax as gpx

mean = gpx.mean_functions.Zero()
kernel = gpx.kernels.RBF()
prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
likelihood = gpx.likelihoods.Gaussian()

model = prior * likelihood  # the joint p(f, y)
```

Conditioning the model on data yields the posterior process, which can then
be queried at any test inputs:

```python
import jax.numpy as jnp

xtrain = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
D = gpx.Dataset(X=xtrain, y=jnp.sin(xtrain))
xtest = jnp.linspace(0.0, 1.0, 50).reshape(-1, 1)

posterior = model.condition(D)  # p(f | D) — equivalently: model | D
predictive = posterior(xtest)
```

$$
\begin{aligned}
k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\ \\
p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
\end{aligned}
$$ (eq-index-conjugate-gp)

<section class="consulting-cta">
    <p>We currently have some <strong>availability for consulting</strong> on how Gaussian processes, Bayesian modelling, and GPJax can be integrated into your team's work. If this sounds relevant to your work, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation below.</p>
</section>

## Learn more

<!-- Every direct child of a `{grid}` must be a `{grid-item-card}` — anything else,
     including an HTML comment, emits `design.grid` and is fatal under `-W`. Hence the
     note about the benchmarks card living out here rather than beside it: that card
     deliberately has no `:link-type:`, because the ASV dashboard is copied into the
     built site by .github/workflows/build_docs.yml and is not a Sphinx document, so
     it cannot be resolved as one. sphinx-design emits a bare `:link:` verbatim, which
     keeps it relative and working in a local build too. -->

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

Install the stable or development version.
:::

:::{grid-item-card} {octicon}`mortar-board` New to Gaussian processes?
:link: examples/intro_to_gps
:link-type: doc

Priors, posteriors and the marginal likelihood from first principles.
:::

:::{grid-item-card} {octicon}`graph` Regression
:link: examples/regression
:link-type: doc

The canonical end-to-end workflow, start to finish.
:::

:::{grid-item-card} {octicon}`book` API reference
:link: reference/index
:link-type: doc

Every public class and function, with source links.
:::

:::{grid-item-card} {octicon}`alert` Sharp bits
:link: sharp_bits
:link-type: doc

The numerical pitfalls worth knowing about before you hit them.
:::

:::{grid-item-card} {octicon}`pulse` Benchmarks
:link: benchmarks/index.html

The ASV dashboard tracking GPJax's performance commit by commit.
:::

::::

## Citing GPJax

If you use GPJax in your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04455#).

```bibtex
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

<!-- Sidebar grouping. `toctree_maxdepth = 0` in conf.py hands depth control to the
     `:maxdepth:` on each toctree below, and `globaltoc_expand_depth = 1` opens every
     top-level entry. So `:maxdepth: 2` is what puts a group's pages into the sidebar
     tree at all, and `:maxdepth: 1` keeps a group to a single line.

     Every group whose pages should be reachable from the sidebar therefore needs
     `:maxdepth: 2`. Groups that are a single page — Getting started, Migrations,
     Project — use `:maxdepth: 1`, because there are no children to reach.

     Note shibuya's fold state is uniform by depth: it cannot open some groups and
     leave others folded. That is why the migration guides live in one page with a
     `##` per release rather than as separate documents under a folded parent. The
     structure gives the tidy single sidebar line for free, with no JavaScript. -->

```{toctree}
:hidden:
:caption: Getting started
:maxdepth: 1

installation
design
sharp_bits
examples/intro_to_gps
examples/intro_to_kernels
examples/regression
examples/classification
examples/poisson
examples/xarray_workflow
examples/natural_gradients
```

```{toctree}
:hidden:
:caption: Accelerating Gaussian processes
:maxdepth: 1

examples/collapsed_vi
examples/uncollapsed_vi
examples/natgrads
examples/dual_svgp
examples/state_space_gps
examples/oilmm
```

```{toctree}
:hidden:
:caption: Applied modelling
:maxdepth: 1

examples/barycentres
examples/graph_kernels
examples/heteroscedastic_inference
examples/infilling_surface_temperature
examples/multioutput
examples/oak
examples/oceanmodelling
examples/spatial_linear_gp
examples/yacht
```

```{toctree}
:hidden:
:caption: Guides for customisation
:maxdepth: 1

examples/constructing_new_kernels
examples/likelihoods_guide
examples/deep_kernels
examples/numpyro_integration
examples/backend
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

reference/index
glossary
```

```{toctree}
:hidden:
:caption: Migrations
:maxdepth: 1

migration
```

```{toctree}
:hidden:
:caption: Project
:maxdepth: 1

contributing
GOVERNANCE
CODE_OF_CONDUCT
references
```

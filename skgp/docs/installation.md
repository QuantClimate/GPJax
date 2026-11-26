# Installation

skgp needs Python 3.12 or newer. Installing it also installs everything it needs to fit a model, including JAX, GPJax and Polars.

::::{tab-set}

:::{tab-item} pip
```bash
pip install skgp
```
:::

:::{tab-item} uv
```bash
uv add skgp
```
:::

::::

To check which version you have:

```bash
python -c "from importlib.metadata import version; print(version('skgp'))"
```

## pandas and pandera

skgp reads Polars dataframes without anything else installed. To pass pandas dataframes, install pandas too; skgp then returns pandas dataframes for them.

```bash
pip install pandas
```

To check your data with your own schema (the `schema=` setting), use any object with a `.validate(frame)` method. A [pandera](https://pandera.readthedocs.io/) schema is one; skgp does not depend on pandera, so install it yourself if you want it:

```bash
pip install "pandera[polars]"
```

## GPUs

skgp fits on the CPU unless JAX is installed with support for your GPU. To fit on a GPU, install that version of JAX as the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) describes. skgp then uses the GPU without any change to your code.

## Development version

skgp's code is in the `skgp/` folder of the [GPJax repository](https://github.com/QuantClimate/GPJax). To work on skgp itself, clone that repository and install skgp with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/QuantClimate/GPJax.git
cd GPJax/skgp
uv sync --package skgp
uv run --package skgp pytest -m ""
```

The last command runs the full test suite, including the slow tests that plain `uv run pytest` skips.

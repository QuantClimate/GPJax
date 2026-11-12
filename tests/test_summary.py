import io

import equinox as eqx
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as npd
import paramax
import pytest
from rich.console import Console
from rich.table import Table

# Match the rest of the suite: stable float64 numerics.
config.update("jax_enable_x64", True)

import gpjax as gpx
from gpjax import summary
from gpjax.gps import Prior
from gpjax.kernels import RBF, Linear
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import (
    Constant,
    Zero,
)
from gpjax.parameters import (
    LowerTriangular,
    NonNegativeReal,
    PositiveReal,
    Real,
    SigmoidBounded,
)


def _frozen_prior():
    """A prior whose lengthscale has been frozen via paramax.non_trainable."""
    prior = Prior(mean_function=Zero(), kernel=RBF())
    return eqx.tree_at(
        lambda m: m.kernel.lengthscale, prior, replace_fn=paramax.non_trainable
    )


def test_paramrecord_fields_exist():
    rec = summary.ParamRecord(
        name="kernel.lengthscale",
        cls="PositiveReal",
        value=jnp.array(1.0),
        bijector="Softplus",
        prior="-",
        trainable=True,
        shape=(),
        dtype="f64",
    )
    assert rec.name == "kernel.lengthscale"
    assert rec.trainable is True
    assert rec.prior == "-"


def test_is_param_leaf_stops_at_parameters():
    assert summary._is_param_leaf(PositiveReal(1.0)) is True
    assert summary._is_param_leaf(jnp.array(1.0)) is False


def test_is_frozen_true_for_frozen_parameter():
    frozen = _frozen_prior()
    assert summary._is_frozen(frozen.kernel.lengthscale) is True
    assert summary._is_frozen(frozen.kernel.variance) is False


def test_is_frozen_false_for_plain_parameter():
    assert summary._is_frozen(PositiveReal(1.0)) is False


@pytest.mark.parametrize(
    ("param", "expected"),
    [
        (PositiveReal(2.0), "Softplus"),
        (NonNegativeReal(0.5), "Softplus"),
        (Real(-3.0), "Identity"),
        (LowerTriangular(jnp.eye(2)), "LowerCholesky"),
    ],
)
def test_bijector_name_mapping(param, expected):
    assert summary._bijector_name(param) == expected


def test_bijector_name_sigmoid_bounded_uses_bounds():
    param = SigmoidBounded(0.5, low=0.0, high=2.0)
    assert summary._bijector_name(param) == "Sigmoid[0, 2]"


def test_bijector_name_falls_back_to_identity_without_constraint():
    # A NonTrainable-wrapped bare array has no `_constraint`.
    wrapped = paramax.non_trainable(jnp.array(1.0))
    assert summary._bijector_name(wrapped) == "Identity"


def test_short_dtype_abbreviates():
    assert summary._short_dtype(jnp.float64) == "f64"
    assert summary._short_dtype(jnp.float32) == "f32"
    assert summary._short_dtype(jnp.int32) == "i32"


def test_format_value_scalar_uses_precision():
    assert summary._format_value(jnp.array(1.0), max_array=4, precision=3) == "1"
    assert (
        summary._format_value(jnp.array(0.123456), max_array=4, precision=3) == "0.123"
    )


def test_format_value_array_truncates():
    out = summary._format_value(jnp.arange(6.0), max_array=4, precision=3)
    assert out == "[0, 1, 2, 3, ...]"


def test_format_value_short_array_not_truncated():
    out = summary._format_value(jnp.array([1.0, 2.0]), max_array=4, precision=3)
    assert out == "[1, 2]"


def _record_by_name(records, name):
    matches = [r for r in records if r.name == name]
    assert matches, f"{name!r} not in {[r.name for r in records]}"
    return matches[0]


def test_collect_isotropic_rbf_prior():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    records = summary._collect(prior)
    names = {r.name for r in records}
    assert "kernel.lengthscale" in names
    assert "kernel.variance" in names

    ls = _record_by_name(records, "kernel.lengthscale")
    assert ls.cls == "PositiveReal"
    assert ls.bijector == "Softplus"
    assert ls.trainable is True
    assert ls.shape == ()
    assert ls.dtype == "f64"


def test_collect_renders_bare_array_leaf():
    # `Constant` holds a bare array; `Zero` wraps its constant to freeze it.
    prior = Prior(mean_function=Constant(), kernel=RBF())
    rec = _record_by_name(summary._collect(prior), "mean_function.constant")
    assert rec.cls == "Array"
    assert rec.bijector == "Identity"
    assert rec.trainable is True


def test_collect_renders_frozen_zero_mean_leaf():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    rec = _record_by_name(summary._collect(prior), "mean_function.constant")
    assert rec.cls == "Array"
    assert rec.trainable is False


def test_collect_composite_kernel_paths():
    prior = Prior(mean_function=Zero(), kernel=RBF() + Linear())
    names = {r.name for r in summary._collect(prior)}
    assert "kernel.kernels[0].lengthscale" in names
    assert "kernel.kernels[1].variance" in names


def test_collect_marks_and_unwraps_frozen_parameter():
    records = summary._collect(_frozen_prior())
    ls = _record_by_name(records, "kernel.lengthscale")
    assert ls.trainable is False
    # paramax.unwrap returns a concrete constrained value even when frozen.
    assert float(np.asarray(ls.value)) > 0.0


def test_collect_variational_family_lower_cholesky():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    posterior = prior * Gaussian()
    q = gpx.variational_families.VariationalGaussian(
        model=posterior, inducing_inputs=jnp.linspace(-3, 3, 5).reshape(-1, 1)
    )
    rec = _record_by_name(summary._collect(q), "variational_root_covariance")
    assert rec.cls == "LowerTriangular"
    assert rec.bijector == "LowerCholesky"
    assert rec.shape == (5, 5)


def test_format_value_traced_value_placeholder():
    captured = {}

    @jax.jit
    def f(x):
        captured["s"] = summary._format_value(x, max_array=4, precision=3)
        return x

    f(jnp.arange(3.0))
    assert captured["s"].startswith("<traced")
    assert "[3]" in captured["s"]


def test_collect_frozen_bare_array_keeps_array_class():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    frozen = eqx.tree_at(
        lambda m: m.mean_function.constant, prior, replace_fn=paramax.non_trainable
    )
    rec = _record_by_name(summary._collect(frozen), "mean_function.constant")
    assert rec.cls == "Array"
    assert rec.trainable is False
    assert rec.bijector == "Identity"


def test_render_returns_table_with_row_per_record():
    records = summary._collect(Prior(mean_function=Zero(), kernel=RBF()))
    table = summary._render(records)
    assert isinstance(table, Table)
    assert table.row_count == len(records)
    assert [c.header for c in table.columns] == list(summary._DEFAULT_COLUMNS)


def test_render_footer_counts_trainable():
    table = summary._render(summary._collect(_frozen_prior()))
    # frozen prior: lengthscale frozen, zero-mean constant frozen, variance trainable.
    assert table.caption == "3 parameters, 1 trainable"


def test_render_respects_column_subset():
    records = summary._collect(Prior(mean_function=Zero(), kernel=RBF()))
    table = summary._render(records, columns=["Parameter", "Trainable"])
    assert [c.header for c in table.columns] == ["Parameter", "Trainable"]


def _capture(model, **kwargs):
    console = Console(record=True, file=io.StringIO(), width=120)
    summary.summarise(model, console=console, **kwargs)
    return console.export_text()


def test_summarise_prints_parameter_names_and_footer():
    text = _capture(Prior(mean_function=Zero(), kernel=RBF()))
    assert "kernel.lengthscale" in text
    assert "Bijector" in text
    assert "parameters," in text  # footer present


def test_summarise_marks_frozen_row():
    text = _capture(_frozen_prior())
    # markup is stripped in export_text, leaving plain yes/no tokens.
    assert "no" in text
    assert "yes" in text


def test_summarise_column_subset_hides_other_columns():
    text = _capture(
        Prior(mean_function=Zero(), kernel=RBF()), columns=["Parameter", "Trainable"]
    )
    assert "Parameter" in text
    assert "Bijector" not in text


def test_summarise_empty_model_renders_empty_table():
    text = _capture(())  # a pytree with no leaves
    assert "0 parameters, 0 trainable" in text
    assert "Parameter" in text  # header row still rendered


def test_summarise_is_jit_safe():
    console = Console(record=True, file=io.StringIO(), width=120)

    @jax.jit
    def f(m):
        summary.summarise(m, console=console)
        return m

    f(Prior(mean_function=Zero(), kernel=RBF()))  # must not raise
    assert "traced" in console.export_text()


def test_summarise_rejects_unknown_columns():
    with pytest.raises(ValueError, match="unknown column"):
        summary.summarise(
            Prior(mean_function=Zero(), kernel=RBF()),
            console=Console(record=True, file=io.StringIO()),
            columns=["Bijektor"],
        )


def test_render_caption_singular_for_one_parameter():
    table = summary._render(summary._collect(PositiveReal(1.0)))
    assert table.caption == "1 parameter, 1 trainable"


def test_summary_mixin_rich_returns_table():
    class _M(summary._SummaryMixin, eqx.Module):
        x: jax.Array

        def __init__(self):
            self.x = jnp.array(1.0)

    assert isinstance(_M().__rich__(), Table)


def test_summary_mixin_mimebundle_has_text_and_html():
    class _M(summary._SummaryMixin, eqx.Module):
        x: jax.Array

        def __init__(self):
            self.x = jnp.array(1.0)

    bundle = _M()._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    html = bundle["text/html"]
    assert "<pre" in html
    assert "<!DOCTYPE" not in html
    assert "body {" not in html
    assert "x" in bundle["text/plain"]


@pytest.mark.parametrize(
    "model_fn",
    [
        lambda: RBF(),
        lambda: Zero(),
        lambda: Gaussian(),
        lambda: Prior(mean_function=Zero(), kernel=RBF()),
        lambda: Prior(mean_function=Zero(), kernel=RBF()) * Gaussian(),
    ],
)
def test_abstract_bases_have_rich_protocol(model_fn):
    model = model_fn()
    assert isinstance(model.__rich__(), Table)
    assert "text/html" in model._repr_mimebundle_()


def test_variational_family_has_rich_protocol():
    prior = Prior(mean_function=Zero(), kernel=RBF())
    posterior = prior * Gaussian()
    q = gpx.variational_families.VariationalGaussian(
        model=posterior, inducing_inputs=jnp.linspace(-3, 3, 5).reshape(-1, 1)
    )
    assert isinstance(q.__rich__(), Table)


def test_summarise_is_publicly_exported():
    assert gpx.summarise is summary.summarise
    assert "summarise" in gpx.__all__


def test_summary_mixin_leaves_repr_to_equinox():
    # The mixin must NOT define its own __repr__; Equinox owns model repr.
    assert "__repr__" not in vars(summary._SummaryMixin)
    assert repr(RBF()).startswith("RBF(")


def test_summarise_validates_columns_for_parameter_free_model():
    with pytest.raises(ValueError, match="unknown column"):
        summary.summarise(
            (), columns=["BadCol"], console=Console(record=True, file=io.StringIO())
        )


def test_format_prior_handles_none_str_and_distribution():
    assert summary._format_prior(None, precision=3) == "-"
    assert summary._format_prior("LogNormal(0, 1)", precision=3) == "LogNormal(0, 1)"
    assert (
        summary._format_prior(npd.LogNormal(0.0, 0.5), precision=3)
        == "LogNormal(loc=0, scale=0.5)"
    )


def test_collect_attaches_priors_by_name():
    model = Prior(mean_function=Zero(), kernel=RBF())
    priors = {"kernel.lengthscale": npd.LogNormal(0.0, 1.0)}
    records = summary._collect(model, priors=priors)
    matched = _record_by_name(records, "kernel.lengthscale")
    assert isinstance(matched.prior, npd.LogNormal)
    # Parameters without a registered prior stay empty.
    assert _record_by_name(records, "kernel.variance").prior is None


def test_summarise_renders_prior_column_when_priors_given():
    model = Prior(mean_function=Zero(), kernel=RBF())
    priors = {
        "kernel.lengthscale": npd.LogNormal(0.0, 1.0),
        "kernel.variance": npd.LogNormal(0.0, 1.0),
    }
    assert "LogNormal" in _capture(model, priors=priors)


def test_summarise_prior_column_defaults_to_dash():
    # Without a priors map the Prior column is present but unpopulated.
    assert "LogNormal" not in _capture(Prior(mean_function=Zero(), kernel=RBF()))

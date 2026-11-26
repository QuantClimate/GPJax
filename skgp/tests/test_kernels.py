"""A kernel given as a GPJax kernel object, and the units of every kernel's values.

Every kernel, named or given, sees the inputs in the units of your columns, so a length scale
or a period is in those units both when you give it and when skgp reports it. A kernel's
variance is relative to the target's variance. A name is the matching GPJax kernel with one
length scale per input, each starting at that input's standard deviation.
"""

import json
import pickle
import re
import warnings

import equinox as eqx
import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import paramax
import polars as pl
import pytest

from helpers import SCALES_FEATURES, capturing_records, scales_data, sine_data
import skgp
from skgp import GPClassifier, GPRegressor, SparseGPClassifier, SparseGPRegressor


def weekly_sales(weeks=156, seed=0):
    """Three years of weekly sales: a slow rise and a season that repeats every 52 weeks."""
    rng = np.random.default_rng(seed)
    week = np.arange(weeks, dtype=float)
    season = 80.0 * np.sin(2 * np.pi * week / 52.0)
    sales = 1_000.0 + 0.5 * week + season + 10.0 * rng.normal(size=weeks)
    return pl.DataFrame({"week": week, "sales": sales})


def rbf(lengthscale):
    """GPJax's RBF kernel with this length scale. GPJax wraps a plain number itself, though
    its annotations ask for a wrapped one, which a type checker would flag at every call."""
    return gpx.kernels.RBF(lengthscale=lengthscale)


def learned_kernel(fitted):
    return json.loads(fitted.to_json())["learned"]["kernel"]


def season_kernel():
    """A trend and a season, written in weeks, the units of the `week` column."""
    return rbf(100.0) + gpx.kernels.Periodic(period=50.0)


@pytest.fixture(scope="module")
def season_fit():
    return GPRegressor("week", "sales", kernel=season_kernel()).fit(weekly_sales())


# The period the season fit learns, in weeks, is checked by the README's example
# (`tests/test_readme.py`).
@pytest.mark.slow
def test_each_learned_value_of_a_gpjax_kernel_is_named_by_its_place(season_fit):
    assert set(learned_kernel(season_fit)) == {
        "kernels[0].lengthscale",
        "kernels[0].variance",
        "kernels[1].lengthscale",
        "kernels[1].variance",
        "kernels[1].period",
    }


@pytest.mark.slow
def test_a_fitted_model_with_a_gpjax_kernel_names_it_by_its_classes(season_fit):
    _, second, third = repr(season_fit).split("\n")

    assert second == (
        "  features week; target sales; kernel RBF + Periodic; mean_function constant; "
        "learned_noise constant"
    )
    assert third.startswith("  learned kernels[0].lengthscale ")
    assert "; kernels[1].period 5" in third
    assert json.loads(season_fit.to_json())["setup"]["kernel"] == "RBF + Periodic"


def test_a_kernel_inside_another_is_named_in_brackets():
    kernel = (gpx.kernels.RBF() + gpx.kernels.Linear()) * gpx.kernels.Periodic()

    model = GPRegressor("x", "y", kernel=kernel)

    assert "kernel=(RBF + Linear) * Periodic" in repr(model)


@pytest.mark.slow
def test_a_name_is_the_gpjax_kernel_with_one_length_scale_per_input_at_its_spread():
    data = scales_data()
    spread = data.select(SCALES_FEATURES).to_numpy().std(axis=0)
    # Made in 64 bits, so that it starts exactly where the named kernel does.
    with jax.enable_x64(True):
        given = gpx.kernels.Matern52(lengthscale=jnp.asarray(spread), n_dims=2)  # ty: ignore[invalid-argument-type]  # as in `rbf`

    named_fit = GPRegressor(SCALES_FEATURES, "temperature").fit(data)
    given_fit = GPRegressor(SCALES_FEATURES, "temperature", kernel=given).fit(data)

    named, by_object = learned_kernel(named_fit), learned_kernel(given_fit)
    assert set(named) == {"lengthscale", "variance"}
    assert set(named["lengthscale"]) == set(SCALES_FEATURES)
    for feature in SCALES_FEATURES:
        assert named["lengthscale"][feature] == pytest.approx(
            by_object["lengthscale"][feature], rel=1e-6
        )
    assert named["variance"] == pytest.approx(by_object["variance"], rel=1e-6)


def test_a_variance_is_relative_to_the_targets_variance():
    data = sine_data(40)
    model = GPRegressor("x", "y")

    plain = learned_kernel(model.fit(data))
    # The same target in other units, a thousand times larger and far from zero.
    other_units = learned_kernel(
        model.fit(data.with_columns(pl.col("y") * 1_000.0 + 5_000.0))
    )

    assert other_units["variance"] == pytest.approx(plain["variance"], rel=1e-6)


@pytest.mark.parametrize(
    "lengthscale", [500.0, jnp.array([500.0])], ids=["a number", "an array of one"]
)
def test_a_kernel_with_one_length_scale_keeps_one(lengthscale):
    # A short fit: this test checks how many lengthscales are reported.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        fitted = GPRegressor(
            SCALES_FEATURES, "temperature", kernel=rbf(lengthscale)
        ).fit(scales_data(), iterations=20)

    # One number, not one for each of the two inputs.
    assert re.fullmatch(
        r"  learned lengthscale \S+; variance \S+; mean_function.constant \S+; noise \S+",
        repr(fitted).split("\n")[2],
    )


def test_a_value_gpjax_holds_as_a_plain_array_is_learned_and_reported():
    # GPJax's polynomial kernel keeps a `shift` given as an array as it is.
    kernel = gpx.kernels.Polynomial(shift=jnp.array(1.0))
    # A short fit: this test checks that the shift is learned and reported.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        fitted = GPRegressor("x", "y", kernel=kernel).fit(sine_data(40), iterations=5)

    assert set(learned_kernel(fitted)) == {"shift", "variance"}
    assert learned_kernel(fitted)["shift"] != pytest.approx(1.0)


@pytest.mark.slow
def test_a_kernel_with_its_own_compute_engine_is_fitted_with_it():
    # Random Fourier features have no kernel function, only their compute engine's features.
    kernel = gpx.kernels.RFF(base_kernel=gpx.kernels.RBF(n_dims=1), num_basis_fns=50)
    data = sine_data(40)

    fitted = GPRegressor("x", "y", kernel=kernel).fit(data)

    x = np.linspace(-1.5, 1.5, 7)
    mean = fitted.predict(pl.DataFrame({"x": x})).mean()
    np.testing.assert_allclose(mean, np.sin(3 * x), atol=0.3)


def test_a_value_held_fixed_in_gpjax_stays_where_you_put_it():
    periodic = gpx.kernels.Periodic(period=52.0)
    periodic = eqx.tree_at(
        lambda k: k.period, periodic, paramax.non_trainable(periodic.period)
    )

    fitted = GPRegressor("week", "sales", kernel=rbf(100.0) + periodic).fit(
        weekly_sales()
    )

    assert learned_kernel(fitted)["kernels[1].period"] == pytest.approx(52.0, abs=1e-5)


# Each kind of fit, on the same 40 rows; the minibatch fits run a fixed number of iterations.
SCALED_FITS = {
    "exact": (GPRegressor("x", "y"), "y", {}),
    "sparse on all rows": (SparseGPRegressor("x", "y", num_inducing=10), "y", {}),
    "sparse on minibatches": (
        SparseGPRegressor("x", "y", num_inducing=10, minibatch_size=20),
        "y",
        {"iterations": 600},
    ),
    "classifier": (GPClassifier("x", "above"), "probability", {}),
    "sparse classifier on all rows": (
        SparseGPClassifier("x", "above", num_inducing=10),
        "probability",
        {},
    ),
    "sparse classifier on minibatches": (
        SparseGPClassifier("x", "above", num_inducing=10, minibatch_size=20),
        "probability",
        {"iterations": 600},
    ),
}


def predicted(model, data, rows, quantity, fit_settings):
    """What the model, fitted to `data`, predicts at `rows`: the mean of a regressor, the
    probability of the second class of a classifier."""
    # The tests compare a fit in one set of units with the same fit in other units.
    # skgp rescales the inputs, so both fits take the same steps, finished or not.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        fitted = model.fit(data, **fit_settings)
    prediction = fitted.predict(rows)
    values = prediction.mean() if quantity == "y" else prediction.probabilities()[:, 1]
    return values, learned_kernel(fitted)


@pytest.mark.slow
@pytest.mark.parametrize("kind", list(SCALED_FITS))
@pytest.mark.parametrize("factor", [10_000.0, 1 / 10_000.0])
def test_a_fit_learns_the_same_curve_whatever_the_units_of_its_input(kind, factor):
    model, quantity, fit_settings = SCALED_FITS[kind]
    data = sine_data(40).with_columns(above=pl.col("y") > 0)
    rows = pl.DataFrame({"x": np.linspace(-1.5, 1.5, 7)})

    plain, plain_learned = predicted(model, data, rows, quantity, fit_settings)
    scaled, scaled_learned = predicted(
        model,
        data.with_columns(pl.col("x") * factor),
        rows.with_columns(pl.col("x") * factor),
        quantity,
        fit_settings,
    )

    # Each fit stops where it judges itself settled, a little apart from the other: about
    # 0.001 in a probability on CI's machines.
    tolerance = 0.05 if "minibatches" in kind else 0.01
    np.testing.assert_allclose(scaled, plain, atol=tolerance)
    assert scaled_learned["lengthscale"]["x"] == pytest.approx(
        plain_learned["lengthscale"]["x"] * factor,
        rel=0.1 if "minibatches" in kind else 0.05,
    )


def test_two_models_with_equal_kernels_are_equal_and_hash_alike():
    first = GPRegressor("x", "y", kernel=rbf(2.0))
    same = GPRegressor("x", "y", kernel=rbf(2.0))
    other = GPRegressor("x", "y", kernel=rbf(3.0))

    assert first == same
    assert hash(first) == hash(same)
    assert first != other
    assert first != GPRegressor("x", "y", kernel="rbf")
    assert len({first, same, other}) == 2


@pytest.mark.parametrize(
    ("features", "active_dims", "how_many"),
    [
        (["x", "z"], [0, 2], "there are only 2 features ('x' and 'z')"),
        (["x"], [1], "there is only 1 feature ('x')"),
    ],
)
def test_a_kernel_that_uses_an_input_past_the_last_feature_is_refused(
    features, active_dims, how_many
):
    with pytest.raises(ValueError, match="active_dims") as error:
        GPRegressor(features, "y", kernel=gpx.kernels.RBF(active_dims=active_dims))

    assert str(error.value) == (
        f"The kernel's RBF has active_dims {active_dims}, but {how_many}: active_dims "
        "are positions in features, counting from 0."
    )


@pytest.mark.slow
def test_a_fitted_model_with_a_gpjax_kernel_predicts_the_same_after_pickling(
    season_fit,
):
    rows = pl.DataFrame({"week": [10.5, 200.0]})

    loaded = pickle.loads(pickle.dumps(season_fit))  # nosec B301: loads what this test just pickled

    np.testing.assert_allclose(
        loaded.predict(rows).mean(), season_fit.predict(rows).mean(), rtol=1e-12
    )


def test_a_verbose_fit_names_a_gpjax_kernel_by_its_classes():
    # A two-step fit: this test checks only the log record that names the kernel.
    with capturing_records() as records, warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        GPRegressor("week", "sales", kernel=season_kernel()).fit(
            weekly_sales(), iterations=2, verbose=True
        )

    messages = [r["message"] for r in records]
    assert "kernel chosen: RBF + Periodic (set by you)" in messages

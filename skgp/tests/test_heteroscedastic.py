"""A regressor with `learned_noise="heteroscedastic"`: noise that varies with the inputs, learned as the
noise process, a second Gaussian process over the logarithm of the noise variance.

Pickling, the `ConvergenceWarning` and 64-bit numbers are tested for these fits with every
other kind, through the `unfinished` fixture."""

import json

import numpy as np
import polars as pl
import pytest

from helpers import sine_data
from skgp import GPRegressor, SparseGPRegressor


def noise_standard_deviation(x):
    """The noise in `growing_noise`: 0.1 at x = 0, growing to 3.1 at x = 10."""
    return 0.1 + 0.3 * x


def growing_noise(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, rows))
    y = 3 * np.sin(x) + noise_standard_deviation(x) * rng.normal(size=rows)
    return pl.DataFrame({"x": x, "y": y})


# Each kind of heteroscedastic fit, with the iterations it may take and how far off the true
# noise it may land: within that factor of it, either way. Adam's steps on minibatches leave
# a fit about as far from the best as they are noisy, and the learned constant mean, which
# the curve near the data can stand in for, wanders with them: over four seeds, 0.42 to 0.91
# where the noise is 0.4, and 0.41, 0.91 and 0.67 after 1,500, 3,000 and 6,000 iterations.
HETEROSCEDASTIC_FITS = {
    "at every row": (
        GPRegressor("x", "y", learned_noise="heteroscedastic"),
        2_000,
        1.25,
    ),
    "sparse on all rows": (
        SparseGPRegressor("x", "y", learned_noise="heteroscedastic", num_inducing=20),
        2_000,
        1.25,
    ),
    "sparse on minibatches": (
        SparseGPRegressor(
            "x",
            "y",
            learned_noise="heteroscedastic",
            num_inducing=20,
            minibatch_size=50,
        ),
        3_000,
        2.5,
    ),
}

# The heteroscedastic fits of the `unfinished` fixture.
UNFINISHED_HETEROSCEDASTIC = [
    "heteroscedastic at every row",
    "heteroscedastic sparse on all rows",
    "heteroscedastic sparse on minibatches",
]


@pytest.fixture(scope="module")
def sparse_fit():
    return HETEROSCEDASTIC_FITS["sparse on all rows"][0].fit(growing_noise())


@pytest.mark.slow
@pytest.mark.parametrize("kind", list(HETEROSCEDASTIC_FITS))
def test_a_heteroscedastic_regressor_learns_noise_that_grows_with_the_input(kind):
    model, iterations, factor = HETEROSCEDASTIC_FITS[kind]
    fitted = model.fit(growing_noise(), iterations=iterations)
    rows = pl.DataFrame({"x": [1.0, 5.0, 9.0]})

    with_noise = fitted.predict(rows).std()
    curve_only = fitted.predict(rows, include_noise=False).std()

    # The noise's own spread at each row, from the two spreads (variances add).
    noise = np.sqrt(with_noise**2 - curve_only**2)
    assert noise[0] < noise[1] < noise[2]
    true_noise = np.array([0.4, 1.6, 2.8])
    assert np.all(true_noise / factor < noise)
    assert np.all(noise < true_noise * factor)


@pytest.mark.parametrize("kind", UNFINISHED_HETEROSCEDASTIC)
def test_far_from_the_data_a_heteroscedastic_fit_predicts_with_the_values_it_reports(
    unfinished, kind
):
    fitted = unfinished[kind].fitted
    learned = json.loads(fitted.to_json())["learned"]
    target_std = sine_data(40)["y"].std(ddof=0)  # the data the fixture fitted
    far = pl.DataFrame({"x": [100.0]})

    curve_only = fitted.predict(far, include_noise=False).std()[0]
    with_noise = fitted.predict(far).std()[0]

    # Far from the data each process is at its prior. The curve's variance is the kernel's,
    # relative to the target's. The noise process has its mean, at which the noise is
    # `noise`, and its variance, the noise kernel's, which raises the mean noise variance a
    # log-normal gives by the exponential of half of it.
    np.testing.assert_allclose(
        curve_only, np.sqrt(learned["kernel"]["variance"]) * target_std, rtol=1e-5
    )
    np.testing.assert_allclose(
        with_noise**2 - curve_only**2,
        learned["noise"] ** 2 * np.exp(learned["noise_kernel"]["variance"] / 2),
        rtol=1e-5,
    )


@pytest.mark.slow
def test_a_heteroscedastic_fit_reports_the_noise_far_from_the_data_and_the_noise_kernel(
    sparse_fit,
):
    description = json.loads(sparse_fit.to_json())
    learned = description["learned"]

    assert description["setup"]["learned_noise"] == "heteroscedastic"
    assert list(learned) == ["kernel", "mean_function", "noise", "noise_kernel"]
    assert set(learned["noise_kernel"]) == {"lengthscale", "variance"}
    assert list(learned["noise_kernel"]["lengthscale"]) == ["x"]
    # Between the least and the most noise in the data, in y's units.
    assert 0.1 < learned["noise"] < 3.1


@pytest.mark.parametrize(
    "kind",
    ["heteroscedastic sparse on all rows", "heteroscedastic sparse on minibatches"],
)
def test_a_sparse_heteroscedastic_fit_reports_how_many_inducing_points_it_has(
    unfinished, kind
):
    description = json.loads(unfinished[kind].fitted.to_json())

    assert description["model"]["num_inducing"] == 10  # what the fixture asked for


@pytest.mark.slow
def test_a_printed_heteroscedastic_fit_names_the_noise_setting_and_the_noise_kernel(
    sparse_fit,
):
    _, second, third = repr(sparse_fit).split("\n")

    assert (
        second == "  features x; target y; kernel matern52; mean_function constant; "
        "learned_noise heteroscedastic"
    )
    assert "; noise_kernel.lengthscale x " in third
    assert "; noise_kernel.variance " in third


@pytest.mark.parametrize(
    ("kind", "model_line", "optimiser_line"),
    [
        (
            "heteroscedastic at every row",
            (
                "model chosen: Gaussian process (variational, with an inducing point at "
                "every row) with Gaussian noise that varies with the inputs"
            ),
            (
                "optimiser chosen: L-BFGS-B on the approximate marginal likelihood, at "
                "most 1 iteration"
            ),
        ),
        (
            "heteroscedastic sparse on all rows",
            (
                "model chosen: sparse Gaussian process (fitted on all rows at once) with "
                "Gaussian noise that varies with the inputs"
            ),
            (
                "optimiser chosen: L-BFGS-B on the approximate marginal likelihood, at "
                "most 1 iteration"
            ),
        ),
        (
            "heteroscedastic sparse on minibatches",
            (
                "model chosen: sparse Gaussian process (fitted on minibatches of 20 "
                "rows) with Gaussian noise that varies with the inputs"
            ),
            "optimiser chosen: Adam on the approximate marginal likelihood, 2 iterations",
        ),
    ],
)
def test_a_verbose_heteroscedastic_fit_names_its_model_and_optimiser(
    unfinished, kind, model_line, optimiser_line
):
    messages = [r["message"] for r in unfinished[kind].records]

    assert model_line in messages
    assert optimiser_line in messages


def test_the_learned_noise_must_be_constant_heteroscedastic_or_none():
    with pytest.raises(ValueError, match=r"^learned_noise must be") as error:
        GPRegressor("x", "y", learned_noise="varying")  # ty: ignore[invalid-argument-type]  # the wrong value is the point

    assert str(error.value) == (
        "learned_noise must be 'constant' or 'heteroscedastic'; got 'varying'."
    )

import warnings

import gpjax as gpx
import jax
import jax.numpy as jnp
import polars as pl
import pytest

from skgp import GPRegressor, SparseGPRegressor

# Five rows already at mean 0 and standard deviation 1 in both columns, which is the scale skgp
# fits in, so the rows can be handed to the GPJax model as they are. The target equals the input.
CENTRED = pl.DataFrame(
    {"x": [-1.5, -0.5, 0.0, 0.5, 1.5], "y": [-1.5, -0.5, 0.0, 0.5, 1.5]}
)

REGRESSORS = {
    "exact": GPRegressor(features=["x"], target="y"),
    "collapsed": SparseGPRegressor(features=["x"], target="y", num_inducing=5),
    "minibatches": SparseGPRegressor(
        features=["x"], target="y", num_inducing=5, minibatch_size=2
    ),
}


@pytest.fixture(scope="module", params=list(REGRESSORS))
def centred_fit(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a line with no noise can stop a fit early
        return REGRESSORS[request.param].fit(CENTRED)


@pytest.mark.slow
def test_the_gpjax_model_conditioned_on_the_training_rows_predicts_the_line(
    centred_fit,
):
    with jax.enable_x64(True):
        training = gpx.Dataset(
            X=jnp.asarray(CENTRED["x"].to_numpy().reshape(-1, 1)),
            y=jnp.asarray(CENTRED["y"].to_numpy().reshape(-1, 1)),
        )
        latent = centred_fit.gpjax_model.condition(training)(jnp.asarray([[0.5]]))

    assert float(latent.mean[0]) == pytest.approx(0.5, abs=0.05)
    assert centred_fit.predict(pl.DataFrame({"x": [0.5]})).mean() == pytest.approx(
        [0.5], abs=0.05
    )


@pytest.mark.slow
def test_nothing_a_user_prints_or_reads_from_a_fit_or_a_prediction_is_a_gpjax_object(
    centred_fit,
):
    prediction = centred_fit.predict(pl.DataFrame({"x": [0.0, 0.5]}))

    for shown in (
        repr(centred_fit),
        str(centred_fit),
        repr(prediction),
        str(prediction),
    ):
        assert "gpjax" not in shown.lower()
    assert not hasattr(prediction, "distribution")
    public = {
        name: getattr(prediction, name)
        for name in dir(prediction)
        if not name.startswith("_")
    }
    assert not [
        name for name, value in public.items() if "gpjax" in type(value).__module__
    ]

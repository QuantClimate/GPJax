"""Saving a fitted model with pickle and loading it back, the way docs/saving.md suggests.

skgp has no save function of its own (decision record 0005), so these tests check that the
way the docs suggest works for every kind of fit.
"""

import pickle
import warnings

import jax
import numpy as np
from pandera.errors import SchemaError
import pandera.polars as pap
import polars as pl
import pytest

from helpers import KINDS, sine_data
import skgp
from skgp import GPRegressor, SparseGPRegressor

NEW_ROWS = pl.DataFrame({"x": [-1.5, 0.0, 1.5]})

# A target already at mean 0 and standard deviation 1, the scale skgp fits in, so a model that
# keeps the training target keeps exactly these five numbers.
TARGET_AT_THE_FITTED_SCALE = [-1.5, 0.5, 0.0, -0.5, 1.5]


@pytest.mark.parametrize(
    "kind",
    [
        "exact",
        "sparse on all rows",
        "sparse on minibatches",
        "heteroscedastic at every row",
        "heteroscedastic sparse on all rows",
        "heteroscedastic sparse on minibatches",
    ],
)
def test_a_pickled_model_predicts_and_describes_itself_as_before(unfinished, kind):
    fitted = unfinished[kind].fitted

    loaded = pickle.loads(pickle.dumps(fitted))

    # 32-bit numbers would differ by about one part in ten million, so this catches them.
    before, after = fitted.predict(NEW_ROWS), loaded.predict(NEW_ROWS)
    np.testing.assert_allclose(after.mean(), before.mean(), rtol=1e-9)
    np.testing.assert_allclose(after.std(), before.std(), rtol=1e-9)
    assert loaded.to_json() == fitted.to_json()  # read from the same saved numbers


@pytest.mark.parametrize(
    "kind",
    ["classifier", "sparse classifier on all rows", "sparse classifier on minibatches"],
)
def test_a_pickled_classifier_predicts_and_describes_itself_as_before(unfinished, kind):
    fitted = unfinished[kind].fitted

    loaded = pickle.loads(pickle.dumps(fitted))

    np.testing.assert_allclose(
        loaded.predict(NEW_ROWS).probabilities(),
        fitted.predict(NEW_ROWS).probabilities(),
        rtol=1e-9,
    )
    assert loaded.to_json() == fitted.to_json()


@pytest.mark.parametrize("kind", KINDS)
def test_a_pickled_model_keeps_its_learned_numbers_64_bit(unfinished, kind):
    loaded = pickle.loads(pickle.dumps(unfinished[kind].fitted))

    arrays = [
        leaf for leaf in jax.tree.leaves(loaded.gpjax_model) if hasattr(leaf, "dtype")
    ]
    assert arrays
    assert {str(array.dtype) for array in arrays} == {"float64"}


def test_a_prediction_does_not_make_the_pickled_model_larger():
    # The first prediction keeps the model conditioned on the training data, whose size
    # grows with the square of the rows (32 MB at 2,000 rows). Pickle leaves it out.
    fitted = GPRegressor(features=["x"], target="y").fit(sine_data(n=40))
    before = len(pickle.dumps(fitted))

    fitted.predict(NEW_ROWS)

    assert len(pickle.dumps(fitted)) == before


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "keeps_the_target"),
    [
        (GPRegressor("x", "y"), True),
        (SparseGPRegressor("x", "y"), True),
        (SparseGPRegressor("x", "y", minibatch_size=2), False),
        (GPRegressor("x", "y", learned_noise="heteroscedastic"), False),
        (SparseGPRegressor("x", "y", learned_noise="heteroscedastic"), False),
        (
            SparseGPRegressor(
                "x", "y", learned_noise="heteroscedastic", minibatch_size=2
            ),
            False,
        ),
    ],
)
def test_only_a_model_that_needs_the_training_target_to_predict_saves_it(
    model, keeps_the_target
):
    # An exact model and a sparse model fitted on all rows at once are conditioned on the
    # training data to predict; every other fit carries its fitted distribution over the
    # inducing points in its place.
    data = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": TARGET_AT_THE_FITTED_SCALE}
    )
    with warnings.catch_warnings():
        # A fit of two iterations has not finished.
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        fitted = model.fit(data, iterations=2)

    target = np.array(TARGET_AT_THE_FITTED_SCALE).tobytes()
    assert (target in pickle.dumps(fitted)) is keeps_the_target


def test_a_pickled_model_keeps_its_pandera_schema_and_still_checks_with_it():
    in_range = pap.Column(float, pap.Check.in_range(-2.0, 2.0))
    schema = pap.DataFrameSchema({"x": in_range})
    fitted = GPRegressor(features=["x"], target="y", schema=schema).fit(sine_data(n=40))

    loaded = pickle.loads(pickle.dumps(fitted))

    assert loaded.model.schema == schema
    with pytest.raises(SchemaError):
        loaded.predict(pl.DataFrame({"x": [5.0]}))

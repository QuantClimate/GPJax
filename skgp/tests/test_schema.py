"""A user's own data checker (`schema`): when it runs, on what, and what it can change."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pandera.polars as pap
import polars as pl
import pytest

from helpers import SCALES_FEATURES, Recorder, good_frame, scales_data
from skgp import GPRegressor


class StrictSchemaError(Exception):
    pass


def test_schema_is_optional():
    assert GPRegressor(features=["x"], target="y").schema is None


def test_a_schema_that_raises_stops_check_fit_and_predict_with_that_error():
    error = StrictSchemaError("y must be positive")
    model = GPRegressor(features=["x"], target="y", schema=Recorder(raises=error))
    passes_then_raises = Recorder()
    fitted = GPRegressor(features=["x"], target="y", schema=passes_then_raises).fit(
        good_frame()
    )
    passes_then_raises.raises = error

    for call, frame in (
        (model.check, good_frame()),
        (model.fit, good_frame()),
        (fitted.predict, good_frame().drop("y")),
    ):
        with pytest.raises(StrictSchemaError, match="y must be positive") as raised:
            call(frame)
        assert raised.value is error


def test_the_schema_runs_before_skgp_checks_so_its_error_wins():
    schema = Recorder(raises=StrictSchemaError("custom rule"))

    with pytest.raises(StrictSchemaError, match="custom rule"):
        GPRegressor(features=["x"], target="y", schema=schema).check(
            good_frame().drop("y")
        )


@pytest.mark.parametrize("make_frame", [pl.DataFrame, pd.DataFrame])
def test_the_schema_receives_the_frame_as_the_caller_passed_it_in_their_own_dataframe_type(
    make_frame,
):
    columns = {"x": [float(i) for i in range(10)], "y": [float(i) for i in range(10)]}
    check_schema, fit_schema = Recorder(), Recorder()
    frame = make_frame(columns)
    short = make_frame({name: values[:4] for name, values in columns.items()})

    GPRegressor(features=["x"], target="y", schema=check_schema).check(frame)
    # a frame too short to fit, so that the checks which follow the schema stop the fit
    with pytest.raises(ValueError, match="The data has 4 rows, but at least 5 rows"):
        GPRegressor(features=["x"], target="y", schema=fit_schema).fit(short)

    (seen_by_check,), (seen_by_fit,) = check_schema.seen, fit_schema.seen
    assert type(seen_by_check) is make_frame
    assert seen_by_check.equals(frame)
    assert type(seen_by_fit) is make_frame
    assert seen_by_fit.equals(short)


def test_skgp_checks_run_on_what_the_schema_returned_and_still_run_after_one_that_accepts_everything():
    returns_nothing, returns_a_short_frame, accepts_everything = (
        Recorder(returns=None),
        Recorder(returns=good_frame(n=4)),
        Recorder(),
    )

    with pytest.raises(
        ValueError, match="The data has 4 rows, but at least 5 rows"
    ):  # the frame as passed
        GPRegressor(features=["x"], target="y", schema=returns_nothing).check(
            good_frame(n=4)
        )
    with pytest.raises(
        ValueError, match="The data has 4 rows, but at least 5 rows"
    ):  # the frame the schema returned
        GPRegressor(features=["x"], target="y", schema=returns_a_short_frame).check(
            good_frame(n=10)
        )
    infinite = good_frame().with_columns(pl.Series("x", [float("inf")] + [1.0] * 9))
    with pytest.raises(
        ValueError, match="Column 'x' has 1 infinite value out of 10 rows"
    ):
        GPRegressor(features=["x"], target="y", schema=accepts_everything).check(
            infinite
        )


def text_numbers():
    """A frame whose columns hold numbers written as text, which skgp alone would refuse."""
    return {
        "x": [str(float(i)) for i in range(10)],
        "y": [str(float(i) ** 2) for i in range(10)],
    }


@pytest.mark.parametrize(
    ("library", "make_schema", "make_frame"),
    [(pap, pap.DataFrameSchema, pl.DataFrame), (pa, pa.DataFrameSchema, pd.DataFrame)],
    ids=["pandera-polars", "pandera-pandas"],
)
def test_skgp_checks_see_the_frame_a_pandera_schema_coerced(
    library, make_schema, make_frame
):
    schema = make_schema(
        {
            "x": library.Column(float, coerce=True),
            "y": library.Column(float, coerce=True),
        }
    )
    frame = make_frame(text_numbers())

    with pytest.raises(ValueError, match="Column 'x' is not numeric"):
        GPRegressor(features=["x"], target="y").check(frame)
    GPRegressor(features=["x"], target="y", schema=schema).check(
        frame
    )  # no error: skgp's checks saw numbers


def test_a_pandera_schema_error_propagates_unchanged():
    polars_schema = pap.DataFrameSchema(
        {"x": pap.Column(float), "y": pap.Column(float, pap.Check.ge(0))}
    )
    pandas_schema = pa.DataFrameSchema(
        {"x": pa.Column(float), "y": pa.Column(float, pa.Check.ge(0))}
    )
    bad = {"x": [float(i) for i in range(10)], "y": [-1.0] * 10}

    with pytest.raises(pa.errors.SchemaError, match="Column 'y' failed"):
        GPRegressor(features=["x"], target="y", schema=polars_schema).check(
            pl.DataFrame(bad)
        )
    with pytest.raises(pa.errors.SchemaError, match="Column 'y' failed"):
        GPRegressor(features=["x"], target="y", schema=pandas_schema).check(
            pd.DataFrame(bad)
        )


@pytest.mark.parametrize(
    "not_a_schema",
    [
        "a string",
        3,
        object(),
        {"validate": 1},
        type("NotCallable", (), {"validate": "nope"})(),
    ],
)
def test_a_value_without_a_callable_validate_method_is_rejected_at_definition_saying_what_is_needed(
    not_a_schema,
):
    with pytest.raises(
        ValueError,
        match=r"^schema must be an object with a \.validate\(frame\) method",
    ):
        GPRegressor(features=["x"], target="y", schema=not_a_schema)


@pytest.mark.slow
def test_a_fitted_model_predicts_from_the_frame_its_schema_coerced():
    coercing = pap.DataFrameSchema(
        {
            "latitude": pap.Column(float, coerce=True),
            "altitude": pap.Column(float, coerce=True),
            "temperature": pap.Column(float, coerce=True, required=False),
        }
    )
    fitted = GPRegressor(
        features=SCALES_FEATURES, target="temperature", schema=coercing
    ).fit(scales_data(n=50))
    numbers = pl.DataFrame({"latitude": [50.5, 51.5], "altitude": [500.0, 2500.0]})
    text = pl.DataFrame({"latitude": ["50.5", "51.5"], "altitude": ["500", "2500"]})

    np.testing.assert_allclose(
        fitted.predict(text).mean(), fitted.predict(numbers).mean()
    )


SCHEMAS = {
    "plain": Recorder,
    "pandera-polars": lambda: pap.DataFrameSchema({"x": pap.Column(float)}),
    "pandera-pandas": lambda: pa.DataFrameSchema({"x": pa.Column(float)}),
}


@pytest.mark.parametrize("kind", SCHEMAS)
def test_a_model_with_a_schema_can_be_compared_and_hashed_and_differs_from_one_without(
    kind,
):
    schema = SCHEMAS[kind]()
    a = GPRegressor(features=["x"], target="y", schema=schema)
    b = GPRegressor(features=["x"], target="y", schema=schema)

    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1}[b] == 1
    assert a != GPRegressor(features=["x"], target="y", kernel="rbf", schema=schema)
    assert a != GPRegressor(features=["x"], target="y")


def test_two_separately_built_but_equal_pandera_schemas_make_equal_models():
    def build():
        return GPRegressor(
            features=["x"],
            target="y",
            schema=pap.DataFrameSchema({"x": pap.Column(float)}),
        )

    assert build() == build()

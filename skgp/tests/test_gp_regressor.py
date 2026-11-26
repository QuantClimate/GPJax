import inspect
import re
from typing import Literal

from gpjax.kernels import AbstractKernel
from gpjax.mean_functions import AbstractMeanFunction
import polars as pl
import pytest

from skgp import GPRegressor, SparseGPRegressor


def test_kernel_defaults_to_matern52():
    assert GPRegressor(features=["x"], target="y").kernel == "matern52"


@pytest.mark.parametrize("regressor", [GPRegressor, SparseGPRegressor])
def test_features_and_target_may_be_given_by_position_and_one_input_by_its_name(
    regressor,
):
    assert regressor("x", "y") == regressor(features=["x"], target="y")
    assert regressor(["x", "z"], "y").features == ("x", "z")


def test_each_mistake_in_the_settings_is_one_plain_line_naming_the_setting_and_what_was_given():
    with pytest.raises(ValueError, match=r"^target must be a column name") as raised:
        GPRegressor(["x"], 3, kernel="periodic")  # ty: ignore[invalid-argument-type]  # the wrong value is the point

    assert str(raised.value) == (
        "target must be a column name, as text; got 3.\n"
        "kernel must be 'rbf', 'matern32' or 'matern52', or a GPJax kernel such as "
        "gpjax.kernels.RBF(); got 'periodic'."
    )


def test_an_unknown_setting_is_named_with_the_closest_real_one():
    with pytest.raises(
        ValueError,
        match=r"^GPRegressor has no setting 'kernal'; did you mean 'kernel'\?$",
    ):
        GPRegressor(["x"], "y", kernal="rbf")  # ty: ignore[unknown-argument]  # the misspelling is the point
    with pytest.raises(ValueError, match=r"^GPRegressor has no setting 'colour'\.$"):
        GPRegressor(["x"], "y", colour="red")  # ty: ignore[unknown-argument]  # the unknown name is the point


@pytest.mark.parametrize(
    ("features", "target", "said"),
    [
        (
            ["x", "z", "x"],
            "y",
            "features names 'x' more than once; name each column once.",
        ),
        (
            ["x", "y"],
            "y",
            "The target 'y' is also in features; a column cannot be both an input and the target.",
        ),
        (
            ["x", ""],
            "y",
            "features includes an empty column name ''; every column needs a name.",
        ),
        (
            ["x"],
            "",
            "target is an empty column name ''; give the name of the target column.",
        ),
    ],
    ids=[
        "feature named twice",
        "target among the features",
        "empty feature",
        "empty target",
    ],
)
def test_a_column_named_twice_or_left_unnamed_is_refused_when_the_model_is_defined(
    features, target, said
):
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        GPRegressor(features, target)


def test_a_model_cannot_be_changed_but_replace_returns_a_new_model_checked_like_any_other():
    model = GPRegressor(["x"], "y")

    with pytest.raises(
        ValueError,
        match=r"never changes, so its kernel cannot be set\. model\.replace\(kernel=\.\.\.\)",
    ):
        model.kernel = "rbf"  # ty: ignore[invalid-assignment]  # changing it is the point
    with pytest.raises(ValueError, match=r"^kernel must be .*; got 'nonsense'\.$"):
        model.replace(kernel="nonsense")
    assert model.replace(kernel="rbf") == GPRegressor(["x"], "y", kernel="rbf")
    assert model.kernel == "matern52"


class Schema:
    def validate(self, frame):
        return None

    def __repr__(self):
        return "Schema()"


def test_a_model_prints_its_columns_first_and_its_schema_last_and_only_when_given():
    assert (
        repr(GPRegressor(["latitude", "altitude"], "temperature"))
        == "GPRegressor(features=['latitude', 'altitude'], target='temperature', kernel='matern52', mean_function='constant', learned_noise='constant', noise_column=None)"
    )
    sparse = repr(SparseGPRegressor(["x"], "y", minibatch_size=4, schema=Schema()))
    assert sparse.startswith(
        "SparseGPRegressor(features=['x'], target='y', kernel='matern52', "
        "mean_function='constant', learned_noise='constant', "
        "minibatch_size=4, "
    )
    assert sparse.endswith(", schema=Schema())")


def test_help_shows_each_setting_with_a_plain_type_and_its_default():
    sparse = str(inspect.signature(SparseGPRegressor))
    settings = inspect.signature(GPRegressor).parameters

    assert [str(settings[name]) for name in ("features", "target", "schema")] == [
        "features: str | collections.abc.Iterable[str]",
        "target: str",
        "schema: Any = None",
    ]
    # Python writes a union of a `Literal` and a class differently from one version to the
    # next, so the kernel's type is compared as a type.
    kernel = settings["kernel"]
    assert kernel.annotation == Literal["rbf", "matern32", "matern52"] | AbstractKernel
    assert kernel.default == "matern52"
    mean_function = settings["mean_function"]
    assert (
        mean_function.annotation
        == Literal["zero", "constant", "linear"] | AbstractMeanFunction
    )
    assert mean_function.default == "constant"
    learned_noise = settings["learned_noise"]
    assert learned_noise.annotation == Literal["constant", "heteroscedastic"] | None
    assert learned_noise.default == "constant"
    assert settings["noise_column"].default is None
    assert list(settings) == [
        "features",
        "target",
        "kernel",
        "mean_function",
        "learned_noise",
        "noise_column",
        "schema",
    ]
    assert (
        "minibatch_size: int | None = None, num_inducing: int | None = None" in sparse
    )
    assert "random_state: int = 0" in sparse
    assert "Annotated" not in sparse


def test_only_a_fitted_model_can_predict():
    assert not hasattr(GPRegressor(features=["x"], target="y"), "predict")


def test_fit_leaves_the_model_unchanged_and_two_fits_of_one_model_are_independent():
    model = GPRegressor(features=["x"], target="y")
    before = model.model_copy()
    data = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]}
    )
    low = model.fit(data)
    high = model.fit(
        data.with_columns(y=pl.col("y") + 10)
    )  # the same curve, exactly 10 higher
    new = pl.DataFrame({"x": [1.5, 2.5]})

    assert model == before
    assert high.predict(new).mean() - low.predict(new).mean() == pytest.approx(
        [10.0, 10.0], abs=0.5
    )

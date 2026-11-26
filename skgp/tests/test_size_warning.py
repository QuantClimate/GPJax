import warnings

import pytest

from helpers import sine_data
import skgp


def exact():
    return skgp.GPRegressor(features=["x"], target="y")


def large_data_warnings(call):
    """What `call` returned and the `LargeDataWarning`s it issued (other warnings are not our concern here)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [w for w in caught if issubclass(w.category, skgp.LargeDataWarning)]


def test_the_default_limit_is_ten_thousand_rows():
    assert skgp.config.size_warning_rows == 10_000


@pytest.mark.parametrize(
    ("value", "said"),
    [
        (-1, "greater than or equal to 0"),
        (True, "valid integer"),  # a bool is not a whole number here
    ],
)
def test_a_warning_limit_that_is_not_a_whole_number_of_zero_or_more_is_refused_when_assigned(
    value, said
):
    with pytest.raises(ValueError, match=rf"(?s)size_warning_rows.*{said}"):
        skgp.config.size_warning_rows = value


@pytest.mark.slow
def test_an_exact_fit_on_more_rows_than_the_limit_warns_once_naming_the_sparse_model_and_still_fits(
    monkeypatch,
):
    data = sine_data(21)
    monkeypatch.setattr(skgp.config, "size_warning_rows", 20)

    fitted, warned = large_data_warnings(lambda: exact().fit(data))

    assert len(warned) == 1
    assert "SparseGPRegressor" in str(warned[0].message)
    assert warned[0].filename == __file__  # points at the user's call to fit
    # the fit went ahead: at its own training points the model is near the data, noise aside
    assert fitted.predict(data.head(2)).mean() == pytest.approx(
        data["y"].head(2).to_numpy(), abs=0.3
    )


def test_as_many_rows_as_the_limit_is_silent(monkeypatch):
    data = sine_data(21)
    monkeypatch.setattr(
        skgp.config, "size_warning_rows", 21
    )  # as many rows as the limit

    _, warned = large_data_warnings(lambda: exact().fit(data))

    assert warned == []


def test_a_sparse_fit_never_warns_about_size(monkeypatch):
    monkeypatch.setattr(skgp.config, "size_warning_rows", 5)
    sparse = skgp.SparseGPRegressor(features=["x"], target="y", num_inducing=5)

    _, warned = large_data_warnings(lambda: sparse.fit(sine_data(12)))

    assert warned == []


@pytest.mark.slow
def test_a_heteroscedastic_fit_at_every_row_on_more_rows_than_the_limit_warns_naming_the_sparse_model(
    monkeypatch,
):
    monkeypatch.setattr(skgp.config, "size_warning_rows", 10)
    model = skgp.GPRegressor(
        features=["x"], target="y", learned_noise="heteroscedastic"
    )

    _, warned = large_data_warnings(lambda: model.fit(sine_data(11), iterations=2))

    (warning,) = warned
    assert str(warning.message) == (
        "11 rows is a lot for a GPRegressor with heteroscedastic noise, which gets slow "
        "and needs a lot of memory as rows grow. Consider SparseGPRegressor, which takes "
        "the same settings. The fit carries on."
    )

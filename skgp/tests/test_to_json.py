from importlib.metadata import version
import json
import re
import warnings

import polars as pl
import pytest

from helpers import NOISY_LINE
import skgp
from skgp import GPRegressor


def reject_constant(constant):
    raise AssertionError(f"{constant} is not valid JSON")


@pytest.fixture(scope="module")
def noisy_line_fit():
    return GPRegressor(features=["x"], target="y", kernel="rbf").fit(NOISY_LINE)


def test_the_json_names_the_setup_the_row_count_and_the_skgp_version(noisy_line_fit):
    description = json.loads(noisy_line_fit.to_json())

    assert set(description) == {
        "skgp_version",
        "training_rows",
        "setup",
        "learned",
        "fitting",
    }
    assert description["skgp_version"] == version("skgp")
    assert description["training_rows"] == 12
    assert description["setup"] == {
        "features": ["x"],
        "target": "y",
        "kernel": "rbf",
        "mean_function": "constant",
        "learned_noise": "constant",
    }
    # One length scale, named after the input.
    assert list(description["learned"]["kernel"]["lengthscale"]) == ["x"]


def test_learned_length_scales_and_noise_are_in_the_users_original_units(
    noisy_line_fit,
):
    learned = json.loads(noisy_line_fit.to_json())["learned"]

    assert set(learned) == {"kernel", "mean_function", "noise"}
    assert set(learned["kernel"]) == {"lengthscale", "variance"}
    # The wobble, not a hundredth of it.
    assert learned["noise"] == pytest.approx(5.0, rel=0.3)
    # In units of x, which runs over 110.
    assert learned["kernel"]["lengthscale"]["x"] > 100


def test_the_json_says_whether_fitting_finished_properly_in_plain_language(
    noisy_line_fit,
):
    text = noisy_line_fit.to_json()

    fitting = json.loads(text, parse_constant=reject_constant)["fitting"]

    assert set(fitting) == {
        "finished_properly",
        "iterations",
        "final_objective",
        "stop_reason",
    }
    assert fitting["iterations"] >= 1
    assert fitting["stop_reason"] == (
        "It stopped because further steps no longer improved the fit."
    )
    for jargon in ("gpjax", "obs_stddev", "likelihood"):
        assert jargon not in text.lower()


def test_the_iteration_count_is_the_number_of_optimiser_steps_taken(unfinished):
    # `unfinished` fits are stopped after one step, by lowering the optimiser's limit.
    fitting = json.loads(unfinished["exact"].fitted.to_json())["fitting"]

    assert fitting["iterations"] == 1
    assert fitting["finished_properly"] is False


def test_a_printed_fitted_model_is_a_short_summary_in_the_users_units(noisy_line_fit):
    first, second, third = repr(noisy_line_fit).split("\n")

    assert first == (
        "FittedGPRegressor made by GPRegressor from 12 rows; the fit finished properly"
    )
    assert second == (
        "  features x; target y; kernel rbf; mean_function constant; "
        "learned_noise constant"
    )
    learned = re.fullmatch(
        r"  learned lengthscale x \S+; variance \S+; mean_function.constant \S+; "
        r"noise (\S+)",
        third,
    )
    assert learned is not None
    assert float(learned[1]) == pytest.approx(5.0, rel=0.3)  # the wobble, in y's units


def test_a_printed_fitted_model_with_many_columns_lists_the_first_ten():
    # Twelve inputs on six rows, cut short after one iteration: only the printing matters.
    columns = {
        f"c{i}": [float(row * (i + 2) % 7) for row in range(6)] for i in range(12)
    }
    rows = pl.DataFrame({**columns, "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
    # A one-step fit on purpose: this test checks how an unfinished fit is printed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        fitted = GPRegressor(features=list(columns), target="y").fit(rows, iterations=1)

    first, second, third = repr(fitted).split("\n")

    assert first.endswith("from 6 rows; the fit did not finish properly")
    assert second == (
        "  features c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 and 2 more; target y; "
        "kernel matern52; mean_function constant; learned_noise constant"
    )
    assert re.fullmatch(
        r"  learned lengthscale c0 \S+, (c[1-9] \S+, ){8}c9 \S+ and 2 more; variance \S+; "
        r"mean_function.constant \S+; noise \S+",
        third,
    )


def test_a_diverged_fit_still_gives_strict_json_with_null_for_non_finite_numbers(
    diverged_fit,
):
    description = json.loads(diverged_fit.to_json(), parse_constant=reject_constant)

    assert description["fitting"]["finished_properly"] is False
    assert description["fitting"]["final_objective"] is None
    assert description["learned"]["noise"] is None

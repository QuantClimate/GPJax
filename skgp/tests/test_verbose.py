"""What a user sees when they turn `verbose=True` on, and that they see nothing otherwise.

All the fits here are on five literal rows, so the numbers in the expected lines can be checked by eye:
x is 1..5 (mean 3, standard deviation 1.414) and y has mean 6 and standard deviation 2.789.
"""

import json
import re
from types import SimpleNamespace

from loguru import logger
import polars as pl
import pytest

from helpers import capturing_records
from skgp import GPRegressor, SparseGPRegressor
from skgp._constants import Constant as C

ROWS = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.1, 3.9, 6.2, 7.8, 10.0]})
NEW_ROWS = pl.DataFrame({"x": [1.5, 2.5, 3.5]})

RESCALING = (
    "standardise features x (shift 3, scale 1.414) and target (shift 6, scale 2.789)"
)
MISSING_VALUE = "Column 'y' has 1 missing value (empty or NaN) out of 5 rows. Remove or fill them first."


def messages(records):
    return [r["message"] for r in records]


@pytest.fixture
def skgp_logging_enabled():
    """The user has switched skgp's logging on, as a user of any loguru library may. Undone afterwards:
    `logger.enable("")` drops every enable and disable setting, which is loguru's starting state."""
    logger.enable("skgp")
    yield
    logger.enable("")


@pytest.fixture(scope="module")
def verbose_run():
    """A verbose fit of the default model on ROWS, and a verbose prediction on NEW_ROWS."""
    with capturing_records() as on_fit:
        fitted = GPRegressor(features=["x"], target="y").fit(ROWS, verbose=True)
    with capturing_records() as on_predict:
        fitted.predict(NEW_ROWS, verbose=True)
    return SimpleNamespace(fitted=fitted, on_fit=on_fit, on_predict=on_predict)


def test_verbose_fit_says_what_it_was_given_what_it_chose_and_how_it_ended(verbose_run):
    lines = messages(verbose_run.on_fit)

    # The optimiser's name and its iteration cap are left out: they may change without a user noticing.
    assert [
        line
        for line in lines
        if not line.startswith(("optimiser chosen", "fit finished"))
    ] == [
        "settings given: features x, target y; set by you: features, target; no schema supplied",
        "data checks passed: columns_present, enough_rows, numeric, no_missing_values, no_infinite_values (5 rows)",
        "kernel chosen: matern52 (the default)",
        "mean function chosen: constant (the default)",
        "model chosen: exact Gaussian process with Gaussian noise",
        f"rescaling chosen: {RESCALING}",
    ]
    # How long it took and what it learned vary a little between machines.
    number = r"-?[\d.]+(?:e[+-]\d+)?"
    assert re.fullmatch(
        rf"fit finished: finished properly after \d+ iterations in [\d.]+ seconds "
        rf"\(final objective {number}\)\. It stopped because further steps no longer "
        rf"improved the fit\. Learned lengthscale x {number}; variance {number}; "
        rf"mean_function\.constant {number}; noise {number}\.",
        lines[-1],
    )


def test_a_verbose_fit_reports_its_iteration_and_objective_as_it_goes(
    monkeypatch, records
):
    # At most once every few seconds normally; after every iteration here, so the test
    # need not wait.
    monkeypatch.setitem(vars(C), "PROGRESS_SECONDS", 0.0)

    fitted = GPRegressor(features=["x"], target="y").fit(ROWS, verbose=True)

    progress = [m for m in messages(records) if m.startswith("fit in progress")]
    iterations = json.loads(fitted.to_json())["fitting"]["iterations"]
    assert len(progress) == iterations
    assert re.fullmatch(
        r"fit in progress: iteration 1, objective -?[\d.]+", progress[0]
    )
    assert progress[-1].startswith(f"fit in progress: iteration {iterations}, ")


@pytest.mark.slow
def test_a_verbose_fit_on_minibatches_shows_the_progress_bar_of_gpjax(capfd):
    model = SparseGPRegressor(
        features=["x"], target="y", num_inducing=3, minibatch_size=2
    )
    model.fit(ROWS, iterations=3)
    quiet = capfd.readouterr().err
    model.fit(ROWS, iterations=3, verbose=True)
    shown = capfd.readouterr().err

    assert quiet == ""
    # Three iterations of two rows make one pass through five rows.
    assert "3/3" in shown


def test_each_record_names_the_step_of_skgp_that_wrote_it(verbose_run):
    # loguru's default output starts with the module and function a record came from.
    origin = {
        r["message"].split(":")[0]: (r["name"], r["function"])
        for r in verbose_run.on_fit
    }

    assert origin["kernel chosen"] == ("skgp._model", "_fit")
    assert origin["optimiser chosen"] == ("skgp._fit_kinds", "_start_optimiser")


def test_a_kernel_the_user_gave_is_reported_as_theirs_even_when_it_equals_the_default():
    with capturing_records() as records:
        GPRegressor(features=["x"], target="y", kernel="matern52").fit(
            ROWS, verbose=True
        )

    assert "kernel chosen: matern52 (set by you)" in messages(records)
    assert (
        "settings given: features x, target y; set by you: features, kernel, target; no schema supplied"
        in messages(records)
    )


def test_a_failed_check_is_reported_as_a_warning_and_the_error_still_reaches_the_caller(
    records,
):
    data = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 2.0, None, 4.0, 5.0]}
    )

    with pytest.raises(ValueError, match=re.escape(MISSING_VALUE)):
        GPRegressor(features=["x"], target="y").fit(data, verbose=True)

    failed = [
        (r["level"].name, r["message"])
        for r in records
        if r["message"].startswith("data checks failed")
    ]
    assert failed == [("WARNING", f"data checks failed: {MISSING_VALUE}")]


def test_verbose_predict_says_what_it_checked_and_how_it_rescaled_the_rows(verbose_run):
    assert messages(verbose_run.on_predict) == [
        "settings given: features x, target y; set by you: features, target; no schema supplied",
        "data checks passed: columns_present, numeric, no_missing_values, no_infinite_values (3 rows)",
        f"rescaling applied: {RESCALING}",
        "prediction made: 3 rows",
    ]


def test_nothing_is_emitted_unless_verbose_even_if_the_user_enabled_skgp_logging(
    records, skgp_logging_enabled
):
    model = GPRegressor(features=["x"], target="y")

    fitted = model.fit(ROWS)
    fitted.predict(NEW_ROWS)
    fitted.predict(NEW_ROWS, verbose=False)
    with pytest.raises(
        ValueError, match="The data is missing 1 column the model needs: 'x'"
    ):
        fitted.predict(pl.DataFrame({"z": [0.0]}))
    assert records == []

    model.fit(
        ROWS, verbose=True
    )  # the same calls do log once asked to, so the silence above is real
    assert messages(records)[0].startswith("settings given: ")

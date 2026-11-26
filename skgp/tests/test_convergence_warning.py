"""`ConvergenceWarning` for every fit by the optimiser, on all rows at once, and none for a
fit on minibatches, which is not judged.

The unfinished fits are made once, in the `unfinished` fixture, by passing a small `iterations`
to `fit` (at most 1 iteration, or 1 pass through the data for the fit on minibatches): that is
deterministic on every platform, unlike data that happens to be hard to fit (an exact
straight line stops abnormally on Linux but converges on macOS). How a fit by the optimiser
is judged and worded, from literal numbers, is tested in test_how_a_fit_ended.py.
"""

import json
import warnings

import polars as pl
import pytest

import helpers
from helpers import UNFINISHED_FITS
import skgp

# The fits of the `unfinished` fixture on minibatches, and those by the optimiser.
ON_MINIBATCHES = [
    kind
    for kind, (model, _) in UNFINISHED_FITS.items()
    if getattr(model, "minibatch_size", None) is not None
]
BY_THE_OPTIMISER = [kind for kind in UNFINISHED_FITS if kind not in ON_MINIBATCHES]


def convergence_warnings(run):
    return [w for w in run.warnings if issubclass(w.category, skgp.ConvergenceWarning)]


@pytest.mark.parametrize("kind", BY_THE_OPTIMISER)
def test_a_fit_that_does_not_finish_properly_warns_once_and_reports_it(
    unfinished, kind
):
    run = unfinished[kind]

    (warning,) = convergence_warnings(run)
    fitting = json.loads(run.fitted.to_json())["fitting"]

    assert str(warning.message).startswith("Fitting did not finish properly")
    assert (
        warning.filename == helpers.__file__
    )  # points at the call to fit, not into skgp
    assert "gpjax" not in str(warning.message).lower()
    assert fitting["finished_properly"] is False
    # With verbose on, the fit also says so in its log, as a warning.
    (finished,) = [r for r in run.records if r["message"].startswith("fit finished: ")]
    assert finished["level"].name == "WARNING"
    assert finished["message"].startswith(
        "fit finished: did not finish properly after "
    )


def test_the_warning_of_the_exact_fit_says_what_to_check_and_how_far_it_got(unfinished):
    (warning,) = convergence_warnings(unfinished["exact"])

    assert str(warning.message) == (
        "Fitting did not finish properly after 1 iteration. It reached its limit of iterations "
        "while the fit was still improving. The fitted model may be less reliable. To let it run "
        "longer, pass a larger iterations to fit. Check the scale of your data (columns or a "
        "target spanning very different magnitudes), and whether it is noise-free: data with no "
        "noise at all, such as an exact formula, can stop a fit early. Predictions are still "
        "available."
    )


@pytest.mark.parametrize("kind", ON_MINIBATCHES)
def test_a_fit_on_minibatches_is_not_judged_wherever_a_user_reads_how_a_fit_ended(
    unfinished, kind
):
    # Two iterations on 40 rows, far too few to fit them, and still no verdict.
    run = unfinished[kind]

    fitting = json.loads(run.fitted.to_json())["fitting"]
    (finished,) = [r for r in run.records if r["message"].startswith("fit finished: ")]

    assert convergence_warnings(run) == []
    assert set(fitting) == {"iterations", "final_objective"}
    assert fitting["iterations"] == 2
    assert finished["level"].name == "INFO"
    assert finished["message"].startswith("fit finished: ran 2 iterations in ")
    assert repr(run.fitted).split("\n")[0].endswith(" from 40 rows")


def test_the_optimisers_own_text_is_only_in_the_log(unfinished):
    run = unfinished["exact"]
    (warning,) = convergence_warnings(run)
    (finished,) = [r for r in run.records if r["message"].startswith("fit finished: ")]
    optimiser_text = finished["extra"]["optimiser_message"]  # SciPy's, in its own words

    assert optimiser_text
    for shown in (run.fitted.to_json(), str(warning.message), finished["message"]):
        assert optimiser_text not in shown


@pytest.mark.slow
def test_an_ordinary_fit_and_predict_emit_no_warning():
    # Eight points of a smooth curve (sin x), which fit without trouble.
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [0.0, 0.84, 0.91, 0.14, -0.76, -0.96, -0.28, 0.66],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = skgp.GPRegressor(features=["x"], target="y").fit(data)
        fitted.predict(data)

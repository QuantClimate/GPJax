"""`SparseGPRegressor`: fitting, its settings, what `to_json()` and the verbose log say about it.

Fits that are only looked at for how they are set up or logged are cut short by lowering a
constant with monkeypatch, so only the setup is judged.
"""

import json
from typing import Any
import warnings

import numpy as np
import polars as pl
import pytest

from helpers import fit_and_watch
import skgp
from skgp import SparseGPRegressor, _inducing

# A rising line in units far from zero, 12 rows, with a few units of wobble: at x = 55 the
# line passes between 1098 and 1124, so a fitted curve should predict about 1111 there.
LINE = pl.DataFrame(
    {
        "x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
        "y": [
            1001.0,
            1019.0,
            1042.0,
            1058.0,
            1083.0,
            1098.0,
            1124.0,
            1139.0,
            1161.0,
            1181.0,
            1200.0,
            1221.0,
        ],
    }
)


def sparse(**settings):
    return SparseGPRegressor(features=["x"], target="y", **settings)


@pytest.mark.slow
@pytest.mark.parametrize("minibatch_size", [None, 4], ids=["collapsed", "minibatches"])
def test_a_sparse_fit_predicts_in_the_units_of_the_data_and_issues_no_warning(
    minibatch_size,
):
    with warnings.catch_warnings():
        # A fit on all rows that did not finish properly would warn.
        warnings.simplefilter("error")
        fitted = sparse(num_inducing=6, minibatch_size=minibatch_size).fit(LINE)

    prediction = fitted.predict(pl.DataFrame({"x": [55.0]}))

    assert prediction.mean() == pytest.approx([1111.0], abs=4.0)
    assert prediction.std()[0] < 15.0  # the wobble is a few units, not hundreds


def test_the_default_number_of_inducing_points_is_a_tenth_of_the_rows_within_a_floor_and_a_ceiling():
    # Called directly: a public fit of 50,000 rows would take far too long to run here.
    assert [_inducing.default_count(rows) for rows in (50, 500, 5_000, 50_000)] == [
        100,
        100,
        500,
        1_000,
    ]


def test_more_inducing_points_than_rows_uses_every_row():
    eight_rows = pl.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0) ** 2})

    fitted = sparse(num_inducing=1_000_000).fit(eight_rows)

    assert json.loads(fitted.to_json())["model"]["num_inducing"] == 8


@pytest.mark.slow
def test_repeated_rows_with_more_inducing_points_than_distinct_rows_fit_silently():
    # Three distinct values of x, ten rows each, so 8 inducing points cannot all be different.
    repeated = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0] * 10,
            "y": [
                0.12,
                0.93,
                0.18,
                0.07,
                0.88,
                0.22,
                0.10,
                0.95,
                0.15,
                0.13,
                0.85,
                0.25,
            ]
            * 2
            + [0.08, 0.91, 0.19, 0.11, 0.9, 0.2],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prediction = (
            sparse(num_inducing=8).fit(repeated).predict(pl.DataFrame({"x": [1.0]}))
        )

    assert prediction.mean() == pytest.approx([0.9], abs=0.1)


# --- random_state: the same data and state give the same model ---


@pytest.fixture
def few_steps():
    """Fits that stop after a few iterations (`few_iterations`) warn that they did not finish,
    which is not what these tests are about."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)
        yield


def few_iterations(minibatch_size):
    """Enough iterations to tell fits apart, and fast: 5 for a fit on all rows, and 3 passes
    through the 12 rows of `LINE` in minibatches of 4."""
    return 5 if minibatch_size is None else 9


@pytest.mark.slow
@pytest.mark.parametrize("minibatch_size", [None, 4], ids=["collapsed", "minibatches"])
def test_the_same_data_and_random_state_give_the_same_model(few_steps, minibatch_size):
    # The only test of the promise that a fit is repeatable: the answer cannot be written down
    # in advance, so one fit is compared with another, to a tolerance.
    rows = pl.DataFrame({"x": [5.0, 25.0, 45.0, 65.0, 85.0, 105.0]})

    iterations = few_iterations(minibatch_size)
    first = (
        sparse(num_inducing=3, minibatch_size=minibatch_size, random_state=3)
        .fit(LINE, iterations=iterations)
        .predict(rows)
        .mean()
    )
    again = (
        sparse(num_inducing=3, minibatch_size=minibatch_size, random_state=3)
        .fit(LINE, iterations=iterations)
        .predict(rows)
        .mean()
    )

    np.testing.assert_allclose(first, again, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("minibatch_size", [None, 4], ids=["collapsed", "minibatches"])
def test_a_different_random_state_gives_a_different_model(few_steps, minibatch_size):
    # random_state seeds where the inducing points start, and the order of the minibatches.
    rows = pl.DataFrame({"x": [5.0, 25.0, 45.0, 65.0, 85.0, 105.0]})

    def predicted_mean(random_state):
        model = sparse(
            num_inducing=3, minibatch_size=minibatch_size, random_state=random_state
        )
        return (
            model.fit(LINE, iterations=few_iterations(minibatch_size))
            .predict(rows)
            .mean()
        )

    assert np.abs(predicted_mean(3) - predicted_mean(4)).max() > 0.1


@pytest.mark.slow
def test_a_fit_on_minibatches_runs_exactly_the_iterations_it_is_given():
    fifty_rows = pl.DataFrame({"x": np.arange(50.0), "y": np.sin(np.arange(50.0))})
    fitted = sparse(num_inducing=5, minibatch_size=20).fit(fifty_rows, iterations=7)

    assert json.loads(fitted.to_json())["fitting"]["iterations"] == 7


# --- settings, refused when the model is defined ---


def test_num_inducing_and_random_state_must_be_whole_numbers():
    for setting, bad, given in (
        ("random_state", 1.5, "1.5"),
        ("num_inducing", "3", "'3'"),
        ("num_inducing", True, "True"),
        ("random_state", None, "None"),
    ):
        with pytest.raises(
            ValueError, match=rf"^{setting} must be a whole number; got {given}\.$"
        ):
            sparse(**{setting: bad})
    model = sparse(num_inducing=np.int64(7), random_state=np.int32(2))
    assert (model.num_inducing, model.random_state) == (7, 2)


def test_a_zero_or_negative_num_inducing_or_minibatch_size_is_refused():
    for bad in (0, -3):
        with pytest.raises(
            ValueError, match=rf"^num_inducing must be above 0; got {bad}\.$"
        ):
            sparse(num_inducing=bad)
        with pytest.raises(
            ValueError, match=rf"^minibatch_size must be above 0; got {bad}\.$"
        ):
            sparse(minibatch_size=bad)


def test_a_random_state_outside_zero_to_the_largest_seed_is_refused():
    with pytest.raises(ValueError, match=r"^random_state must be 0 or more; got -1\.$"):
        sparse(random_state=-1)
    for bad, given in (
        (2**63, r"9,223,372,036,854,775,808\."),
        (10**400, r"10,000,000,.*\.\.\."),  # a long number is cut short
    ):
        with pytest.raises(
            ValueError,
            match=rf"^random_state must be 9,223,372,036,854,775,807 or less; got {given}$",
        ):
            sparse(random_state=bad)
    assert sparse(random_state=2**63 - 1).random_state == 2**63 - 1


def test_the_defaults_are_no_minibatches_and_random_state_zero():
    model = SparseGPRegressor(features=["x"], target="y")

    assert (model.minibatch_size, model.random_state) == (None, 0)


# --- what a fit says about itself: to_json and the verbose log ---

# Three fits of the 12-row `LINE`, each described next to what it must report: the `model` block of
# `to_json()` and the sparse lines of the verbose log (the lines a user reads while it fits).
CASES: dict[str, dict[str, Any]] = {
    "num_inducing given": {
        "settings": {"num_inducing": 6, "random_state": 7},
        "model": {
            "class": "SparseGPRegressor",
            "num_inducing": 6,
            "minibatch_size": None,
            "random_state": 7,
        },
        "log": [
            "inducing points chosen: 6 (set by you), starting at k-means centres of the rescaled inputs",
            "random state: 7",
            "model chosen: sparse Gaussian process (fitted on all rows at once)",
            "optimiser chosen: L-BFGS-B on the approximate marginal likelihood",
        ],
    },
    "num_inducing left to skgp": {
        "settings": {},
        "model": {
            "class": "SparseGPRegressor",
            "num_inducing": 12,  # 12 rows is below the floor of 100, so every row is used
            "minibatch_size": None,
            "random_state": 0,
        },
        "log": [
            "inducing points chosen: 12 (the default), starting at k-means centres of the rescaled inputs",
            "random state: 0",
        ],
    },
    "minibatches": {
        "settings": {"num_inducing": 6, "minibatch_size": 4},
        "model": {
            "class": "SparseGPRegressor",
            "num_inducing": 6,
            "minibatch_size": 4,
            "random_state": 0,
        },
        "log": [
            "model chosen: sparse Gaussian process (fitted on minibatches of 4 rows)",
            "optimiser chosen: natural gradients alternating with Adam on the approximate marginal likelihood",
        ],
    },
}


@pytest.fixture(scope="module", params=list(CASES))
def watched(request):
    """One verbose fit of `LINE`, cut short to one iteration, or one pass through the data on
    minibatches (only how it is set up and described is looked at), and what it logged."""
    settings = CASES[request.param]["settings"]
    run = fit_and_watch(
        sparse(**settings),
        LINE,
        iterations=1 if "minibatch_size" not in settings else 3,
    )
    run.case = request.param
    return run


def test_to_json_adds_the_class_and_the_sparse_settings_beside_the_old_fields(watched):
    def refuse_non_finite(constant):
        raise AssertionError(f"{constant} is not strict JSON")

    description = json.loads(watched.fitted.to_json(), parse_constant=refuse_non_finite)

    assert set(description) == {
        "skgp_version",
        "training_rows",
        "setup",
        "learned",
        "fitting",
        "model",
    }
    assert description["training_rows"] == 12
    assert description["model"] == CASES[watched.case]["model"]


def test_a_verbose_fit_says_what_it_chose_in_plain_lines(watched):
    # The one place the sparse log lines are pinned, as a user reads them while the fit runs.
    messages = [record["message"] for record in watched.records]

    for expected in CASES[watched.case]["log"]:
        assert any(message.startswith(expected) for message in messages), (
            f"no log line starts {expected!r}: {messages}"
        )
    for jargon in ("collapsed", "evidence lower bound", "natural gradients:"):
        assert not [message for message in messages if jargon in message]

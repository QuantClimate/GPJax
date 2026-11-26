"""`SparseGPClassifier`: what it predicts, its two kinds of fit, and how it describes itself."""

import json
import warnings

import pytest

from helpers import WEATHER_GROUPS
import skgp
from skgp import GPClassifier, SparseGPClassifier


@pytest.mark.slow
def test_on_all_rows_each_training_row_is_more_likely_its_own_class():
    fitted = SparseGPClassifier("x", "weather", num_inducing=5).fit(WEATHER_GROUPS)

    prediction = fitted.predict(WEATHER_GROUPS)

    assert prediction.classes == ("dry", "rain")
    assert list(prediction.most_likely()) == WEATHER_GROUPS["weather"].to_list()


def test_to_json_describes_its_sparse_settings(unfinished):
    description = json.loads(
        unfinished["sparse classifier on minibatches"].fitted.to_json()
    )

    assert description["model"] == {
        "class": "SparseGPClassifier",
        "num_inducing": 10,
        "minibatch_size": 20,
        "random_state": 0,
    }


@pytest.mark.parametrize(
    ("kind", "model_line", "optimiser_line"),
    [
        (
            "sparse classifier on all rows",
            (
                "model chosen: sparse Gaussian process (fitted on all rows at once) with "
                "a Bernoulli likelihood"
            ),
            (
                "optimiser chosen: L-BFGS-B on the approximate marginal likelihood, at "
                "most 1 iteration"
            ),
        ),
        (
            "sparse classifier on minibatches",
            (
                "model chosen: sparse Gaussian process (fitted on minibatches of 20 "
                "rows) with a Bernoulli likelihood"
            ),
            (
                "optimiser chosen: natural gradients alternating with Adam on the "
                "approximate marginal likelihood, 2 iterations"
            ),
        ),
    ],
)
def test_a_verbose_fit_says_which_model_and_optimiser_it_chose(
    unfinished, kind, model_line, optimiser_line
):
    messages = [record["message"] for record in unfinished[kind].records]

    assert model_line in messages
    assert optimiser_line in messages


def test_it_never_warns_about_size(monkeypatch):
    monkeypatch.setattr(skgp.config, "size_warning_rows", 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SparseGPClassifier("x", "weather", num_inducing=5).fit(
            WEATHER_GROUPS, iterations=1
        )

    assert not [w for w in caught if issubclass(w.category, skgp.LargeDataWarning)]


def test_data_checked_by_one_classifier_is_accepted_by_the_other():
    checked = GPClassifier("x", "weather").check(WEATHER_GROUPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", skgp.ConvergenceWarning)  # 1 iteration is short
        fitted = SparseGPClassifier("x", "weather", num_inducing=5).fit(
            checked, iterations=1
        )

    assert fitted.classes == ("dry", "rain")

"""How a fit by the optimiser ended, judged from what the optimiser left behind, given here
as literal numbers. A fit on minibatches is not judged (test_convergence_warning.py).

These call `skgp._how_a_fit_ended` directly (CODING_STANDARDS.md, "Through the public API"):
no public fit can be made to end in each of these ways. The objective is lower for a better
fit, and the thresholds are scaled by the rows it sums over.
"""

import numpy as np
import pytest

from skgp import _how_a_fit_ended


# Every fit here is on 1,000 rows, so a fit had settled if no entry of its gradient is above
# 0.1 (1e-4 per row), or if its objective changed by no more than 0.01 (1e-5 per row) over
# its last 3 iterations. The objectives are the last few of the fit's iterations.
@pytest.mark.parametrize(
    (
        "objectives",
        "final_objective",
        "gradient",
        "reached_the_iteration_limit",
        "stop_reason",
        "finished_properly",
        "advises_a_larger_iterations",
    ),
    [
        # Seen on ordinary rows (two inputs, noise 0.1): the line search gave up after 75
        # iterations, yet the fit had learned a noise of 0.098, its gradient was 0.02 at most
        # and its objective had changed by 0.0002 over its last 3 iterations.
        (
            [-412.3000, -412.3001, -412.3002, -412.3002],
            -412.3002,
            [0.02, -0.015],
            False,
            "It stopped because further steps no longer improved the fit.",
            True,
            False,
        ),
        # Seen in a sparse fit: creeping on by 0.001 an iteration until it ran out of
        # iterations, 0.003 over its last 3, though its gradient was above 0.1.
        (
            [250.004, 250.003, 250.002, 250.001],
            250.001,
            [0.3, -0.2],
            True,
            (
                "It reached its limit of iterations, but by then further steps no longer "
                "improved the fit by much."
            ),
            True,
            False,
        ),
        # Seen in a sparse fit whose inducing points sat on top of each other: a gradient of
        # 2 where its objective had stopped changing.
        (
            [-80.5, -80.5, -80.5, -80.5],
            -80.5,
            [2.0, -2.0],
            False,
            "It stopped because further steps no longer improved the fit.",
            True,
            False,
        ),
        # Seen in a fit cut off after one iteration: a gradient of about 500 and no history
        # to judge.
        (
            [130.0],
            130.0,
            [500.0, -350.0],
            True,
            "It reached its limit of iterations while the fit was still improving.",
            False,
            True,
        ),
        # Seen in a fit cut off after 10 of the 15 iterations it needs: still changing by
        # 0.06 over its last 3 iterations.
        (
            [101.0, 100.98, 100.96, 100.94],
            100.94,
            [1.5, -0.4],
            True,
            "It reached its limit of iterations while the fit was still improving.",
            False,
            True,
        ),
        # Stopped on its own after two iterations, too few for its objective to be judged,
        # with a gradient of 0.01 at most.
        (
            [55.2, 55.1],
            55.1,
            [0.01, -0.004],
            False,
            "It stopped because further steps no longer improved the fit.",
            True,
            False,
        ),
        # The line search gave up with a gradient of 2 and a change of 0.5 over the last 3
        # iterations.
        (
            [90.0, 89.8, 89.6, 89.5],
            89.5,
            [2.0, -1.0],
            False,
            (
                "It stopped because it could not find a step that improved the fit, though "
                "the fit had not settled."
            ),
            False,
            False,
        ),
        # Undefined where it started (targets near the largest float), so it stopped before
        # its first iteration.
        (
            [],
            np.nan,
            [np.nan, np.nan],
            False,
            (
                "It stopped because the fit broke down: its numbers became infinite or "
                "undefined."
            ),
            False,
            False,
        ),
    ],
)
def test_a_fit_by_the_optimiser_is_judged_by_where_it_ended_and_not_by_the_optimisers_verdict(
    objectives,
    final_objective,
    gradient,
    reached_the_iteration_limit,
    stop_reason,
    finished_properly,
    advises_a_larger_iterations,
):
    optimisation = _how_a_fit_ended.on_all_rows(
        objectives,
        final_objective,
        np.array(gradient),
        reached_the_iteration_limit=reached_the_iteration_limit,
        optimiser_message="",  # scipy's own text, which plays no part
        rows=1_000,
    )
    warning = _how_a_fit_ended.warning_for_a_regressor(optimisation)

    assert optimisation.stop_reason == stop_reason
    assert optimisation.finished_properly is finished_properly
    assert ("pass a larger iterations to fit" in warning) is advises_a_larger_iterations


# The warning of a regressor and of a classifier with the advice to run longer is tested through `fit`,
# in test_convergence_warning.py and test_gp_classifier.py.
@pytest.mark.parametrize(
    ("warning_of", "optimisation", "warning"),
    [
        (
            _how_a_fit_ended.warning_for_a_regressor,
            _how_a_fit_ended.Optimisation(
                finished_properly=False,
                iterations=40,
                final_objective=89.5,
                stop_reason="It stopped because it could not find a step that improved "
                "the fit, though the fit had not settled.",
                optimiser_message="ABNORMAL: ",
                ran_out_of_iterations=False,
            ),
            (
                "Fitting did not finish properly after 40 iterations. It stopped because it "
                "could not find a step that improved the fit, though the fit had not settled. "
                "The fitted model may be less reliable. Check the scale of your data (columns "
                "or a target spanning very different magnitudes), and whether it is "
                "noise-free: data with no noise at all, such as an exact formula, can stop a "
                "fit early. Predictions are still available."
            ),
        ),
        (
            _how_a_fit_ended.warning_for_a_classifier,
            _how_a_fit_ended.Optimisation(
                finished_properly=False,
                iterations=25,
                final_objective=14.2,
                stop_reason="It stopped because it could not find a step that improved "
                "the fit, though the fit had not settled.",
                optimiser_message="ABNORMAL: ",
                ran_out_of_iterations=False,
            ),
            (
                "Fitting did not finish properly after 25 iterations. It stopped because it "
                "could not find a step that improved the fit, though the fit had not settled. "
                "The fitted model may be less reliable. Predictions are still available."
            ),
        ),
    ],
    ids=["a regressor", "a classifier"],
)
def test_the_warning_of_a_regressor_and_of_a_classifier_reads_in_full_when_more_iterations_would_not_help(
    warning_of, optimisation, warning
):
    assert warning_of(optimisation) == warning

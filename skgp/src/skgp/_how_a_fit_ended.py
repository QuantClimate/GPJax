"""How a fit ended: whether it finished properly, why it stopped in plain words, and what
the `ConvergenceWarning` says when it did not finish.

Only a fit by the optimiser, on all rows at once, is judged. It is judged from what the
optimiser's run left behind, given as plain numbers: the objective after each iteration, the
gradient where it stopped, whether scipy stopped it at its limit of iterations, and how many
rows the objective sums over. A fit on minibatches, by Adam alone or alternating with
natural gradients, runs exactly the iterations it is given, and skgp does not judge whether
it converged (decision record 0008). Nothing here needs GPJax or JAX. The thresholds are
fields of `Constant`.
"""

from dataclasses import dataclass

import numpy as np

from skgp._constants import Constant as C
from skgp._text import count


@dataclass(frozen=True)
class Optimisation:
    """How the search for the best hyperparameters ended. A fit on minibatches is not
    judged: it has its `iterations` and `final_objective`, and nothing else."""

    iterations: int
    final_objective: float
    # Judged from where the fit ended, not by the optimiser. None: not judged.
    finished_properly: bool | None = None
    stop_reason: str | None = None  # why it stopped, in plain words for the user
    optimiser_message: str | None = (
        None  # the optimiser's own text, for the verbose log
    )
    # Whether more iterations could have helped: the fit used them all and had not settled.
    ran_out_of_iterations: bool = False


def on_all_rows(
    objectives: list[float],
    final_objective: float,
    final_gradient: np.ndarray,
    *,
    reached_the_iteration_limit: bool,
    optimiser_message: str,
    rows: int,
) -> Optimisation:
    """How a fit on all rows at once, by the optimiser, ended. `objectives` is the objective
    after each of its iterations; `final_objective` and `final_gradient` are the objective
    and its gradient where it stopped; `reached_the_iteration_limit` is whether scipy
    stopped it at its limit of iterations; the objective sums over `rows` rows. Whether it
    finished properly is `_settled`'s verdict. `optimiser_message` is the optimiser's own
    text, kept on one line for the verbose log."""
    settled = _settled(final_gradient, objectives, rows)
    return Optimisation(
        iterations=len(objectives),
        final_objective=final_objective,
        finished_properly=settled,
        stop_reason=_why_it_stopped(
            final_objective, reached_the_iteration_limit, settled
        ),
        optimiser_message=" ".join(optimiser_message.split()),
        ran_out_of_iterations=reached_the_iteration_limit and not settled,
    )


def on_minibatches(objectives: np.ndarray, iterations_per_pass: int) -> Optimisation:
    """How a fit on minibatches ended: how many iterations it ran, and its final objective.
    `objectives` is the objective of each iteration on its own minibatch, so it is noisy; the
    final objective is their mean over the last pass, `iterations_per_pass` iterations, as
    many as it takes to see every row once on average. Whether the fit converged is not
    judged."""
    return Optimisation(
        iterations=len(objectives),
        final_objective=float(objectives[-iterations_per_pass:].mean()),
    )


def _settled(gradient: np.ndarray, objectives: list[float], rows: int) -> bool:
    """Whether a fit by the optimiser had settled where it stopped, judged from where it
    ended and not from the optimiser's own verdict.

    The optimiser can stop on a good fit and still report a failure. On 1,000 ordinary rows
    (two inputs, noise 0.1) its line search gave up after 75 iterations, yet the fit had
    learned a noise of 0.098, no entry of its gradient was above 2e-5 per row, and its
    objective had changed by 2e-7 per row over its last 3 iterations. A sparse fit on all
    rows can creep on by about 1e-6 per row an iteration until it runs out of iterations,
    and one whose inducing points sit on top of each other can keep gradient entries of
    2e-3 per row where its objective has stopped changing. A fit cut off after one
    iteration has gradient entries of about 0.5 per row and no history to judge; one cut
    off after 10 of the 15 iterations it needs still changes by 6e-5 per row over 3
    iterations. So a fit had settled if either:

    - its gradient is small: no entry is larger than `C.GRADIENT_TOLERANCE` per row, or
    - its objective had stopped changing: over its last `C.SETTLED_ITERATIONS` iterations it
      changed by no more than `C.SETTLED_TOLERANCE` per row in all.

    The objective is a sum over the rows, so both limits grow with them. A gradient or an
    objective that is not a finite number never counts as settled.
    """
    gradient_is_small = bool(np.max(np.abs(gradient)) <= C.GRADIENT_TOLERANCE * rows)
    stopped_changing = len(objectives) > C.SETTLED_ITERATIONS and bool(
        abs(objectives[-1 - C.SETTLED_ITERATIONS] - objectives[-1])
        <= C.SETTLED_TOLERANCE * rows
    )
    return gradient_is_small or stopped_changing


def _why_it_stopped(
    final_objective: float, reached_the_iteration_limit: bool, settled: bool
) -> str:
    """Why a fit by the optimiser stopped, in plain words. `settled` is `_settled`'s verdict."""
    if not np.isfinite(final_objective):
        return "It stopped because the fit broke down: its numbers became infinite or undefined."
    if reached_the_iteration_limit:
        if settled:
            return (
                "It reached its limit of iterations, but by then further steps no longer "
                "improved the fit by much."
            )
        return "It reached its limit of iterations while the fit was still improving."
    if settled:
        return "It stopped because further steps no longer improved the fit."
    return (
        "It stopped because it could not find a step that improved the fit, though the "
        "fit had not settled."
    )


def warning_for_a_regressor(optimisation: Optimisation) -> str:
    """What the `ConvergenceWarning` says when a regressor's fit by the optimiser did not
    finish."""
    return (
        f"{_did_not_finish(optimisation)} Check the "
        "scale of your data (columns or a target spanning very different magnitudes), and "
        "whether it is noise-free: data with no noise at all, such as an exact formula, can "
        "stop a fit early. Predictions are still available."
    )


def warning_for_a_classifier(optimisation: Optimisation) -> str:
    """What the `ConvergenceWarning` says when a classifier's fit by the optimiser did not
    finish. A regressor's advice about the scale of the target and noise-free data does not
    apply to classes."""
    return f"{_did_not_finish(optimisation)} Predictions are still available."


def _did_not_finish(optimisation: Optimisation) -> str:
    """How the `ConvergenceWarning` starts, for a regressor and for a classifier: how far
    the fit got, why it stopped, and, when more iterations could have helped, the advice to
    let it run longer."""
    more_iterations = (
        " To let it run longer, pass a larger iterations to fit."
        if optimisation.ran_out_of_iterations
        else ""
    )
    return (
        f"Fitting did not finish properly after "
        f"{count(optimisation.iterations, 'iteration')}. "
        f"{optimisation.stop_reason} The fitted model may be less reliable."
        f"{more_iterations}"
    )

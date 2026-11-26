"""Verbose log records.

skgp emits records through loguru only when a call is made with `verbose=True`.
With `verbose=False` a `Reporter` does nothing at all, so skgp is silent by
construction and never touches loguru's global state (`logger.enable` and
`logger.disable`).

Each record has named fields, which loguru puts in `record["extra"]` for sinks to
read (for example a JSON sink), and a one-line message in plain words that
carries the key values, so loguru's default output is useful on its own. The
message is built in one place, `_MESSAGES` below, from the very same fields: each
event has one function whose parameters are exactly that event's fields, so a
message cannot show a value the fields do not hold, and adding or dropping a
field without updating its message fails loudly.
"""

from collections.abc import Callable
from typing import Any

from loguru import logger
import numpy as np

from skgp._constants import Constant as C
from skgp._text import count, listed
from skgp._types import Likelihood


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _number(value: float) -> str:
    return f"{value:.4g}"


def _numbers(values: list[float]) -> str:
    return ", ".join(_number(v) for v in values)


# How `model chosen` names each likelihood, in plain words.
_LIKELIHOOD_SHOWN: dict[Likelihood, str] = {
    "Gaussian": "Gaussian noise",
    "HeteroscedasticGaussian": "Gaussian noise that varies with the inputs",
    "Bernoulli": "a Bernoulli likelihood",
}


def learned_text(
    *,
    kernel: dict[str, Any],
    mean_function: dict[str, Any] | None = None,
    noise: float | None = None,
    noise_kernel: dict[str, Any] | None = None,
) -> str:
    """What a fit learned, as `fit finished` and a printed fitted model show it: each value
    of the kernel by its place in the kernel, then each value of the mean function, after
    `mean_function.`, then the noise, then each value of the noise kernel, after
    `noise_kernel.`. A `"zero"` mean function learns nothing, a classifier learns no noise,
    and only a regressor with heteroscedastic noise has a noise kernel."""
    parts = [f"{place} {_value_text(value)}" for place, value in kernel.items()]
    for place, value in (mean_function or {}).items():
        parts.append(f"mean_function.{place} {_value_text(value)}")
    if noise is not None:
        parts.append(f"noise {_number(noise)}")
    for place, value in (noise_kernel or {}).items():
        parts.append(f"noise_kernel.{place} {_value_text(value)}")
    return "; ".join(parts)


def _value_text(value: Any) -> str:
    """A value of a kernel or a mean function: one number, a number for each input by its name, or a list, the
    first `C.MAX_COLUMNS_LISTED` numbers of the last two shown."""
    if isinstance(value, dict):
        return listed([f"{name} {_number(number)}" for name, number in value.items()])
    if isinstance(value, list):
        return listed([_number(number) for number in np.ravel(value).tolist()])
    return _number(value)


def _duration(seconds: float) -> str:
    if seconds < C.SECONDS_PER_MINUTE:
        return f"{seconds:.1f} seconds"
    return f"{seconds / C.SECONDS_PER_MINUTE:.1f} minutes"


def _settings_given(
    *, features: list[str], target: str, given: list[str], schema_supplied: bool
) -> str:
    schema = "schema supplied" if schema_supplied else "no schema supplied"
    return (
        f"settings given: features {', '.join(features)}, target {target}; "
        f"set by you: {', '.join(given) or 'nothing'}; {schema}"
    )


def _kernel_chosen(*, kernel: str, is_default: bool) -> str:
    return f"kernel chosen: {kernel} ({_default_or_yours(is_default)})"


def _mean_function_chosen(*, mean_function: str, is_default: bool) -> str:
    return f"mean function chosen: {mean_function} ({_default_or_yours(is_default)})"


def _default_or_yours(is_default: bool) -> str:
    return "the default" if is_default else "set by you"


def _rescaling(
    verb: str,
    *,
    method: str,
    features: list[str],
    feature_shift: list[float],
    feature_scale: list[float],
    target_shift: float | None = None,
    target_scale: float | None = None,
) -> str:
    # A classifier's target holds classes, which are not rescaled: it has no shift or scale.
    features_text = (
        f"rescaling {verb}: {method} features {', '.join(features)} "
        f"(shift {_numbers(feature_shift)}, scale {_numbers(feature_scale)})"
    )
    if target_shift is None or target_scale is None:
        return features_text
    return (
        f"{features_text} and target (shift {_number(target_shift)}, "
        f"scale {_number(target_scale)})"
    )


def _fit_finished(
    *,
    iterations: int,
    elapsed_seconds: float,
    final_objective: float,
    kernel: dict[str, Any],
    mean_function: dict[str, Any] | None = None,
    finished_properly: bool | None = None,
    stop_reason: str | None = None,
    optimiser_message: str | None = None,
    noise: float | None = None,
    noise_kernel: dict[str, Any] | None = None,
) -> str:
    # `optimiser_message` is the optimiser's own text, kept in the fields for anyone who
    # wants it but left out of the message, which says the same in plain words.
    learned = learned_text(
        kernel=kernel,
        mean_function=mean_function,
        noise=noise,
        noise_kernel=noise_kernel,
    )
    ran = (
        f"{count(iterations, 'iteration')} in {_duration(elapsed_seconds)} "
        f"(final objective {_number(final_objective)})"
    )
    if finished_properly is None:
        # A fit on minibatches, which is not judged: it has no verdict and no stop reason.
        return f"fit finished: ran {ran}. Learned {learned}."
    outcome = "finished properly" if finished_properly else "did not finish properly"
    return f"fit finished: {outcome} after {ran}. {stop_reason} Learned {learned}."


def _optimiser_chosen(
    *,
    optimiser: str,
    objective: str,
    max_iterations: int | None = None,
    iterations: int | None = None,
) -> str:
    # An optimiser that stops by itself has `max_iterations`; one that runs a set number
    # of iterations has `iterations`.
    if iterations is None:
        assert max_iterations is not None
        how_long = f"at most {count(max_iterations, 'iteration')}"
    else:
        how_long = count(iterations, "iteration")
    return f"optimiser chosen: {optimiser} on the {objective}, {how_long}"


def _data_checks_passed(*, checks: list[str], n_rows: int, passed: bool) -> str:
    return f"data checks passed: {', '.join(checks)} ({count(n_rows, 'row')})"


# One function per event, keyed by the event name. The name is the start of the message.
_MESSAGES: dict[str, Callable[..., str]] = {
    "settings given": _settings_given,
    "kernel chosen": _kernel_chosen,
    "mean function chosen": _mean_function_chosen,
    "data checks passed": _data_checks_passed,
    "data checks failed": lambda *, passed, error: (
        f"data checks failed: {_one_line(error)}"
    ),
    "data already checked": lambda *, n_rows: (
        f"data already checked: {count(n_rows, 'row')}, not checked again"
    ),
    "model chosen": lambda *, model, likelihood: (
        f"model chosen: {model} with {_LIKELIHOOD_SHOWN[likelihood]}"
    ),
    "classes coded": lambda *, target, class_0, class_1: (
        f"classes coded: in {target}, {class_0!r} is 0 and {class_1!r} is 1"
    ),
    "rescaling chosen": lambda **fields: _rescaling("chosen", **fields),
    "rescaling applied": lambda **fields: _rescaling("applied", **fields),
    "optimiser chosen": _optimiser_chosen,
    "inducing points chosen": lambda *, count, is_default, starting: (
        f"inducing points chosen: {count:,} ({_default_or_yours(is_default)}), "
        f"starting at {starting}"
    ),
    "random state": lambda *, random_state: f"random state: {random_state}",
    "fit in progress": lambda *, iteration, objective: (
        f"fit in progress: iteration {iteration:,}, objective {_number(objective)}"
    ),
    "fit finished": _fit_finished,
    "prediction made": lambda *, n_rows: f"prediction made: {count(n_rows, 'row')}",
    "columns ignored": lambda *, columns: (
        f"columns ignored: {', '.join(columns)} (the model does not use them)"
    ),
}


class Reporter:
    """Emits the records of one `fit` or `predict` call, or nothing when not verbose."""

    def __init__(self, call: str, verbose: bool) -> None:
        self._call = call
        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        """Whether this call emits records."""
        return self._verbose

    def info(self, event: str, /, **fields: Any) -> None:
        self._emit("INFO", event, fields)

    def warning(self, event: str, /, **fields: Any) -> None:
        self._emit("WARNING", event, fields)

    def _emit(self, level: str, event: str, fields: dict[str, Any]) -> None:
        if self._verbose:
            # Depth 2 skips this method and `info` or `warning`, so the record names the
            # skgp function that reported it, not this one.
            logger.opt(depth=2).bind(call=self._call, **fields).log(
                level, _MESSAGES[event](**fields)
            )

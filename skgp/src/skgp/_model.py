"""What every model shares: its settings, its data checks, how it is fitted, and the parts of a
fitted model that do not depend on what it predicts.

`_regressor.py` builds `GPRegressor`, `SparseGPRegressor` and `FittedGPRegressor` on it, and
`_classifier.py` builds `GPClassifier`, `SparseGPClassifier` and `FittedGPClassifier`.
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import difflib
import functools
from importlib.metadata import version
import inspect
import operator
import time
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Self,
    Union,
    get_args,
    get_origin,
)
import warnings

from gpjax.kernels import AbstractKernel
from gpjax.mean_functions import AbstractMeanFunction
from narwhals.stable.v2.typing import IntoDataFrame
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import ErrorDetails

from skgp import _checked, _frames, _gp, _how_a_fit_ended, _log
from skgp._checked import CheckedData
from skgp._config import config
from skgp._constants import Constant as C
from skgp._fit_kinds import FitKind
from skgp._log import Reporter
from skgp._scaling import Standardiser
from skgp._score import Score
from skgp._text import count, joined, listed, quoted
from skgp._types import (
    Frame,
    Kernel,
    KernelName,
    Likelihood,
    MeanFunction,
    MeanFunctionName,
    Task,
    WholeNumber,
)
from skgp._warnings import ConvergenceWarning, LargeDataWarning


def _plain(annotation: Any) -> Any:
    """`annotation` without pydantic's `Annotated[...]` details, so `help()` shows
    `int | None`."""
    if get_origin(annotation) is Annotated:
        return _plain(get_args(annotation)[0])
    if get_origin(annotation) in (Union, UnionType):
        return functools.reduce(operator.or_, map(_plain, get_args(annotation)))
    return annotation


def _valid_kernel(kernel: Any) -> Kernel:
    """`kernel` if it is one of the names or a GPJax kernel; raises `ValueError` otherwise."""
    if isinstance(kernel, AbstractKernel) or (
        isinstance(kernel, str) and kernel in get_args(KernelName)
    ):
        return kernel
    names = joined([repr(name) for name in get_args(KernelName)], "or")
    raise ValueError(
        f"kernel must be {names}, or a GPJax kernel such as gpjax.kernels.RBF(); "
        f"got {_shown(kernel)}."
    )


def _valid_mean_function(mean_function: Any) -> MeanFunction:
    """`mean_function` if it is one of the names or a GPJax mean function; raises
    `ValueError` otherwise."""
    if isinstance(mean_function, AbstractMeanFunction) or (
        isinstance(mean_function, str) and mean_function in get_args(MeanFunctionName)
    ):
        return mean_function
    names = joined([repr(name) for name in get_args(MeanFunctionName)], "or")
    raise ValueError(
        f"mean_function must be {names}, or a GPJax mean function such as "
        f"gpjax.mean_functions.Constant(); got {_shown(mean_function)}."
    )


class _ShownAsIs(str):
    """Text that a printed model shows without quotes, as the name of a GPJax kernel or mean
    function is shown."""

    def __repr__(self) -> str:
        return str(self)


with warnings.catch_warnings():
    # `schema` is the natural name for this setting, but pydantic warns that it hides
    # its own deprecated `BaseModel.schema()` method. skgp does not use that method.
    warnings.filterwarnings(
        "ignore", message='Field name "schema"', category=UserWarning
    )

    class _Settings(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")

        schema: Any = None
        """Your own data checker: any object with a `.validate(frame)` method that
        raises on bad data. A pandera schema works; skgp does not depend on pandera.

        It runs first, inside `check` (so at `fit` and `predict`), on the dataframe
        exactly as you passed it, in your own dataframe type, because a schema may
        fix types. skgp's own checks then always run too and cannot be switched off.

        - If `validate` returns a dataframe (pandera returns the validated, possibly
          coerced, frame), skgp's checks and the model use that returned frame.
        - If it returns `None`, skgp's checks use the frame you passed.
        - Whatever it raises propagates unchanged and stops the call.
        - It also sees prediction rows, which have no target column, so it must
          accept that (in pandera, mark the target column `required=False`).
        - A `CheckedData` passed to `fit` or `predict` is not validated again.

        A value with no callable `validate` is rejected when the model is defined.
        The schema is not included in a fitted model's `to_json()`. Two models are
        equal only if their schemas are equal.
        """

        @field_validator("schema")
        @classmethod
        def _schema_can_validate(cls, schema: Any) -> Any:
            if schema is not None and not callable(getattr(schema, "validate", None)):
                raise ValueError(
                    "schema must be an object with a .validate(frame) method, such as a "
                    f"pandera schema; got {type(schema).__name__}, which has no callable "
                    "validate method"
                )
            return schema


class _Model(_Settings):
    """What every model description shares: the columns, the kernel, the schema, the data
    checks and how it is fitted. A `CheckedData` accepts any model of the same task."""

    task: ClassVar[Task]
    """What the model predicts, which decides how its target is checked."""

    settings_not_taken: ClassVar[dict[str, str]] = {}
    """Settings another model takes and this one does not, each with the reason, which the
    refusal gives."""

    features: tuple[str, ...] = Field(min_length=1)
    """The columns of your dataframe that are the model's inputs, in order: a list of
    column names, or a single name for one input. Each column is named once, and the
    target is not among them."""

    target: str
    """The column of your dataframe that the model learns to predict."""

    kernel: Annotated[Kernel, PlainValidator(_valid_kernel)] = "matern52"
    """The kind of curve the model expects: a name, or a GPJax kernel.

    - By name: `"matern52"` (the default) for a smooth curve, `"matern32"` for a rougher
      one, or `"rbf"` for a very smooth one. Each has one length scale per input, learned
      from the data, starting at that input's standard deviation.
    - A GPJax kernel, such as `gpjax.kernels.RBF() + gpjax.kernels.Periodic(period=52.0)`,
      is fitted as you give it. Its values are where the fit starts, and a value held with
      `paramax.non_trainable` stays where you put it. A kernel with one length scale keeps
      one; give one per input, such as `lengthscale=jnp.array([1.0, 500.0])`, for more.

    Every kernel sees the inputs in the units of your columns, so a value that measures
    distance along an input, such as a length scale or a period, is in those units, both as
    you give it and as the fit reports it. A date input, an input column of dates or of
    dates and times, is in days since 1970-01-01, so
    `gpjax.kernels.Periodic(period=365.25)` repeats every year. (The length scale of GPJax's `Periodic` kernel
    measures a fraction of the period instead, and has no units.) A regressor's
    kernel variance is relative to the target's variance, so 1 means the curve varies as
    much as the target does. A kernel's `active_dims` are positions in `features`, counting
    from 0. Any other value, and `active_dims` past the last feature, are refused when the
    model is defined."""

    mean_function: Annotated[MeanFunction, PlainValidator(_valid_mean_function)] = (
        "constant"
    )
    """What the curve is far from the training data, where the data says nothing: a name, or
    a GPJax mean function. Near the data, the curve follows the data whatever the mean
    function is.

    - `"constant"` (the default): one number, learned from the data. A regressor's starts
      at the target's average.
    - `"zero"`: zero in the target's units, held there: for a target that goes back to 0
      far from the data, such as a difference from a long-term average.
    - `"linear"`: a number plus a slope along each input, all learned, so that a prediction
      far from the data keeps rising or falling along each input as the data does. A slope is in the target's units per unit of
      its input's column: per day along a date input. The slope is one value, learned, so
      a prediction's spread does not include any doubt about it, which matters more the
      further from the data you predict.
    - A GPJax mean function, such as `gpjax.mean_functions.Constant(15.0)`, sees the inputs
      in the units of your columns and gives the curve in the target's units. Its values
      are where the fit starts, and a value held with `paramax.non_trainable` stays where
      you put it. On a fit on minibatches, each step moves a value by about a twentieth
      of the target's standard deviation (of 1 for a classifier), or by about a
      twentieth of itself if it must be positive and is above 1, so start it near where it belongs: the
      names start at the target's average.

    With heteroscedastic noise, the mean function is the curve's; the noise process keeps
    its own learned constant mean.

    A classifier's curve is the one it turns into probabilities through the probit, so its
    mean function has no units: far from the data, the probability of the second class
    goes towards the probit of the mean function there, nearer 0.5 the less the model
    knows. Any other value is refused when the model is defined."""

    def __init__(
        self, features: str | Iterable[str], target: str, **settings: Any
    ) -> None:
        # `features` and `target` may be given by position. A mistake in any setting is
        # raised as one plain line per problem, not as pydantic's own report. Each class
        # also declares its settings in a typed signature for editors and type checkers.
        try:
            BaseModel.__init__(self, **settings, features=features, target=target)
        except ValidationError as error:
            raise ValueError(
                "\n".join(_problem(type(self), found) for found in error.errors())
            ) from None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        # `help()` and editors show each class's settings with plain annotations, such as
        # `int | None`, and the schema last.
        super().__pydantic_init_subclass__(**kwargs)
        signature = inspect.signature(cls)
        cls.__signature__ = signature.replace(
            parameters=[
                parameter.replace(annotation=_plain(parameter.annotation))
                for parameter in sorted(
                    signature.parameters.values(),
                    key=lambda parameter: parameter.name == "schema",
                )
            ]
        )

    @field_validator("features", mode="before")
    @classmethod
    def _one_name_is_one_input(cls, features: Any) -> Any:
        return (features,) if isinstance(features, str) else features

    @field_validator("features")
    @classmethod
    def _each_input_named_once(cls, features: tuple[str, ...]) -> tuple[str, ...]:
        if "" in features:
            raise ValueError(
                "features includes an empty column name ''; every column needs a name."
            )
        repeated = [name for name, n in Counter(features).items() if n > 1]
        if repeated:
            raise ValueError(
                f"features names {quoted(repeated)} more than once; name each column once."
            )
        return features

    @field_validator("target")
    @classmethod
    def _target_named(cls, target: str) -> str:
        if not target:
            raise ValueError(
                "target is an empty column name ''; give the name of the target column."
            )
        return target

    @model_validator(mode="after")
    def _kernel_stays_within_the_features(self) -> Self:
        if isinstance(self.kernel, str):
            return self
        beyond = _gp.beyond_the_inputs(self.kernel, len(self.features))
        if beyond is not None:
            part, dims = beyond
            n = len(self.features)
            raise ValueError(
                f"The kernel's {type(part).__name__} has active_dims {dims}, but there "
                f"{'is' if n == 1 else 'are'} only {count(n, 'feature')} "
                f"({quoted(list(self.features))}): active_dims are positions in features, "
                "counting from 0."
            )
        return self

    @model_validator(mode="after")
    def _target_is_not_an_input(self) -> Self:
        if self.target in self.features:
            raise ValueError(
                f"The target '{self.target}' is also in features; a column cannot be "
                "both an input and the target."
            )
        return self

    if not TYPE_CHECKING:  # so that type checkers still see a frozen model

        def __setattr__(self, name: str, value: Any) -> None:
            raise ValueError(
                f"A {type(self).__name__} never changes, so its {name} cannot be set. "
                f"model.replace({name}=...) returns a new model with the change."
            )

    def __hash__(self) -> int:
        # Not hashing `schema`: schema objects (pandera's, for one) need not be hashable, and
        # a GPJax kernel or mean function is not, so each is hashed by the name skgp shows for
        # it. Models equal in every setting, schema included, still hash the same.
        return hash(
            (
                self.features,
                self.target,
                _gp.kernel_shown(self.kernel),
                _gp.mean_function_shown(self.mean_function),
            )
        )

    def __repr_args__(self) -> Iterator[tuple[str, Any]]:
        # How a model prints: the columns, then the other settings in the order they are
        # declared, and the schema last, only when one was given.
        yield "features", list(self.features)
        yield "target", self.target
        for name in type(self).model_fields:
            if name == "kernel" and not isinstance(self.kernel, str):
                yield name, _ShownAsIs(_gp.kernel_shown(self.kernel))
            elif name == "mean_function" and not isinstance(self.mean_function, str):
                yield name, _ShownAsIs(_gp.mean_function_shown(self.mean_function))
            elif name not in ("features", "target", "schema"):
                yield name, getattr(self, name)
        if self.schema is not None:
            yield "schema", self.schema

    def replace(self, **changes: Any) -> Self:
        """A new model like this one with `changes` made, for example
        `model.replace(kernel="rbf")`. This one is unchanged.

        The new model is checked as any new model is, so a mistake raises `ValueError`.
        A setting you never gave keeps its default unless you change it.
        """
        given = {name: getattr(self, name) for name in self.model_fields_set}
        return type(self)(**{**given, **changes})

    def check(
        self,
        data: IntoDataFrame,
        *,
        for_prediction: bool = False,
        verbose: bool = False,
    ) -> CheckedData:
        """Run every data check of this model on `data` and return it as `CheckedData`.

        Training data (the default) needs the feature and target columns and
        at least 5 rows. Each input must hold numbers, or dates, or dates and times: a
        date input, which the model counts in days since 1970-01-01 in UTC (a date and
        time with no time zone is taken as UTC). A regressor's target must hold numbers; a
        classifier's must hold exactly two different values, of any type. A
        `GPRegressor`'s `noise_column`, if it has one, must hold numbers that are zero or
        more; rows to predict need it only for a prediction with noise, and it is
        checked whenever they have it. With `for_prediction=True`
        only the feature columns are needed, and any number of rows, and a target
        column is ignored. Any column that is neither an input, the target nor the
        column of known noise is ignored. Your `schema`, if you gave one, runs first; whatever it raises
        propagates. Raises `ValueError`, naming the column at fault, when one of
        skgp's own checks fails.

        `fit` and `predict` accept the result without checking it again, on this
        model or any other of the same task (regression or classification) with the
        same features, target, schema and column of known noise.

        With `verbose=True`, skgp emits log records through `loguru`, as `fit` and
        `predict` do: the settings given, which checks ran and whether they passed,
        and the names of any columns ignored. With the default `False` it emits
        nothing.
        """
        report = Reporter("check", verbose)
        _report_settings_given(report, self)
        return _checked.check(
            self,
            _frames.to_frame(data),
            for_prediction=for_prediction,
            report=report,
        )

    def _known_noise_column(self) -> str | None:
        """The column of known noise: none, unless the model takes a `noise_column`."""
        return None

    @property
    def likelihood(self) -> Likelihood:
        """How the target is modelled given the curve. The `model chosen` record names it,
        and the fit on minibatches, which serves regressors and classifiers alike, fits with
        it."""
        raise NotImplementedError

    @property
    def fit_kind(self) -> FitKind[Any]:
        """How this model is fitted (see `skgp._fit_kinds`)."""
        raise NotImplementedError

    def _fit[F: _Fitted](
        self,
        fitted_class: type[F],
        data: IntoDataFrame | CheckedData,
        iterations: int,
        verbose: bool,
    ) -> F:
        """What `fit` does for every model: check the data, rescale the inputs, prepare the
        target, fit, log and warn. The public `fit` of each kind of model calls it with the
        class of fitted model it returns."""
        started = time.monotonic()
        _require_whole_number_above_zero(
            iterations, "iterations", "how many iterations the fit may take"
        )
        report = Reporter("fit", verbose)
        _report_settings_given(report, self)
        frame = _checked.checked(self, data, for_prediction=False, report=report)._frame
        report.info(
            "kernel chosen",
            kernel=_gp.kernel_shown(self.kernel),
            is_default="kernel" not in self.model_fields_set,
        )
        report.info(
            "mean function chosen",
            mean_function=_gp.mean_function_shown(self.mean_function),
            is_default="mean_function" not in self.model_fields_set,
        )
        raw_x = _frames.columns_to_array(frame, self.features)
        kind = self.fit_kind
        large_data_message = kind.large_data_message(raw_x.shape[0])
        if large_data_message is not None and raw_x.shape[0] > config.size_warning_rows:
            # Past this method and the public `fit`, to whoever called it.
            warnings.warn(large_data_message, LargeDataWarning, stacklevel=3)
        report.info(
            "model chosen", model=kind.description(self), likelihood=self.likelihood
        )
        x_scaler = Standardiser.fit(raw_x)
        y, target_fields, target_rescaling = self._prepared_target(frame, report)
        prior = _gp.prior_to_fit(
            self.kernel,
            self.mean_function,
            x_scaler,
            self._target_scaler(target_fields),
        )
        report.info(
            "rescaling chosen",
            **_rescaling_fields(self.features, x_scaler),
            **target_rescaling,
        )
        x = x_scaler.apply(raw_x)
        x.flags.writeable = False
        y.flags.writeable = False
        fitted_gp, optimisation = kind.fit(
            self,
            prior,
            x,
            y,
            iterations=int(iterations),
            report=report,
            known_noise_variance=self._known_noise_variance(frame, target_fields),
        )
        fitted = fitted_class(
            model=self,
            _training_rows=len(x),
            _date_features=tuple(
                name
                for name in self.features
                if _frames.holds_dates_or_times(frame, name)
            ),
            _fitted_gp=fitted_gp,
            _optimisation=optimisation,
            _x_scaler=x_scaler,
            **target_fields,
        )
        # A fit on minibatches is not judged (`finished_properly` is None), so it never
        # warns, and its record has no verdict.
        did_not_finish = optimisation.finished_properly is False
        (report.warning if did_not_finish else report.info)(
            "fit finished",
            **_those_given(
                finished_properly=optimisation.finished_properly,
                iterations=optimisation.iterations,
                elapsed_seconds=time.monotonic() - started,
                final_objective=optimisation.final_objective,
                stop_reason=optimisation.stop_reason,
                optimiser_message=optimisation.optimiser_message,
            ),
            **fitted._learned(),
        )
        if did_not_finish:
            warnings.warn(
                kind.did_not_finish_message(optimisation),
                ConvergenceWarning,
                stacklevel=3,  # past this method and the public `fit`
            )
        return fitted

    def _target_scaler(self, target_fields: dict[str, Any]) -> Standardiser | None:
        """How the target was rescaled (see `_prepared_target`, which gives
        `target_fields`): not at all, unless a subclass rescales it."""
        return None

    def _known_noise_variance(
        self, frame: Frame, target_fields: dict[str, Any]
    ) -> np.ndarray | None:
        """The variance of each row's known noise, rescaled as the target was (see
        `_prepared_target`, which gives `target_fields`): none, unless the model takes a
        `noise_column`."""
        return None

    def _prepared_target(
        self, frame: Frame, report: Reporter
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
        """The target as the model is fitted to it, one number per row; the fields of the
        fitted model that say how it was made from your column (a regressor's rescaling, a
        classifier's classes); and the target's part of the `rescaling chosen` log record."""
        raise NotImplementedError


class _SparseModel(_Model):
    """What every sparse model description shares: how many inducing points summarise the
    data, whether the fit works through it in minibatches, and the seed of everything random
    in the fit. `SparseGPRegressor` and `SparseGPClassifier` take these settings from here."""

    minibatch_size: Annotated[WholeNumber, Field(gt=0)] | None = None
    """How many rows each iteration of the fit looks at. Leave it unset to fit on all the
    rows at once. A whole number above zero fits on random minibatches of that size, one
    per iteration, for exactly the `iterations` given to `fit`. skgp does not judge
    whether such a fit converged: it issues no `ConvergenceWarning`, and `to_json()` gives
    the iterations it ran and its final objective, with no `finished_properly`. Zero or a
    negative number is refused."""

    num_inducing: Annotated[WholeNumber, Field(gt=0)] | None = None
    """How many inducing points summarise the data, a whole number above zero. If it is
    more than the number of rows in the training data, every row is used. Leave it
    unset to let skgp choose."""

    random_state: Annotated[WholeNumber, Field(ge=0, le=C.MAX_RANDOM_STATE)] = 0
    """A whole number from 0 to 9,223,372,036,854,775,807 that seeds everything random in
    the fit (the k-means starting points and the order of the minibatches). The same data,
    settings and `random_state` give the same model; change it to see how much a fit
    depends on it."""


@dataclass(frozen=True, eq=False, kw_only=True)
class _Fitted:
    """What every fitted model shares, whatever it predicts. It never changes.

    Each subclass is a dataclass with `repr=False`, so that it keeps the printed summary
    defined here, and declares `model` again with the classes that can make it."""

    model: _Model
    _training_rows: int
    # The fitted Gaussian process, which predicts, says what was learned and pickles itself
    # (see `_gp.FittedGP`).
    _fitted_gp: _gp.FittedGP
    _optimisation: _how_a_fit_ended.Optimisation
    # The shift and scale of each input column used before fitting. The kernel turns them
    # back, so its values are in the units of the columns (see `_gp.prior_to_fit`).
    _x_scaler: Standardiser
    # The date inputs of the training data, counted in days since 1970-01-01. They must be
    # date inputs in rows to predict too, and every other input must hold numbers.
    _date_features: tuple[str, ...]
    # The skgp version that fitted this model. A pickled model keeps it, so `to_json()`
    # still names it when the model is loaded with another version.
    _skgp_version: str = field(default_factory=lambda: version("skgp"))

    def __repr__(self) -> str:
        """A short summary: which description made the model, its columns, target, kernel
        and mean function, the number of rows, whether the fit finished properly and what it learned,
        in your units. A fit on minibatches is not judged, so its summary does not say how
        it ended. A long list of columns is cut short; `to_json()` has them all."""
        model = self.model
        finished = {
            True: "; the fit finished properly",
            False: "; the fit did not finish properly",
            None: "",
        }[self._optimisation.finished_properly]
        return (
            f"{type(self).__name__} made by {type(model).__name__} from "
            f"{self._training_rows:,} rows{finished}\n"
            f"  features {listed(self._features_shown())}; target {self._target_shown()}; "
            f"kernel {_gp.kernel_shown(model.kernel)}; "
            f"mean_function {_gp.mean_function_shown(model.mean_function)}"
            f"{''.join(f'; {name} {value}' for name, value in self._setup_after_kernel().items())}\n"
            f"  learned {_log.learned_text(**self._learned())}"
        )

    @property
    def gpjax_model(self) -> Any:
        """The underlying fitted GPJax model, for cases skgp does not cover.

        See docs/advanced.md.
        """
        return self._fitted_gp.gpjax_model

    def check(self, data: IntoDataFrame, *, verbose: bool = False) -> CheckedData:
        """Run every data check on rows to predict and return them as `CheckedData`.

        The same as `check(data, for_prediction=True)` on the model that was fitted,
        and also that each date input of the training data is a date input here, and
        every other input holds numbers: the feature columns are needed, the target is
        not, and any number of rows will do. Pass the result to `predict`, which does
        not check it again, apart from the date inputs. With
        `verbose=True`, skgp logs what the checks did.
        """
        checked = self.model.check(data, for_prediction=True, verbose=verbose)
        self._require_date_inputs_where_they_were(checked._frame)
        return checked

    def _score(self, data: IntoDataFrame | CheckedData, scoring: Any) -> "Score":
        """What `score` does for every fitted model: check `data` once as rows to predict,
        with their known targets, predict them, and score the prediction by each scorer of
        `scoring` (see `skgp.sklearn.Scoring`)."""
        # `skgp.sklearn` builds on the fitted models, so it is imported here, when scoring.
        from skgp import sklearn as skgp_sklearn

        model = self.model
        checked = _checked.checked(
            model, data, for_prediction=True, report=Reporter("score", False)
        )
        frame = checked._frame
        if len(frame) == 0:
            raise ValueError(
                "The data has no rows; scoring needs at least 1 row with a known target."
            )
        _checked.check_known_targets(model, frame)
        self._require_known_classes(frame)
        observed = self._observed(frame[model.target].to_numpy())
        prediction = self.predict(checked)
        return Score(
            skgp_sklearn._scores(prediction, observed, scoring),
            number_of_rows=len(frame),
        )

    def _observed(self, target: np.ndarray) -> np.ndarray:
        """The known targets as the scorers compare the predictions with them: as they
        are, unless a subclass codes them."""
        return target

    def predict(self, data: Any) -> Any:
        """What the model expects at each row of `data` (see each subclass)."""
        raise NotImplementedError

    def _require_known_classes(self, frame: Frame) -> None:
        """Raise `ValueError` if the targets of `frame` hold a class the model was not
        fitted to: nothing to check, unless the model is a classifier."""

    def _rows_to_predict(
        self,
        data: IntoDataFrame | CheckedData,
        *,
        verbose: bool,
        suffix: Any,
        batch_size: Any,
    ) -> tuple[Reporter, Frame, np.ndarray]:
        """The start of `predict`: the reporter of the call, the checked rows of `data` and
        their inputs rescaled as the training inputs were, after checking the `suffix` and
        `batch_size` given to `predict` and logging the settings and the rescaling."""
        if not isinstance(suffix, str):
            raise ValueError(
                "suffix is text added to the names of the columns skgp adds, so it "
                f"must be a string; got {suffix!r}."
            )
        _require_whole_number_above_zero(
            batch_size, "batch_size", "how many rows are predicted at a time"
        )
        report = Reporter("predict", verbose)
        model = self.model
        _report_settings_given(report, model)
        frame = _checked.checked(model, data, for_prediction=True, report=report)._frame
        self._require_date_inputs_where_they_were(frame)
        report.info(
            "rescaling applied",
            **_rescaling_fields(model.features, self._x_scaler),
            **self._target_rescaling_fields(),
        )
        x_new = self._x_scaler.apply(_frames.columns_to_array(frame, model.features))
        return report, frame, x_new

    def _require_date_inputs_where_they_were(self, frame: Frame) -> None:
        """Raise `ValueError` for the first input of the rows to predict that holds dates
        where the training data held numbers, or the other way round."""
        for name in self.model.features:
            held_dates = name in self._date_features
            if _frames.holds_dates_or_times(frame, name) == held_dates:
                continue
            if held_dates:
                raise ValueError(
                    f"Column '{name}' held dates when the model was fitted, so the rows to "
                    f"predict must hold dates there too; it holds {frame.schema[name]} values."
                )
            raise ValueError(
                f"Column '{name}' held numbers when the model was fitted, so the rows to "
                "predict must hold numbers there too, not dates."
            )

    def _features_shown(self) -> list[str]:
        """The inputs as the printed summary names them, each date input with its unit."""
        units = self._units_of_date_inputs()
        return [
            f"{name} ({units[name]})" if name in units else name
            for name in self.model.features
        ]

    def _units_of_date_inputs(self) -> dict[str, str]:
        """Each date input, with its unit, as `print` and `to_json()` give them."""
        return {name: _frames.DAYS_SINCE_1970_01_01 for name in self._date_features}

    def _target_rescaling_fields(self) -> dict[str, float]:
        """The target's part of the `rescaling applied` log record: nothing, unless a
        subclass rescaled its target."""
        return {}

    def _setup_after_kernel(self) -> dict[str, Any]:
        """The settings that `to_json()["setup"]` and the printed model give after the
        kernel: none, unless a subclass has more."""
        return {}

    def _target_shown(self) -> str:
        """The target as the printed summary names it."""
        return self.model.target

    def _target_section(self) -> dict[str, Any]:
        """What `to_json()` says about the target's values, after `setup`: nothing, unless a
        subclass says otherwise."""
        return {}

    def _learned(self) -> dict[str, Any]:
        """What fitting learned: the `learned` section of `to_json()`, also shown by the
        printed model and the `fit finished` log record (see `_gp.FittedGP.learned`)."""
        return self._fitted_gp.learned(self.model.features)

    def _json_text(self) -> str:
        """What `to_json()` returns, as strict JSON: a number that is NaN or infinite is
        written as `null`."""
        model = self.model
        optimisation = self._optimisation
        description = {
            "skgp_version": self._skgp_version,
            "training_rows": self._training_rows,
            "setup": {
                "features": list(model.features),
                **(
                    {"date_features": self._units_of_date_inputs()}
                    if self._date_features
                    else {}
                ),
                "target": model.target,
                "kernel": _gp.kernel_shown(model.kernel),
                "mean_function": _gp.mean_function_shown(model.mean_function),
                **self._setup_after_kernel(),
            },
            **self._target_section(),
            "learned": self._learned(),
            "fitting": _those_given(
                finished_properly=optimisation.finished_properly,
                iterations=optimisation.iterations,
                final_objective=optimisation.final_objective,
                stop_reason=optimisation.stop_reason,
            ),
        }
        if isinstance(model, _SparseModel):
            description["model"] = {
                "class": type(model).__name__,
                "num_inducing": self._fitted_gp.number_of_inducing_points,
                "minibatch_size": model.minibatch_size,
                "random_state": model.random_state,
            }
        return _STRICT_JSON.dump_json(description).decode()


_STRICT_JSON = TypeAdapter(dict[str, Any], config=ConfigDict(ser_json_inf_nan="null"))


def _those_given(**fields: Any) -> dict[str, Any]:
    """The `fields` that have a value, in the order given: one that is None is left out."""
    return {name: value for name, value in fields.items() if value is not None}


def _require_whole_number_above_zero(value: Any, name: str, meaning: str) -> None:
    """Raise `ValueError` unless `value`, given to a method as `name`, which is `meaning`, is a
    whole number above zero. A numpy integer counts; a bool does not."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(
            f"{name} is {meaning}, so it must be a whole number above zero; got {value!r}."
        )


def _rescaling_fields(
    features: tuple[str, ...], x_scaler: Standardiser
) -> dict[str, Any]:
    """The inputs' part of the `rescaling chosen` and `rescaling applied` log records."""
    return {
        "method": "standardise",
        "features": list(features),
        "feature_shift": x_scaler.shift.tolist(),
        "feature_scale": x_scaler.scale.tolist(),
    }


def _report_settings_given(report: Reporter, model: _Model) -> None:
    """The record `fit`, `predict` and `check` start with: what the user gave explicitly."""
    report.info(
        "settings given",
        features=list(model.features),
        target=model.target,
        given=sorted(model.model_fields_set),
        schema_supplied=model.schema is not None,
    )


# What a setting must be, for each kind of problem pydantic reports, in plain words. The
# fields come from pydantic's own report: `expected` lists the allowed values, and `gt`,
# `ge` and `le` are bounds.
_MUST_BE = {
    "missing": "{setting} is required.",
    "literal_error": "{setting} must be {expected}; got {given}.",
    "int_type": "{setting} must be a whole number; got {given}.",
    "bool_type": "{setting} must be True or False; got {given}.",
    "greater_than": "{setting} must be above {gt}; got {given}.",
    "greater_than_equal": "{setting} must be {ge} or more; got {given}.",
    "less_than_equal": "{setting} must be {le} or less; got {given}.",
    "string_type": "{setting} must be a column name, as text; got {given}.",
    "tuple_type": "{setting} must be a column name or a list of column names; got {given}.",
    "too_short": "{setting} must name at least one column; got {given}.",
}


def _problem(model_class: type[_Model], found: ErrorDetails) -> str:
    """One problem pydantic found in the settings of a `model_class`, as one plain line that
    names the setting and what was given."""
    setting = str(found["loc"][0]) if found["loc"] else "settings"
    context = found.get("ctx", {})
    if found["type"] == "value_error":  # skgp's own checks, already in plain words
        return str(context["error"])
    if found["type"] == "extra_forbidden":
        reason = model_class.settings_not_taken.get(setting)
        close = difflib.get_close_matches(setting, list(model_class.model_fields), n=1)
        hint = (
            f": {reason}"
            if reason
            else f"; did you mean '{close[0]}'?"
            if close
            else "."
        )
        return f"{model_class.__name__} has no setting '{setting}'{hint}"
    if len(found["loc"]) > 1:  # one entry of a list, such as one name in features
        setting = f"each entry of {setting}"
    given = _shown(found["input"])
    template = _MUST_BE.get(found["type"])
    if template is None:
        return f"{setting} is not valid ({found['msg']}); got {given}."
    bounds = {
        name: _shown(value) for name, value in context.items() if name != "expected"
    }
    return template.format(
        setting=setting, given=given, expected=context.get("expected"), **bounds
    )


def _shown(value: Any) -> str:
    """`value` as a problem names what was given: a whole number with commas, anything else
    as Python writes it, cut short if long."""
    text = (
        f"{value:,}"
        if isinstance(value, int) and not isinstance(value, bool)
        else repr(value)
    )
    if len(text) <= C.MAX_VALUE_CHARACTERS_SHOWN:
        return text
    ellipsis = "..."
    return f"{text[: C.MAX_VALUE_CHARACTERS_SHOWN - len(ellipsis)]}{ellipsis}"

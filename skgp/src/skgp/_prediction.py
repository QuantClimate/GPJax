from collections.abc import Callable
from dataclasses import dataclass
import html
from statistics import NormalDist
from typing import Any

import numpy as np

from skgp import _frames
from skgp._constants import Constant as C
from skgp._model import _require_whole_number_above_zero
from skgp._text import count, quoted
from skgp._types import Frame


def _seeded_generator(n_samples: Any, random_state: Any) -> np.random.Generator:
    """The numpy random number generator seeded with `random_state`, after checking that
    `n_samples` is a whole number above zero and `random_state` a whole number from 0 to
    `C.MAX_RANDOM_STATE`. A numpy integer counts; a bool does not."""
    _require_whole_number_above_zero(n_samples, "n_samples", "how many samples to give")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or not 0 <= random_state <= C.MAX_RANDOM_STATE
    ):
        raise ValueError(
            "random_state is the seed of the samples, so it must be a whole number from 0 "
            f"to {C.MAX_RANDOM_STATE:,}; got {random_state!r}."
        )
    return np.random.default_rng(int(random_state))


def _joint_normal_samples(
    mean: np.ndarray,
    covariance: np.ndarray,
    generator: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """`n_samples` samples of the joint normal distribution with `mean` and `covariance`,
    as an array of shape (rows, n_samples), the shape scikit-learn's `sample_y` gives. The
    covariance is factorised by its eigenvalues, which works when rounding makes it a
    little less than positive definite, as for the curve at two equal rows. No rows give
    no values."""
    if len(mean) == 0:
        return np.empty((0, n_samples))
    samples = generator.multivariate_normal(
        mean, covariance, size=n_samples, method="eigh"
    )
    return samples.T


@dataclass(frozen=True, eq=False, repr=False)
class _RowsAndPredictions:
    """What both kinds of prediction share: the rows you asked about, the suffix of the
    columns skgp adds, and how the two are shown together as a table."""

    _rows: Frame  # every column of the rows passed to `predict`, as they were checked
    suffix: str
    """The text added to the names of the columns skgp adds: what you passed to `predict`
    as `suffix`, empty by default."""

    def _added_columns(self) -> dict[str, np.ndarray]:
        """The columns `to_frame()` adds by default, by their names without the suffix."""
        raise NotImplementedError

    def _title(self) -> str:
        """The line printed above the table."""
        raise NotImplementedError

    def _suffixed(self, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name + self.suffix: values for name, values in columns.items()}

    def _with_rows(self, columns: dict[str, np.ndarray]) -> Any:
        """Your rows followed by `columns`, already suffixed, in your dataframe type. Raises
        `ValueError` if one of your columns has the name of one of them."""
        clashing = [name for name in columns if name in self._rows.columns]
        if clashing:
            raise ValueError(self._clash_message(clashing))
        return _frames.to_native(_frames.with_columns(self._rows, columns))

    def __repr__(self) -> str:
        title, table = self._shown()
        return f"{title}\n{table!r}"

    def _repr_html_(self) -> str:
        """The printed form as HTML, so a notebook shows the table as a table."""
        title, table = self._shown()
        table_html = getattr(table, "_repr_html_", lambda: None)()
        if table_html is None:
            table_html = f"<pre>{html.escape(repr(table))}</pre>"
        return f"<p>{html.escape(title)}</p>\n{table_html}"

    def _shown(self) -> tuple[str, Any]:
        """A title and the table `to_frame()` gives, for printing. When your columns
        clash with skgp's names, the table holds only the predictions instead, with
        your row labels, so printing never fails."""
        title = self._title()
        columns = self._suffixed(self._added_columns())
        if any(name in self._rows.columns for name in columns):
            title += (
                " (your own columns are not shown, because some share a name with "
                "the predictions; pass suffix= to predict to show them)"
            )
            return title, _frames.to_native(
                _frames.columns_for_rows_of(self._rows, columns)
            )
        return title, _frames.to_native(_frames.with_columns(self._rows, columns))

    def _clash_message(self, clashing: list[str]) -> str:
        has = "a column named" if len(clashing) == 1 else "columns named"
        message = (
            f"Your rows already have {has} {quoted(clashing)}, which skgp would add for "
            "the predictions"
        )
        if self.suffix:
            return (
                f"{message}, even with suffix '{self.suffix}'. Pass a different "
                "suffix= to predict."
            )
        return (
            f"{message}. Pass suffix= to predict, for example suffix='_predicted', to "
            "add that text to the name of every column skgp adds."
        )


@dataclass(frozen=True, eq=False, repr=False)
class Prediction(_RowsAndPredictions):
    """What a fitted regressor expects at each row you asked about, in your original units.

    By default it is for a new observation, so the spread includes measurement noise.
    Made with `predict(..., include_noise=False)` it is for the underlying curve
    without the noise: the same mean, with a smaller spread. `includes_noise` says
    which, and so does its printed form.

    The model's expectation is one joint normal distribution over all rows:
    `mean()`, `std()` and `interval()` read its per-row values, and `covariance()`
    gives how the rows move together. `to_frame()` puts the predictions next to
    your rows, and printing a prediction shows that table.

    Two predictions are equal only if they are the same object.
    """

    includes_noise: bool
    """True for a new observation, measurement noise included; False for the
    underlying curve without it."""
    _mean: np.ndarray
    _variance: np.ndarray
    # Builds the joint covariance of all rows on demand; it is never built otherwise.
    _joint_covariance: Callable[[], np.ndarray]

    def mean(self) -> np.ndarray:
        """The predicted mean of each row, in the target's units, as a numpy array.

        It is the same whether or not the prediction includes measurement noise.
        """
        return self._mean.copy()

    def std(self) -> np.ndarray:
        """The predicted standard deviation of each row, in the target's units, as a numpy array.

        It includes measurement noise when `includes_noise` is True, and is the
        spread of the underlying curve alone when it is False.
        """
        return np.sqrt(self._variance)

    def covariance(self) -> np.ndarray:
        """How the predictions for the rows move together: an n-by-n numpy array, for n rows.

        Row `i` and column `j` hold the covariance between rows `i` and `j`, in the
        target's units squared, in the order of the rows passed to `predict`. Its
        diagonal is `std() ** 2`. Like `std()`, it includes measurement noise only
        when `includes_noise` is True: the noise of each row is its own, so it adds
        to the diagonal only.

        This builds the whole matrix, so memory and time grow with the square
        of the number of rows (20,000 rows take about 3 GB). `mean()`, `std()`
        and `interval()` do not build it.
        """
        return self._joint_covariance()

    def sample(self, n_samples: int = 1, random_state: int = 0) -> np.ndarray:
        """Random samples of what the model expects, in the target's units: a numpy array
        of n rows by `n_samples` columns, for n rows, the shape scikit-learn's `sample_y`
        gives (its numbers for the same seed differ, as its random number generator does).

        Each column is one sample: a value at every row, chosen at random from the joint
        normal distribution over all the rows (`mean()` and `covariance()`), so that the
        rows hang together as the model expects; nearby rows get similar values. A sample
        includes measurement noise when `includes_noise` is True, so it is one possible
        set of new observations, and is one possible underlying curve when it is False.
        With heteroscedastic noise the noise at a row is not quite normal, so there the
        samples of new observations are a normal approximation with the right mean and
        covariance.

        `n_samples` is a whole number above zero, 1 by default. `random_state` is a whole
        number from 0 to 9,223,372,036,854,775,807, 0 by default, as in scikit-learn: the
        same prediction and `random_state` give the same samples. It builds the
        covariance, so memory and time grow with the square of the number of rows.
        """
        generator = _seeded_generator(n_samples, random_state)
        return _joint_normal_samples(
            self.mean(), self.covariance(), generator, int(n_samples)
        )

    def to_frame(self, interval: float | None = None) -> Any:
        """Your rows, every column you passed, followed by `mean` and `std` columns.

        It is the dataframe type passed to `predict`, with the same rows in the same
        order. Give `interval` a level between 0 and 1, such as 0.95, to also add
        `lower` and `upper` columns: the bounds `interval(level)` gives.

        If you passed `suffix` to `predict`, it is added to the name of every column
        skgp adds (`mean_predicted` for `suffix="_predicted"`). If one of your
        columns already has the name of a column skgp would add, this raises
        `ValueError` naming it; pass a `suffix` to `predict` to avoid the clash.
        `mean()`, `std()`, `interval()` and `covariance()` are not affected.
        """
        columns = self._added_columns()
        if interval is not None:
            columns |= self._bounds(interval, "interval")
        return self._with_rows(self._suffixed(columns))

    def interval(self, level: float = 0.95) -> Any:
        """Lower and upper bounds that each row falls between with probability `level`.

        Only the `lower` and `upper` columns (with your `suffix`, if you gave one),
        one row per row passed to `predict`, in the dataframe type passed to
        `predict`. The bounds are for a new observation when `includes_noise` is
        True, and for the underlying curve when it is False.
        """
        bounds = self._suffixed(self._bounds(level, "level"))
        return _frames.to_native(_frames.columns_for_rows_of(self._rows, bounds))

    def _added_columns(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean(), "std": self.std()}

    def _title(self) -> str:
        what = (
            "a new observation, measurement noise included"
            if self.includes_noise
            else "the underlying curve, without measurement noise"
        )
        return f"Prediction for {what}, {count(len(self._mean), 'row')}:"

    def _bounds(self, level: float, argument: str) -> dict[str, np.ndarray]:
        """`lower` and `upper` at `level`; `argument` is what the caller called the level."""
        if not 0 < level < 1:
            raise ValueError(
                f"{argument} must be between 0 and 1 (exclusive), got {level}"
            )
        z = NormalDist().inv_cdf(0.5 + level / 2)
        mean, std = self.mean(), self.std()
        return {"lower": mean - z * std, "upper": mean + z * std}


@dataclass(frozen=True, eq=False, repr=False)
class ClassPrediction(_RowsAndPredictions):
    """What a fitted classifier expects at each row you asked about: the probability of each
    of its two classes, and the more likely one.

    `classes` are the two values of your target column, in the order the model coded them.
    `probabilities()` and `most_likely()` give the results as numpy arrays, and
    `to_frame()` puts them next to your rows; printing a prediction shows that table.

    Two predictions are equal only if they are the same object.
    """

    classes: tuple[Any, Any]
    """The two values of your target column: the one coded 0 first, then the one coded 1.
    The probabilities are in this order."""
    _probability_of_second: np.ndarray  # of `classes[1]`, for each row
    # Builds the mean and the joint covariance of the underlying curve at all rows on
    # demand, for `sample`; they are never built otherwise.
    _curve_mean_and_covariance: Callable[[], tuple[np.ndarray, np.ndarray]]
    # The probability of `classes[1]` where the underlying curve has the values given.
    _probability_given_curve: Callable[[np.ndarray], np.ndarray]

    def probabilities(self) -> np.ndarray:
        """The probability of each class at each row: a numpy array of n rows by 2 columns,
        for n rows. Column 0 is for `classes[0]` and column 1 for `classes[1]`, so each row
        sums to 1."""
        second = self._probability_of_second
        return np.column_stack([1.0 - second, second])

    def most_likely(self) -> np.ndarray:
        """The more likely class at each row, as a numpy array of your target's values. A row
        whose two probabilities are exactly equal gets `classes[0]`."""
        return np.asarray(self.classes)[self.probabilities().argmax(axis=1)]

    def sample(self, n_samples: int = 1, random_state: int = 0) -> np.ndarray:
        """Random samples of the class at each row: a numpy array of n rows by `n_samples`
        columns, for n rows, holding the values of your target column.

        Each column is one sample, chosen at random jointly over all the rows: first one
        possible underlying curve, the one the model turns into probabilities, from its
        joint normal distribution at every row, so that nearby rows hang together; then
        at each row `classes[1]` with the probability that curve gives there, and
        `classes[0]` otherwise. Over many samples a row gets each class as often as
        `probabilities()` says, and two rows close together tend to get the same class.

        `n_samples` is a whole number above zero, 1 by default. `random_state` is a whole
        number from 0 to 9,223,372,036,854,775,807, 0 by default: the same prediction and
        `random_state` give the same samples. It builds the latent curve's covariance, so
        memory and time grow with the square of the number of rows.
        """
        generator = _seeded_generator(n_samples, random_state)
        mean, covariance = self._curve_mean_and_covariance()
        curves = _joint_normal_samples(mean, covariance, generator, int(n_samples))
        probability = self._probability_given_curve(curves)
        is_second = generator.uniform(size=curves.shape) < probability
        return np.asarray(self.classes)[is_second.astype(int)]

    def to_frame(self) -> Any:
        """Your rows, every column you passed, followed by the probability of each class and
        the more likely class.

        The probability columns are named `probability_` followed by the class, such as
        `probability_rain`, in the order of `classes`, and the last column is
        `most_likely_class`. It is the dataframe type passed to `predict`, with the same
        rows in the same order.

        If you passed `suffix` to `predict`, it is added to the name of every column
        skgp adds. If one of your columns already has the name of a column skgp would
        add, this raises `ValueError` naming it; pass a `suffix` to `predict` to avoid
        the clash.
        """
        return self._with_rows(self._suffixed(self._added_columns()))

    def _added_columns(self) -> dict[str, np.ndarray]:
        probabilities = self.probabilities()
        return {
            f"probability_{self.classes[0]}": probabilities[:, 0],
            f"probability_{self.classes[1]}": probabilities[:, 1],
            "most_likely_class": self.most_likely(),
        }

    def _title(self) -> str:
        return (
            f"Prediction of the classes {self.classes[0]!r} and {self.classes[1]!r}, "
            f"{count(len(self._probability_of_second), 'row')}:"
        )

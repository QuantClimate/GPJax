"""The only place that knows about dataframe libraries.

Any eager dataframe narwhals supports (Polars, pandas, ...) goes in, and
results come back as the same kind of dataframe. Everything else in skgp works
with `Frame` and numpy arrays and never branches on the dataframe library.
"""

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np

from skgp._constants import Constant as C
from skgp._types import Frame

DAYS_SINCE_1970_01_01 = "days since 1970-01-01"
"""The unit of a date input, as `print` and `to_json()` name it."""


def to_frame(native: Any) -> Frame:
    """Wrap a caller's dataframe. Raises `TypeError` for anything that is not an eager dataframe."""
    return nw.from_native(native, eager_only=True)


def to_native(frame: Frame) -> Any:
    """The caller's own dataframe type, exactly as it was passed in."""
    return frame.to_native()


def snapshot(frame: Frame) -> Frame:
    """A frame that later changes to the caller's own dataframe can never reach.

    Immutable dataframes (Polars) are shared at no cost; mutable ones (pandas and
    the like) are copied once, here.
    """
    return frame.clone()


def columns_to_array(frame: Frame, columns: tuple[str, ...]) -> np.ndarray:
    """The named columns as a float64 array of shape (rows, len(columns)).

    A column of dates, or of dates and times, becomes the number of days since the start
    of 1970-01-01 in UTC, with the time of day as a fraction of a day. A date and time with
    a time zone counts from that instant in UTC; one without a time zone is taken as it is,
    as if it were in UTC.
    """
    for name in columns:
        if _pandas_dates_and_times_narwhals_cannot_read(frame, name):
            in_utc = frame[name].to_native().dt.tz_convert("UTC")
            frame = frame.with_columns(
                nw.from_native(in_utc, series_only=True).alias(name)
            )
    return frame.select([_as_numbers(frame, name) for name in columns]).to_numpy()


def _as_numbers(frame: Frame, name: str) -> nw.Expr:
    column = nw.col(name)
    if holds_dates_or_times(frame, name):
        days = column.dt.timestamp("us").cast(nw.Float64) / C.MICROSECONDS_PER_DAY
        return days.alias(name)
    return column.cast(nw.Float64)


def holds_dates_or_times(frame: Frame, column: str) -> bool:
    """Whether `column` holds dates, or dates and times, rather than numbers or text."""
    return isinstance(
        frame.schema[column], (nw.Date, nw.Datetime)
    ) or _pandas_dates_and_times_narwhals_cannot_read(frame, column)


def _pandas_dates_and_times_narwhals_cannot_read(frame: Frame, column: str) -> bool:
    """Whether `column` is a pandas column of dates and times in a time zone whose name
    narwhals cannot read, such as "America/New_York", so that it calls the column's type
    unknown. Converted to UTC, narwhals reads it."""
    return isinstance(frame.schema[column], nw.Unknown) and str(
        frame[column].to_native().dtype
    ).startswith("datetime64[")


def distinct_values(frame: Frame, column: str) -> list[Any]:
    """The different values in a column, as Python values in sorted order: numbers from
    smallest, false before true, text alphabetically."""
    return sorted(frame[column].unique().to_list())


def is_value(frame: Frame, column: str, value: Any) -> np.ndarray:
    """1.0 for each row whose value in `column` is `value`, and 0.0 for every other row."""
    return frame.select((nw.col(column) == value).cast(nw.Float64)).to_numpy()[:, 0]


def frame_like(frame: Frame, columns: dict[str, np.ndarray]) -> Frame:
    """A new frame from numpy columns, in the same dataframe library as `frame`."""
    return nw.from_dict(columns, backend=nw.get_native_namespace(frame))


def columns_for_rows_of(frame: Frame, columns: dict[str, np.ndarray]) -> Frame:
    """Only the numpy columns, one value per row of `frame`; none of its own columns are kept.

    Row labels (a pandas index) are kept and rows are matched by position, so the
    names of `frame`'s columns can never clash with the new ones.
    """
    return with_columns(frame.drop(*frame.columns), columns)


def with_columns(frame: Frame, columns: dict[str, np.ndarray]) -> Frame:
    """`frame` followed by the numpy columns. Rows are matched by position, never by index."""
    return nw.concat([frame, frame_like(frame, columns)], how="horizontal")


def count_below_zero(frame: Frame, column: str) -> int:
    """How many values of a numeric column are below zero."""
    return int(frame.select((nw.col(column) < 0).sum()).item())


def count_missing_and_infinite(
    frame: Frame, columns: tuple[str, ...]
) -> dict[str, tuple[int, int]]:
    """For each numeric column, (number of missing values, number of infinite values).

    Missing means null or NaN. Polars and pandas treat these differently on
    their own, so this counts them the same way on both.
    """
    expressions = []
    for name in columns:
        column = nw.col(name)
        if frame.schema[name].is_float():
            infinite = (column == float("inf")) | (column == float("-inf"))
            expressions += [
                infinite.fill_null(False).sum().alias(f"{name}\0infinite"),
                column.is_finite().fill_null(False).sum().alias(f"{name}\0finite"),
            ]
        else:  # integers and decimals can only be null
            expressions.append(column.null_count().alias(f"{name}\0null"))
    result = frame.select(expressions)
    row = dict(zip(result.columns, result.row(0), strict=True))
    counts = {}
    for name in columns:
        if frame.schema[name].is_float():
            infinite = int(row[f"{name}\0infinite"])
            counts[name] = (
                len(frame) - int(row[f"{name}\0finite"]) - infinite,
                infinite,
            )
        else:
            counts[name] = (int(row[f"{name}\0null"]), 0)
    return counts

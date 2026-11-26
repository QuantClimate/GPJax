"""Where the inducing points of a sparse model start."""

import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

from skgp._constants import Constant as C


def default_count(rows: int) -> int:
    """How many inducing points to use for `rows` training rows when the user gave no number."""
    return min(
        C.MAX_INDUCING_POINTS_IF_NOT_GIVEN,
        max(
            C.MIN_INDUCING_POINTS_IF_NOT_GIVEN,
            round(C.INDUCING_POINTS_SHARE_OF_ROWS * rows),
        ),
    )


def k_means_subset_rows(rows: int, count: int) -> int | None:
    """How many of `rows` training rows k-means runs on when it looks for `count` centres, or
    None when it runs on all of them."""
    subset_rows = max(C.K_MEANS_SUBSET_ROWS, count)
    return subset_rows if rows > subset_rows else None


def k_means_centres(x: np.ndarray, count: int, seed: int) -> np.ndarray:
    """The centres of a k-means clustering of the rows of `x`, as starting inducing points.

    `count` is capped at the number of rows, so a count above it gives one centre per row.
    With more than `C.K_MEANS_SUBSET_ROWS` rows (or `count` rows, if that is more) the
    clustering runs on a random subset of that many rows. The same `x`, `count` and
    `seed` always give the same centres.
    """
    subset_rows = k_means_subset_rows(len(x), count)
    if subset_rows is not None:
        x = x[np.random.default_rng(seed).choice(len(x), subset_rows, replace=False)]
    with warnings.catch_warnings():
        # With fewer distinct rows than centres, scipy's choice of starting centres
        # divides zero by zero once every distinct row is taken, and some clusters stay
        # empty; it warns about both. Repeated centres are harmless, so the user is not
        # told. Any other warning reaches them.
        warnings.filterwarnings(
            "ignore", message="One of the clusters is empty", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        centres, _ = kmeans2(
            x, min(count, len(x)), minit="++", seed=np.random.default_rng(seed)
        )
    return centres

"""Every number, and every name of an optimiser, that sets how skgp fits, checks or reports,
as the fields of `Constant`.

What stays where it is used instead is in CODING_STANDARDS.md ("Every fixed value in one
place").
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class _Constant:
    # Data checks and rescaling.

    MIN_TRAINING_ROWS: int = 5
    """The fewest rows of training data a model will be fitted to."""

    CLASSIFIER_CLASSES: int = 2
    """How many different values the target of a classifier's training data must hold, one
    for each class."""

    ROUNDING_TOLERANCE: float = 1e-12
    """A column whose spread is at most this many times the size of its mean never varies.

    Its spread is zero, or rounding noise when its value is not exactly representable, so it
    gets a scale of 1 when it is rescaled."""

    MICROSECONDS_PER_DAY: float = 86_400_000_000.0
    """How many microseconds a day has. A column of dates or dates and times becomes days since
    1970-01-01, counted from the microseconds dataframes store."""

    # The inducing points of a sparse model.

    INDUCING_POINTS_SHARE_OF_ROWS: float = 0.1
    """The share of the training rows a sparse model uses as inducing points when the user
    gives no number, within the two bounds below. A starting guess, as are the two bounds."""

    MIN_INDUCING_POINTS_IF_NOT_GIVEN: int = 100
    """The fewest inducing points a sparse model gets when the user gives no number. If that is
    more than the number of training rows, every row is used."""

    MAX_INDUCING_POINTS_IF_NOT_GIVEN: int = 1_000
    """The most inducing points a sparse model gets when the user gives no number."""

    K_MEANS_SUBSET_ROWS: int = 10_000
    """With more training rows than this, k-means runs on a random subset of this many rows
    (or of as many rows as it looks for centres, if that is more), so that finding where the
    inducing points start does not take longer than the fit. A starting guess."""

    MAX_RANDOM_STATE: int = 2**63 - 1
    """The largest seed JAX accepts, a signed 64-bit whole number; numpy needs it not negative."""

    # The model a fit starts from. A regressor's target is shifted to mean 0 and scaled to
    # standard deviation 1; the kernel sees the inputs in the units of the user's columns.

    STARTING_LENGTHSCALE: float = 1.0
    """The length scale of a kernel given by name along each input when a fit starts, in
    standard deviations of that input."""

    STARTING_VARIANCE: float = 1.0
    """The variance of a kernel given by name when a fit starts: relative to the target's
    variance for a regressor, on the probit scale for a classifier."""

    STARTING_NOISE_STANDARD_DEVIATION: float = 0.5
    """The standard deviation of the noise on each observation when a fit starts. A
    heteroscedastic fit starts with this noise at every row."""

    NOISE_KERNEL: str = "matern52"
    """The noise kernel, by name: the kernel of the noise process, the logarithm of the noise
    variance, of a regressor with heteroscedastic noise. It starts as the kernel of that name
    does."""

    # Fitting with the optimiser: an exact fit, a sparse fit on all rows at once, or a
    # classifier's fit. The most iterations it may take is the `iterations` given to `fit`.
    # Whether a fit had settled is judged from where it ended
    # (`skgp._how_a_fit_ended._settled`).

    OPTIMISER: str = "L-BFGS-B"
    """The scipy optimiser that fits the model."""

    SEARCH_LOGARITHM_ABOVE: float = 10.0
    """The optimiser searches over the logarithm of a positive value that starts above this,
    and over the number GPJax holds for any other, the inverse of softplus of the value. That
    number moves a value by about one per unit step once the value is above 1, a tenth of it
    or more up to here. Beyond, such steps would barely move it, and that is where a column in
    large units puts its length scales."""

    GRADIENT_TOLERANCE: float = 1e-4
    """A fit had settled if no entry of its gradient is larger than this per row of training
    data, in rescaled units."""

    SETTLED_ITERATIONS: int = 3
    """A fit also had settled if its objective at its last iteration differs from its objective
    this many iterations earlier by no more than `SETTLED_TOLERANCE` per row."""

    SETTLED_TOLERANCE: float = 1e-5
    """The most a fit's objective may differ between its last iteration and the iteration
    `SETTLED_ITERATIONS` before it, per row of training data in rescaled units, for the fit to
    have settled."""

    # Fitting on minibatches: a sparse fit with a `minibatch_size`, which runs exactly the
    # `iterations` given to `fit`, one minibatch each. Whether such a fit converged is not
    # judged (decision record 0008).

    OPTIMISER_ON_MINIBATCHES: str = "natural gradients alternating with Adam"
    """The optimiser of a fit on minibatches, as the `optimiser chosen` record names it: a
    natural-gradient step on the fitted distribution over the inducing points, then an Adam
    step on everything else."""

    OPTIMISER_ON_MINIBATCHES_WITH_HETEROSCEDASTIC_NOISE: str = "Adam"
    """The optimiser of a fit on minibatches of a regressor with heteroscedastic noise, as the
    `optimiser chosen` record names it. GPJax takes no natural-gradient step on its two fitted
    distributions over the inducing points, so the Adam step moves them too."""

    ADAM_LEARNING_RATE: float = 0.05
    """The learning rate of the Adam optimiser in a fit on minibatches (the `optim` of
    GPJax's `fit_natgrads`, or of its `fit` with heteroscedastic noise)."""

    ADAM_STEP_IN_PROPORTION_ABOVE: float = 1.0
    """In a fit on minibatches, the Adam step on a positive value that GPJax holds as a number
    above this is multiplied by that number. GPJax holds a positive value as the inverse of
    softplus of it, which is about the value once that is above 1, so each step moves such a
    value by about the same share of it, whatever the units of the column. Below, GPJax's
    softplus already does, and the step is left as it is."""

    NATURAL_GRADIENT_LEARNING_RATE: float = 0.1
    """The learning rate of the natural-gradient steps in a fit on minibatches (the
    `natgrad_lr` of GPJax's `fit_natgrads`)."""

    # Messages and logs.

    PROGRESS_SECONDS: float = 5.0
    """The shortest time, in seconds, between two reports of how a verbose fit by the optimiser
    is going."""

    MAX_COLUMNS_LISTED: int = 10
    """A list of columns, or of the values in a column, longer than this is cut short where a
    person reads it."""

    MAX_VALUE_CHARACTERS_SHOWN: int = 60
    """The most characters an error message shows of a value it names. A longer value is cut to
    this many characters, the last three of them "..."."""

    SECONDS_PER_MINUTE: int = 60
    """The number of seconds in a minute. How long a fit took is given in seconds under a
    minute, and in minutes after."""


Constant: Final[_Constant] = _Constant()
"""Every number, and every name of an optimiser, that sets how skgp fits, checks or reports."""

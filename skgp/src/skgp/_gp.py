"""The only place that talks to GPJax and JAX.

Each fit returns a `FittedGP`, which predicts, says what the fit learned and pickles
itself, so no other module needs to know which GPJax model it holds. Everything that runs
JAX code happens inside a scoped 64-bit context, and the arrays that come out are numpy.
The context is thread-local and only covers operations run inside it, so each public step
re-enters it.
"""

from collections.abc import Callable, Iterator
import dataclasses
from dataclasses import dataclass
from functools import cached_property
import operator
from typing import Any

import equinox as eqx
import gpjax as gpx
from gpjax.kernels import AbstractKernel, ProductKernel, SumKernel
from gpjax.kernels.base import CombinationKernel
from gpjax.mean_functions import (
    AbstractMeanFunction,
    CombinationMeanFunction,
    ProductMeanFunction,
    SumMeanFunction,
)
from gpjax.parameters import NonNegativeReal, PositiveReal, val
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
import optax
import paramax
from paramax import AbstractUnwrappable
from scipy.optimize import OptimizeResult, minimize

from skgp import _how_a_fit_ended
from skgp._constants import Constant as C
from skgp._scaling import Standardiser
from skgp._types import Kernel, Likelihood, MeanFunction, OnIteration

_KERNELS = {
    "rbf": gpx.kernels.RBF,
    "matern32": gpx.kernels.Matern32,
    "matern52": gpx.kernels.Matern52,
}


class _InYourUnits(AbstractKernel):
    """A kernel given the inputs in the units of the user's columns.

    skgp fits on inputs rescaled to mean 0 and standard deviation 1, which keeps the inducing
    points and the optimiser's steps on the same footing whatever the columns' units. This
    kernel turns the rescaled inputs back, times `scale` plus `shift`, before `kernel` sees
    them, so every value of `kernel` that measures an input (the length scale of an RBF
    kernel, a period) is in the units of the user's columns. `kernel` computes with its own
    compute engine. `shift` and `scale` are held where they are during the fit."""

    # Equinox makes no constructor of its own for a kernel that has one, so these fields
    # need no defaults, though the base kernel's fields have them.
    kernel: AbstractKernel  # ty: ignore[invalid-dataclass-field-order]
    shift: Any  # ty: ignore[invalid-dataclass-field-order]
    scale: Any  # ty: ignore[invalid-dataclass-field-order]

    def __init__(self, kernel: AbstractKernel, x_scaler: Standardiser) -> None:
        self.kernel = kernel
        self.shift = paramax.non_trainable(
            jnp.asarray(x_scaler.shift, dtype=jnp.float64)
        )
        self.scale = paramax.non_trainable(
            jnp.asarray(x_scaler.scale, dtype=jnp.float64)
        )
        super().__init__()

    def _in_column_units(self, x: Any) -> Any:
        return x * val(self.scale) + val(self.shift)

    def __call__(self, x: Any, y: Any) -> Any:
        return self.kernel(self._in_column_units(x), self._in_column_units(y))

    # GPJax asks the prior's kernel for these three; each goes to `kernel`'s own compute
    # engine, which a kernel such as RFF needs.
    def cross_covariance(self, x: Any, y: Any) -> Any:
        return self.kernel.cross_covariance(
            self._in_column_units(x), self._in_column_units(y)
        )

    def gram(self, x: Any) -> Any:
        return self.kernel.gram(self._in_column_units(x))

    def diagonal(self, x: Any) -> Any:
        return self.kernel.diagonal(self._in_column_units(x))


def prior_to_fit(
    kernel: Kernel,
    mean_function: MeanFunction,
    x_scaler: Standardiser,
    y_scaler: Standardiser | None,
) -> Any:
    """The GPJax prior a fit starts from and learns, for the model's `kernel` and
    `mean_function` settings, inputs rescaled by `x_scaler`, and a regressor's target rescaled
    by `y_scaler` (None for a classifier, whose target is not rescaled): see `_kernel_to_fit`
    and `_mean_function_to_fit`."""
    with jax.enable_x64(True):
        return gpx.gps.Prior(
            kernel=_kernel_to_fit(kernel, x_scaler),
            mean_function=_mean_function_to_fit(mean_function, x_scaler, y_scaler),
        )


def _kernel_to_fit(kernel: Kernel, x_scaler: Standardiser) -> Any:
    """The GPJax kernel a fit starts from and learns, for `kernel`, the model's setting, and
    inputs rescaled by `x_scaler` (see `_InYourUnits`).

    A name is the matching GPJax kernel with one length scale per input, each starting at
    `C.STARTING_LENGTHSCALE` standard deviations of that input, and a variance starting at
    `C.STARTING_VARIANCE`. A GPJax kernel starts at its own values, made 64-bit: one made
    outside JAX's 64-bit mode holds 32-bit numbers."""
    with jax.enable_x64(True):
        if isinstance(kernel, str):
            chosen = _named_kernel(kernel, x_scaler.scale)
        else:
            chosen = jax.tree.map(_in_64_bits, kernel)
        return _InYourUnits(chosen, x_scaler)


def _named_kernel(name: str, scale: Any) -> AbstractKernel:
    """The GPJax kernel of `name` with one length scale per input, each starting at
    `C.STARTING_LENGTHSCALE` times that input's `scale`, its standard deviation, and a
    variance starting at `C.STARTING_VARIANCE`. Must run inside the 64-bit context."""
    return _KERNELS[name](
        lengthscale=jnp.asarray(  # ty: ignore[invalid-argument-type]  # GPJax wraps plain values itself
            C.STARTING_LENGTHSCALE * scale, dtype=jnp.float64
        ),
        variance=C.STARTING_VARIANCE,  # ty: ignore[invalid-argument-type]
        n_dims=len(scale),
    )


def _noise_kernel_to_fit(curve_kernel: Any) -> Any:
    """The noise kernel of a heteroscedastic model whose curve has `curve_kernel`, an
    `_InYourUnits`: the kernel named `C.NOISE_KERNEL`, which starts as one given by that name
    does and sees the inputs in the units of the user's columns, as `curve_kernel` does. Must
    run inside the 64-bit context."""
    noise_kernel = _named_kernel(C.NOISE_KERNEL, val(curve_kernel.scale))
    return eqx.tree_at(lambda wrapper: wrapper.kernel, curve_kernel, noise_kernel)


def _in_64_bits(leaf: Any) -> Any:
    if isinstance(leaf, (jax.Array, np.ndarray)) and jnp.issubdtype(
        leaf.dtype, jnp.floating
    ):
        return jnp.asarray(leaf, dtype=jnp.float64)
    return leaf


def _held(values: Any) -> Any:
    """`values` as 64-bit numbers that a fit holds where they are. Must run inside the 64-bit
    context."""
    return paramax.non_trainable(jnp.asarray(values, dtype=jnp.float64))


class _MeanFunction(AbstractMeanFunction):
    """A mean function as skgp fits it: on inputs rescaled by `x_scaler` and a curve rescaled
    by the target's shift and scale (0 and 1 for a classifier, whose curve is not rescaled).
    It holds the rescaling where it is, so that it can say what it learned in the units of the
    user's columns and target."""

    # Equinox makes no constructor of its own for a module that has one.
    x_shift: Any
    x_scale: Any
    target_shift: Any
    target_scale: Any

    def __init__(self, x_scaler: Standardiser, y_scaler: Standardiser | None) -> None:
        self.x_shift = _held(x_scaler.shift)
        self.x_scale = _held(x_scaler.scale)
        self.target_shift = _held(0.0 if y_scaler is None else y_scaler.shift[0])
        self.target_scale = _held(1.0 if y_scaler is None else y_scaler.scale[0])

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        """Every value the fit learned, by name, in the units of the user's columns and
        target: none, unless a subclass learns some."""
        return {}

    def _in_target_units(self, rescaled: Any) -> float:
        return float(val(self.target_shift) + val(self.target_scale) * rescaled)


class _Constant(_MeanFunction):
    """`"constant"`: one number, learned, starting where the rescaled target's average is (0),
    so a regressor's starts at the target's average."""

    constant: Any

    def __init__(self, x_scaler: Standardiser, y_scaler: Standardiser | None) -> None:
        super().__init__(x_scaler, y_scaler)
        self.constant = jnp.asarray(0.0, dtype=jnp.float64)

    def __call__(self, x: Any) -> Any:
        return jnp.full((x.shape[0], 1), val(self.constant), dtype=x.dtype)

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        return {"constant": self._in_target_units(np.asarray(val(self.constant)))}


class _Zero(_Constant):
    """`"zero"`: zero in the target's units, held there, so it learns nothing. On the
    rescaled curve that is minus the target's shift divided by its scale; a classifier's
    curve, which is not rescaled, is held at 0."""

    def __init__(self, x_scaler: Standardiser, y_scaler: Standardiser | None) -> None:
        super().__init__(x_scaler, y_scaler)
        self.constant = _held(-val(self.target_shift) / val(self.target_scale))

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        return {}


class _Linear(_MeanFunction):
    """`"linear"`: a constant plus a slope along each rescaled input, all learned, starting at
    0, so the fit starts from the target's average."""

    constant: Any
    slope: Any

    def __init__(self, x_scaler: Standardiser, y_scaler: Standardiser | None) -> None:
        super().__init__(x_scaler, y_scaler)
        self.constant = jnp.asarray(0.0, dtype=jnp.float64)
        self.slope = jnp.zeros(len(x_scaler.scale), dtype=jnp.float64)

    def __call__(self, x: Any) -> Any:
        return (self.constant + x @ self.slope)[:, None]

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        """The `intercept`, the mean where every input is 0, in the target's units, and the
        `slope` along each input, in the target's units per unit of the input's column. A
        rescaled input is the column minus its shift, divided by its scale, so a slope `b` on
        it is `b` times the target's scale divided by the column's scale in the user's units,
        and the mean at every input 0 is the constant minus each slope times its shift
        divided by its scale."""
        slopes = np.asarray(self.slope, dtype=np.float64)
        x_shift = np.asarray(val(self.x_shift), dtype=np.float64)
        x_scale = np.asarray(val(self.x_scale), dtype=np.float64)
        target_scale = float(val(self.target_scale))
        at_every_input_zero = float(self.constant) - float(
            np.sum(slopes * x_shift / x_scale)
        )
        return {
            "intercept": self._in_target_units(at_every_input_zero),
            "slope": dict(
                zip(features, map(float, slopes * target_scale / x_scale), strict=True)
            ),
        }


class _YourMeanFunction(_MeanFunction):
    """A GPJax mean function the user gave, which sees the inputs in the units of the user's
    columns and gives the curve in the target's units: the rescaled inputs are turned back
    before it sees them, and what it gives is rescaled as the target was."""

    mean_function: AbstractMeanFunction

    def __init__(
        self,
        mean_function: AbstractMeanFunction,
        x_scaler: Standardiser,
        y_scaler: Standardiser | None,
    ) -> None:
        super().__init__(x_scaler, y_scaler)
        self.mean_function = jax.tree.map(_in_64_bits, mean_function)

    def __call__(self, x: Any) -> Any:
        in_your_units = self.mean_function(x * val(self.x_scale) + val(self.x_shift))
        return (in_your_units - val(self.target_shift)) / val(self.target_scale)

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        """Every value of the GPJax mean function, by its place in it, as `_values_by_place`
        gives them, with every input of the model as the inputs it sees."""
        return _values_by_place(
            self.mean_function, AbstractMeanFunction, lambda part: list(features)
        )


_NAMED_MEAN_FUNCTIONS: dict[str, type[_MeanFunction]] = {
    "zero": _Zero,
    "constant": _Constant,
    "linear": _Linear,
}


def _mean_function_to_fit(
    mean_function: MeanFunction, x_scaler: Standardiser, y_scaler: Standardiser | None
) -> _MeanFunction:
    """The mean function a fit starts from and learns, for `mean_function`, the model's
    setting: one of skgp's by name, or the user's GPJax mean function, seeing the units of the
    user's columns and target (see `_YourMeanFunction`). Must run inside the 64-bit
    context."""
    if isinstance(mean_function, str):
        return _NAMED_MEAN_FUNCTIONS[mean_function](x_scaler, y_scaler)
    return _YourMeanFunction(mean_function, x_scaler, y_scaler)


class _KnownAndLearnedNoise(gpx.likelihoods.Gaussian):
    """GPJax's Gaussian likelihood for a regressor with known noise: the noise variance at
    each training row is that row's known variance, `known_noise_variance`, plus the learned
    noise variance, the square of `obs_stddev`. With no learned noise, `obs_stddev` is held
    at zero. GPJax's exact fit reads the noise at the training rows from `noise_vector`, and
    adds only `obs_stddev` at new rows, whose known noise skgp adds itself."""

    known_noise_variance: Any = None

    def __init__(self, known_noise_variance: np.ndarray, *, learns_noise: bool) -> None:
        super().__init__(obs_stddev=C.STARTING_NOISE_STANDARD_DEVIATION)
        self.known_noise_variance = paramax.non_trainable(
            jnp.asarray(known_noise_variance, dtype=jnp.float64)
        )
        if not learns_noise:
            self.obs_stddev = paramax.non_trainable(jnp.asarray(0.0, dtype=jnp.float64))

    def noise_vector(self, n: int) -> Any:
        return val(self.known_noise_variance) + jnp.square(val(self.obs_stddev))


def _starting_model(
    prior: Any,
    likelihood: Likelihood = "Gaussian",
    *,
    known_noise_variance: np.ndarray | None = None,
    learns_noise: bool = True,
) -> Any:
    """The model a fit starts from: the Gaussian process `prior` (see `prior_to_fit`) times
    the `likelihood`. A Gaussian likelihood with a `known_noise_variance`
    at each training row adds it to the learned noise, or uses it alone without
    `learns_noise` (see `_KnownAndLearnedNoise`). Must run inside the 64-bit context."""
    # A heteroscedastic model starts from `_starting_heteroscedastic_family` instead.
    assert likelihood != "HeteroscedasticGaussian"
    if likelihood == "Bernoulli":
        return prior * gpx.likelihoods.Bernoulli()  # ty: ignore[missing-argument]  # GPJax gives the integrator a default
    if known_noise_variance is not None:
        return prior * _KnownAndLearnedNoise(
            known_noise_variance, learns_noise=learns_noise
        )
    return prior * gpx.likelihoods.Gaussian(
        obs_stddev=C.STARTING_NOISE_STANDARD_DEVIATION
    )


def kernel_shown(kernel: Kernel) -> str:
    """The kernel as skgp shows it: a name as given, a GPJax kernel by its class, and a sum
    or a product of kernels written out with `+` or `*`, in brackets inside another:
    `(RBF + Linear) * Periodic`."""
    return kernel if isinstance(kernel, str) else _classes_shown(kernel, inside=False)


def mean_function_shown(mean_function: MeanFunction) -> str:
    """The mean function as skgp shows it: a name as given, and a GPJax mean function as
    `kernel_shown` shows a GPJax kernel, such as `Constant + Constant`."""
    if isinstance(mean_function, str):
        return mean_function
    return _classes_shown(mean_function, inside=False)


# The sign each sum or product of kernels or of mean functions is written out with.
_SIGNS = {
    SumKernel: " + ",
    ProductKernel: " * ",
    SumMeanFunction: " + ",
    ProductMeanFunction: " * ",
}


def _classes_shown(part: AbstractKernel | AbstractMeanFunction, *, inside: bool) -> str:
    sign = _SIGNS.get(type(part))
    if isinstance(part, (CombinationKernel, CombinationMeanFunction)) and sign:
        parts = part.kernels if isinstance(part, CombinationKernel) else part.means
        text = sign.join(_classes_shown(each, inside=True) for each in parts)
        return f"({text})" if inside else text
    return type(part).__name__


def _parts[T: eqx.Module](
    whole: T, kind: type[T], place: str = ""
) -> Iterator[tuple[str, T]]:
    """Every part of `whole` that is a `kind` (a kernel or a mean function), `whole` first,
    each with its place: `""` for `whole` itself, `"kernels[1]."` for the second kernel of a
    sum."""
    yield place, whole
    for name, value in _fields(whole):
        if isinstance(value, kind):
            yield from _parts(value, kind, f"{place}{name}.")
        elif isinstance(value, (tuple, list)):
            for index, part in enumerate(value):
                if isinstance(part, kind):
                    yield from _parts(part, kind, f"{place}{name}[{index}].")


def _fields(module: eqx.Module) -> Iterator[tuple[str, Any]]:
    """Each field of `module` that it holds, by name, with its value."""
    for field in dataclasses.fields(module):
        if hasattr(module, field.name):
            yield field.name, getattr(module, field.name)


def beyond_the_inputs(
    kernel: AbstractKernel, n_inputs: int
) -> tuple[AbstractKernel, Any] | None:
    """The first kernel in `kernel` whose `active_dims` point past the last of `n_inputs`
    inputs, with those `active_dims`, or None if every one stays within them."""
    for _, part in _parts(kernel, AbstractKernel):
        dims = part.active_dims
        if isinstance(dims, slice):
            beyond = (dims.start is not None and dims.start >= n_inputs) or (
                dims.stop is not None and dims.stop > n_inputs
            )
        else:
            beyond = any(not -n_inputs <= dim < n_inputs for dim in dims)
        if beyond:
            return part, dims
    return None


def _values_of(fitted_kernel: Any, features: tuple[str, ...]) -> dict[str, Any]:
    """Every value of the kernel inside `fitted_kernel`, an `_InYourUnits`, by its place in
    the kernel: `lengthscale`, or `kernels[1].period` for a value of the second kernel of a
    sum. The kernel saw the inputs in the units of the user's columns and a regressor's
    target rescaled to standard deviation 1 (see `_InYourUnits`). A value with one number
    per input of its kernel is a dict by input name, a single number a float, and anything
    else a list."""

    def inputs_of(part: AbstractKernel) -> list[str]:
        dims = part.active_dims
        if isinstance(dims, slice):
            return list(features[dims])
        return [features[dim] for dim in dims]

    return _values_by_place(fitted_kernel.kernel, AbstractKernel, inputs_of)


def _values_by_place[T: eqx.Module](
    whole: T, kind: type[T], inputs_of: Callable[[T], list[str]]
) -> dict[str, Any]:
    """Every value in `whole`, a kernel or a mean function (its `kind`), by its place in it
    (see `_parts`), each with the inputs `inputs_of` a part gives: a value with one number per
    input is a dict by input name, a single number a float, and anything else a list."""
    values: dict[str, Any] = {}
    with jax.enable_x64(True):
        for place, part in _parts(whole, kind):
            for name, held in _fields(part):
                value = val(held) if isinstance(held, AbstractUnwrappable) else held
                if eqx.is_inexact_array(value):
                    numbers = np.asarray(value, dtype=np.float64)
                    values[f"{place}{name}"] = _by_input(numbers, inputs_of(part))
    return values


def _by_input(numbers: np.ndarray, inputs: list[str]) -> Any:
    if numbers.shape == (len(inputs),):
        return dict(zip(inputs, map(float, numbers), strict=True))
    if numbers.size == 1:
        return float(numbers.reshape(()))
    return numbers.tolist()


_POSITIVE = (PositiveReal, NonNegativeReal)


def _step_scale_of_a_positive_value(held: Any) -> Any:
    """What an Adam step on a number that holds a positive value is multiplied by: the number
    itself, but no less than `C.ADAM_STEP_IN_PROPORTION_ABOVE` (see `_for_each_number`)."""
    return jnp.maximum(C.ADAM_STEP_IN_PROPORTION_ABOVE, held)


def _for_each_number(tree: Any, if_positive: Any, otherwise: Any) -> Any:
    """`tree`, a model or its trainable part, with each number that holds a positive value
    replaced by `if_positive` of it, and every other number by `otherwise` of it.

    GPJax holds a positive value as the inverse of softplus of it, a number close to the
    value itself once the value is above 1. So a length scale in the units of a column that
    runs to thousands is held as a number in the thousands, which an optimiser's step of the
    usual size would barely move, and one in the hundred-thousandths as a number near -12.
    The optimisers below treat these numbers apart, so that a step changes a value in
    proportion to it, whatever the units of the column."""

    def replace(node: Any) -> Any:
        if isinstance(node, _POSITIVE):
            return jax.tree.map(if_positive, node)
        return otherwise(node)

    return jax.tree.map(replace, tree, is_leaf=lambda node: isinstance(node, _POSITIVE))


def _inverse_softplus(value: Any) -> Any:
    return value + jnp.log(-jnp.expm1(-value))


def _steps_in_proportion() -> optax.GradientTransformation:
    """An optax step that multiplies the update of each number that holds a positive value
    by that number, as it is at the time, when it is above `C.ADAM_STEP_IN_PROPORTION_ABOVE`
    (see `_for_each_number`), and the update of every other number of a user's GPJax mean
    function by the target's scale, because that number is in the target's units, not in the
    rescaled units of every other number (see `_YourMeanFunction`). Without it, a mean
    function's constant on a target in the thousands would move by about the learning rate,
    a few hundredths, at each step."""

    def init(params: Any) -> Any:
        return optax.EmptyState()

    def update(updates: Any, state: Any, params: Any = None) -> tuple[Any, Any]:
        assert params is not None  # GPJax passes them, held values included
        scales = _for_each_number(
            params, _step_scale_of_a_positive_value, jnp.ones_like
        )

        def in_the_targets_units(part: Any, scales_of_part: Any) -> Any:
            if not isinstance(part, _YourMeanFunction):
                return scales_of_part
            target_scale = val(part.target_scale)
            yours = _for_each_number(
                part.mean_function,
                _step_scale_of_a_positive_value,
                lambda number: jnp.full_like(number, target_scale),
            )
            return eqx.tree_at(lambda m: m.mean_function, scales_of_part, yours)

        scales = jax.tree.map(
            in_the_targets_units,
            params,
            scales,
            is_leaf=lambda part: isinstance(part, _YourMeanFunction),
        )
        return jax.tree.map(operator.mul, updates, scales), state

    return optax.GradientTransformation(init, update)


def probability_given_curve(curve: np.ndarray) -> np.ndarray:
    """The probability of the class coded 1 where a classifier's underlying curve has the
    values `curve`, as GPJax's `Bernoulli` likelihood gives it: the probit, kept between
    0.001 and 0.999."""
    with jax.enable_x64(True):
        probability = gpx.likelihoods.inv_probit(jnp.asarray(curve, dtype=jnp.float64))
        return np.asarray(probability, dtype=np.float64)


def _dataset(x: np.ndarray, y: np.ndarray) -> Any:
    return gpx.Dataset(
        X=jnp.asarray(x, dtype=jnp.float64),
        y=jnp.asarray(y.reshape(-1, 1), dtype=jnp.float64),
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class FittedGP:
    """The fitted Gaussian process inside a fitted model: the fitted GPJax model, with
    whatever else it needs to predict. Every fit below returns one, of the kind that suits
    what it fitted, so no caller asks which GPJax class is inside.

    It predicts at new rows, says what the fit learned and pickles itself. Its numbers are in
    fitted units: the inputs, and a regressor's target, rescaled to mean 0 and standard
    deviation 1."""

    gpjax_model: Any
    """The fitted GPJax model itself, which `fitted.gpjax_model` gives (see
    docs/advanced.md)."""

    def mean_and_variance(
        self,
        x_new: np.ndarray,
        *,
        batch_size: int,
        include_noise: bool,
        known_noise_variance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance at each row of `x_new`, of a new observation (noise included)
        with `include_noise`, or of the underlying curve without. A new observation of a
        regressor with known noise has its row's `known_noise_variance` added too.

        A classifier's new observation is its class, so with `include_noise` the mean is the
        probability of the class coded 1. GPJax's `Bernoulli` likelihood gives it from the
        latent curve's mean and variance at the row, as the probit of the mean divided by the
        square root of one plus the variance, so a row the model knows little about gets a
        probability nearer 0.5. GPJax keeps every probability between 0.001 and 0.999.

        Rows are predicted `batch_size` at a time, so that memory does not grow with the
        number of rows predicted.
        """
        with jax.enable_x64(True):
            means, variances = [], []
            for start in range(
                0, max(len(x_new), 1), batch_size
            ):  # one (empty) batch for no rows
                predictive = self._predictive(
                    x_new[start : start + batch_size],
                    dense=False,
                    include_noise=include_noise,
                )
                means.append(np.asarray(predictive.mean, dtype=np.float64))
                variances.append(np.asarray(predictive.variance, dtype=np.float64))
        variance = np.concatenate(variances)
        if include_noise and known_noise_variance is not None:
            variance = variance + known_noise_variance
        return np.concatenate(means), variance

    def covariance(
        self,
        x_new: np.ndarray,
        *,
        include_noise: bool,
        known_noise_variance: np.ndarray | None = None,
    ) -> np.ndarray:
        """The full covariance between the rows of `x_new`, of a new observation (noise
        included) with `include_noise`, or of the underlying curve without. A new
        observation of a regressor with known noise has its row's `known_noise_variance` added
        to its own variance: known noise at one row is independent of that at another.

        Shape (rows, rows): memory grows with the square of the number of rows.
        """
        with jax.enable_x64(True):
            covariance = np.asarray(
                self._predictive(
                    x_new, dense=True, include_noise=include_noise
                ).covariance_matrix,
                dtype=np.float64,
            )
        if include_noise and known_noise_variance is not None:
            covariance = covariance + np.diag(known_noise_variance)
        return covariance

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        """Every value the fit learned, by name, in this order, for inputs named `features`:

        - `kernel`: every value of the user's kernel (see `_values_of`).
        - `mean_function`, unless it learned nothing: every value of the mean function, in
          the units of the user's columns and target (see `_MeanFunction.learned`).
        - `noise`, for a regressor that learns noise: the standard deviation of the noise on each observation,
          in fitted units. With heteroscedastic noise it is the median noise far from the
          training data, where the noise process has its learned mean, a constant: the
          exponential of half of it. A prediction there adds a larger noise variance, the
          mean the log-normal gives, which also counts how unsure the noise process is.
        - `noise_kernel`, only with heteroscedastic noise: every value of the noise kernel.
          Its variance is that of the noise process, the logarithm of the noise variance, so
          it has no units.
        """
        raise NotImplementedError

    @property
    def number_of_inducing_points(self) -> int:
        """How many inducing points a variational family has: with heteroscedastic noise
        those of the curve, as many as the noise process has. Only a sparse model is asked."""
        raise NotImplementedError

    @cached_property
    def _conditioned(self) -> Any:
        """The GPJax model conditioned, which every prediction is read from, made on the
        first prediction and kept, so later predictions and `covariance` reuse it.

        GPJax does the expensive work here, once, and the result keeps it, so querying it
        again is cheap. That kept work takes memory that grows with the square of the
        training rows for an exact model (about 72 MB at 3,000 rows and 800 MB at 10,000),
        with the inducing points times the training rows for a sparse regression model
        fitted on all rows at once, and with the square of the inducing points for every
        other model (a GPClassifier, and a GPRegressor with heteroscedastic noise, has one at
        every row).
        """
        with jax.enable_x64(True):
            return self._condition()

    def _condition(self) -> Any:
        """The GPJax model conditioned (see `_conditioned`). Must run inside the 64-bit
        context."""
        raise NotImplementedError

    def _predictive(
        self, x_new: np.ndarray, *, dense: bool, include_noise: bool
    ) -> Any:
        """GPJax's distribution at the rows `x_new`: of a new observation with
        `include_noise`, of the underlying curve (before the noise is added) without. With
        `dense` it holds the covariance between all the rows, otherwise only each row's
        variance. Must run inside the 64-bit context."""
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        """What pickle saves: everything but the conditioned model that the first
        prediction keeps, which can be hundreds of megabytes; the next prediction after
        loading makes it again. The GPJax model's arrays are saved as numpy arrays: JAX
        makes an array 32-bit when it is unpickled outside the 64-bit context, which skgp
        enters only while it runs, and a numpy array keeps its type."""
        state = vars(self).copy()
        state.pop("_conditioned", None)
        state["gpjax_model"] = jax.tree.map(
            lambda leaf: np.asarray(leaf) if isinstance(leaf, jax.Array) else leaf,
            self.gpjax_model,
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """A pickled `FittedGP` rebuilt, with the GPJax model's arrays 64-bit JAX arrays
        again."""
        with jax.enable_x64(True):
            gpjax_model = jax.tree.map(
                lambda leaf: (
                    jnp.asarray(leaf) if isinstance(leaf, np.ndarray) else leaf
                ),
                state["gpjax_model"],
            )
        vars(self).update(state, gpjax_model=gpjax_model)


def _mean_function_learned(prior: Any, features: tuple[str, ...]) -> dict[str, Any]:
    """`{"mean_function": ...}`, what the mean function of `prior` learned, or nothing if it
    learned nothing."""
    with jax.enable_x64(True):
        learned = prior.mean_function.learned(features)
    return {"mean_function": learned} if learned else {}


@dataclass(frozen=True, eq=False, kw_only=True)
class _OneProcess(FittedGP):
    """A `FittedGP` whose GPJax model has one Gaussian process, the curve, which GPJax
    conditions as a whole.

    It keeps the training data, as `training_data`, the rescaled inputs and target, only
    when its GPJax model needs them to predict: an exact model, and a sparse regression
    model fitted on all rows at once, whose family works out its distribution over the
    inducing points from the data when it is conditioned. Every other GPJax model carries
    its fitted distribution over the inducing points in their place, and is conditioned on
    nothing. `learns_noise` says
    whether the likelihood is Gaussian, with a noise the fit learned, as a regressor's is,
    or GPJax's `Bernoulli`, as a classifier's is."""

    training_data: tuple[np.ndarray, np.ndarray] | None = None
    learns_noise: bool = True

    @property
    def _prior_times_likelihood(self) -> Any:
        """The model of the prior times the likelihood: the GPJax model itself, or the one
        inside it."""
        raise NotImplementedError

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        model = self._prior_times_likelihood
        values: dict[str, Any] = {
            "kernel": _values_of(model.prior.kernel, features),
            **_mean_function_learned(model.prior, features),
        }
        if self.learns_noise:
            with jax.enable_x64(True):
                values["noise"] = float(val(model.likelihood.obs_stddev))
        return values

    def _condition(self) -> Any:
        if self.training_data is None:
            return self.gpjax_model.condition(None)
        return self.gpjax_model.condition(_dataset(*self.training_data))

    def _predictive(
        self, x_new: np.ndarray, *, dense: bool, include_noise: bool
    ) -> Any:
        curve = self._conditioned(
            jnp.asarray(x_new, dtype=jnp.float64),
            covariance="dense" if dense else "diagonal",
        )
        return (
            self._prior_times_likelihood.likelihood(curve) if include_noise else curve
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class _ExactModel(_OneProcess):
    """A `FittedGP` of GPJax's model of the prior times a Gaussian likelihood, which
    `fit_exact` fitted, with the training data it is conditioned on to predict. It has no
    inducing points."""

    @property
    def _prior_times_likelihood(self) -> Any:
        return self.gpjax_model


@dataclass(frozen=True, eq=False, kw_only=True)
class _VariationalFamily(_OneProcess):
    """A `FittedGP` of a GPJax variational family of one curve, which holds the model of
    the prior times the likelihood, at `model`, and the inducing points."""

    @property
    def _prior_times_likelihood(self) -> Any:
        return self.gpjax_model.model

    @property
    def number_of_inducing_points(self) -> int:
        return len(val(self.gpjax_model.inducing_inputs))


@dataclass(frozen=True, eq=False, kw_only=True)
class _HeteroscedasticFamily(FittedGP):
    """A `FittedGP` of GPJax's heteroscedastic variational family: the curve and the noise
    process, each with a fitted distribution over its inducing points, which stands for the
    training data. GPJax conditions each process on its own fitted distribution, and not
    the two together.

    GPJax's family holds three copies of the model: at `model`, in the curve's distribution
    at `signal_variational.model`, and the noise process alone in the noise process's
    distribution at `noise_variational.model`. A fit trains the curve's prior only in the
    second and the noise process's prior only in the third, and leaves the first, which
    `learned` reads and `docs/advanced.md` points users at, where it started. So the family
    is kept with the fitted model in every place it holds one, whichever fit made it."""

    def __post_init__(self) -> None:
        family = self.gpjax_model
        with jax.enable_x64(True):
            fitted_model = eqx.tree_at(
                lambda m: (m.prior, m.noise_model),
                family.model,
                (family.signal_variational.model.prior, family.noise_variational.model),
            )
            every_copy_fitted = eqx.tree_at(
                lambda f: (f.model, f.signal_variational.model),
                family,
                (fitted_model, fitted_model),
            )
        # How a frozen dataclass sets its own field after it is made.
        object.__setattr__(self, "gpjax_model", every_copy_fitted)

    def learned(self, features: tuple[str, ...]) -> dict[str, Any]:
        model = self.gpjax_model.model
        with jax.enable_x64(True):
            log_variance = np.asarray(
                val(model.noise_prior.mean_function.constant)
            ).ravel()[0]
        return {
            "kernel": _values_of(model.prior.kernel, features),
            **_mean_function_learned(model.prior, features),
            "noise": float(np.exp(log_variance / 2)),
            "noise_kernel": _values_of(model.noise_prior.kernel, features),
        }

    @property
    def number_of_inducing_points(self) -> int:
        return len(val(self.gpjax_model.signal_variational.inducing_inputs))

    def _condition(self) -> Any:
        return (
            self.gpjax_model.signal_variational.condition(None),
            self.gpjax_model.noise_variational.condition(None),
        )

    def _predictive(
        self, x_new: np.ndarray, *, dense: bool, include_noise: bool
    ) -> Any:
        """See `FittedGP._predictive`. The noise variance at each row is the mean the
        log-normal gives from the noise process there, which counts how unsure the process
        is, and adds to the row's own variance only."""
        rows = jnp.asarray(x_new, dtype=jnp.float64)
        curve_posterior, noise_process_posterior = self._conditioned
        curve = curve_posterior(rows, covariance="dense" if dense else "diagonal")
        if not include_noise:
            return curve
        return self.gpjax_model.model.likelihood(
            curve, noise_process_posterior(rows, covariance="diagonal")
        )


def fit_exact(
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    known_noise_variance: np.ndarray | None,
    learns_noise: bool,
    max_iterations: int,
    on_iteration: OnIteration,
) -> tuple[FittedGP, _how_a_fit_ended.Optimisation]:
    """Maximise the marginal likelihood of a model with the `prior` in at most
    `max_iterations` iterations, calling `on_iteration` after each. Each training row's
    noise variance is its `known_noise_variance`, if given, plus the learned noise variance, if
    `learns_noise`. Returns a `FittedGP` that keeps the training data: its GPJax model is
    conditioned on them to predict."""
    with jax.enable_x64(True):
        data = _dataset(x, y)
        model = _starting_model(
            prior,
            known_noise_variance=known_noise_variance,
            learns_noise=learns_noise,
        )
        fitted, optimisation = _minimise(
            model,
            lambda m: -gpx.objectives.conjugate_mll(m, data),
            len(x),
            max_iterations,
            on_iteration,
        )
    return (
        _ExactModel(
            gpjax_model=fitted, training_data=(x, y), learns_noise=learns_noise
        ),
        optimisation,
    )


def _starting_family(
    family_class: Any,
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    inducing_inputs: np.ndarray,
    *,
    likelihood: Likelihood = "Gaussian",
) -> tuple[Any, Any]:
    """The data and the variational family of `family_class` a variational fit starts from,
    with the inducing points where they were put. Must run inside the 64-bit context."""
    family = family_class(
        model=_starting_model(prior, likelihood),
        inducing_inputs=jnp.asarray(inducing_inputs, dtype=jnp.float64),
    )
    return _dataset(x, y), family


def fit_sparse_regression_on_all_rows(
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    inducing_inputs: np.ndarray,
    *,
    max_iterations: int,
    on_iteration: OnIteration,
) -> tuple[FittedGP, _how_a_fit_ended.Optimisation]:
    """Maximise the collapsed bound on all the data in at most `max_iterations` iterations,
    moving the inducing points too, and calling `on_iteration` after each iteration.

    Returns a `FittedGP` of the fitted variational family, which holds the model and the
    inducing points, with the training data: the family works out its distribution over the
    inducing points from them when it is conditioned to predict.
    """
    with jax.enable_x64(True):
        data, family = _starting_family(
            gpx.variational_families.CollapsedVariationalGaussian,
            prior,
            x,
            y,
            inducing_inputs,
        )
        fitted, optimisation = _minimise(
            family,
            lambda f: -gpx.objectives.collapsed_elbo(f, data),
            len(x),
            max_iterations,
            on_iteration,
        )
    return _VariationalFamily(gpjax_model=fitted, training_data=(x, y)), optimisation


def fit_sparse_on_minibatches(
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    inducing_inputs: np.ndarray,
    minibatch_size: int,
    *,
    iterations: int,
    likelihood: Likelihood,
    random_state: int,
    show_progress: bool,
) -> tuple[FittedGP, _how_a_fit_ended.Optimisation]:
    """Fit a sparse model with the `likelihood` on random minibatches of `minibatch_size`
    rows for exactly `iterations` iterations, one minibatch each, moving the inducing points
    too.

    Each iteration takes a natural-gradient step on the fitted distribution over the
    inducing points, then an Adam step on everything else; the step on a positive value that
    GPJax holds as a number above `C.ADAM_STEP_IN_PROPORTION_ABOVE` is multiplied by that
    number (see `_steps_in_proportion`). GPJax takes no natural-gradient step on the two
    distributions of a heteroscedastic model, so there the Adam step moves them too. With
    `show_progress`, GPJax shows its own progress bar. Returns a `FittedGP` of the fitted
    variational family, which holds the model, the inducing points and the fitted
    distribution over them (two, for a heteroscedastic model), and so keeps no training
    data.
    """
    iterations_per_pass = -(-len(x) // minibatch_size)

    def negative_heteroscedastic_elbo(family: Any, batch: Any) -> Any:
        return -gpx.objectives.heteroscedastic_elbo(family, batch)

    def negative_elbo(family: Any, batch: Any) -> Any:
        return -gpx.objectives.elbo(family, batch)

    with jax.enable_x64(True):
        # What the two GPJax fits below are given alike.
        fit_settings: dict[str, Any] = {
            "optim": optax.chain(
                optax.adam(C.ADAM_LEARNING_RATE), _steps_in_proportion()
            ),
            "key": jax.random.key(random_state),
            "num_iters": iterations,
            "batch_size": minibatch_size,
            "verbose": show_progress,
        }
        if likelihood == "HeteroscedasticGaussian":
            data, family = _starting_heteroscedastic_family(
                prior, x, y, inducing_inputs, held=False
            )
            fitted, objectives = gpx.fit(
                model=family,
                objective=negative_heteroscedastic_elbo,
                train_data=data,
                **fit_settings,
            )
            fitted_gp: FittedGP = _HeteroscedasticFamily(gpjax_model=fitted)
        else:
            data, family = _starting_family(
                gpx.variational_families.WhitenedVariationalGaussian,
                prior,
                x,
                y,
                inducing_inputs,
                likelihood=likelihood,
            )
            fitted, objectives = gpx.fit_natgrads(
                model=family,
                objective=negative_elbo,
                train_data=data,
                natgrad_lr=C.NATURAL_GRADIENT_LEARNING_RATE,
                **fit_settings,
            )
            fitted_gp = _VariationalFamily(
                gpjax_model=fitted, learns_noise=likelihood == "Gaussian"
            )
        objectives = np.asarray(objectives, dtype=np.float64)
    return fitted_gp, _how_a_fit_ended.on_minibatches(objectives, iterations_per_pass)


def fit_classifier_on_all_rows(
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iterations: int,
    on_iteration: OnIteration,
    inducing_inputs: np.ndarray | None = None,
) -> tuple[FittedGP, _how_a_fit_ended.Optimisation]:
    """Fit a classifier to the 0s and 1s in `y` on all rows at once, maximising the evidence
    lower bound in at most `max_iterations` iterations, and calling `on_iteration` after each.

    The model is the Gaussian process prior with GPJax's `Bernoulli` likelihood, whose link
    is the probit, fitted by variational inference. GPJax's collapsed bound refuses a
    non-Gaussian likelihood, so the fitted distribution over the inducing points is fitted
    together with the kernel settings.

    - With no `inducing_inputs` (a `GPClassifier`) there is an inducing point at every
      training row, held there (`paramax.non_trainable`), so the inducing points summarise
      nothing: the fitted distribution is over the latent values at the training rows
      themselves.
    - Otherwise (a `SparseGPClassifier`) the inducing points start at `inducing_inputs` and
      move during the fit.

    Returns a `FittedGP` of the fitted variational family, which holds the model, the
    inducing points and the fitted distribution over them, and so keeps no training data.
    """
    with jax.enable_x64(True):
        data, family = _starting_family(
            gpx.variational_families.WhitenedVariationalGaussian,
            prior,
            x,
            y,
            x if inducing_inputs is None else inducing_inputs,
            likelihood="Bernoulli",
        )
        if inducing_inputs is None:
            family = eqx.tree_at(
                lambda f: f.inducing_inputs,
                family,
                paramax.non_trainable(family.inducing_inputs),
            )
        fitted, optimisation = _minimise(
            family,
            lambda f: -gpx.objectives.elbo(f, data),
            len(x),
            max_iterations,
            on_iteration,
        )
    return _VariationalFamily(gpjax_model=fitted, learns_noise=False), optimisation


def _starting_heteroscedastic_family(
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    inducing_inputs: np.ndarray,
    *,
    held: bool,
) -> tuple[Any, Any]:
    """The data and the variational family a heteroscedastic fit starts from. Must run
    inside the 64-bit context.

    The model has two Gaussian processes: the curve, with the `prior`, and the noise process, the logarithm of the noise variance, with the noise kernel (see
    `_noise_kernel_to_fit`) and a constant mean that starts at the logarithm of
    `C.STARTING_NOISE_STANDARD_DEVIATION` squared and is learned. The noise variance is the
    exponential of the noise process. Each process has a fitted distribution over inducing
    points that start at `inducing_inputs`, held there with `held`. GPJax's own family fits
    those distributions as they are, and at an inducing point at every row an L-BFGS-B fit
    of them stopped at a flat curve with all the variation as noise; fitted whitened, as a
    classifier's is, the same fit found the curve and the noise. So the family is GPJax's
    with whitened distributions in place of its own."""
    likelihood = gpx.likelihoods.HeteroscedasticGaussian(  # ty: ignore[missing-argument]  # GPJax gives the integrator a default
        noise_transform=gpx.likelihoods.LogNormalTransform()
    )
    starting_noise_mean = gpx.mean_functions.Constant(
        jnp.log(jnp.asarray(C.STARTING_NOISE_STANDARD_DEVIATION, jnp.float64) ** 2)
    )
    model = gpx.gps.HeteroscedasticModel(
        prior=prior,
        likelihood=likelihood,
        noise_prior=gpx.gps.Prior(
            mean_function=starting_noise_mean,
            kernel=_noise_kernel_to_fit(prior.kernel),
        ),
    )
    starting_inputs = jnp.asarray(inducing_inputs, dtype=jnp.float64)
    whitened = gpx.variational_families.WhitenedVariationalGaussian
    family = eqx.tree_at(
        lambda f: (f.signal_variational, f.noise_variational),
        gpx.variational_families.HeteroscedasticVariationalFamily(
            model, inducing_inputs=starting_inputs
        ),
        # GPJax starts each distribution where its prior is when given none.
        (
            whitened(model=model, inducing_inputs=starting_inputs),  # ty: ignore[missing-argument]
            whitened(model=model.noise_model, inducing_inputs=starting_inputs),  # ty: ignore[missing-argument]
        ),
    )
    if held:
        held_inputs = paramax.non_trainable(starting_inputs)
        family = eqx.tree_at(
            lambda f: (
                f.signal_variational.inducing_inputs,
                f.noise_variational.inducing_inputs,
            ),
            family,
            (held_inputs, held_inputs),
        )
    return _dataset(x, y), family


def fit_heteroscedastic_on_all_rows(
    prior: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iterations: int,
    on_iteration: OnIteration,
    inducing_inputs: np.ndarray | None = None,
) -> tuple[FittedGP, _how_a_fit_ended.Optimisation]:
    """Fit a regressor with heteroscedastic noise (see `_starting_heteroscedastic_family`)
    on all rows at once, maximising the evidence lower bound in at most `max_iterations`
    iterations, and calling `on_iteration` after each. GPJax has no exact fit of it.

    - With no `inducing_inputs` (a `GPRegressor`) there is an inducing point at every
      training row, held there, for each process, as in a `GPClassifier`.
    - Otherwise (a `SparseGPRegressor`) both processes' inducing points start at
      `inducing_inputs` and move during the fit, each set on its own.

    Returns a `FittedGP` of the fitted heteroscedastic variational family, which holds the
    fitted distribution over each process's inducing points, and so keeps no training data.
    """
    with jax.enable_x64(True):
        data, family = _starting_heteroscedastic_family(
            prior,
            x,
            y,
            x if inducing_inputs is None else inducing_inputs,
            held=inducing_inputs is None,
        )
        fitted, optimisation = _minimise(
            family,
            lambda f: -gpx.objectives.heteroscedastic_elbo(f, data),
            len(x),
            max_iterations,
            on_iteration,
        )
    return _HeteroscedasticFamily(gpjax_model=fitted), optimisation


def _minimise(
    model: Any,
    objective_of_model: Any,
    rows: int,
    max_iterations: int,
    on_iteration: OnIteration,
) -> tuple[Any, _how_a_fit_ended.Optimisation]:
    """L-BFGS-B on every trainable number in `model`, whose objective sums over `rows` rows,
    for at most `max_iterations` iterations, calling `on_iteration` after each. Must run
    inside the 64-bit context.

    A number wrapped in `paramax.non_trainable` (the rescaling a kernel and a mean function
    hold, a classifier's inducing points) is held where it is, not given to the optimiser. The optimiser searches over the
    logarithm of each positive value that starts above `C.SEARCH_LOGARITHM_ABOVE`, so its
    steps change such a value in proportion to it, whatever the units of the column (see
    `_for_each_number`), and over every other number as GPJax holds it. The logarithm is
    kept to such values because whether a fit had settled is judged by its gradient (see
    `_how_a_fit_ended._settled`), which is larger on a logarithm: on data with no noise,
    where a fit creeps along until it stops, that would call fits unsettled that predict as
    well as ever."""
    params, static = eqx.partition(
        model,
        eqx.is_array,
        is_leaf=lambda node: isinstance(node, paramax.NonTrainable),
    )
    held_at_start, unravel = ravel_pytree(params)
    holds_a_positive_value, _ = ravel_pytree(
        _for_each_number(params, jnp.ones_like, jnp.zeros_like)
    )
    by_logarithm = (holds_a_positive_value > 0) & (
        jax.nn.softplus(held_at_start) > C.SEARCH_LOGARITHM_ABOVE
    )

    def model_at(searched):
        # Exponentiating only the logarithms keeps a large number elsewhere from
        # overflowing, which would make the gradient undefined.
        logarithms = jnp.where(by_logarithm, searched, 0.0)
        held = jnp.where(by_logarithm, _inverse_softplus(jnp.exp(logarithms)), searched)
        return eqx.combine(unravel(held), static)

    @jax.jit
    def value_and_grad(searched):
        return jax.value_and_grad(lambda s: objective_of_model(model_at(s)))(searched)

    def objective(vector):
        value, grad = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(grad, dtype=np.float64)

    objectives: list[float] = []  # after each iteration

    def after_each_iteration(intermediate_result: OptimizeResult) -> None:
        objectives.append(float(intermediate_result.fun))
        on_iteration(len(objectives), objectives[-1])

    searched_at_start = jnp.where(
        by_logarithm, jnp.log(jax.nn.softplus(held_at_start)), held_at_start
    )
    result = minimize(
        objective,
        np.asarray(searched_at_start),
        jac=True,
        method=C.OPTIMISER,
        options={"maxiter": max_iterations},
        callback=after_each_iteration,
    )
    fitted = model_at(jnp.asarray(result.x))
    return fitted, _how_a_fit_ended.on_all_rows(
        objectives,
        float(result.fun),
        np.asarray(result.jac, dtype=np.float64),
        # SciPy's code for reaching its limit on the number of steps.
        reached_the_iteration_limit=result.status == 1,
        optimiser_message=str(result.message),
        rows=rows,
    )

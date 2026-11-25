"""Human-readable model summaries for GPJax pytrees.

Render any GPJax model (kernel, prior, posterior, likelihood, variational
family, ...) as a flat ``rich`` table -- one row per parameter -- in the
spirit of GPflow's ``print_summary``. Use the free function :func:`summarise`,
or rely on the ``__rich__`` / ``_repr_mimebundle_`` hooks attached to the
user-facing abstract bases for ``rich.print`` and notebook auto-rendering.
"""

import io
import typing as tp

import jax
import jax.core
import numpy as np
from numpyro.distributions import biject_to
import paramax
from paramax import AbstractUnwrappable
from rich import box
from rich.console import Console
from rich.table import Table

from gpjax.parameters import SigmoidBounded

__all__ = ["summarise"]

# Default GPflow-style column set, in display order.
_DEFAULT_COLUMNS: tuple[str, ...] = (
    "Parameter",
    "Class",
    "Value",
    "Bijector",
    "Prior",
    "Trainable",
    "Shape",
    "Dtype",
)

# Friendly labels for the numpyro transforms returned by ``biject_to``.
_BIJECTOR_NAMES: dict[str, str] = {
    "SoftplusTransform": "Softplus",
    "IdentityTransform": "Identity",
    "SoftplusLowerCholeskyTransform": "LowerCholesky",
}


class ParamRecord(tp.NamedTuple):
    """A single rendered parameter row (rendering-agnostic)."""

    name: str
    cls: str
    value: tp.Any
    bijector: str
    prior: tp.Any
    trainable: bool
    shape: tuple[int, ...]
    dtype: str


def _is_param_leaf(x: tp.Any) -> bool:
    """Stop pytree traversal at parameter objects."""
    return isinstance(x, AbstractUnwrappable)


def _is_frozen(leaf: tp.Any) -> bool:
    """True iff ``leaf``'s subtree contains a ``paramax.NonTrainable``."""
    leaves = jax.tree_util.tree_leaves(
        leaf, is_leaf=lambda y: isinstance(y, paramax.NonTrainable)
    )
    return any(isinstance(x, paramax.NonTrainable) for x in leaves)


def _bijector_name(leaf: tp.Any) -> str:
    """Friendly bijector label for a parameter leaf."""
    if isinstance(leaf, SigmoidBounded):
        return f"Sigmoid[{leaf.low:g}, {leaf.high:g}]"
    constraint = getattr(leaf, "_constraint", None)
    if constraint is None:
        return "Identity"
    raw = type(biject_to(constraint)).__name__
    return _BIJECTOR_NAMES.get(raw, raw)


def _short_dtype(dtype: tp.Any) -> str:
    """Abbreviate a dtype name, e.g. float64 -> f64."""
    name = np.dtype(dtype).name
    return (
        name.replace("float", "f")
        .replace("complex", "c")
        .replace("uint", "u")
        .replace("int", "i")
    )


def _is_traced(value: tp.Any) -> bool:
    """True for abstract values seen under a ``jax.jit`` trace."""
    return isinstance(value, jax.core.Tracer)


def _format_value(value: tp.Any, *, max_array: int, precision: int) -> str:
    """Format a parameter value; never crashes on traced values."""
    if _is_traced(value):
        return f"<traced {_short_dtype(value.dtype)}{list(value.shape)}>"
    arr = np.asarray(value)
    if arr.ndim == 0:
        return f"{float(arr):.{precision}g}"
    flat = arr.reshape(-1)
    shown = ", ".join(f"{float(v):.{precision}g}" for v in flat[:max_array])
    if flat.size > max_array:
        shown += ", ..."
    return f"[{shown}]"


def _format_prior(prior: tp.Any, *, precision: int) -> str:
    """Friendly label for a parameter's prior (or ``-`` when absent).

    Accepts ``None`` (renders ``-``), a pre-formatted string, or any object
    exposing NumPyro-style ``arg_constraints`` (e.g. a ``numpyro`` distribution),
    which is rendered as ``Name(arg=value, ...)``.
    """
    if prior is None:
        return "-"
    if isinstance(prior, str):
        return prior
    arg_constraints = getattr(type(prior), "arg_constraints", None)
    if isinstance(arg_constraints, dict) and arg_constraints:
        parts = []
        for arg in arg_constraints:
            try:
                value = float(np.asarray(getattr(prior, arg)))
            except (TypeError, ValueError, AttributeError):
                parts = []
                break
            parts.append(f"{arg}={value:.{precision}g}")
        if parts:
            return f"{type(prior).__name__}({', '.join(parts)})"
    return type(prior).__name__


def _collect(
    model: tp.Any, *, priors: tp.Mapping[str, tp.Any] | None = None
) -> list[ParamRecord]:
    """Walk ``model`` and produce one :class:`ParamRecord` per parameter.

    ``priors`` optionally maps a parameter's name (as shown in the Parameter
    column) to a prior object, populating the otherwise-empty Prior column.
    """
    prior_map = priors or {}
    records: list[ParamRecord] = []
    paths_leaves, _ = jax.tree_util.tree_flatten_with_path(
        model, is_leaf=_is_param_leaf
    )
    for path, leaf in paths_leaves:
        is_param = _is_param_leaf(leaf)
        if not (is_param or isinstance(leaf, (jax.Array, np.ndarray))):
            continue
        value = paramax.unwrap(leaf) if is_param else leaf
        # A frozen *bare* array stops traversal at a NonTrainable wrapper; label
        # it like its unfrozen form ("Array"), not "NonTrainable".
        if not is_param or isinstance(leaf, paramax.NonTrainable):
            cls = "Array"
        else:
            cls = type(leaf).__name__
        name = jax.tree_util.keystr(path).lstrip(".")
        records.append(
            ParamRecord(
                name=name,
                cls=cls,
                value=value,
                bijector=_bijector_name(leaf) if is_param else "Identity",
                prior=prior_map.get(name),
                trainable=not _is_frozen(leaf),
                shape=tuple(getattr(value, "shape", ())),
                dtype=_short_dtype(value.dtype) if hasattr(value, "dtype") else "?",
            )
        )
    return records


def _render(
    records: list[ParamRecord],
    *,
    columns: tp.Sequence[str] = _DEFAULT_COLUMNS,
    title: str | None = None,
    max_array: int = 4,
    precision: int = 3,
) -> Table:
    """Render collected records into a ``rich.Table``."""
    table = Table(title=title, box=box.ROUNDED, title_justify="left")
    for column in columns:
        table.add_column(column, overflow="fold")

    accessors: dict[str, tp.Callable[[ParamRecord], str]] = {
        "Parameter": lambda r: r.name,
        "Class": lambda r: r.cls,
        "Value": lambda r: _format_value(
            r.value, max_array=max_array, precision=precision
        ),
        "Bijector": lambda r: r.bijector,
        "Prior": lambda r: _format_prior(r.prior, precision=precision),
        "Trainable": lambda r: (
            "[green]yes[/green]" if r.trainable else "[dim red]no[/dim red]"
        ),
        "Shape": lambda r: str(r.shape),
        "Dtype": lambda r: r.dtype,
    }

    n_trainable = 0
    for record in records:
        cells = [accessors[column](record) for column in columns]
        table.add_row(*cells, style=None if record.trainable else "dim")
        n_trainable += int(record.trainable)

    plural = "" if len(records) == 1 else "s"
    table.caption = f"{len(records)} parameter{plural}, {n_trainable} trainable"
    table.caption_justify = "left"
    return table


def summarise(
    model: tp.Any,
    *,
    columns: tp.Sequence[str] | None = None,
    console: Console | None = None,
    max_array: int = 4,
    precision: int = 3,
    title: str | None = None,
    priors: tp.Mapping[str, tp.Any] | None = None,
) -> None:
    """Print a ``rich`` table summarising a GPJax model's parameters.

    Renders one row per parameter -- showing the constrained value, bijector,
    trainability, shape, and dtype -- for any GPJax model (kernel, prior,
    posterior, likelihood, or variational family).

    Args:
        model: Any GPJax pytree (kernel, prior, posterior, likelihood,
            variational family, ...).
        columns: Subset/ordering of columns to show. Defaults to the full
            GPflow-style column set.
        console: Target ``rich.Console``; defaults to a fresh one.
        max_array: Maximum number of array elements shown per value.
        precision: Significant figures for numeric values.
        title: Table title; defaults to the model's class name.
        priors: Optional mapping from a parameter's name (as shown in the
            Parameter column) to a prior object (e.g. a NumPyro distribution),
            used to populate the Prior column. Defaults to ``None`` (all ``-``).

    Example:
        >>> import gpjax as gpx
        >>> kernel = gpx.kernels.RBF()
        >>> gpx.summarise(kernel)  # doctest: +SKIP
    """
    cols = tuple(columns) if columns is not None else _DEFAULT_COLUMNS
    unknown = [c for c in cols if c not in _DEFAULT_COLUMNS]
    if unknown:
        raise ValueError(
            f"unknown column(s) {unknown}; valid columns are {list(_DEFAULT_COLUMNS)}"
        )
    table = _render(
        _collect(model, priors=priors),
        columns=cols,
        title=title if title is not None else type(model).__name__,
        max_array=max_array,
        precision=precision,
    )
    target = console if console is not None else Console()
    target.print(table)


class _SummaryMixin:
    """Adds ``rich`` / notebook rendering to user-facing GPJax bases.

    ``repr`` is intentionally left to Equinox; this only powers
    ``rich.print(model)`` and Jupyter auto-rendering.
    """

    def __rich__(self) -> Table:
        return _render(_collect(self), title=type(self).__name__)

    def _repr_mimebundle_(
        self, include: tp.Any = None, exclude: tp.Any = None
    ) -> dict[str, str]:
        console = Console(record=True, file=io.StringIO(), width=120)
        console.print(self.__rich__())
        return {
            "text/plain": console.export_text(clear=False),
            "text/html": console.export_html(
                inline_styles=True,
                code_format='<pre style="font-family:Menlo,monospace">{code}</pre>',
            ),
        }

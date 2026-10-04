"""Matplotlib styling for GPJax figures.

`use_style` applies QuantClimate's "Calibrated Ledger" style when the optional
`qc-core` package is installed, and the Matplotlib style that ships with GPJax
when it is not. Matplotlib is imported only when `use_style` is called, so it
stays an optional dependency of GPJax.
"""

from pathlib import Path
import typing as tp

__all__ = ["STYLE_PATH", "use_style"]

#: The fallback Matplotlib style file that ships with GPJax.
STYLE_PATH: Path = Path(__file__).with_name("gpjax.mplstyle")


def use_style() -> tp.Literal["ledger", "gpjax"]:
    """Apply the GPJax plotting style to Matplotlib globally.

    If `qc-core` is installed, this applies its Calibrated Ledger style, which
    also registers the ledger colour names (for example, `color="oxblood"`).
    Otherwise, it applies the style file at `STYLE_PATH`.

    Returns:
        The name of the style that was applied: `"ledger"` or `"gpjax"`.

    Raises:
        ImportError: If Matplotlib is not installed.
    """
    try:
        from matplotlib import style
    except ImportError as error:
        raise ImportError(
            "gpjax.plotting needs Matplotlib. Install it with `pip install matplotlib`."
        ) from error

    try:
        from qc_core import plotting as qc_plotting
    except ImportError:
        style.use(STYLE_PATH)
        return "gpjax"

    qc_plotting.use_ledger_style()
    return "ledger"

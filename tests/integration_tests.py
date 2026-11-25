# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: docs
#     language: python
#     name: python3
# ---

# %%
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
import os
import sys

# %%
from typing import (
    Any,
)

import gpjax
import jax.numpy as jnp
import jupytext

# %%
get_last = lambda x: x[-1]


# %%
@dataclass
class Result:
    path: str
    comparisons: field(default_factory=dict)  # type: ignore
    precision: int = 1
    compare_history: bool = True

    def __post_init__(self):
        self.name: str = self.path.split("/")[-1].split(".")[0].replace("_", "-")
        self.failures: list = []

    def _compare(
        self,
        observed_variables: dict[str, Any],
        variable_name: str,
        true_value: float,
        operation: Callable[[Any], Any],
    ):
        if variable_name == "history" and not self.compare_history:
            return
        value = operation(observed_variables[variable_name])
        if not abs(true_value - value) < self.precision:
            message = (
                f"{self.name}: {variable_name} drifted from golden value "
                f"{true_value} (got {value}, precision {self.precision})"
            )
            print(message)
            self.failures.append(message)

    def test(self):
        notebook = jupytext.read(self.path)
        contents = ""
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                if c["source"].startswith("%"):
                    pass
                else:
                    contents += c["source"]
            contents += "\n"

        contents = contents.replace('plt.style.use("./gpjax.mplstyle")', "").replace(
            "plt.show()", ""
        )
        lines = contents.split("\n")
        contents = "\n".join([line for line in lines if not line.startswith("%")])

        loc = {}
        # weird bug in interactive interpreter: lambda functions
        # don't have access to the global scope of the executed file
        # so we need to pass gpjax in the globals explicitly
        # since it's used in a lambda function inside the examples
        _globals = globals()
        _globals["gpx"] = gpjax
        # The notebooks do `from utils import ...`, which resolves against the
        # directory they live in (docs/examples/) when Sphinx executes them.
        # This script runs from the repository root, so put that directory on
        # sys.path for the duration of the exec.
        notebook_dir = os.path.dirname(os.path.abspath(self.path))
        sys.path.insert(0, notebook_dir)
        try:
            exec(contents, _globals, loc)
        finally:
            sys.path.remove(notebook_dir)
        for k, v in self.comparisons.items():
            truth, op = v
            self._compare(
                observed_variables=loc, variable_name=k, true_value=truth, operation=op
            )
        if self.failures:
            raise AssertionError(
                f"{self.name}: golden-value drift detected:\n"
                + "\n".join(self.failures)
            )


# %%
regression = Result(
    path="docs/examples/regression.py",
    comparisons={
        "history": (55.07405622, get_last),
        "predictive_mean": (37.91222107, jnp.sum),
        "predictive_std": (202.36889441, jnp.sum),
    },
)
regression.test()

# %%
sparse = Result(
    path="docs/examples/collapsed_vi.py",
    comparisons={
        "history": (1851.11700608, get_last),
        "predictive_mean": (1.37497714, jnp.sum),
        "predictive_std": (248.32254630, jnp.sum),
    },
)
sparse.test()

# %%
stochastic = Result(
    path="docs/examples/uncollapsed_vi.py",
    comparisons={
        "history": (59440.08265547, get_last),
        "meanf": (-55.18585235, jnp.sum),
        "sigma": (555.41381240, jnp.sum),
    },
)
stochastic.test()

# %%
heteroscedastic = Result(
    path="docs/examples/heteroscedastic_inference.py",
    comparisons={
        "history": (-139.22405213, get_last),
    },
)
heteroscedastic.test()

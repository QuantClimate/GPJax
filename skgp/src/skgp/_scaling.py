"""Shifting and scaling of columns, and converting results back.

Nothing else in skgp knows how the numbers were scaled.
"""

from dataclasses import dataclass

import numpy as np

from skgp._constants import Constant as C


@dataclass(frozen=True, eq=False)
class Standardiser:
    """Shifts each column to mean 0 and scales it to standard deviation 1.

    Two standardisers are equal only if they are the same object, which also makes them
    hashable; comparing their arrays would be ambiguous."""

    shift: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardiser":
        shift = values.mean(axis=0)
        spread = values.std(axis=0)
        # A column that never varies has a spread of zero, or of rounding noise
        # when its value is not exactly representable. Use a scale of 1 for it.
        never_varies = spread <= C.ROUNDING_TOLERANCE * np.abs(shift)
        return cls(shift=shift, scale=np.where(never_varies, 1.0, spread))

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.shift) / self.scale

    def undo(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.shift

    def undo_variance(self, variances: np.ndarray) -> np.ndarray:
        return variances * self.scale**2

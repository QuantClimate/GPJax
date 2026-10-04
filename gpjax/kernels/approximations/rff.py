"""Compute Random Fourier Feature (RFF) kernel approximations."""

import beartype.typing as tp
import jax.random as jr
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import BasisFunctionComputation
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.typing import (
    Array,
    KeyArray,
)


class RFF(AbstractKernel):
    r"""Computes an approximation of the kernel using Random Fourier Features.

    All stationary kernels are equivalent to the Fourier transform of a probability
    distribution. We call the corresponding distribution the spectral density. Using
    a finite number of basis functions, we can compute the spectral density using a
    Monte-Carlo approximation. This is done by sampling from the spectral density and
    computing the Fourier transform of the samples. The kernel is then approximated by
    the inner product of the Fourier transform of the samples with the Fourier
    transform of the data.

    The key reference for this implementation is the following papers:
    - 'Random Features for Large-Scale Kernel Machines' by Rahimi and Recht (2008).
    - 'On the Error of Random Fourier Features' by Sutherland and Schneider (2015).
    """

    compute_engine: BasisFunctionComputation
    base_kernel: StationaryKernel
    num_basis_fns: int
    frequencies: tp.Union[Float[Array, "M D"], None]

    def __init__(
        self,
        base_kernel: AbstractKernel,
        num_basis_fns: int = 50,
        frequencies: tp.Union[Float[Array, "M D"], None] = None,
        compute_engine: BasisFunctionComputation = BasisFunctionComputation(),
        key: KeyArray = jr.key(0),
    ):
        r"""Initialise the RFF kernel.

        Args:
            base_kernel (StationaryKernel): The base kernel to be approximated.
            num_basis_fns (int): The number of basis functions to use in the approximation.
            frequencies (Float[Array, "M D"] | None): The frequencies to use in the approximation.
                If None, the frequencies are sampled from the spectral density of the base
                kernel.
            compute_engine (BasisFunctionComputation): The computation engine to use for
                the basis function computation.
            key (KeyArray): The random key to use for sampling the frequencies.
        """
        self._check_valid_base_kernel(base_kernel)

        if frequencies is None:
            n_dims = base_kernel.n_dims
            if n_dims is None:
                raise ValueError(
                    "Expected the number of dimensions to be specified for the base kernel. "
                    "Please specify the n_dims argument for the base kernel."
                )
            # The spectral measure is a D-dimensional distribution already
            # carrying the lengthscale as its inverse scale, so the draws are
            # the frequencies ω themselves — no further rescaling downstream.
            frequencies = base_kernel.spectral_density.sample(
                key, (num_basis_fns,)
            )  # [M, D]

        self.base_kernel = base_kernel
        self.num_basis_fns = num_basis_fns
        self.frequencies = frequencies

        super().__init__(compute_engine=compute_engine)

    def __call__(self, x: Float[Array, "D 1"], y: Float[Array, "D 1"]) -> None:
        """Superfluous for RFFs."""
        raise RuntimeError("RFFs do not have a kernel function.")

    @staticmethod
    def _check_valid_base_kernel(kernel: AbstractKernel):
        r"""Verify that the base kernel is valid for RFF approximation.

        Args:
            kernel (AbstractKernel): The kernel to be checked.
        """
        if not isinstance(kernel, StationaryKernel):
            raise TypeError(
                "RFF needs a stationary kernel with a spectral density, but got "
                f"{type(kernel).__name__}. Pathwise sampling (`sample_approx`) "
                "uses RFF, so it does not support this kernel. Sample from the "
                "predictive distribution instead."
            )

        # check that the kernel has a spectral density
        _ = kernel.spectral_density

    def compute_features(self, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            x: A $N \times D$ array of inputs.

        Returns:
            Float[Array, "N L"]: A $N \times L$ array of features where $L = 2M$.
        """
        return self.compute_engine.compute_features(self, x)

from gpjax.kernels.approximations import RFF
from gpjax.kernels.nonstationary import (
    Linear,
    Polynomial,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
    StationaryKernel,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro.distributions as npd
import pytest

config.update("jax_enable_x64", True)
_jitter = 1e-6


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10])
@pytest.mark.parametrize("n_dims", [1, 3])
def test_frequency_sampler(
    kernel: type[StationaryKernel], num_basis_fns: int, n_dims: int
):
    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel=base_kernel, num_basis_fns=num_basis_fns)
    assert approximate.frequencies.shape == (num_basis_fns, n_dims)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10])
@pytest.mark.parametrize("n_dims", [1, 3])
@pytest.mark.parametrize("n_data", [50, 100])
def test_gram(
    kernel: type[StationaryKernel], num_basis_fns: int, n_dims: int, n_data: int
):
    key = jr.key(123)
    x = jr.uniform(key, shape=(n_data, 1), minval=-3.0, maxval=3.0).reshape(-1, 1)
    if n_dims > 1:
        x = jnp.hstack([x] * n_dims)
    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel=base_kernel, num_basis_fns=num_basis_fns)

    linop = approximate.gram(x)

    # Check the return type
    assert isinstance(linop, lx.AbstractLinearOperator)

    Kxx = linop.as_matrix() + jnp.eye(n_data) * _jitter

    # Check that the shape is correct
    assert Kxx.shape == (n_data, n_data)

    # Check that the Gram matrix is PSD
    evals, _ = jnp.linalg.eigh(Kxx)
    assert jnp.all(evals > 0)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10])
@pytest.mark.parametrize("n_dims", [1, 3])
@pytest.mark.parametrize("n_datas", [(50, 100), (100, 50)])
def test_cross_covariance(
    kernel: type[StationaryKernel],
    num_basis_fns: int,
    n_dims: int,
    n_datas: tuple[int, int],
):
    nd1, nd2 = n_datas
    key = jr.key(123)
    x1 = jr.uniform(key, shape=(nd1, 1), minval=-3.0, maxval=3.0)
    if n_dims > 1:
        x1 = jnp.hstack([x1] * n_dims)
    x2 = jr.uniform(key, shape=(nd2, 1), minval=-3.0, maxval=3.0)
    if n_dims > 1:
        x2 = jnp.hstack([x2] * n_dims)

    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel=base_kernel, num_basis_fns=num_basis_fns)
    Kxx = approximate.cross_covariance(x1, x2)

    # Check the return type
    assert isinstance(Kxx, jax.Array)

    # Check that the shape is correct
    assert Kxx.shape == (nd1, nd2)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("n_dim", [1, 3])
def test_improvement(kernel: type[StationaryKernel], n_dim: int):
    n_data = 100
    key = jr.key(123)

    x = jr.uniform(key, minval=-3.0, maxval=3.0, shape=(n_data, n_dim))
    base_kernel = kernel(active_dims=list(range(n_dim)))
    exact_linop = base_kernel.gram(x).as_matrix()

    crude_approximation = RFF(base_kernel=base_kernel, num_basis_fns=10)
    c_linop = crude_approximation.gram(x).as_matrix()

    better_approximation = RFF(base_kernel=base_kernel, num_basis_fns=100)
    b_linop = better_approximation.gram(x).as_matrix()

    c_delta = jnp.linalg.norm(exact_linop - c_linop, ord="fro")
    b_delta = jnp.linalg.norm(exact_linop - b_linop, ord="fro")

    # The frobenius norm of the difference between the exact and approximate
    # should improve as we increase the number of basis functions
    assert c_delta > b_delta


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_exactness(kernel: type[StationaryKernel]):
    kernel = kernel(n_dims=1)

    n_data = 100
    key = jr.key(123)

    x = jr.uniform(key, minval=-3.0, maxval=3.0, shape=(n_data, 1))
    exact_linop = kernel.gram(x).as_matrix()

    better_approximation = RFF(base_kernel=kernel, num_basis_fns=300)
    b_linop = better_approximation.gram(x).as_matrix()

    max_delta = jnp.max(exact_linop - b_linop)
    assert max_delta < 0.1


@pytest.mark.parametrize("kernel", [Polynomial, Linear])
def test_nonstationary_raises_error(kernel):
    with pytest.raises(TypeError):
        RFF(base_kernel=kernel(1), num_basis_fns=10)


@pytest.mark.parametrize(
    "kernel",
    [RationalQuadratic, PoweredExponential, Periodic],
)
def test_missing_spectral_density_raises_error(kernel):
    with pytest.raises(NotImplementedError):
        RFF(base_kernel=kernel(), num_basis_fns=10)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_stochastic_init(kernel: type[StationaryKernel]):
    with pytest.raises(ValueError, match="Please specify the n_dims argument"):
        # n_dims is not specified, but should be
        RFF(base_kernel=kernel(), num_basis_fns=10)

    kernel = kernel(n_dims=1)

    k1 = RFF(base_kernel=kernel, num_basis_fns=10, key=jr.key(123))
    k2 = RFF(base_kernel=kernel, num_basis_fns=10, key=jr.key(42))

    assert (k1.frequencies != k2.frequencies).any()


@pytest.mark.parametrize("n_dims", [2, 3])
def test_matern_rff_gram_matches_isotropic(n_dims):
    """In d>1 the Matérn RFF Gram must converge to the exact isotropic Matérn
    Gram. The tensor-product bug (issue #2) plateaus at relF ~0.05 (d=2) /
    ~0.12 (d=3) and never improves; the fix reaches <0.02 by M~=20k."""
    key = jr.key(123)
    n_data = 25
    x = jr.uniform(key, shape=(n_data, n_dims), minval=-2.0, maxval=2.0)

    base_kernel = Matern32(active_dims=list(range(n_dims)))
    approx = RFF(base_kernel=base_kernel, num_basis_fns=20000, key=jr.key(0))

    exact_gram = base_kernel.gram(x).as_matrix()
    approx_gram = approx.gram(x).as_matrix()

    rel_frobenius = jnp.linalg.norm(approx_gram - exact_gram) / jnp.linalg.norm(
        exact_gram
    )
    assert rel_frobenius < 0.02


@pytest.mark.parametrize("n_dims", [1, 3])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_rff_effective_frequencies_unchanged(n_dims: int, lengthscale: float):
    """Folding the lengthscale into the spectral measure is bit-identical.

    Previously the measure was standardised and `compute_features` divided the
    draws by ℓ. Now the measure carries diag(ℓ)⁻¹ directly and no rescaling
    happens downstream. Both routes must give the same effective frequencies.
    """
    base_kernel = Matern32(n_dims=n_dims, lengthscale=lengthscale)
    approx = RFF(base_kernel=base_kernel, num_basis_fns=64, key=jr.key(5))

    # Reference: the pre-fix route — standardised draw, then divide by ℓ.
    standardised = npd.MultivariateStudentT(
        df=3, loc=jnp.zeros(n_dims), scale_tril=jnp.eye(n_dims)
    )
    reference = standardised.sample(jr.key(5), (64,)) / lengthscale

    assert jnp.array_equal(approx.frequencies, reference)

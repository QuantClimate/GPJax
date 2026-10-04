# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from itertools import product
from typing import Any

import equinox as eqx
from gpjax.dataset import Dataset
from gpjax.fit import fit
from gpjax.gps import Prior
from gpjax.kernels import location_functions
from gpjax.kernels.approximations import RFF
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.kernels.nonstationary import (
    ArcCosine,
    Gibbs,
    Linear,
    Polynomial,
    VaryingAmplitude,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
    White,
)
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Zero
from gpjax.objectives import conjugate_mll
from gpjax.parameters import NonNegativeReal, val
from gpjax.summary import _collect
from hypothesis import (
    given,
    strategies as st,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import optax as ox
from paramax import AbstractUnwrappable
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def params_product(params: dict[str, list]) -> list[dict[str, Any]]:
    return [
        dict(zip(params.keys(), values, strict=False))
        for values in product(*params.values())
    ]


TESTED_KERNELS = [
    (
        ArcCosine,
        params_product(
            {
                "order": [0, 1, 2],
                "weight_variance": [0.1, 1.0],
                "bias_variance": [0.1, 1.0],
            }
        ),
    ),
    (Linear, [{}]),
    (Polynomial, params_product({"degree": [1, 2, 3], "shift": [1e-6, 0.1, 1.0]})),
]

VARIANCES = [0.1]


@pytest.fixture
def kernel_request(
    kernel,
    params,
    variance,
):
    return kernel, params, variance


# NOTE: no marks here. `kernel`, `params` and `variance` are supplied by the
# parametrize marks on each consuming test. Marking a fixture is a no-op and is
# a hard error from pytest 9.1.
@pytest.fixture
def test_init(kernel_request):
    kernel, params, variance = kernel_request
    return kernel(**params, variance=variance)


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_override_paramtype(kernel_request):
    kernel, params, variance = kernel_request

    new_params = {}  # otherwise we change the fixture and next test fails
    for param, value in params.items():
        if param in ("degree", "order"):
            continue
        new_params[param] = value

    k = kernel(**new_params, variance=NonNegativeReal(variance))
    assert isinstance(k.variance, NonNegativeReal)

    for param in params:
        if param in ("degree", "order", "weight_variance", "bias_variance"):
            continue
        # Parameter is now a raw value, not a Static object
        assert not isinstance(getattr(k, param), AbstractUnwrappable)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
def test_init_defaults(kernel: type[AbstractKernel]):
    # Initialise kernel
    k = kernel()

    # Check that the parameters are set correctly
    assert isinstance(k.compute_engine, type(AbstractKernelComputation()))
    assert isinstance(k.variance, NonNegativeReal)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_variances(kernel: type[AbstractKernel], variance):
    # Initialise kernel
    k = kernel(variance=variance)

    # Check that the parameters are set correctly
    assert isinstance(k.variance, NonNegativeReal)
    assert jnp.allclose(k.variance.unwrap(), jnp.asarray(variance))


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
def test_gram(test_init: AbstractKernel, n: int):
    # kernel is initialized in the test_init fixture
    k = test_init
    n_dims = k.n_dims or 1

    # Inputs
    x = jnp.linspace(0.0, 1.0, n * n_dims).reshape(n, n_dims)

    # Test gram matrix
    Kxx = k.gram(x)
    assert isinstance(Kxx, lx.AbstractLinearOperator)
    assert Kxx.as_matrix().shape == (n, n)
    assert jnp.all(jnp.linalg.eigvalsh(Kxx.as_matrix() + jnp.eye(n) * 1e-6) > 0.0)


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
@pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
def test_cross_covariance(test_init: AbstractKernel, n_a: int, n_b: int):
    # kernel is initialized in the test_init fixture
    k = test_init
    n_dims = k.n_dims or 1

    # Inputs
    x = jnp.linspace(0.0, 1.0, n_a * n_dims).reshape(n_a, n_dims)
    y = jnp.linspace(0.0, 1.0, n_b * n_dims).reshape(n_b, n_dims)

    # Test cross covariance matrix
    Kxy = k.cross_covariance(x, y)
    assert isinstance(Kxy, jax.Array)
    assert Kxy.shape == (n_a, n_b)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_arccosine_special_case(order: int):
    """For certain values of weight variance (1.0) and bias variance (0.0), we can test
    our calculations using the Monte Carlo expansion of the arccosine kernel, e.g.
    see Eq. (1) of https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf.
    """
    kernel = ArcCosine(
        weight_variance=jnp.array([1.0, 1.0]), bias_variance=1e-25, order=order
    )

    # Inputs close(ish) together
    a = jnp.array([[0.0, 0.0]])
    b = jnp.array([[2.0, 2.0]])

    # calc cross-covariance exactly
    Kab_exact = kernel.cross_covariance(a, b)

    # calc cross-covariance using samples
    weights = jax.random.normal(jr.key(123), (10_000, 2))  # [S, d]
    weights_a = jnp.matmul(weights, a.T)  # [S, 1]
    weights_b = jnp.matmul(weights, b.T)  # [S, 1]
    H_a = jnp.heaviside(weights_a, 0.5)
    H_b = jnp.heaviside(weights_b, 0.5)
    integrands = H_a * H_b * (weights_a**order) * (weights_b**order)
    Kab_approx = 2.0 * jnp.mean(integrands)

    assert jnp.max(Kab_approx - Kab_exact) < 1e-4


def test_arccosine_variances_wrapped_and_trainable():
    """All three ArcCosine variances must be NonNegativeReal so they are
    trainable (not silently frozen) and constrained (cannot go negative)."""
    kernel = ArcCosine(
        order=1,
        n_dims=2,
        weight_variance=1.0,
        bias_variance=1.0,
        variance=1.0,
    )
    assert isinstance(kernel.weight_variance, NonNegativeReal)
    assert isinstance(kernel.bias_variance, NonNegativeReal)
    assert isinstance(kernel.variance, NonNegativeReal)

    # Every variance must land in the trainable (array) partition, not static.
    params, _ = eqx.partition(kernel, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    assert len(leaves) >= 3  # weight, bias, variance all present as arrays


def test_arccosine_variance_stays_positive_under_optimisation():
    """A gradient step that would push weight_variance negative must instead
    leave the constrained value positive and the Gram finite/PSD."""
    x = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    kernel = ArcCosine(order=1, n_dims=1, weight_variance=jnp.array(0.05))
    params, static = eqx.partition(kernel, eqx.is_array)

    def loss(p):
        k = eqx.combine(p, static)
        return jnp.sum(k.gram(x).as_matrix())  # gradient pushes variances down

    grads = jax.grad(loss)(params)
    stepped = jax.tree_util.tree_map(lambda leaf, g: leaf - 100.0 * g, params, grads)
    k_new = eqx.combine(stepped, static)
    assert val(k_new.weight_variance) > 0.0
    gram = k_new.gram(x).as_matrix()
    assert jnp.all(jnp.isfinite(gram))


# ---------------------------------------------------------------------------
# Location-function kernels: VaryingAmplitude and Gibbs.
#
# Inputs have two space columns (0, 1) and one covariate column (2).
# ---------------------------------------------------------------------------

INPUTS = jr.uniform(jr.key(0), (40, 3), minval=-2.0, maxval=2.0)
RADIAL_KERNELS = [
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    PoweredExponential,
]


def _min_eigenvalue(kernel: AbstractKernel) -> float:
    return float(jnp.linalg.eigvalsh(kernel.gram(INPUTS).as_matrix()).min())


def _covariate_fn(weight: float) -> location_functions.Linear:
    return location_functions.Linear(active_dims=[2], weights=[weight])


@pytest.mark.parametrize("base", RADIAL_KERNELS)
@pytest.mark.parametrize("wrapper", ["amplitude", "gibbs"])
def test_zero_location_function_gives_the_base_kernel(base, wrapper):
    base_kernel = base(active_dims=[0, 1], lengthscale=0.8, variance=1.7)
    fn = location_functions.Constant()
    kernel = (
        VaryingAmplitude(base_kernel, amplitude=fn)
        if wrapper == "amplitude"
        else Gibbs(base_kernel, lengthscale=fn)
    )
    assert jnp.allclose(
        kernel.gram(INPUTS).as_matrix(), base_kernel.gram(INPUTS).as_matrix()
    )


def test_varying_amplitude_scales_the_base_kernel():
    base_kernel = Matern32(active_dims=[0, 1])
    kernel = VaryingAmplitude(base_kernel, amplitude=_covariate_fn(0.6))
    sigma = jnp.exp(0.6 * INPUTS[:, 2])
    expected = sigma[:, None] * sigma[None, :] * base_kernel.gram(INPUTS).as_matrix()
    assert jnp.allclose(kernel.gram(INPUTS).as_matrix(), expected)


def test_varying_amplitude_diagonal_is_the_local_variance():
    base_kernel = RBF(active_dims=[0, 1], variance=2.0)
    kernel = VaryingAmplitude(base_kernel, amplitude=_covariate_fn(-0.4))
    diagonal = kernel.diagonal(INPUTS).as_matrix().diagonal()
    assert jnp.allclose(diagonal, 2.0 * jnp.exp(2 * -0.4 * INPUTS[:, 2]))


def test_varying_amplitude_accepts_a_nonstationary_base_kernel():
    kernel = VaryingAmplitude(Linear(active_dims=[0]), amplitude=_covariate_fn(0.3))
    assert _min_eigenvalue(kernel) > -1e-8


def test_varying_amplitude_rejects_rff():
    rff = RFF(base_kernel=RBF(n_dims=3), num_basis_fns=5)
    with pytest.raises(TypeError, match="one pair of points"):
        VaryingAmplitude(rff, amplitude=location_functions.Constant())


def test_wrappers_do_not_take_active_dims():
    with pytest.raises(TypeError):
        Gibbs(RBF(), lengthscale=location_functions.Constant(), active_dims=[0])


def _paciorek_schervish(x, y, ell, base_lengthscale, variance):
    """Direct Paciorek-Schervish Matern-3/2 with Sigma(x) = ell(x)^2 diag(l0^2)."""
    sigma_x = jnp.diag((ell(x) * base_lengthscale) ** 2)
    sigma_y = jnp.diag((ell(y) * base_lengthscale) ** 2)
    sigma = (sigma_x + sigma_y) / 2
    prefactor = (
        jnp.linalg.det(sigma_x) ** 0.25
        * jnp.linalg.det(sigma_y) ** 0.25
        / jnp.sqrt(jnp.linalg.det(sigma))
    )
    h = x[:2] - y[:2]
    r = jnp.sqrt(h @ jnp.linalg.solve(sigma, h) + 1e-36)
    return variance * prefactor * (1 + jnp.sqrt(3.0) * r) * jnp.exp(-jnp.sqrt(3.0) * r)


def test_gibbs_matches_the_paciorek_schervish_formula():
    base_lengthscale = jnp.array([0.7, 1.3])
    fn = _covariate_fn(0.9)
    kernel = Gibbs(
        Matern32(active_dims=[0, 1], lengthscale=base_lengthscale, variance=2.0),
        lengthscale=fn,
    )
    ell = lambda x: jnp.exp(fn(x))
    expected = jax.vmap(
        lambda a: jax.vmap(
            lambda b: _paciorek_schervish(a, b, ell, base_lengthscale, 2.0)
        )(INPUTS)
    )(INPUTS)
    assert jnp.allclose(kernel.gram(INPUTS).as_matrix(), expected)


@pytest.mark.parametrize("base", RADIAL_KERNELS)
def test_gibbs_keeps_the_base_variance(base):
    kernel = Gibbs(
        base(active_dims=[0, 1], variance=1.7), lengthscale=_covariate_fn(1.2)
    )
    diagonal = kernel.diagonal(INPUTS).as_matrix().diagonal()
    assert jnp.allclose(diagonal, 1.7)


def test_gibbs_correlation_is_shorter_where_the_lengthscale_is_smaller():
    kernel = Gibbs(Matern32(active_dims=[0, 1]), lengthscale=_covariate_fn(1.0))
    low = jnp.array([[0.0, 0.0, -1.0], [0.5, 0.0, -1.0]])
    high = low.at[:, 2].set(1.0)
    assert kernel(low[0], low[1]) < kernel(high[0], high[1])


@pytest.mark.parametrize(
    "base",
    [
        Periodic(),
        White(),
        RBF() + Matern32(),
        RBF() * Matern32(),
        Linear(),
    ],
    ids=["periodic", "white", "sum", "product", "linear"],
)
def test_gibbs_rejects_bases_that_are_not_isotropic_radial(base):
    with pytest.raises(TypeError, match="isotropic radial"):
        Gibbs(base, lengthscale=location_functions.Constant())


@given(
    weights=st.lists(st.floats(min_value=-1.5, max_value=1.5), min_size=2, max_size=2),
    base_index=st.integers(min_value=0, max_value=len(RADIAL_KERNELS) - 1),
)
def test_location_function_kernels_are_positive_definite(weights, base_index):
    base_kernel = RADIAL_KERNELS[base_index](active_dims=[0, 1], lengthscale=0.5)
    gibbs = Gibbs(base_kernel, lengthscale=_covariate_fn(weights[0]))
    kernel = VaryingAmplitude(gibbs, amplitude=_covariate_fn(weights[1]))
    gram = kernel.gram(INPUTS).as_matrix()
    assert jnp.allclose(gram, gram.T)
    assert _min_eigenvalue(kernel) > -1e-8 * jnp.max(jnp.diag(gram))


def test_location_function_kernels_fit_and_have_finite_gradients():
    y = jnp.sin(INPUTS[:, :1] * 2.0) * jnp.exp(0.5 * INPUTS[:, 2:3])
    data = Dataset(X=INPUTS, y=y)
    kernel = VaryingAmplitude(
        Gibbs(Matern52(active_dims=[0, 1]), lengthscale=_covariate_fn(0.0)),
        amplitude=_covariate_fn(0.0),
    )
    model = Prior(mean_function=Zero(), kernel=kernel) * Gaussian()

    objective = lambda m, d: -conjugate_mll(m, d)
    grads = jax.jit(jax.grad(objective))(model, data)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))

    fitted, history = fit(
        model=model,
        objective=objective,
        train_data=data,
        optim=ox.adam(0.05),
        num_iters=30,
        verbose=False,
    )
    assert history[-1] < history[0]
    weight = val(fitted.prior.kernel.amplitude.weights)
    assert not jnp.allclose(weight, 0.0)


def test_summary_shows_location_function_parameters():
    kernel = Gibbs(Matern32(active_dims=[0, 1]), lengthscale=_covariate_fn(0.1))
    names = {r.name for r in _collect(Prior(mean_function=Zero(), kernel=kernel))}
    assert "kernel.lengthscale.weights" in names
    assert "kernel.base_kernel.lengthscale" in names


@pytest.mark.parametrize("wrapper", ["amplitude", "gibbs"])
def test_rff_names_the_kernel_it_cannot_approximate(wrapper):
    fn = location_functions.Constant()
    kernel = (
        VaryingAmplitude(RBF(), amplitude=fn)
        if wrapper == "amplitude"
        else Gibbs(RBF(), lengthscale=fn)
    )
    with pytest.raises(TypeError, match=f"{type(kernel).__name__}.*sample_approx"):
        RFF(base_kernel=kernel)

# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

from collections.abc import Callable

import equinox as eqx
import gpjax as gpx
import gpjax.distributions
from gpjax.gps import JointModel
import gpjax.linalg.utils
from gpjax.parameters import (
    LowerTriangular,
    Real,
    val,
)
import gpjax.variational_families
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    DualVariationalGaussian,
    GraphVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import (
    Array,
    Float,
)
import lineax as lx
import networkx as nx
import numpy as np
import numpyro.distributions as npd
from numpyro.distributions import Distribution as NumpyroDistribution
import pytest

from tests._categorical_helpers import (
    whitened_family,
    zero_mean_rbf_prior,
)
from tests._dual_helpers import (
    DUAL_JITTER,
    build_dual,
    matched_variational_gaussian as _matched_variational_gaussian,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning"
)


def test_abstract_variational_family():
    # Test that the abstract class cannot be instantiated.
    with pytest.raises(TypeError):
        AbstractVariationalFamily()

    # Create a dummy variational family class with abstract methods implemented.
    class DummyPosterior:
        @property
        def __class__(self) -> type:
            return JointModel

    class DummyVariationalFamily(AbstractVariationalFamily):
        def predict(self, x: Float[Array, "N D"]) -> npd.MultivariateNormal:
            return npd.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(x.shape[1]))

        def prior_kl(self) -> Float[Array, ""]:
            return jnp.array(0.0)

        def condition(self, train_data):
            return None

    # A family that forgets `prior_kl` must not be instantiable: every
    # ELBO-style objective subtracts it, so the base class declares it.
    class KLFreeVariationalFamily(AbstractVariationalFamily):
        def predict(self, x: Float[Array, "N D"]) -> npd.MultivariateNormal:
            return npd.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(x.shape[1]))

        def condition(self, train_data):
            return None

    with pytest.raises(TypeError):
        KLFreeVariationalFamily(model=DummyPosterior())

    # `condition` is part of the same contract: `q | D` must work for every
    # family, so a class that omits it is not a variational family either.
    class ConditionFreeVariationalFamily(AbstractVariationalFamily):
        def predict(self, x: Float[Array, "N D"]) -> npd.MultivariateNormal:
            return npd.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(x.shape[1]))

        def prior_kl(self) -> Float[Array, ""]:
            return jnp.array(0.0)

    with pytest.raises(TypeError):
        ConditionFreeVariationalFamily(model=DummyPosterior())

    # Test that the dummy variational family can be instantiated.
    dummy_variational_family = DummyVariationalFamily(model=DummyPosterior())
    assert isinstance(dummy_variational_family, AbstractVariationalFamily)


# Functions to test variational family parameter shapes upon initialisation.
def vector_shape(n_inducing: int) -> tuple[int, int]:
    """Shape of a vector with n_inducing rows and 1 column."""
    return (n_inducing, 1)


def matrix_shape(n_inducing: int) -> tuple[int, int]:
    """Shape of a matrix with n_inducing rows and 1 column."""
    return (n_inducing, n_inducing)


# Functions to test variational parameter values upon initialisation.
def vector_val(val: float) -> Callable[[int], Float[Array, "n_inducing 1"]]:
    """Vector of shape (n_inducing, 1) filled with val."""

    def vector_val_fn(n_inducing: int):
        return val * jnp.ones(vector_shape(n_inducing))

    return vector_val_fn


def diag_matrix_val(
    val: float,
) -> Callable[[int], Float[Array, "n_inducing n_inducing"]]:
    """Diagonal matrix of shape (n_inducing, n_inducing) filled with val."""

    def diag_matrix_fn(n_inducing: int) -> Float[Array, "n_inducing n_inducing"]:
        return jnp.eye(n_inducing) * val

    return diag_matrix_fn


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize(
    "variational_family",
    [
        VariationalGaussian,
        WhitenedVariationalGaussian,
        DualVariationalGaussian,
    ],
)
def test_variational_gaussians(
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
) -> None:
    # Initialise variational family:
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )
    likelihood = gpx.likelihoods.Gaussian()
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)

    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)

    posterior = prior * likelihood
    q = variational_family(model=posterior, inducing_inputs=inducing_inputs)

    # Test init:
    assert q.num_inducing == n_inducing
    assert isinstance(q, AbstractVariationalFamily)

    if isinstance(q, DualVariationalGaussian):
        # The dual family stores sites, not moments, and both default to zero so that
        # q(u) = p(u) at initialisation.
        assert val(q.dual_vector).shape == vector_shape(n_inducing)
        assert val(q.dual_matrix).shape == matrix_shape(n_inducing)
        assert (val(q.dual_vector) == 0.0).all()
        assert (val(q.dual_matrix) == 0.0).all()
    else:
        assert q.variational_mean.unwrap().shape == vector_shape(n_inducing)
        assert q.variational_root_covariance.unwrap().shape == matrix_shape(n_inducing)
        assert (q.variational_mean.unwrap() == vector_val(0.0)(n_inducing)).all()
        assert (
            q.variational_root_covariance.unwrap() == diag_matrix_val(1.0)(n_inducing)
        ).all()

    # Test KL
    kl = q.prior_kl()
    assert isinstance(kl, jnp.ndarray)
    assert kl.shape == ()
    # The dual family initialises exactly at q(u) = p(u), where the KL is zero up to
    # the round-off of tr(R^{-1} Kzz) - M; the moment families start strictly inside.
    assert kl >= (-1e-10 if isinstance(q, DualVariationalGaussian) else 0.0)

    # Test predictions
    predictive_dist = q(test_inputs)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize(
    "removed_name",
    ["NaturalVariationalGaussian", "ExpectationVariationalGaussian"],
)
def test_removed_families_are_gone(removed_name: str) -> None:
    """The natural/expectation parameterisations were superseded by `fit_natgrads`.

    They were parameterisation-only classes with no optimiser attached; natural-gradient
    steps are now taken directly on `VariationalGaussian` and
    `WhitenedVariationalGaussian`. This guards against them creeping back in.
    """
    assert not hasattr(gpjax.variational_families, removed_name)
    assert removed_name not in gpjax.variational_families.__all__


def test_psd_helper_is_gone() -> None:
    """`_psd`'s only callers lived inside the two deleted classes.

    Checked separately from the class names because `_psd` was private and never
    exported, so the `__all__` assertion above would be vacuous for it. The guard is
    against the dead helper returning alongside the classes, not a reservation of the
    name for all time.
    """
    assert not hasattr(gpjax.variational_families, "_psd")


@pytest.mark.parametrize("n_test", [10, 20])
@pytest.mark.parametrize("n_inducing", [10, 20])
@pytest.mark.parametrize(
    "variational_family",
    [
        GraphVariationalGaussian,
    ],
)
def test_graph_variational_gaussian(
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
) -> None:
    G = nx.barbell_graph(100, 0)
    L = nx.laplacian_matrix(G).toarray()

    kernel = gpx.kernels.GraphKernel(
        laplacian=L,
        lengthscale=2.3,
        variance=3.2,
        smoothness=6.1,
    )
    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Bernoulli()

    inducing_inputs = jnp.array(
        np.random.randint(low=1, high=100, size=(n_inducing, 1))
    ).astype(jnp.int64)

    test_inputs = jnp.array(np.random.randint(low=0, high=1, size=(n_test, 1))).astype(
        jnp.int64
    )

    posterior = prior * likelihood
    q = variational_family(model=posterior, inducing_inputs=inducing_inputs)
    # Test KL
    kl = q.prior_kl()
    assert isinstance(kl, jnp.ndarray)
    assert kl.shape == ()
    assert kl >= 0.0

    # Test predictions
    predictive_dist = q(test_inputs)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_datapoints", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize("point_dim", [1, 2])
def test_collapsed_variational_gaussian(
    n_test: int, n_inducing: int, n_datapoints: int, point_dim: int
) -> None:
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.key(123), shape=x.shape) * 0.1
    x = jnp.hstack([x] * point_dim)
    D = gpx.Dataset(X=x, y=y)

    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)
    test_inputs = jnp.hstack([test_inputs] * point_dim)

    posterior = prior * gpx.likelihoods.Gaussian()

    variational_family = CollapsedVariationalGaussian(
        model=posterior,
        inducing_inputs=inducing_inputs,
    )

    # We should raise an error for non-Gaussian likelihoods:
    with pytest.raises(TypeError):
        CollapsedVariationalGaussian(
            model=prior * gpx.likelihoods.Bernoulli(),
            inducing_inputs=inducing_inputs,
        )

    # Test init
    assert variational_family.num_inducing == n_inducing
    assert (variational_family.inducing_inputs.unwrap() == inducing_inputs).all()
    assert variational_family.model.likelihood.obs_stddev.unwrap() == 1.0

    # Test predictions
    predictive_dist = variational_family(test_inputs, D)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


# ---------------------------------------------------------------------------
# Closed-form prior KL (issue #665)
#
# `WhitenedVariationalGaussian.prior_kl` and `VariationalGaussian.prior_kl`
# used to densify `S = sqrt sqrt^T` and hand it to the generic Gaussian KL,
# which then re-factorised a matrix whose Cholesky factor is already stored in
# `variational_root_covariance`. The tests below pin the numerical values and
# gradients of the old implementation and assert the number of Cholesky
# factorisations the new implementation is allowed to perform.
# ---------------------------------------------------------------------------


def _kl_variational_mean(num_inducing: int) -> Float[Array, "N 1"]:
    """Deterministic, non-trivial variational mean."""
    index = jnp.arange(num_inducing, dtype=jnp.float64)
    return (0.3 * jnp.cos(1.7 * index) - 0.15 * index).reshape(-1, 1)


def _kl_variational_root(num_inducing: int) -> Float[Array, "N N"]:
    """Deterministic, non-trivial lower-triangular root with positive diagonal."""
    row = jnp.arange(num_inducing, dtype=jnp.float64)[:, None]
    col = jnp.arange(num_inducing, dtype=jnp.float64)[None, :]
    off_diagonal = 0.2 * jnp.sin(0.9 * row + 1.3 * col)
    diagonal = 0.5 + 0.4 * jnp.cos(0.6 * jnp.arange(num_inducing, dtype=jnp.float64))
    return jnp.tril(off_diagonal, -1) + jnp.diag(diagonal)


def _build_kl_family(
    family: type[VariationalGaussian], num_inducing: int
) -> VariationalGaussian:
    """Build a fully deterministic variational family for KL regression tests."""
    kernel = gpx.kernels.RBF(lengthscale=jnp.array(0.7), variance=jnp.array(1.3))
    mean_function = gpx.mean_functions.Constant(jnp.array([0.4]))
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function)
    posterior = prior * gpx.likelihoods.Gaussian()
    inducing_inputs = jnp.linspace(-3.0, 3.0, num_inducing).reshape(-1, 1)
    return family(
        model=posterior,
        inducing_inputs=inducing_inputs,
        variational_mean=_kl_variational_mean(num_inducing),
        variational_root_covariance=_kl_variational_root(num_inducing),
    )


# KL values recorded from the pre-#665 implementation (densify + generic KL),
# in float64. These are hard-coded literals so that the test is a genuine
# regression guard rather than a comparison against the code under test.
_RECORDED_PRIOR_KL = {
    ("WhitenedVariationalGaussian", 1): 0.055360515657826292,
    ("WhitenedVariationalGaussian", 3): 0.4558012885085363,
    ("WhitenedVariationalGaussian", 5): 2.2228894799193393,
    ("WhitenedVariationalGaussian", 12): 12.540706158684808,
    ("VariationalGaussian", 1): 0.051927405288060224,
    ("VariationalGaussian", 3): 0.89833938800292357,
    ("VariationalGaussian", 5): 3.0465143546546192,
    ("VariationalGaussian", 12): 35.666594348171607,
}

# Gradients recorded from the pre-#665 implementation at ``num_inducing=3``,
# taken with respect to the *unconstrained* parameters that `fit` optimises.
_RECORDED_PRIOR_KL_GRADS = {
    "WhitenedVariationalGaussian": {
        "variational_mean": np.array(
            [[0.3], [-0.18865334828865737], [-0.5900394577738384]]
        ),
        "variational_root_covariance": np.array(
            [
                0.15666538192549667,
                0.19476952617563908,
                0.00831613248665811,
                -0.12527973849920676,
                -0.21121593177991954,
                -0.430429665953372,
            ]
        ),
        "inducing_inputs": np.zeros((3, 1)),
    },
    "VariationalGaussian": {
        "variational_mean": np.array(
            [[-0.07687652189052713], [-0.452723814014418], [-0.7615217319894719]]
        ),
        "variational_root_covariance": np.array(
            [
                0.12042525327023475,
                0.14981022922180554,
                0.00633143799283199,
                -0.24853831088039383,
                -0.31926350826875094,
                -0.5011713127475177,
            ]
        ),
        "inducing_inputs": np.array(
            [
                [-9.6628829749081168e-05],
                [-2.0328302385444184e-04],
                [2.9991185360352303e-04],
            ]
        ),
    },
}


def _textbook_gaussian_kl(mean_q, cov_q, mean_p, cov_p):
    """KL[q || p] for two multivariate Gaussians, written from the textbook.

    Deliberately independent of any GPJax linear-algebra helper so that this
    reference cannot drift with changes to ``gpjax.distributions``.
    """
    dim = mean_q.shape[0]
    trace = jnp.trace(jnp.linalg.solve(cov_p, cov_q))
    diff = mean_p - mean_q
    mahalanobis = diff @ jnp.linalg.solve(cov_p, diff)
    _, log_det_p = jnp.linalg.slogdet(cov_p)
    _, log_det_q = jnp.linalg.slogdet(cov_q)
    return 0.5 * (trace + mahalanobis - dim + log_det_p - log_det_q)


def _reference_prior_kl(family):
    """Reference prior KL built from the dense covariance matrices."""
    variational_mean = val(family.variational_mean).reshape(-1)
    variational_sqrt = val(family.variational_root_covariance)
    cov_q = variational_sqrt @ variational_sqrt.T
    num_inducing = variational_sqrt.shape[-1]

    if isinstance(family, WhitenedVariationalGaussian):
        mean_p = jnp.zeros_like(variational_mean)
        cov_p = jnp.eye(num_inducing, dtype=variational_sqrt.dtype)
    else:
        inducing_inputs = val(family.inducing_inputs)
        mean_p = family.model.prior.mean_function(inducing_inputs).reshape(-1)
        cov_p = family.model.prior.kernel.gram(inducing_inputs).as_matrix()
        cov_p = (
            cov_p + jnp.eye(num_inducing, dtype=cov_p.dtype) * family.model.prior.jitter
        )

    return _textbook_gaussian_kl(variational_mean, cov_q, mean_p, cov_p)


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("num_inducing", [1, 3, 5, 12])
def test_prior_kl_matches_recorded_values(family, num_inducing):
    """The closed form must reproduce the pre-#665 KL values."""
    q = _build_kl_family(family, num_inducing)
    expected = _RECORDED_PRIOR_KL[(family.__name__, num_inducing)]
    np.testing.assert_allclose(np.float64(q.prior_kl()), expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("num_inducing", [1, 2, 3, 5, 12])
def test_prior_kl_matches_textbook_reference(family, num_inducing):
    """The closed form must agree with an independent dense-matrix reference."""
    q = _build_kl_family(family, num_inducing)
    np.testing.assert_allclose(
        np.float64(q.prior_kl()),
        np.float64(_reference_prior_kl(q)),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
def test_prior_kl_gradients_match_recorded_values(family):
    """Gradients must match those of the pre-#665 implementation."""
    q = _build_kl_family(family, 3)
    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    recorded = _RECORDED_PRIOR_KL_GRADS[family.__name__]

    np.testing.assert_allclose(
        np.asarray(grads.variational_mean.value),
        recorded["variational_mean"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grads.variational_root_covariance._flat),
        recorded["variational_root_covariance"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grads.inducing_inputs.value),
        recorded["inducing_inputs"],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("num_inducing", [2, 5, 12])
def test_prior_kl_gradients_match_textbook_reference(family, num_inducing):
    """Gradients must agree with an independent dense-matrix reference."""
    q = _build_kl_family(family, num_inducing)
    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    reference_grads = eqx.filter_grad(_reference_prior_kl)(q)

    np.testing.assert_allclose(
        np.asarray(grads.variational_mean.value),
        np.asarray(reference_grads.variational_mean.value),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(grads.variational_root_covariance._flat),
        np.asarray(reference_grads.variational_root_covariance._flat),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(grads.inducing_inputs.value),
        np.asarray(reference_grads.inducing_inputs.value),
        rtol=1e-9,
        atol=1e-11,
    )


def _count_cholesky_calls(monkeypatch, thunk) -> dict[str, int]:
    """Count Cholesky factorisations performed while evaluating ``thunk``."""
    counts = {"cholesky_factor": 0, "dense_cholesky": 0}
    original_cholesky_factor = gpjax.linalg.utils.cholesky_factor
    original_dense_cholesky = jnp.linalg.cholesky

    def counting_cholesky_factor(operator):
        counts["cholesky_factor"] += 1
        return original_cholesky_factor(operator)

    def counting_dense_cholesky(matrix):
        counts["dense_cholesky"] += 1
        return original_dense_cholesky(matrix)

    # ``gpjax.variational_families`` is deliberately absent: its only
    # ``cholesky_factor`` call site lived in the removed
    # ``ExpectationVariationalGaussian``, so the module no longer imports the name.
    for module in (gpjax.linalg.utils, gpjax.distributions):
        monkeypatch.setattr(module, "cholesky_factor", counting_cholesky_factor)
    monkeypatch.setattr(jnp.linalg, "cholesky", counting_dense_cholesky)

    thunk()
    return counts


@pytest.mark.parametrize(
    ("family", "expected_dense_cholesky"),
    [(WhitenedVariationalGaussian, 0), (VariationalGaussian, 1)],
    ids=["WhitenedVariationalGaussian", "VariationalGaussian"],
)
def test_prior_kl_factorisation_count(monkeypatch, family, expected_dense_cholesky):
    """The KL must not re-factorise a matrix whose root it already holds.

    Before #665 the counts were two dense Choleskys for the whitened family and
    four for ``VariationalGaussian``. The whitened KL needs none;
    ``VariationalGaussian`` needs exactly one, of ``Kzz``.
    """
    q = _build_kl_family(family, 4)
    counts = _count_cholesky_calls(monkeypatch, q.prior_kl)

    assert counts["dense_cholesky"] == expected_dense_cholesky
    assert counts["cholesky_factor"] == 0


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
def test_prior_kl_is_jit_grad_and_vmap_compatible(family):
    q = _build_kl_family(family, 4)
    eager = q.prior_kl()

    jitted = eqx.filter_jit(lambda model: model.prior_kl())
    np.testing.assert_allclose(np.float64(jitted(q)), np.float64(eager), rtol=1e-12)

    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    assert jnp.all(jnp.isfinite(grads.variational_mean.value))
    assert jnp.all(jnp.isfinite(grads.variational_root_covariance._flat))

    def kl_from_mean(mean_value):
        model = eqx.tree_at(lambda t: t.variational_mean.value, q, mean_value)
        return model.prior_kl()

    zero_mean = jnp.zeros_like(val(q.variational_mean))
    batch = jnp.stack(
        [val(q.variational_mean), val(q.variational_mean) + 0.5, zero_mean]
    )
    batched = jax.vmap(kl_from_mean)(batch)

    assert batched.shape == (3,)
    np.testing.assert_allclose(
        np.float64(batched[0]), np.float64(eager), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.float64(batched[2]),
        np.float64(kl_from_mean(zero_mean)),
        rtol=1e-12,
        atol=1e-14,
    )


def test_whitened_prior_kl_is_zero_when_q_equals_the_prior():
    """KL[N(0, I) || N(0, I)] must be zero."""
    q = _build_kl_family(WhitenedVariationalGaussian, 6)
    q = eqx.tree_at(
        lambda t: (t.variational_mean, t.variational_root_covariance),
        q,
        (Real(jnp.zeros((6, 1))), LowerTriangular(jnp.eye(6))),
    )
    np.testing.assert_allclose(np.float64(q.prior_kl()), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "family",
    [WhitenedVariationalGaussian, VariationalGaussian],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prior_kl_is_non_negative(family, seed):
    num_inducing = 6
    key_mean, key_root = jr.split(jr.key(seed))
    root = jnp.tril(jr.normal(key_root, (num_inducing, num_inducing)) * 0.2, -1)
    root = root + jnp.diag(0.3 + jnp.abs(jr.normal(key_root, (num_inducing,))))

    q = _build_kl_family(family, num_inducing)
    q = eqx.tree_at(
        lambda t: (t.variational_mean, t.variational_root_covariance),
        q,
        (Real(jr.normal(key_mean, (num_inducing, 1))), LowerTriangular(root)),
    )
    assert q.prior_kl() >= 0.0


# ---------------------------------------------------------------------------
# DualVariationalGaussian -- the t-SVGP site parameterisation (arXiv:2111.03412)
# ---------------------------------------------------------------------------
_DUAL_JITTER = DUAL_JITTER


def _build_dual_family(
    num_inducing: int,
    seed: int | None = None,
    jitter: float = _DUAL_JITTER,
    variance: float = 1.3,
) -> DualVariationalGaussian:
    """Build a dual family, optionally at random positive semi-definite sites."""
    kernel = gpx.kernels.RBF(
        lengthscale=jnp.array(0.7), variance=jnp.array(float(variance))
    )
    mean_function = gpx.mean_functions.Constant(jnp.array([0.4]))
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function, jitter=jitter)
    posterior = prior * gpx.likelihoods.Gaussian()
    inducing_inputs = jnp.linspace(-3.0, 3.0, num_inducing).reshape(-1, 1)

    return build_dual(posterior, inducing_inputs, seed=seed)


@pytest.mark.parametrize("num_inducing", [1, 4, 9])
def test_dual_prior_kl_zero_at_initialisation(num_inducing: int) -> None:
    """Zero sites mean q(u) = p(u), so R = Kzz and the KL vanishes."""
    q = _build_dual_family(num_inducing)
    np.testing.assert_allclose(np.float64(q.prior_kl()), 0.0, atol=1e-12)


@pytest.mark.parametrize("seed", list(range(10)))
def test_dual_prior_kl_non_negative(seed: int) -> None:
    q = _build_dual_family(6, seed=seed)
    assert q.prior_kl() >= -1e-12


@pytest.mark.parametrize("seed", list(range(10)))
def test_dual_prior_kl_matches_the_moment_family(seed: int) -> None:
    """The dual KL must equal the moment family's KL at the same q(u).

    Non-negativity alone rules out very little: a dropped term or a doubled
    trace stays non-negative. ``VariationalGaussian.prior_kl`` is pinned to a
    dense textbook reference by
    ``test_prior_kl_matches_textbook_reference``, so evaluating it at the
    dual family's implied moments turns the site parameterisation into a
    checked quantity rather than a merely finite one.
    """
    q_dual = _build_dual_family(6, seed=seed)
    q_moment = _matched_variational_gaussian(q_dual)
    np.testing.assert_allclose(
        np.float64(q_dual.prior_kl()),
        np.float64(q_moment.prior_kl()),
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize("n_test", [1, 7])
@pytest.mark.parametrize("seed", [0, 3])
def test_dual_predict_matches_variational_gaussian_at_matched_moments(
    n_test: int, seed: int
) -> None:
    """The dual predictive must agree with the moment family it is equivalent to."""
    q_dual = _build_dual_family(5, seed=seed)
    q_moment = _matched_variational_gaussian(q_dual)
    test_inputs = jnp.linspace(-3.0, 3.0, n_test).reshape(-1, 1)

    dual_dist = q_dual.predict(test_inputs)
    moment_dist = q_moment.predict(test_inputs)

    np.testing.assert_allclose(
        np.asarray(dual_dist.mean),
        np.asarray(moment_dist.mean),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(dual_dist.covariance()),
        np.asarray(moment_dist.covariance()),
        rtol=1e-9,
        atol=1e-10,
    )


def test_dual_prior_kl_is_jit_grad_and_vmap_compatible() -> None:
    q = _build_dual_family(4, seed=1)
    eager = q.prior_kl()

    jitted = eqx.filter_jit(lambda model: model.prior_kl())
    np.testing.assert_allclose(np.float64(jitted(q)), np.float64(eager), rtol=1e-12)

    grads = eqx.filter_grad(lambda model: model.prior_kl())(q)
    assert jnp.all(jnp.isfinite(grads.dual_vector.value))
    assert jnp.all(jnp.isfinite(grads.dual_matrix.value))

    def kl_from_vector(vector_value):
        model = eqx.tree_at(lambda t: t.dual_vector.value, q, vector_value)
        return model.prior_kl()

    zero_vector = jnp.zeros_like(val(q.dual_vector))
    batch = jnp.stack([val(q.dual_vector), val(q.dual_vector) + 0.5, zero_vector])
    batched = jax.vmap(kl_from_vector)(batch)

    assert batched.shape == (3,)
    np.testing.assert_allclose(
        np.float64(batched[0]), np.float64(eager), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.float64(batched[2]),
        np.float64(kl_from_vector(zero_vector)),
        rtol=1e-12,
        atol=1e-14,
    )


def test_dual_working_matrices_reconstruct_r() -> None:
    """``Lr`` is lower triangular and satisfies ``Lr Lr^T = Kzz + Kzz L2 Kzz``."""
    q = _build_dual_family(6, seed=5)
    gram, _, root_working = q._working_matrices()
    dual_matrix = val(q.dual_matrix)

    working = gram + gram @ dual_matrix @ gram
    np.testing.assert_allclose(
        np.asarray(root_working @ root_working.T), np.asarray(working), rtol=1e-10
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.triu(root_working, 1)), np.zeros((6, 6))
    )


@pytest.mark.parametrize(("variance", "num_inducing"), [(1e4, 80), (1e3, 50)])
def test_dual_working_matrices_survive_a_large_variance_kernel(
    variance: float, num_inducing: int
) -> None:
    """A badly scaled Kzz must not make chol(R) return NaN.

    Forming ``R = Kzz + Kzz L2 Kzz`` explicitly carries a rounding error of order
    ``||Kzz||^2 ||L2|| eps``, which for these settings dwarfs
    ``lambda_min(R) ~ jitter`` and made ``jnp.linalg.cholesky`` return NaN silently --
    poisoning every later ``fit_natgrads`` iterate. Factorising in the Kzz basis
    instead is unconditionally safe. The default 1e-6 jitter is used deliberately;
    the failure is invisible at the 1e-8 the other dual tests run at.
    """
    q = _build_dual_family(num_inducing, seed=6, jitter=1e-6, variance=float(variance))
    inputs = jnp.linspace(-3.0, 3.0, 9).reshape(-1, 1)

    _, _, root_working = q._working_matrices()
    assert jnp.all(jnp.isfinite(root_working))
    assert jnp.isfinite(q.prior_kl())
    assert jnp.all(jnp.isfinite(jnp.stack(q.marginals(inputs))))


def test_dual_family_cholesky_budget(monkeypatch) -> None:
    """Both routines factorise Kzz and R once each, and nothing else."""
    q = _build_dual_family(4, seed=2)
    inputs = jnp.linspace(-3.0, 3.0, 11).reshape(-1, 1)

    kl_counts = _count_cholesky_calls(monkeypatch, q.prior_kl)
    assert kl_counts["dense_cholesky"] == 2
    assert kl_counts["cholesky_factor"] == 0

    marginal_counts = _count_cholesky_calls(monkeypatch, lambda: q.marginals(inputs))
    assert marginal_counts["dense_cholesky"] == 2
    assert marginal_counts["cholesky_factor"] == 0


def test_dual_marginals_include_jitter() -> None:
    """``marginals`` must inflate every variance by exactly ``jitter``.

    ``VariationalGaussian.predict`` runs ``add_jitter`` on its output covariance, so
    the per-point variances ``elbo`` sees carry the same offset. Without it,
    ``dual_elbo`` misses ``elbo`` at matched moments by ``N * jitter / (2 sigma^2)``.
    """
    q = _build_dual_family(5, seed=4)
    inputs = jnp.linspace(-3.0, 3.0, 13).reshape(-1, 1)

    gram, root_gram, root_working = q._working_matrices()
    cross = q.model.prior.kernel.cross_covariance(val(q.inducing_inputs), inputs)
    diagonal = jnp.diag(q.model.prior.kernel.gram(inputs).as_matrix())
    prior_projection = jsp.linalg.solve_triangular(root_gram, cross, lower=True)
    site_projection = jsp.linalg.solve_triangular(root_working, cross, lower=True)
    analytic = (
        diagonal
        - jnp.sum(jnp.square(prior_projection), axis=0)
        + jnp.sum(jnp.square(site_projection), axis=0)
    )
    del gram

    _, variance = q.marginals(inputs)
    np.testing.assert_allclose(
        np.asarray(variance - analytic), q.model.prior.jitter, rtol=0.0, atol=1e-15
    )


# ---------------------------------------------------------------------------
# CollapsedPosterior.prior_kl -- oracle tests
#
# ``CollapsedPosterior.prior_kl`` evaluates the KL from Titsias' analytically
# optimal q*(u) to the prior p(u) through the factorisation the collapsed
# bound already holds: S* = Lz B^-1 Lz^T and a centred mean
# sigma^-1 Lz B^-1 A (y - m(x)). Asserting only that the result is finite
# would pass for a sign error, a dropped term or a factor of two, so the
# reference below is built along a route that shares no algebra with it:
# q*(u) is re-derived by Bayes' rule in the projected (DTC) linear model and
# handed to the same dense textbook Gaussian KL used by the uncollapsed
# families above.
# ---------------------------------------------------------------------------

_COLLAPSED_KERNELS = {
    "rbf": lambda: gpx.kernels.RBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.3)),
    "matern32": lambda: gpx.kernels.Matern32(
        lengthscale=jnp.array(0.6), variance=jnp.array(0.9)
    ),
    "matern12": lambda: gpx.kernels.Matern12(
        lengthscale=jnp.array(1.1), variance=jnp.array(2.0)
    ),
}

# Inducing configurations: a coarse grid, a finer grid whose points sit within
# a lengthscale of each other (so Kzz is genuinely correlated), a scattered
# two-dimensional set, and the M = 1 edge case where ``tr(B^-1) - M`` is a
# single scalar.
_COLLAPSED_INDUCING = {
    "grid-4-1d": lambda: jnp.linspace(-2.0, 2.0, 4).reshape(-1, 1),
    "grid-9-1d": lambda: jnp.linspace(-3.0, 3.0, 9).reshape(-1, 1),
    "scatter-6-2d": lambda: jr.uniform(
        jr.key(11), (6, 2), minval=-2.5, maxval=2.5, dtype=jnp.float64
    ),
    "single-1d": lambda: jnp.array([[0.3]]),
}


def _collapsed_train_data(num_dims: int, num_data: int = 17) -> gpx.Dataset:
    """Deterministic training data of the requested input dimension."""
    inputs = jr.uniform(
        jr.key(5), (num_data, num_dims), minval=-3.0, maxval=3.0, dtype=jnp.float64
    )
    outputs = jnp.sin(1.3 * inputs[:, :1]) + 0.15 * jr.normal(
        jr.key(7), (num_data, 1), dtype=jnp.float64
    )
    return gpx.Dataset(X=inputs, y=outputs)


def _build_collapsed_family(
    kernel_name: str, inducing_name: str, observation_noise: float
) -> tuple[CollapsedVariationalGaussian, gpx.Dataset]:
    """Build a collapsed family and matching data, fully deterministically."""
    inducing_inputs = _COLLAPSED_INDUCING[inducing_name]()
    prior = gpx.gps.Prior(
        kernel=_COLLAPSED_KERNELS[kernel_name](),
        mean_function=gpx.mean_functions.Constant(jnp.array(0.4)),
    )
    model = prior * gpx.likelihoods.Gaussian(
        obs_stddev=jnp.array(float(observation_noise))
    )
    family = CollapsedVariationalGaussian(model=model, inducing_inputs=inducing_inputs)
    return family, _collapsed_train_data(inducing_inputs.shape[-1])


def _titsias_optimal_q(family: CollapsedVariationalGaussian, train_data: gpx.Dataset):
    r"""Titsias' optimal :math:`q^{\star}(u)`, re-derived rather than quoted.

    The collapsed bound is maximised by the exact posterior over :math:`u` of
    the projected ("DTC") Gaussian linear model

    .. math::

        u \sim \mathcal{N}(m_z, K_{zz}), \qquad
        y \mid u \sim \mathcal{N}\big(m_x + W(u - m_z),\, \sigma^2 I\big),
        \qquad W = K_{xz}K_{zz}^{-1},

    so its moments follow from the textbook conjugate Gaussian update alone:

    .. math::

        S^{\star} = \big(K_{zz}^{-1} + \sigma^{-2}W^{\top}W\big)^{-1},
        \qquad
        \mu^{\star} = m_z + \sigma^{-2}S^{\star}W^{\top}(y - m_x).

    Neither :math:`L_z`, nor :math:`A`, nor :math:`B` appears: the whole
    factorisation the implementation is built on is bypassed, which is what
    makes the comparison evidence rather than a restatement. ``K_zz`` is read
    throughout as the stabilised ``K_zz + jitter I`` that conditioning
    actually uses, so the two routes describe the same prior.

    Args:
        family: The collapsed variational family.
        train_data: The data the optimal distribution is solved against.

    Returns:
        tuple: ``(mean_q, cov_q, mean_p, cov_p)`` as dense arrays, with the
        means flattened to one dimension.
    """
    model = family.model
    kernel = model.prior.kernel
    mean_function = model.prior.mean_function
    noise_variance = val(model.likelihood.obs_stddev) ** 2

    inducing_inputs = val(family.inducing_inputs)
    num_inducing = inducing_inputs.shape[0]
    identity = jnp.eye(num_inducing, dtype=inducing_inputs.dtype)

    prior_cov = kernel.gram(inducing_inputs).as_matrix() + model.prior.jitter * identity
    prior_mean = mean_function(inducing_inputs).reshape(-1)

    # W = Kxz Kzz^{-1}, the DTC projection of u onto the training inputs.
    cross_cov = kernel.cross_covariance(inducing_inputs, train_data.X)
    projection = jnp.linalg.solve(prior_cov, cross_cov).T
    residual = (train_data.y - mean_function(train_data.X)).reshape(-1)

    precision = jnp.linalg.inv(prior_cov) + projection.T @ projection / noise_variance
    optimal_cov = jnp.linalg.inv(precision)
    optimal_mean = prior_mean + optimal_cov @ (projection.T @ residual) / noise_variance
    return optimal_mean, optimal_cov, prior_mean, prior_cov


def _oracle_collapsed_prior_kl(
    family: CollapsedVariationalGaussian, train_data: gpx.Dataset
) -> Float[Array, ""]:
    """KL[q*(u) || p(u)] from the re-derived moments and the textbook formula."""
    optimal_mean, optimal_cov, prior_mean, prior_cov = _titsias_optimal_q(
        family, train_data
    )
    return _textbook_gaussian_kl(optimal_mean, optimal_cov, prior_mean, prior_cov)


@pytest.mark.parametrize("observation_noise", [0.37, 1.2])
@pytest.mark.parametrize("inducing_name", list(_COLLAPSED_INDUCING))
@pytest.mark.parametrize("kernel_name", list(_COLLAPSED_KERNELS))
def test_collapsed_prior_kl_matches_gaussian_kl_oracle(
    kernel_name: str, inducing_name: str, observation_noise: float
) -> None:
    """The collapsed KL must equal KL[q*(u) || p(u)] computed densely.

    This is the only assertion that constrains the *value* of
    ``CollapsedPosterior.prior_kl``: a sign flip on any of its four terms, a
    dropped ``- M``, or a factor of two moves the result far outside the
    tolerance below.
    """
    family, train_data = _build_collapsed_family(
        kernel_name, inducing_name, observation_noise
    )
    optimal_mean, optimal_cov, prior_mean, prior_cov = _titsias_optimal_q(
        family, train_data
    )

    # Guard against a degenerate configuration in which q* has collapsed onto
    # the prior and every term of the KL is separately zero.
    assert jnp.linalg.norm(optimal_mean - prior_mean) > 1e-2
    assert jnp.max(jnp.abs(optimal_cov - prior_cov)) > 1e-2

    reference = _textbook_gaussian_kl(optimal_mean, optimal_cov, prior_mean, prior_cov)
    assert reference > 1e-2

    np.testing.assert_allclose(
        np.float64(family.condition(train_data).prior_kl),
        np.float64(reference),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("inducing_name", list(_COLLAPSED_INDUCING))
@pytest.mark.parametrize("kernel_name", list(_COLLAPSED_KERNELS))
def test_collapsed_optimal_q_moments_match_the_dtc_posterior(
    kernel_name: str, inducing_name: str
) -> None:
    r"""The cached factors must encode the moments the KL claims they do.

    ``prior_kl`` reads :math:`S^{\star} = L_z B^{-1} L_z^{\top}` and
    :math:`\tilde m^{\star} = \sigma^{-1} L_z B^{-1} A (y - m(x))` off the
    conditioning cache. Reassembling both from that cache and comparing them
    with the conjugate update splits a KL failure into "the moments are
    wrong" and "the KL of correct moments is assembled wrong", which the
    value test alone cannot distinguish.
    """
    family, train_data = _build_collapsed_family(kernel_name, inducing_name, 0.37)
    posterior = family.condition(train_data)

    num_inducing = posterior.cholesky_b.shape[0]
    inverse_b = jsp.linalg.cho_solve(
        (posterior.cholesky_b, True), jnp.eye(num_inducing, dtype=jnp.float64)
    )
    cached_cov = posterior.cholesky_kzz @ inverse_b @ posterior.cholesky_kzz.T
    cached_centred_mean = (
        posterior.cholesky_kzz
        @ jsp.linalg.cho_solve(
            (posterior.cholesky_b, True), posterior.scaled_cross @ posterior.residual
        )
    ).reshape(-1) / jnp.sqrt(posterior.observation_variance)

    optimal_mean, optimal_cov, prior_mean, _ = _titsias_optimal_q(family, train_data)

    np.testing.assert_allclose(
        np.asarray(cached_cov), np.asarray(optimal_cov), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(prior_mean + cached_centred_mean),
        np.asarray(optimal_mean),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("inducing_name", ["grid-9-1d", "scatter-6-2d"])
@pytest.mark.parametrize("kernel_name", ["rbf", "matern32"])
def test_collapsed_prior_kl_gradients_match_the_oracle(
    kernel_name: str, inducing_name: str
) -> None:
    """``collapsed_elbo`` differentiates the KL, so its gradient must be right too.

    A value test cannot see an error that cancels at the evaluation point but
    not in the derivative -- for instance a term differentiated through the
    wrong factor.
    """
    family, train_data = _build_collapsed_family(kernel_name, inducing_name, 0.5)

    grads = eqx.filter_grad(lambda model: model.prior_kl(train_data))(family)
    reference_grads = eqx.filter_grad(
        lambda model: _oracle_collapsed_prior_kl(model, train_data)
    )(family)

    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    reference_leaves = jax.tree.leaves(
        eqx.filter(reference_grads, eqx.is_inexact_array)
    )

    assert leaves
    assert len(leaves) == len(reference_leaves)
    for leaf, reference_leaf in zip(leaves, reference_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(leaf), np.asarray(reference_leaf), rtol=1e-8, atol=1e-10
        )


def test_collapsed_prior_kl_is_sugar_over_the_conditioned_posterior() -> None:
    """``family.prior_kl(D)`` must be exactly ``family.condition(D).prior_kl``."""
    family, train_data = _build_collapsed_family("rbf", "grid-9-1d", 0.37)
    np.testing.assert_array_equal(
        np.asarray(family.prior_kl(train_data)),
        np.asarray(family.condition(train_data).prior_kl),
    )


def test_collapsed_prior_kl_vanishes_when_the_data_are_uninformative() -> None:
    """As sigma grows, q*(u) returns to p(u) and the KL must fall to zero.

    An anchor that does not go through the dense oracle at all: with a huge
    observation noise ``B -> I``, every term of the KL vanishes separately,
    and any surviving constant (a mislaid ``- M``, say) shows up as a
    non-zero limit.
    """
    family, train_data = _build_collapsed_family("rbf", "grid-9-1d", 1e6)
    assert 0.0 <= family.prior_kl(train_data) < 1e-8


# --- Families with more than one latent function (Categorical) ---


def _three_class_model():
    return zero_mean_rbf_prior() * gpx.likelihoods.Categorical(num_classes=3)


def test_whitened_family_defaults_have_one_mean_column_and_root_per_class():
    inducing_inputs = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    family = gpx.variational_families.WhitenedVariationalGaussian(
        model=_three_class_model(), inducing_inputs=inducing_inputs
    )
    assert val(family.variational_mean).shape == (4, 3)
    root = val(family.variational_root_covariance)
    assert root.shape == (3, 4, 4)
    assert jnp.allclose(root, jnp.stack([jnp.eye(4)] * 3))


def test_whitened_prior_kl_with_three_classes_is_the_sum_of_the_class_divergences():
    family, _, mean, root = whitened_family()
    standard_normal = gpjax.distributions.GaussianDistribution(
        loc=jnp.zeros(3), scale=lx.MatrixLinearOperator(jnp.eye(3))
    )
    expected = sum(
        gpjax.distributions.GaussianDistribution(
            loc=mean[:, class_index],
            scale=lx.MatrixLinearOperator(root[class_index] @ root[class_index].T),
        ).kl_divergence(standard_normal)
        for class_index in range(3)
    )
    assert jnp.allclose(family.prior_kl(), expected)


@pytest.mark.parametrize(
    "family_name",
    [
        "VariationalGaussian",
        "DualVariationalGaussian",
        "CollapsedVariationalGaussian",
        "GraphVariationalGaussian",
    ],
)
def test_families_without_multi_class_support_raise(family_name):
    family_class = getattr(gpx.variational_families, family_name)
    inducing_inputs = jnp.linspace(-1.0, 1.0, 3).reshape(-1, 1)
    if family_name == "GraphVariationalGaussian":
        inducing_inputs = jnp.arange(3).reshape(-1, 1)
    with pytest.raises(ValueError, match="WhitenedVariationalGaussian"):
        family_class(model=_three_class_model(), inducing_inputs=inducing_inputs)


@pytest.mark.parametrize(
    ("likelihood", "mean_shape", "root_shape", "wrong_argument"),
    [
        # One latent function: the mean needs one column.
        (gpx.likelihoods.Bernoulli(), (3, 2), (3, 3), "variational_mean"),
        # Three latent functions: the root needs one (M, M) matrix per class.
        (
            gpx.likelihoods.Categorical(num_classes=3),
            (3, 3),
            (3, 3),
            "variational_root_covariance",
        ),
    ],
    ids=["mean-columns", "root-per-class"],
)
def test_whitened_family_rejects_moments_of_the_wrong_shape(
    likelihood, mean_shape, root_shape, wrong_argument
):
    model = zero_mean_rbf_prior() * likelihood
    with pytest.raises(ValueError, match=wrong_argument):
        gpx.variational_families.WhitenedVariationalGaussian(
            model=model,
            inducing_inputs=jnp.linspace(-1.0, 1.0, 3).reshape(-1, 1),
            variational_mean=jnp.zeros(mean_shape),
            variational_root_covariance=jnp.eye(3) * jnp.ones(root_shape),
        )

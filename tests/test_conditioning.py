"""Contract tests for :mod:`gpjax.conditioning`.

Four properties are pinned here, none of them covered elsewhere in the suite.

1. **Pytree round-trip.** Every ``Posterior`` is an immutable ``eqx.Module``
   that JAX flattens and rebuilds at every transformation boundary. A field
   that silently became static -- or stopped being static -- would corrupt
   those boundaries with nothing else noticing, so every conditioned mode is
   flattened, unflattened, and required to come back with identical leaves,
   an identical treedef, and identical predictions.

2. **jit-ability.** ``jax.jit(lambda m, d: m.condition(d).log_marginal_likelihood)``
   must compile and agree with the eager value. jit-ability is a first-class
   constraint in this repository, so ``jax.grad`` and ``jax.vmap`` through
   ``condition`` are pinned too, per mode rather than for the conjugate path
   alone.

3. **``sample_approx`` refuses multi-output loudly.** ADR-0001 records the
   refusal as a consequence of the pathwise sampler being single-output. The
   ``ValueError`` had no test, so neither the refusal nor its message was
   protected.

4. **Conformance.** Every conditionable object in the library takes the same
   ``condition(train_data)``, treats ``model | D`` as sugar for it, and
   returns from ``model.predict(x, D)`` a distribution *numerically
   identical* to ``model.condition(D)(x)``. That last identity is the
   property ``CONTEXT.md`` calls non-negotiable: ``predict`` is sugar and
   never a second implementation.

The two documented exclusions -- :class:`~gpjax.gps.HeteroscedasticModel` and
:class:`~gpjax.variational_families.HeteroscedasticVariationalFamily` -- are
pinned to raise ``NotImplementedError`` naming the alternative.
"""

from collections.abc import Callable
import inspect
import re
from typing import (
    Any,
    NamedTuple,
)

import equinox as eqx
import gpjax as gpx
from gpjax.conditioning import (
    CollapsedPosterior,
    ExactPosterior,
    LatentPosterior,
    Posterior,
    SparsePosterior,
)
from gpjax.gps import (
    ConjugateModel,
    HeteroscedasticModel,
    JointModel,
    NonConjugateModel,
)
from gpjax.state_space.conditioning import StateSpacePosterior
from gpjax.state_space.gps import (
    StateSpaceConjugateModel,
    StateSpacePrior,
)
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    DualVariationalGaussian,
    GraphVariationalGaussian,
    HeteroscedasticVariationalFamily,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
import jax
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
import pytest

from tests._categorical_helpers import (
    three_class_data,
    whitened_family,
    zero_mean_rbf_prior,
)
from tests._dual_helpers import random_dual_sites

_NUM_TRAIN = 10
_NUM_INDUCING = 5
_NUM_TEST = 6


# ---------------------------------------------------------------------------
# Model and data builders
# ---------------------------------------------------------------------------


def _regression_data() -> gpx.Dataset:
    """A small, sorted, float64 regression set (sorted for the Kalman path)."""
    inputs = jnp.linspace(0.0, 1.0, _NUM_TRAIN).reshape(-1, 1)
    outputs = jnp.sin(3.0 * inputs) + 0.1 * jnp.cos(11.0 * inputs)
    return gpx.Dataset(X=inputs, y=outputs)


def _classification_data() -> gpx.Dataset:
    data = _regression_data()
    return gpx.Dataset(X=data.X, y=(data.y > 0.0).astype(jnp.float64))


def _test_inputs():
    return jnp.linspace(-0.1, 1.1, _NUM_TEST).reshape(-1, 1)


def _inducing_inputs():
    return jnp.linspace(0.0, 1.0, _NUM_INDUCING).reshape(-1, 1)


def _conjugate_model() -> ConjugateModel:
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array(0.4)),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.3)),
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.37))


def _nonconjugate_model() -> NonConjugateModel:
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array(0.2)),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array(0.6), variance=jnp.array(1.1)),
    )
    model = prior * gpx.likelihoods.Bernoulli()
    return model.init_latent(_NUM_TRAIN, key=jr.key(7))


def _state_space_model() -> StateSpaceConjugateModel:
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(
            lengthscale=jnp.array(0.9), variance=jnp.array(1.2)
        ),
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.3))


def _heteroscedastic_model() -> HeteroscedasticModel:
    signal_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array(0.8)),
    )
    noise_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array(0.5)),
    )
    return HeteroscedasticModel(
        prior=signal_prior,
        likelihood=gpx.likelihoods.HeteroscedasticGaussian(),
        noise_prior=noise_prior,
    )


def _nontrivial_moments(num_inducing: int):
    """Deterministic, non-default variational mean and root for a Gaussian family.

    Default moments leave the predictive equal to the prior, which would make
    the gradient and round-trip assertions far weaker than they look.
    """
    index = jnp.arange(num_inducing, dtype=jnp.float64)
    mean = (0.3 * jnp.cos(1.7 * index) - 0.15 * index).reshape(-1, 1)
    row, col = index[:, None], index[None, :]
    off_diagonal = 0.2 * jnp.sin(0.9 * row + 1.3 * col)
    diagonal = 0.5 + 0.4 * jnp.cos(0.6 * index)
    root = jnp.tril(off_diagonal, -1) + jnp.diag(diagonal)
    return mean, root


def _pathwise_model() -> ConjugateModel:
    """A conjugate model whose kernel has a well-defined spectral density.

    The pathwise (Fourier-feature) sampler needs `n_dims` fixed on the
    kernel; an ARD (array-valued) lengthscale does that implicitly, whereas
    the scalar lengthscale used by `_conjugate_model` leaves it `None`.
    """
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array(0.3)),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array([0.3]), variance=jnp.array(0.9)),
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.2))


def _pathwise_family(family_cls, variational_mean=None, variational_root=None):
    mean, root = _nontrivial_moments(_NUM_INDUCING)
    return family_cls(
        model=_pathwise_model(),
        inducing_inputs=_inducing_inputs(),
        variational_mean=mean if variational_mean is None else variational_mean,
        variational_root_covariance=root
        if variational_root is None
        else variational_root,
    )


def _multi_output_model():
    coregionalization = gpx.parameters.CoregionalizationMatrix(
        num_outputs=2, rank=1, key=jr.key(3)
    )
    kernel = gpx.kernels.ICMKernel(
        base_kernel=gpx.kernels.RBF(),
        coregionalization_matrix=coregionalization,
    )
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    return prior * gpx.likelihoods.MultiOutputGaussian(num_outputs=2)


def _multi_output_data() -> gpx.Dataset:
    inputs = jnp.linspace(0.0, 1.0, _NUM_TRAIN).reshape(-1, 1)
    outputs = jnp.column_stack(
        [jnp.sin(3.0 * inputs.squeeze()), jnp.cos(3.0 * inputs.squeeze())]
    )
    return gpx.Dataset(X=inputs, y=outputs)


# ---------------------------------------------------------------------------
# The conditionable registry
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    """One conditionable object plus everything a contract check needs.

    Attributes:
        conditionable: The object exposing ``condition``/``__or__``/``predict``.
        train_data: The dataset passed to ``condition``.
        test_inputs: Where the conditioned process is queried.
        posterior_type: The exact ``Posterior`` subclass ``condition`` returns.
        scalar_quantity: The evidence-like scalar the posterior exposes, or
            ``None`` for the sparse families, which expose no such scalar.
        alternative_targets: A second, valid ``y`` for the same inputs, used to
            vmap ``condition`` over training targets. ``None`` where vmapping
            over targets is incoherent -- the sparse families carry $q(u)$
            internally and ignore ``train_data`` entirely.
    """

    conditionable: Any
    train_data: gpx.Dataset
    test_inputs: Any
    posterior_type: type
    scalar_quantity: Callable[[Posterior], Any] | None = None
    alternative_targets: Any = None


def _conjugate_case() -> _Case:
    data = _regression_data()
    return _Case(
        conditionable=_conjugate_model(),
        train_data=data,
        test_inputs=_test_inputs(),
        posterior_type=ExactPosterior,
        scalar_quantity=lambda posterior: posterior.log_marginal_likelihood,
        alternative_targets=2.0 * data.y,
    )


def _nonconjugate_case() -> _Case:
    data = _classification_data()
    return _Case(
        conditionable=_nonconjugate_model(),
        train_data=data,
        test_inputs=_test_inputs(),
        posterior_type=LatentPosterior,
        scalar_quantity=lambda posterior: posterior.log_posterior_density,
        # Still binary, so the Bernoulli log-density stays well defined.
        alternative_targets=1.0 - data.y,
    )


def _state_space_case() -> _Case:
    data = _regression_data()
    return _Case(
        conditionable=_state_space_model(),
        train_data=data,
        test_inputs=_test_inputs(),
        posterior_type=StateSpacePosterior,
        scalar_quantity=lambda posterior: posterior.log_marginal_likelihood,
        alternative_targets=2.0 * data.y,
    )


def _variational_case() -> _Case:
    mean, root = _nontrivial_moments(_NUM_INDUCING)
    family = VariationalGaussian(
        model=_conjugate_model(),
        inducing_inputs=_inducing_inputs(),
        variational_mean=mean,
        variational_root_covariance=root,
    )
    return _Case(
        conditionable=family,
        train_data=_regression_data(),
        test_inputs=_test_inputs(),
        posterior_type=SparsePosterior,
    )


def _whitened_case() -> _Case:
    mean, root = _nontrivial_moments(_NUM_INDUCING)
    family = WhitenedVariationalGaussian(
        model=_conjugate_model(),
        inducing_inputs=_inducing_inputs(),
        variational_mean=mean,
        variational_root_covariance=root,
    )
    return _Case(
        conditionable=family,
        train_data=_regression_data(),
        test_inputs=_test_inputs(),
        posterior_type=SparsePosterior,
    )


def _dual_case() -> _Case:
    dual_vector, dual_matrix = random_dual_sites(3, _NUM_INDUCING)
    family = DualVariationalGaussian(
        model=_conjugate_model(),
        inducing_inputs=_inducing_inputs(),
        dual_vector=dual_vector,
        dual_matrix=dual_matrix,
    )
    return _Case(
        conditionable=family,
        train_data=_regression_data(),
        test_inputs=_test_inputs(),
        posterior_type=SparsePosterior,
    )


def _graph_case() -> _Case:
    graph = nx.barbell_graph(6, 0)
    laplacian = jnp.asarray(nx.laplacian_matrix(graph).toarray(), dtype=jnp.float64)
    kernel = gpx.kernels.GraphKernel(
        laplacian=laplacian, lengthscale=2.3, variance=3.2, smoothness=6.1
    )
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Constant(), kernel=kernel)
    model = prior * gpx.likelihoods.Bernoulli()

    mean, root = _nontrivial_moments(_NUM_INDUCING)
    family = GraphVariationalGaussian(
        model=model,
        inducing_inputs=jnp.arange(0, 10, 2).reshape(-1, 1).astype(jnp.int64),
        variational_mean=mean,
        variational_root_covariance=root,
    )
    # The graph family carries q(u) internally and never reads `train_data`;
    # the node indices are stored as float64 purely so that `Dataset` does not
    # emit its precision warning, which pytest is configured to treat as an
    # error.
    node_indices = jnp.arange(12, dtype=jnp.float64).reshape(-1, 1)
    labels = jnp.asarray(np.resize([0.0, 1.0], 12), dtype=jnp.float64).reshape(-1, 1)
    return _Case(
        conditionable=family,
        train_data=gpx.Dataset(X=node_indices, y=labels),
        test_inputs=jnp.arange(1, 12, 2).reshape(-1, 1).astype(jnp.int64),
        posterior_type=SparsePosterior,
    )


def _collapsed_case() -> _Case:
    data = _regression_data()
    family = CollapsedVariationalGaussian(
        model=_conjugate_model(), inducing_inputs=_inducing_inputs()
    )
    return _Case(
        conditionable=family,
        train_data=data,
        test_inputs=_test_inputs(),
        posterior_type=CollapsedPosterior,
        scalar_quantity=lambda posterior: posterior.elbo_bound,
        alternative_targets=2.0 * data.y,
    )


_CASE_BUILDERS = {
    "conjugate": _conjugate_case,
    "nonconjugate": _nonconjugate_case,
    "state_space": _state_space_case,
    "variational": _variational_case,
    "whitened": _whitened_case,
    "dual": _dual_case,
    "graph": _graph_case,
    "collapsed": _collapsed_case,
}

_CASE_NAMES = list(_CASE_BUILDERS)
_DATA_CONSUMING_CASES = [
    name
    for name in _CASE_NAMES
    if _CASE_BUILDERS[name]().alternative_targets is not None
]


def _build(case_name: str) -> _Case:
    return _CASE_BUILDERS[case_name]()


def _assert_identical(first, second) -> None:
    """Two predictive distributions must agree bit for bit, not merely closely.

    ``predict`` is documented as one line of sugar over ``condition``. Anything
    short of exact equality means a second derivation crept back in.
    """
    np.testing.assert_array_equal(np.asarray(first.mean), np.asarray(second.mean))
    np.testing.assert_array_equal(
        np.asarray(first.variance), np.asarray(second.variance)
    )


# ---------------------------------------------------------------------------
# 1. Pytree round-trip
# ---------------------------------------------------------------------------


def test_the_registry_exercises_every_posterior_in_the_module():
    """Guard against a new ``Posterior`` subclass slipping past these tests."""
    covered = {_build(name).posterior_type for name in _CASE_NAMES}
    assert {
        ExactPosterior,
        LatentPosterior,
        SparsePosterior,
        CollapsedPosterior,
    } <= covered


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_condition_returns_the_declared_posterior(case_name):
    case = _build(case_name)
    posterior = case.conditionable.condition(case.train_data)
    assert isinstance(posterior, Posterior)
    assert type(posterior) is case.posterior_type


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_posterior_survives_a_pytree_round_trip(case_name):
    case = _build(case_name)
    posterior = case.conditionable.condition(case.train_data)

    leaves, treedef = jax.tree_util.tree_flatten(posterior)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    rebuilt_leaves, rebuilt_treedef = jax.tree_util.tree_flatten(rebuilt)

    assert type(rebuilt) is type(posterior)
    # Static metadata lives in the treedef, so treedef equality is exactly the
    # "same static metadata" assertion.
    assert rebuilt_treedef == treedef
    assert len(rebuilt_leaves) == len(leaves)
    for original, restored in zip(leaves, rebuilt_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(restored))

    _assert_identical(posterior(case.test_inputs), rebuilt(case.test_inputs))


@pytest.mark.parametrize("case_name", _DATA_CONSUMING_CASES)
def test_pytree_round_trip_preserves_the_evidence_scalar(case_name):
    """The cached factorisation, not just the predictive, must survive."""
    case = _build(case_name)
    posterior = case.conditionable.condition(case.train_data)

    leaves, treedef = jax.tree_util.tree_flatten(posterior)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    np.testing.assert_array_equal(
        np.asarray(case.scalar_quantity(rebuilt)),
        np.asarray(case.scalar_quantity(posterior)),
    )


def test_sparse_posterior_whitened_flag_is_static_and_survives_flattening():
    """``whitened`` selects a different algebra, so it must never be a leaf."""
    data = _regression_data()
    plain = _build("variational").conditionable.condition(data)
    whitened = _build("whitened").conditionable.condition(data)

    assert plain.whitened is False
    assert whitened.whitened is True

    # A static field discriminates the treedefs; a leaf would not.
    assert jax.tree_util.tree_structure(plain) != jax.tree_util.tree_structure(whitened)
    for posterior in (plain, whitened):
        leaves, treedef = jax.tree_util.tree_flatten(posterior)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.whitened is posterior.whitened


def test_exact_posterior_keeps_its_dataset_metadata_through_a_round_trip():
    """``Dataset.n_total`` is pytree aux data; conditioning must not drop it."""
    data = _regression_data()
    minibatch = gpx.Dataset(X=data.X, y=data.y, n_total=4 * _NUM_TRAIN)
    posterior = _conjugate_model().condition(minibatch)

    leaves, treedef = jax.tree_util.tree_flatten(posterior)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.train_data.n_total == 4 * _NUM_TRAIN
    assert rebuilt.train_data.full_size == 4 * _NUM_TRAIN


# ---------------------------------------------------------------------------
# 2. jit, grad and vmap
# ---------------------------------------------------------------------------


def test_the_headline_jit_expression_compiles_and_matches():
    """``jax.jit(lambda m, d: m.condition(d).log_marginal_likelihood)``."""
    model, data = _conjugate_model(), _regression_data()
    compiled = jax.jit(lambda m, d: m.condition(d).log_marginal_likelihood)

    np.testing.assert_allclose(
        np.asarray(compiled(model, data)),
        np.asarray(model.condition(data).log_marginal_likelihood),
        rtol=1e-10,
    )


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_condition_and_query_are_jittable(case_name):
    case = _build(case_name)

    def query(conditionable, train_data, test_inputs):
        return conditionable.condition(train_data)(test_inputs).mean

    eager = query(case.conditionable, case.train_data, case.test_inputs)
    compiled = jax.jit(query)(case.conditionable, case.train_data, case.test_inputs)

    np.testing.assert_allclose(
        np.asarray(compiled), np.asarray(eager), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("case_name", _DATA_CONSUMING_CASES)
def test_conditioned_scalars_are_jittable(case_name):
    case = _build(case_name)
    quantity = case.scalar_quantity

    def evaluate(conditionable, train_data):
        return quantity(conditionable.condition(train_data))

    eager = evaluate(case.conditionable, case.train_data)
    compiled = jax.jit(evaluate)(case.conditionable, case.train_data)

    assert jnp.shape(eager) == ()
    np.testing.assert_allclose(
        np.asarray(compiled), np.asarray(eager), rtol=1e-10, atol=1e-12
    )


def test_grad_through_condition_matches_finite_differences():
    """Differentiating the evidence through ``condition`` gives the right slope."""
    data = _regression_data()

    def evidence(lengthscale):
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Constant(jnp.array(0.4)),
            kernel=gpx.kernels.RBF(lengthscale=lengthscale, variance=jnp.array(1.3)),
        )
        model = prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.37))
        return model.condition(data).log_marginal_likelihood

    lengthscale = jnp.asarray(0.8)
    step = 1e-6
    numeric = (evidence(lengthscale + step) - evidence(lengthscale - step)) / (2 * step)

    np.testing.assert_allclose(
        np.asarray(jax.grad(evidence)(lengthscale)), np.asarray(numeric), rtol=1e-6
    )


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_condition_is_differentiable_in_every_mode(case_name):
    case = _build(case_name)

    def summed_mean(conditionable):
        posterior = conditionable.condition(case.train_data)
        return jnp.sum(posterior(case.test_inputs).mean)

    gradient = eqx.filter_grad(summed_mean)(case.conditionable)
    leaves = jax.tree_util.tree_leaves(gradient)

    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0.0)) for leaf in leaves)


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_queries_vmap_over_blocks_of_test_inputs(case_name):
    """Condition once, then query a batch of test blocks under ``vmap``."""
    case = _build(case_name)
    posterior = case.conditionable.condition(case.train_data)
    blocks = jnp.stack([case.test_inputs, case.test_inputs])

    means = jax.vmap(lambda block: posterior(block).mean)(blocks)

    assert means.shape == (2, case.test_inputs.shape[0])
    reference = np.asarray(posterior(case.test_inputs).mean)
    for row in means:
        np.testing.assert_allclose(np.asarray(row), reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case_name", _DATA_CONSUMING_CASES)
def test_condition_vmaps_over_training_targets(case_name):
    """``condition`` itself is batched, not merely the query that follows it.

    Only the modes whose maths reads ``train_data`` appear here: the sparse
    variational families accept ``train_data`` for interface uniformity and
    ignore it, so batching over targets would assert nothing about them.
    """
    case = _build(case_name)
    quantity = case.scalar_quantity
    stacked = jnp.stack([case.train_data.y, case.alternative_targets])

    def evaluate(targets):
        data = gpx.Dataset(X=case.train_data.X, y=targets)
        return quantity(case.conditionable.condition(data))

    batched = jax.vmap(evaluate)(stacked)

    assert batched.shape == (2,)
    np.testing.assert_allclose(
        np.asarray(batched[0]),
        np.asarray(evaluate(case.train_data.y)),
        rtol=1e-10,
        atol=1e-12,
    )
    # The two targets are genuinely different, so the batch must be too --
    # otherwise vmap could be silently broadcasting a single conditioning.
    assert not bool(jnp.allclose(batched[0], batched[1]))


# ---------------------------------------------------------------------------
# 3. sample_approx refuses multi-output
# ---------------------------------------------------------------------------


_MULTI_OUTPUT_REFUSAL = "sample_approx does not support multi-output likelihoods yet."


def test_sample_approx_refuses_a_multi_output_posterior():
    posterior = _multi_output_model().condition(_multi_output_data())
    with pytest.raises(ValueError, match=re.escape(_MULTI_OUTPUT_REFUSAL)):
        posterior.sample_approx(2, jr.key(1))


def test_sample_approx_sugar_refuses_a_multi_output_model():
    """The ``ConjugateModel`` shortcut must refuse identically, not differently."""
    model, data = _multi_output_model(), _multi_output_data()
    with pytest.raises(ValueError, match=re.escape(_MULTI_OUTPUT_REFUSAL)):
        model.sample_approx(2, data, jr.key(1))


def test_sample_approx_still_draws_for_a_single_output_posterior():
    """Positive control: the refusal above must not be over-broad."""
    # The pathwise sampler needs the kernel's spectral density, which in turn
    # needs a known input dimension -- hence the length-1 (ARD) lengthscale.
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array(0.4)),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array([0.8]), variance=jnp.array(1.3)),
    )
    model = prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.37))
    posterior = model.condition(_regression_data())
    sample_fn = posterior.sample_approx(3, jr.key(1), num_features=16)

    draws = sample_fn(_test_inputs())

    assert draws.shape == (_NUM_TEST, 3)
    assert bool(jnp.all(jnp.isfinite(draws)))


# ---------------------------------------------------------------------------
# 3b. sample_approx on SparsePosterior (decoupled / Matheron sampling, #673)
# ---------------------------------------------------------------------------

_PATHWISE_FAMILIES = [VariationalGaussian, WhitenedVariationalGaussian]


@pytest.mark.parametrize("family_cls", _PATHWISE_FAMILIES)
def test_sparse_sample_approx_matches_the_analytic_predictive(family_cls):
    """Monte-Carlo mean/covariance of the decoupled sampler must reproduce the
    sparse posterior's closed-form predictive moments, for both the
    unwhitened and whitened parameterisations."""
    posterior = _pathwise_family(family_cls).condition(_regression_data())
    grid = jnp.linspace(-0.2, 1.2, 8).reshape(-1, 1)

    sample_fn = posterior.sample_approx(6000, jr.key(0), num_features=500)
    draws = sample_fn(grid)  # [8, 6000]

    empirical_mean = jnp.mean(draws, axis=-1)
    empirical_cov = jnp.cov(draws)

    exact = posterior(grid, covariance="dense")
    np.testing.assert_allclose(
        np.asarray(empirical_mean), np.asarray(exact.mean), atol=0.06
    )

    exact_cov = np.asarray(exact.covariance())
    rel_frobenius = np.linalg.norm(
        np.asarray(empirical_cov) - exact_cov
    ) / np.linalg.norm(exact_cov)
    assert rel_frobenius < 0.15


@pytest.mark.parametrize("family_cls", _PATHWISE_FAMILIES)
def test_sparse_sample_approx_is_deterministic_and_finite(family_cls):
    posterior = _pathwise_family(family_cls).condition(_regression_data())
    grid = _test_inputs()

    draws_a = posterior.sample_approx(4, jr.key(1), num_features=32)(grid)
    draws_b = posterior.sample_approx(4, jr.key(1), num_features=32)(grid)
    draws_c = posterior.sample_approx(4, jr.key(2), num_features=32)(grid)

    np.testing.assert_array_equal(np.asarray(draws_a), np.asarray(draws_b))
    assert not bool(jnp.allclose(draws_a, draws_c))
    assert draws_a.shape == (_NUM_TEST, 4)
    assert bool(jnp.all(jnp.isfinite(draws_a)))


def test_sparse_sample_approx_is_jittable():
    grid = _test_inputs()

    def draw(conditionable, train_data, key):
        posterior = conditionable.condition(train_data)
        return posterior.sample_approx(4, key, num_features=32)(grid)

    family, data, key = (
        _pathwise_family(VariationalGaussian),
        _regression_data(),
        jr.key(5),
    )
    eager = draw(family, data, key)
    compiled = jax.jit(draw)(family, data, key)

    np.testing.assert_allclose(
        np.asarray(compiled), np.asarray(eager), rtol=1e-10, atol=1e-12
    )


def test_sparse_sample_approx_is_differentiable_in_the_variational_mean():
    """Gradients must flow from the sampled functions back to q(u)'s mean."""
    grid = _test_inputs()
    key = jr.key(9)
    _, root = _nontrivial_moments(_NUM_INDUCING)

    def summed_draws(variational_mean):
        posterior = _pathwise_family(
            VariationalGaussian,
            variational_mean=variational_mean,
            variational_root=root,
        ).condition(_regression_data())
        return jnp.sum(posterior.sample_approx(3, key, num_features=32)(grid))

    mean0, _ = _nontrivial_moments(_NUM_INDUCING)
    gradient = jax.grad(summed_draws)(mean0)

    assert gradient.shape == mean0.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(gradient != 0.0))


def test_sparse_sample_approx_vmaps_over_keys():
    posterior = _pathwise_family(VariationalGaussian).condition(_regression_data())
    grid = _test_inputs()
    keys = jr.split(jr.key(11), 4)

    batched = jax.vmap(lambda k: posterior.sample_approx(3, k, num_features=32)(grid))(
        keys
    )

    assert batched.shape == (4, _NUM_TEST, 3)
    assert bool(jnp.all(jnp.isfinite(batched)))
    # Independent keys must give independent draws, not a broadcast repeat.
    assert not bool(jnp.allclose(batched[0], batched[1]))


# ---------------------------------------------------------------------------
# 4. Conformance to the condition contract
# ---------------------------------------------------------------------------


_CONDITIONABLE_CLASSES = [
    JointModel,
    ConjugateModel,
    NonConjugateModel,
    HeteroscedasticModel,
    StateSpaceConjugateModel,
    AbstractVariationalFamily,
    VariationalGaussian,
    WhitenedVariationalGaussian,
    GraphVariationalGaussian,
    DualVariationalGaussian,
    CollapsedVariationalGaussian,
    HeteroscedasticVariationalFamily,
]


@pytest.mark.parametrize(
    "conditionable_cls", _CONDITIONABLE_CLASSES, ids=lambda cls: cls.__name__
)
def test_condition_has_the_one_uniform_signature(conditionable_cls):
    """``condition(train_data)``, required, everywhere -- universal or not at all.

    The maintainer's stated requirement for v1.0 is that a patchy conditioning
    API must be avoided at all costs, so this checks the shape of the signature
    directly rather than inferring it from behaviour.
    """
    signature = inspect.signature(conditionable_cls.condition)
    parameters = list(signature.parameters.values())[1:]

    assert parameters, f"{conditionable_cls.__name__}.condition takes no train_data"
    train_data = parameters[0]
    assert train_data.name == "train_data"
    assert train_data.default is inspect.Parameter.empty
    assert train_data.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    # Anything beyond it must be keyword-only, so `obj.condition(D)` is always
    # a complete call.
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:]
    )


@pytest.mark.parametrize(
    "conditionable_cls", _CONDITIONABLE_CLASSES, ids=lambda cls: cls.__name__
)
def test_every_conditionable_defines_the_or_operator(conditionable_cls):
    assert callable(getattr(conditionable_cls, "__or__", None))


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_or_is_sugar_for_condition(case_name):
    case = _build(case_name)

    via_operator = case.conditionable | case.train_data
    via_method = case.conditionable.condition(case.train_data)

    assert type(via_operator) is type(via_method)
    _assert_identical(via_operator(case.test_inputs), via_method(case.test_inputs))


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_predict_is_sugar_and_never_a_second_implementation(case_name):
    """``model.predict(x, D)`` must equal ``model.condition(D)(x)`` exactly."""
    case = _build(case_name)

    via_predict = case.conditionable.predict(case.test_inputs, case.train_data)
    via_condition = case.conditionable.condition(case.train_data)(case.test_inputs)

    _assert_identical(via_predict, via_condition)


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_posterior_predict_is_sugar_for_calling_the_posterior(case_name):
    """On the posterior itself, ``predict(t, D)`` is ``self(t)``; ``D`` is ignored."""
    case = _build(case_name)
    posterior = case.conditionable.condition(case.train_data)

    _assert_identical(
        posterior.predict(case.test_inputs, case.train_data),
        posterior(case.test_inputs),
    )


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_conditioning_twice_gives_the_same_process(case_name):
    """``Posterior`` is immutable, so conditioning is a pure function of state."""
    case = _build(case_name)

    first = case.conditionable.condition(case.train_data)
    second = case.conditionable.condition(case.train_data)

    _assert_identical(first(case.test_inputs), second(case.test_inputs))


# ---------------------------------------------------------------------------
# The documented exclusions
# ---------------------------------------------------------------------------


def test_heteroscedastic_model_refuses_to_condition_and_names_the_alternative():
    model = _heteroscedastic_model()
    with pytest.raises(NotImplementedError) as excinfo:
        model.condition(_regression_data())

    message = str(excinfo.value)
    assert "HeteroscedasticVariationalFamily" in message
    assert "heteroscedastic_elbo" in message


def test_heteroscedastic_model_or_refuses_identically():
    model = _heteroscedastic_model()
    with pytest.raises(NotImplementedError, match="HeteroscedasticVariationalFamily"):
        model | _regression_data()


def test_heteroscedastic_family_refuses_to_condition_and_names_the_alternative():
    family = HeteroscedasticVariationalFamily(
        model=_heteroscedastic_model(), inducing_inputs=_inducing_inputs()
    )
    with pytest.raises(NotImplementedError) as excinfo:
        family.condition(_regression_data())

    message = str(excinfo.value)
    assert "predict_latents" in message
    assert "signal_variational" in message


# --- Conditioning with more than one latent function (Categorical) ---


def test_log_posterior_density_with_three_classes_at_the_prior_mean():
    # A zero latent with a zero mean function gives f = 0, so each of the 4
    # labels has probability 1/3, and each of the 4 * 3 whitened values adds
    # the standard-normal log density at zero, -log(2 pi) / 2.
    data = three_class_data()
    model = zero_mean_rbf_prior() * gpx.likelihoods.Categorical(num_classes=3)
    density = model.condition(data).log_posterior_density
    expected = 4 * jnp.log(1.0 / 3.0) + 12 * (-0.5 * jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(density, expected)


@pytest.mark.parametrize("covariance", ["dense", "diagonal"])
def test_latent_predictive_with_three_classes_is_blocked_with_a_shared_covariance(
    covariance,
):
    data = three_class_data()
    latent = jr.normal(jr.key(0), (4, 3))
    model = gpx.gps.NonConjugateModel(
        zero_mean_rbf_prior(), gpx.likelihoods.Categorical(num_classes=3), latent=latent
    )
    test_inputs = jnp.linspace(-1.5, 1.5, 5).reshape(-1, 1)
    predictive = model.condition(data)(test_inputs, covariance=covariance)

    one_class_predictives = [
        gpx.gps.NonConjugateModel(
            zero_mean_rbf_prior(),
            gpx.likelihoods.Bernoulli(),
            latent=latent[:, [class_index]],
        ).condition(data)(test_inputs, covariance="dense")
        for class_index in range(3)
    ]
    expected_mean = jnp.concatenate(
        [class_predictive.mean for class_predictive in one_class_predictives]
    )
    expected_covariance = jnp.kron(jnp.eye(3), one_class_predictives[0].covariance())
    assert predictive.mean.shape == (15,)
    assert jnp.allclose(predictive.mean, expected_mean)
    if covariance == "dense":
        assert jnp.allclose(predictive.covariance(), expected_covariance)
    assert jnp.allclose(predictive.variance, jnp.diag(expected_covariance))


@pytest.mark.parametrize("covariance", ["dense", "diagonal"])
def test_whitened_predictive_with_three_classes_is_blocked_and_block_diagonal(
    covariance,
):
    family, inducing_inputs, mean, root = whitened_family()
    data = three_class_data()
    test_inputs = jnp.linspace(-1.5, 1.5, 5).reshape(-1, 1)
    predictive = family.condition(data)(test_inputs, covariance=covariance)

    bernoulli_model = zero_mean_rbf_prior() * gpx.likelihoods.Bernoulli()
    one_class_predictives = [
        gpx.variational_families.WhitenedVariationalGaussian(
            model=bernoulli_model,
            inducing_inputs=inducing_inputs,
            variational_mean=mean[:, [class_index]],
            variational_root_covariance=root[class_index],
        ).condition(data)(test_inputs, covariance="dense")
        for class_index in range(3)
    ]
    expected_mean = jnp.concatenate(
        [class_predictive.mean for class_predictive in one_class_predictives]
    )
    expected_covariance = jax.scipy.linalg.block_diag(
        *[class_predictive.covariance() for class_predictive in one_class_predictives]
    )
    assert predictive.mean.shape == (15,)
    assert jnp.allclose(predictive.mean, expected_mean)
    if covariance == "dense":
        assert jnp.allclose(predictive.covariance(), expected_covariance)
    assert jnp.allclose(predictive.variance, jnp.diag(expected_covariance))


def test_sparse_sample_approx_rejects_more_than_one_latent_function():
    family, *_ = whitened_family()
    posterior = family.condition(three_class_data())
    with pytest.raises(ValueError, match="one latent function"):
        posterior.sample_approx(num_samples=2, key=jr.key(0))

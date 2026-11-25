"""Tests for the Categorical (softmax multi-class) likelihood."""

from gpjax.distributions import GaussianDistribution
from gpjax.likelihoods import Categorical
import jax
from jax import config
import jax.numpy as jnp
import lineax as lx
import pytest

config.update("jax_enable_x64", True)


def test_link_function_gives_softmax_probabilities():
    likelihood = Categorical(num_classes=3)
    f = jnp.array([[0.0, jnp.log(2.0), 0.0]])
    assert jnp.allclose(
        likelihood.link_function(f).probs, jnp.array([[0.25, 0.5, 0.25]])
    )


def test_log_prob_takes_float_class_indices_of_shape_n_by_one():
    likelihood = Categorical(num_classes=3)
    f = jnp.array([[0.0, jnp.log(2.0), 0.0], [0.0, 0.0, 0.0]])
    y = jnp.array([[1.0], [2.0]])
    log_prob = likelihood.link_function(f).log_prob(y)
    assert log_prob.shape == (2,)
    assert jnp.allclose(log_prob, jnp.log(jnp.array([0.5, 1.0 / 3.0])))


def test_num_latent_functions_is_the_number_of_classes():
    assert Categorical(num_classes=4).num_latent_functions == 4


@pytest.mark.parametrize("dense", [True, False])
def test_predict_applies_the_moment_scaled_softmax_to_blocked_latents(dense):
    # Two test points, three classes, blocked ordering: index c * M + m.
    # Point 0 has means (0, 2 log 2, 0) and variance 24 / pi, so each mean is
    # scaled by 1 / sqrt(1 + pi * (24 / pi) / 8) = 1 / 2 to (0, log 2, 0).
    # Point 1 has zero means, so all classes are equally likely.
    big = 24.0 / jnp.pi
    loc = jnp.array([0.0, 0.0, 2.0 * jnp.log(2.0), 0.0, 0.0, 0.0])
    variance = jnp.array([big, 0.0, big, 0.0, big, 0.0])
    scale = (
        lx.MatrixLinearOperator(jnp.diag(variance))
        if dense
        else lx.DiagonalLinearOperator(variance)
    )
    predictive = Categorical(num_classes=3).predict(
        GaussianDistribution(loc=loc, scale=scale)
    )
    expected = jnp.array([[0.25, 0.5, 0.25], [1 / 3, 1 / 3, 1 / 3]])
    assert jnp.allclose(predictive.probs, expected)
    assert jnp.allclose(
        predictive.log_prob(jnp.array([[1.0], [0.0]])), jnp.log(jnp.array([0.5, 1 / 3]))
    )


# --- NonConjugateModel with C latent functions ---


def _three_class_data(n: int = 4):
    import gpjax as gpx

    x = jnp.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = jnp.array([[0.0], [1.0], [2.0], [1.0]])[:n]
    return gpx.Dataset(X=x, y=y)


def _prior():
    import gpjax as gpx

    return gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
    )


def test_prior_times_categorical_holds_one_latent_column_per_class():
    import gpjax as gpx

    model = _prior() * Categorical(num_classes=3)
    assert isinstance(model, gpx.gps.NonConjugateModel)
    model = model.init_latent(5)
    from gpjax.parameters import val

    assert val(model.latent).shape == (5, 3)


def test_log_posterior_density_at_the_prior_mean():
    # A zero latent with a zero mean function gives f = 0, so each label has
    # probability 1/3, and each of the N * C whitened values contributes the
    # standard-normal log density at zero, -log(2 pi) / 2.
    data = _three_class_data()
    model = _prior() * Categorical(num_classes=3)
    density = model.condition(data).log_posterior_density
    expected = 4 * jnp.log(1.0 / 3.0) + 12 * (-0.5 * jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(density, expected)


@pytest.mark.parametrize("covariance", ["dense", "diagonal"])
def test_latent_predictive_is_blocked_with_one_shared_covariance(covariance):
    import gpjax as gpx
    import jax.random as jr

    data = _three_class_data()
    latent = jr.normal(jr.key(0), (4, 3))
    model = gpx.gps.NonConjugateModel(
        _prior(), Categorical(num_classes=3), latent=latent
    )
    xtest = jnp.linspace(-1.5, 1.5, 5).reshape(-1, 1)
    predictive = model.condition(data)(xtest, covariance=covariance)

    single = [
        gpx.gps.NonConjugateModel(
            _prior(), gpx.likelihoods.Bernoulli(), latent=latent[:, [c]]
        ).condition(data)(xtest, covariance="dense")
        for c in range(3)
    ]
    expected_mean = jnp.concatenate([s.mean for s in single])
    assert predictive.mean.shape == (15,)
    assert jnp.allclose(predictive.mean, expected_mean)
    expected_cov = jnp.kron(jnp.eye(3), single[0].covariance())
    if covariance == "dense":
        assert jnp.allclose(predictive.covariance(), expected_cov)
    assert jnp.allclose(predictive.variance, jnp.diag(expected_cov))


# --- WhitenedVariationalGaussian with C latent functions ---


def _whitened_family(num_inducing: int = 3, seed: int = 0):
    import gpjax as gpx
    import jax.random as jr

    model = _prior() * Categorical(num_classes=3)
    z = jnp.linspace(-1.0, 1.0, num_inducing).reshape(-1, 1)
    key_mean, key_root = jr.split(jr.key(seed))
    mean = jr.normal(key_mean, (num_inducing, 3))
    root = jnp.tril(jr.normal(key_root, (3, num_inducing, num_inducing)))
    root = root.at[:, jnp.arange(num_inducing), jnp.arange(num_inducing)].set(
        jnp.abs(jnp.diagonal(root, axis1=1, axis2=2)) + 0.5
    )
    family = gpx.variational_families.WhitenedVariationalGaussian(
        model=model,
        inducing_inputs=z,
        variational_mean=mean,
        variational_root_covariance=root,
    )
    return family, z, mean, root


def test_whitened_family_defaults_have_one_mean_column_and_root_per_class():
    import gpjax as gpx
    from gpjax.parameters import val

    model = _prior() * Categorical(num_classes=3)
    z = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    family = gpx.variational_families.WhitenedVariationalGaussian(
        model=model, inducing_inputs=z
    )
    assert val(family.variational_mean).shape == (4, 3)
    root = val(family.variational_root_covariance)
    assert root.shape == (3, 4, 4)
    assert jnp.allclose(root, jnp.stack([jnp.eye(4)] * 3))


def test_whitened_prior_kl_is_the_sum_of_the_per_class_divergences():
    family, _, mean, root = _whitened_family()
    standard = GaussianDistribution(
        loc=jnp.zeros(3), scale=lx.MatrixLinearOperator(jnp.eye(3))
    )
    expected = sum(
        GaussianDistribution(
            loc=mean[:, c], scale=lx.MatrixLinearOperator(root[c] @ root[c].T)
        ).kl_divergence(standard)
        for c in range(3)
    )
    assert jnp.allclose(family.prior_kl(), expected)


@pytest.mark.parametrize("covariance", ["dense", "diagonal"])
def test_whitened_predictive_is_blocked_and_block_diagonal(covariance):
    import gpjax as gpx

    family, z, mean, root = _whitened_family()
    data = _three_class_data()
    xtest = jnp.linspace(-1.5, 1.5, 5).reshape(-1, 1)
    predictive = family.condition(data)(xtest, covariance=covariance)

    bernoulli_model = _prior() * gpx.likelihoods.Bernoulli()
    single = [
        gpx.variational_families.WhitenedVariationalGaussian(
            model=bernoulli_model,
            inducing_inputs=z,
            variational_mean=mean[:, [c]],
            variational_root_covariance=root[c],
        ).condition(data)(xtest, covariance="dense")
        for c in range(3)
    ]
    assert predictive.mean.shape == (15,)
    assert jnp.allclose(predictive.mean, jnp.concatenate([s.mean for s in single]))
    expected_cov = jax.scipy.linalg.block_diag(*[s.covariance() for s in single])
    if covariance == "dense":
        assert jnp.allclose(predictive.covariance(), expected_cov)
    assert jnp.allclose(predictive.variance, jnp.diag(expected_cov))


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
    import gpjax as gpx

    family = getattr(gpx.variational_families, family_name)
    model = _prior() * Categorical(num_classes=3)
    z = jnp.linspace(-1.0, 1.0, 3).reshape(-1, 1)
    if family_name == "GraphVariationalGaussian":
        z = jnp.arange(3).reshape(-1, 1)
    with pytest.raises(ValueError, match="WhitenedVariationalGaussian"):
        family(model=model, inducing_inputs=z)


# --- elbo and fit ---


def test_elbo_with_a_fixed_key_is_finite_and_deterministic():
    import gpjax as gpx
    import jax.random as jr

    family, *_ = _whitened_family()
    data = _three_class_data()
    first = gpx.objectives.elbo(family, data, key=jr.key(7))
    second = gpx.objectives.elbo(family, data, key=jr.key(7))
    assert jnp.isfinite(first)
    assert first == second


def test_elbo_without_a_key_raises():
    import gpjax as gpx

    family, *_ = _whitened_family()
    with pytest.raises(ValueError, match="key="):
        gpx.objectives.elbo(family, _three_class_data())


def test_fit_separates_three_classes():
    import gpjax as gpx
    import jax.random as jr
    import optax as ox

    # Three classes in three bands of the input: class c for x in band c.
    x = jnp.linspace(-3.0, 3.0, 60).reshape(-1, 1)
    labels = jnp.digitize(x[:, 0], jnp.array([-1.0, 1.0]))
    data = gpx.Dataset(X=x, y=labels.astype(jnp.float64).reshape(-1, 1))

    model = _prior() * Categorical(num_classes=3)
    z = jnp.linspace(-3.0, 3.0, 10).reshape(-1, 1)
    family = gpx.variational_families.WhitenedVariationalGaussian(
        model=model, inducing_inputs=z
    )

    fit_key, elbo_key = jr.split(jr.key(0))

    def objective(q, d):
        return -gpx.objectives.elbo(q, d, key=elbo_key)

    fitted, history = gpx.fit(
        model=family,
        objective=objective,
        train_data=data,
        optim=ox.adam(0.05),
        key=fit_key,
        num_iters=300,
        verbose=False,
    )
    assert history[-1] < history[0]

    probs = model.likelihood.predict(fitted.condition(data)(x)).probs
    accuracy = jnp.mean(jnp.argmax(probs, axis=1) == labels)
    assert accuracy > 0.9


def test_fit_natgrads_rejects_more_than_one_latent_function():
    import gpjax as gpx
    import optax as ox

    family, *_ = _whitened_family()
    with pytest.raises(ValueError, match="one latent function"):
        gpx.fit_natgrads(
            model=family,
            objective=lambda q, d: -gpx.objectives.elbo(q, d, key=jax.random.key(0)),
            train_data=_three_class_data(),
            optim=ox.adam(0.01),
            num_iters=2,
            verbose=False,
        )

"""Tests for gpjax.kernels.additive.transforms."""

from jax import config

config.update("jax_enable_x64", True)

from gpjax.kernels.additive.transforms import (
    SinhArcsinhTransform,
    fit_all_normalising_flows,
    fit_normalising_flow,
)
import jax
import jax.numpy as jnp
import jax.random as jr
from numpyro.distributions.transforms import ComposeTransform
import pytest


class TestSinhArcsinhTransform:
    """Tests for SinhArcsinhTransform NumPyro subclass."""

    def test_identity_at_neutral_params(self):
        """sinh(1 * arcsinh(x) - 0) = x — neutral params give identity."""
        t = SinhArcsinhTransform(skewness=jnp.float64(0.0), tailweight=jnp.float64(1.0))
        x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert jnp.allclose(t(x), x, atol=1e-12)

    @pytest.mark.parametrize(
        ("skewness", "tailweight"),
        [(0.0, 1.0), (0.5, 1.2), (-0.3, 0.8), (1.0, 2.0)],
    )
    def test_forward_inverse_roundtrip(self, skewness, tailweight):
        """t.inv(t(x)) ≈ x for various parameter settings."""
        t = SinhArcsinhTransform(
            skewness=jnp.float64(skewness),
            tailweight=jnp.float64(tailweight),
        )
        x = jnp.linspace(-3.0, 3.0, 50)
        assert jnp.allclose(t.inv(t(x)), x, atol=1e-10)

    @pytest.mark.parametrize(
        ("skewness", "tailweight"),
        [(0.0, 1.0), (0.5, 1.2), (-0.3, 0.8)],
    )
    def test_inverse_forward_roundtrip(self, skewness, tailweight):
        """t(t.inv(y)) ≈ y."""
        t = SinhArcsinhTransform(
            skewness=jnp.float64(skewness),
            tailweight=jnp.float64(tailweight),
        )
        y = jnp.linspace(-3.0, 3.0, 50)
        assert jnp.allclose(t(t.inv(y)), y, atol=1e-10)

    def test_log_abs_det_jacobian_vs_grad(self):
        """ldj matches d/dx log|dy/dx| computed via jax.grad."""
        skewness = jnp.float64(0.5)
        tailweight = jnp.float64(1.3)
        t = SinhArcsinhTransform(skewness=skewness, tailweight=tailweight)

        x_vals = jnp.linspace(-2.0, 2.0, 20)

        # jax.grad of the forward pass gives dy/dx for scalar inputs
        grad_fn = jax.grad(lambda xi: t(xi))
        for xi in x_vals:
            ldj_analytic = t.log_abs_det_jacobian(xi, t(xi))
            dydx = grad_fn(xi)
            ldj_numerical = jnp.log(jnp.abs(dydx))
            assert jnp.allclose(ldj_analytic, ldj_numerical, atol=1e-10), (
                f"ldj mismatch at x={xi}: analytic={ldj_analytic}, "
                f"numerical={ldj_numerical}"
            )

    def test_jit_compatible(self):
        """Forward pass works under jax.jit."""
        t = SinhArcsinhTransform(skewness=jnp.float64(0.5), tailweight=jnp.float64(1.2))
        x = jnp.linspace(-2.0, 2.0, 10)
        y_eager = t(x)
        y_jit = jax.jit(t)(x)
        assert jnp.allclose(y_eager, y_jit, atol=1e-14)

    def test_gradient_flows(self):
        """Gradients w.r.t. skewness are finite and non-zero."""
        x = jnp.array([0.5, -0.3, 1.0])

        def f(skewness):
            t = SinhArcsinhTransform(skewness=skewness, tailweight=jnp.float64(1.2))
            return jnp.sum(t(x))

        g = jax.grad(f)(jnp.float64(0.5))
        assert jnp.isfinite(g)
        assert not jnp.allclose(g, 0.0)

    def test_tree_flatten_unflatten(self):
        """Roundtrip through JAX tree flatten/unflatten preserves params."""
        t = SinhArcsinhTransform(skewness=jnp.float64(0.3), tailweight=jnp.float64(1.5))
        leaves, treedef = jax.tree.flatten(t)
        t2 = jax.tree.unflatten(treedef, leaves)
        x = jnp.array([0.5, -1.0, 2.0])
        assert jnp.allclose(t(x), t2(x), atol=1e-14)

    def test_skewness_shifts_distribution(self):
        """Positive skewness should shift outputs negatively for x=0."""
        x = jnp.float64(0.0)
        t_pos = SinhArcsinhTransform(
            skewness=jnp.float64(1.0), tailweight=jnp.float64(1.0)
        )
        t_neg = SinhArcsinhTransform(
            skewness=jnp.float64(-1.0), tailweight=jnp.float64(1.0)
        )
        # sinh(-1) < 0 < sinh(1)
        assert t_pos(x) < 0.0
        assert t_neg(x) > 0.0


class TestFitNormalisingFlow:
    """Tests for fit_normalising_flow."""

    def test_returns_compose_transform(self):
        """Output is a ComposeTransform."""
        key = jr.PRNGKey(0)
        x = jnp.exp(jr.normal(key, shape=(100,))) + 1.0
        flow = fit_normalising_flow(x)
        assert isinstance(flow, ComposeTransform)

    def test_forward_inverse_roundtrip(self):
        """flow.inv(flow(x)) ≈ x on training data."""
        key = jr.PRNGKey(1)
        x = jnp.exp(jr.normal(key, shape=(100,))) + 1.0
        flow = fit_normalising_flow(x)
        x_rt = flow.inv(flow(x))
        assert jnp.allclose(x_rt, x, atol=1e-4)

    def test_approximate_normality(self):
        """Transformed log-normal data should have mean ≈ 0, std ≈ 1."""
        key = jr.PRNGKey(2)
        x = jnp.exp(0.5 * jr.normal(key, shape=(500,))) + 2.0
        flow = fit_normalising_flow(x)
        z = flow(x)
        assert jnp.abs(jnp.mean(z)) < 0.3, f"mean={jnp.mean(z)}"
        assert jnp.abs(jnp.std(z) - 1.0) < 0.3, f"std={jnp.std(z)}"

    def test_deterministic(self):
        """Same input produces the same flow."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f1 = fit_normalising_flow(x)
        f2 = fit_normalising_flow(x)
        z1 = f1(x)
        z2 = f2(x)
        assert jnp.allclose(z1, z2, atol=1e-10)

    def test_compose_has_four_parts(self):
        """The returned ComposeTransform has exactly 4 steps."""
        key = jr.PRNGKey(3)
        x = jnp.exp(jr.normal(key, shape=(50,))) + 1.0
        flow = fit_normalising_flow(x)
        assert len(flow.parts) == 4


class TestFitAllNormalisingFlows:
    """Tests for fit_all_normalising_flows."""

    def test_returns_correct_length(self):
        """Returns one flow per column."""
        key = jr.PRNGKey(4)
        X = jr.normal(key, shape=(50, 3)) + 5.0  # shift positive
        flows = fit_all_normalising_flows(X)
        assert len(flows) == 3

    def test_each_is_compose_transform(self):
        """Each element is a ComposeTransform."""
        key = jr.PRNGKey(5)
        X = jr.normal(key, shape=(50, 2)) + 5.0
        flows = fit_all_normalising_flows(X)
        for f in flows:
            assert isinstance(f, ComposeTransform)

    def test_columns_fitted_independently(self):
        """Fitting D columns together matches fitting each one separately."""
        key = jr.PRNGKey(6)
        X = jnp.exp(jr.normal(key, shape=(80, 2))) + 1.0
        flows_all = fit_all_normalising_flows(X)
        flow_0 = fit_normalising_flow(X[:, 0])
        flow_1 = fit_normalising_flow(X[:, 1])
        assert jnp.allclose(flows_all[0](X[:, 0]), flow_0(X[:, 0]), atol=1e-10)
        assert jnp.allclose(flows_all[1](X[:, 1]), flow_1(X[:, 1]), atol=1e-10)

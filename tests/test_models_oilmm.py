"""Tests for OILMM (Orthogonal Instantaneous Linear Mixing Model)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)


class TestOrthogonalMixingMatrix:
    """Tests for OrthogonalMixingMatrix parameter class."""

    def test_initialization(self):
        """Test basic initialization."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(42)
        mix = OrthogonalMixingMatrix(num_outputs=5, num_latent_gps=2, key=key)

        assert mix.num_outputs == 5
        assert mix.num_latent_gps == 2
        assert mix.U_latent.unwrap().shape == (5, 2)
        assert mix.S.unwrap().shape == (2,)
        assert mix.obs_noise_variance.unwrap().shape == ()
        assert mix.latent_noise_variance.unwrap().shape == (2,)

    def test_U_orthonormality(self):
        """Test that U has orthonormal columns via SVD."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(123)
        mix = OrthogonalMixingMatrix(num_outputs=10, num_latent_gps=3, key=key)

        U = mix.U
        UTU = U.T @ U
        expected = jnp.eye(3)

        assert jnp.allclose(UTU, expected, atol=1e-6)

    def test_H_matrix_shape_and_composition(self):
        """Test H = U S^(1/2) has correct shape and composition."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(456)
        p, m = 7, 3
        mix = OrthogonalMixingMatrix(num_outputs=p, num_latent_gps=m, key=key)

        H = mix.H
        assert H.shape == (p, m)

        # Check H = U * sqrt(S) (broadcasting)
        U = mix.U
        sqrt_S = jnp.sqrt(mix.S.unwrap())
        expected_H = U * sqrt_S[None, :]
        assert jnp.allclose(H, expected_H, atol=1e-10)

    def test_T_matrix_is_pseudo_inverse(self):
        """Test T = S^(-1/2) U^T is the left pseudo-inverse of H."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(789)
        mix = OrthogonalMixingMatrix(num_outputs=8, num_latent_gps=4, key=key)

        T = mix.T
        H = mix.H

        # T H should equal I_m (identity in latent space)
        TH = T @ H
        expected = jnp.eye(mix.num_latent_gps)
        assert jnp.allclose(TH, expected, atol=1e-6)

    def test_projected_noise_variance_diagonal(self):
        """Test projected noise is diagonal: sigma^2 S^(-1) + D."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix
        from gpjax.parameters import NonNegativeReal, PositiveReal

        key = jax.random.PRNGKey(101)
        mix = OrthogonalMixingMatrix(num_outputs=6, num_latent_gps=2, key=key)

        # Set specific noise values for testing using eqx.tree_at
        mix = eqx.tree_at(lambda m: m.obs_noise_variance, mix, PositiveReal(0.5))
        mix = eqx.tree_at(
            lambda m: m.latent_noise_variance,
            mix,
            NonNegativeReal(jnp.array([0.1, 0.2])),
        )
        mix = eqx.tree_at(lambda m: m.S, mix, PositiveReal(jnp.array([2.0, 4.0])))

        proj_noise = mix.projected_noise_variance

        # Expected: sigma^2/S + D = 0.5/[2.0, 4.0] + [0.1, 0.2]
        expected = jnp.array([0.5 / 2.0 + 0.1, 0.5 / 4.0 + 0.2])
        assert jnp.allclose(proj_noise, expected, atol=1e-10)


class TestOILMMModel:
    """Tests for OILMMModel."""

    def test_initialization(self):
        """Test OILMMModel initialization."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        assert model.num_outputs == 3
        assert model.num_latent_gps == 2
        assert len(model.latent_priors) == 2

    def test_project_observations(self):
        """Test projection: y_latent = T @ y."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(123)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=5,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Create test data
        N, P = 10, 5
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.sin(2 * jnp.pi * X) + jr.normal(key, (N, P)) * 0.1
        dataset = gpx.Dataset(X=X, y=y)

        # Project
        X_out, y_projected = model._project_observations(dataset)

        # Check shapes
        assert X_out.shape == (N, 1)
        assert y_projected.shape == (2, N)  # [M, N]

        # Verify projection: T @ y^T = y_latent
        T = model.mixing_matrix.T  # [M, P]
        expected = T @ y.T  # [M, P] @ [P, N] = [M, N]
        assert jnp.allclose(y_projected, expected, atol=1e-10)

    def test_condition(self):
        """Test conditioning creates M independent posteriors."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, OILMMPosterior

        key = jax.random.PRNGKey(456)
        kernel = gpx.kernels.Matern52()

        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Create training data
        N, P = 20, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        # Condition
        posterior = model.condition(dataset)

        # Check type
        assert isinstance(posterior, OILMMPosterior)
        assert posterior.num_latent_gps == 2
        assert len(posterior.latent_posteriors) == 2

    def test_conditioned_posteriors_use_correct_noise(self):
        """Test that each latent posterior gets correct projected noise."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel
        from gpjax.parameters import NonNegativeReal, PositiveReal

        key = jax.random.PRNGKey(789)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=6,
            num_latent_gps=3,
            kernel=kernel,
            key=key,
        )

        # Set known noise values using eqx.tree_at
        model = eqx.tree_at(
            lambda m: m.mixing_matrix.obs_noise_variance,
            model,
            PositiveReal(0.5),
        )
        model = eqx.tree_at(
            lambda m: m.mixing_matrix.latent_noise_variance,
            model,
            NonNegativeReal(jnp.array([0.1, 0.2, 0.3])),
        )
        model = eqx.tree_at(
            lambda m: m.mixing_matrix.S,
            model,
            PositiveReal(jnp.array([1.0, 2.0, 4.0])),
        )

        # Create data and condition
        N = 15
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, 6))
        dataset = gpx.Dataset(X=X, y=y)

        posterior = model.condition(dataset)

        # Verify each posterior has correct noise.
        # Gaussian likelihood wraps obs_stddev in NonNegativeReal, so
        # we use .unwrap() to get the raw array, then square to get variance.
        expected_noise_vars = model.mixing_matrix.projected_noise_variance
        for i in range(3):
            lik = posterior.latent_posteriors[i].likelihood
            # lik.obs_stddev is a NonNegativeReal -- get raw value
            obs_var = lik.obs_stddev.unwrap() ** 2
            expected = expected_noise_vars[i]
            assert jnp.allclose(obs_var, expected, atol=1e-6), (
                f"Latent GP {i}: expected noise var {expected}, got {obs_var}"
            )


class TestOILMMPosterior:
    """Tests for OILMMPosterior prediction."""

    def test_predict_mean_shape(self):
        """Test prediction mean has correct shape [NP]."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Create and condition on training data
        N_train, P = 20, 3
        X_train = jnp.linspace(0, 1, N_train).reshape(-1, 1)
        y_train = jr.normal(key, (N_train, P))
        train_data = gpx.Dataset(X=X_train, y=y_train)

        posterior = model.condition(train_data)

        # Predict at test points
        N_test = 10
        X_test = jnp.linspace(0.1, 0.9, N_test).reshape(-1, 1)
        pred = posterior.predict(X_test)

        # Check mean shape (blocked order, index p * N + n: [NP])
        assert pred.mean.shape == (N_test * P,)
        assert jnp.all(jnp.isfinite(pred.mean))

    def test_predict_mean_reconstruction(self):
        """Test mean is correctly reconstructed: f_mean = H @ latent_means."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(123)
        kernel = gpx.kernels.Matern52()

        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Condition
        N, P = 15, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)
        posterior = model.condition(dataset)

        # Predict
        X_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
        pred = posterior.predict(X_test)

        # Manually compute expected mean using each latent posterior + its dataset
        latent_means = jnp.array(
            [post(X_test).mean for post in posterior.latent_posteriors]
        )  # [M, N_test]
        H = model.mixing_matrix.H  # [P, M]
        expected_mean = jnp.einsum("pm,mn->pn", H, latent_means)  # [P, N_test]
        expected_mean_flat = expected_mean.ravel()  # [P * N_test], blocked

        assert jnp.allclose(pred.mean, expected_mean_flat, atol=1e-6)

    def test_predict_full_covariance_shape(self):
        """Test full covariance has shape [NP, NP]."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(456)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 12, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        posterior = model.condition(gpx.Dataset(X=X, y=y))

        X_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
        pred = posterior.predict(X_test, covariance="dense")

        expected_shape = (5 * P, 5 * P)
        assert pred.covariance().shape == expected_shape
        assert jnp.all(jnp.isfinite(pred.covariance()))

    def test_predict_covariance_is_psd(self):
        """Test covariance is positive semi-definite."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(789)
        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=gpx.kernels.Matern32(),
            key=key,
        )

        N, P = 10, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        posterior = model.condition(gpx.Dataset(X=X, y=y))

        X_test = jnp.linspace(0.1, 0.9, 8).reshape(-1, 1)
        pred = posterior.predict(X_test, covariance="dense")
        cov = pred.covariance()

        # Check PSD via eigenvalues
        eigvals = jnp.linalg.eigvalsh(cov)
        assert jnp.all(eigvals >= -1e-6), "Covariance not PSD"


class TestOILMMConstructors:
    """Tests for OILMM convenience constructors."""

    def test_create_oilmm(self):
        """Test basic create_oilmm constructor."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm

        key = jax.random.PRNGKey(42)
        model = create_oilmm(
            num_outputs=5,
            num_latent_gps=2,
            key=key,
            kernel=gpx.kernels.RBF(),
        )

        assert model.num_outputs == 5
        assert model.num_latent_gps == 2

    def test_create_oilmm_with_kernels(self):
        """Test constructor with custom kernels per latent."""
        import warnings

        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_with_kernels

        key = jax.random.PRNGKey(123)
        kernels = [gpx.kernels.RBF(), gpx.kernels.Matern52()]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = create_oilmm_with_kernels(
                latent_kernels=kernels,
                num_outputs=6,
                key=key,
            )

        assert model.num_outputs == 6
        assert model.num_latent_gps == 2
        # Verify each prior has the correct kernel
        assert isinstance(model.latent_priors[0].kernel, gpx.kernels.RBF)
        assert isinstance(model.latent_priors[1].kernel, gpx.kernels.Matern52)

    def test_create_oilmm_from_data(self):
        """Test data-informed initialization."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_from_data

        key = jax.random.PRNGKey(456)

        # Create dataset with structure
        N = 50
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        # Create correlated outputs
        base = jnp.sin(2 * jnp.pi * X.squeeze())
        y = jnp.column_stack(
            [
                base,
                base + jr.normal(key, (N,)) * 0.1,
                -base,
                jr.normal(key, (N,)),
            ]
        )
        dataset = gpx.Dataset(X=X, y=y)

        model = create_oilmm_from_data(
            dataset=dataset,
            num_latent_gps=2,
            key=key,
            kernel=gpx.kernels.RBF(),
        )

        assert model.num_outputs == 4
        assert model.num_latent_gps == 2

        # U should have orthonormal columns (from SVD projection)
        U = model.mixing_matrix.U
        assert U.shape == (4, 2)
        assert jnp.allclose(U.T @ U, jnp.eye(2), atol=1e-6)


class TestOILMMIntegration:
    """End-to-end integration tests for OILMM workflow."""

    def test_full_workflow_create_condition_predict(self):
        """Test complete OILMM workflow: create, condition, predict."""
        import gpjax as gpx

        key = jax.random.PRNGKey(42)

        # 1. Create model
        model = gpx.models.create_oilmm(
            num_outputs=3,
            num_latent_gps=2,
            key=key,
            kernel=gpx.kernels.RBF(),
        )

        # 2. Generate synthetic training data
        N_train, P = 30, 3
        X_train = jnp.linspace(0, 1, N_train).reshape(-1, 1)
        # Create correlated outputs
        base1 = jnp.sin(2 * jnp.pi * X_train.squeeze())
        base2 = jnp.cos(2 * jnp.pi * X_train.squeeze())
        y_train = jnp.column_stack(
            [
                base1 + 0.5 * base2,
                base2,
                -base1 + base2,
            ]
        )
        y_train += jr.normal(key, y_train.shape) * 0.05  # Add noise
        train_data = gpx.Dataset(X=X_train, y=y_train)

        # 3. Condition on observations
        posterior = model.condition(train_data)

        # Verify posterior type
        assert isinstance(posterior, gpx.models.OILMMPosterior)
        assert posterior.num_latent_gps == 2

        # 4. Predict at test points (train data stored internally)
        N_test = 10
        X_test = jnp.linspace(0.1, 0.9, N_test).reshape(-1, 1)
        pred = posterior.predict(X_test, covariance="dense")

        # 5. Verify prediction properties
        assert pred.mean.shape == (N_test * P,)
        assert pred.covariance().shape == (N_test * P, N_test * P)
        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))

        # Verify covariance is PSD
        eigvals = jnp.linalg.eigvalsh(pred.covariance())
        assert jnp.all(eigvals >= -1e-6)

    def test_oilmm_vs_independent_gps_sanity(self):
        """Sanity check: OILMM with m=p should behave reasonably."""
        import gpjax as gpx

        key = jax.random.PRNGKey(123)
        P = 2

        # OILMM with m=p (maximal latent GPs)
        model = gpx.models.create_oilmm(
            num_outputs=P,
            num_latent_gps=P,
            key=key,
            kernel=gpx.kernels.Matern52(),
        )

        # Simple data
        N = 20
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        # Condition and predict
        posterior = model.condition(dataset)
        pred = posterior.predict(X[:10])

        # Basic sanity: predictions should be finite and reasonable scale
        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.std(pred.mean) < 10.0  # Not exploding

    def test_jit_compatibility(self):
        """Test that OILMM prediction internals are JIT-compatible.

        GaussianDistribution is not a registered JAX pytree, so we JIT the
        computation and extract raw arrays inside the traced function.
        """
        import gpjax as gpx

        key = jax.random.PRNGKey(456)
        model = gpx.models.create_oilmm(
            num_outputs=4,
            num_latent_gps=2,
            key=key,
        )

        # Create data and condition
        N = 15
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, 4))
        dataset = gpx.Dataset(X=X, y=y)
        posterior = model.condition(dataset)

        # JIT a function that returns raw arrays (not GaussianDistribution)
        @jax.jit
        def predict_arrays(X_test):
            pred = posterior.predict(X_test)
            return pred.loc, pred.covariance()

        X_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)

        # Run JIT version
        mean_jit, cov_jit = predict_arrays(X_test)

        # Run non-JIT version
        pred_normal = posterior.predict(X_test)

        # Results should match
        assert jnp.allclose(mean_jit, pred_normal.mean, atol=1e-10)
        assert jnp.allclose(cov_jit, pred_normal.covariance(), atol=1e-10)


class TestOILMMMLL:
    """Tests for oilmm_mll log marginal likelihood."""

    def test_returns_finite_scalar(self):
        """Test MLL returns a finite scalar."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(42)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 20, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        mll = oilmm_mll(model, data)
        assert mll.shape == ()
        assert jnp.isfinite(mll)

    def test_correction_terms_nonzero(self):
        """Correction terms make MLL differ from naive latent sum."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(123)
        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 15, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        full_mll = oilmm_mll(model, data)

        # Compute just the sum of latent MLLs
        posterior = model.condition(data)
        naive_sum = sum(
            post.log_marginal_likelihood for post in posterior.latent_posteriors
        )

        # They should differ due to correction terms
        assert not jnp.allclose(full_mll, naive_sum, atol=1e-6)

    def test_gradient_flows(self):
        """Gradients flow through all OILMM parameters."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(456)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 10, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        # Use eqx.partition to split into differentiable and static parts
        params, static = eqx.partition(model, eqx.is_array)

        def loss_fn(params):
            m = eqx.combine(params, static)
            return -oilmm_mll(m, data)

        grads = jax.grad(loss_fn)(params)

        # Check no NaN gradients in any leaf
        flat_grads = jax.tree.leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g)), f"NaN gradient found: {g}"

    def test_small_scale_brute_force(self):
        """For tiny problem, verify against brute-force MOGP MLL."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll
        from gpjax.parameters import NonNegativeReal, PositiveReal

        key = jax.random.PRNGKey(789)
        n, p, m = 5, 3, 2

        model = OILMMModel(
            num_outputs=p,
            num_latent_gps=m,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        # Fix parameters for deterministic comparison using eqx.tree_at
        model = eqx.tree_at(
            lambda mod: mod.mixing_matrix.obs_noise_variance,
            model,
            PositiveReal(0.1),
        )
        model = eqx.tree_at(
            lambda mod: mod.mixing_matrix.latent_noise_variance,
            model,
            NonNegativeReal(jnp.zeros(m)),
        )
        model = eqx.tree_at(
            lambda mod: mod.mixing_matrix.S,
            model,
            PositiveReal(jnp.array([2.0, 1.5])),
        )

        X = jnp.linspace(0, 1, n).reshape(-1, 1)
        y = jr.normal(key, (n, p))
        data = gpx.Dataset(X=X, y=y)

        oilmm_val = oilmm_mll(model, data)

        # Brute-force: compute full NP x NP covariance
        H = model.mixing_matrix.H  # [P, M]
        sigma2 = model.mixing_matrix.obs_noise_variance.unwrap()

        # Compute each latent kernel matrix
        latent_Ks = []
        for i in range(m):
            prior = model.latent_priors[i]
            K = prior.kernel.cross_covariance(X, X)  # [N, N]
            latent_Ks.append(K)

        latent_K_block = jax.scipy.linalg.block_diag(*latent_Ks)  # [MN, MN]
        H_kron_I = jnp.kron(H, jnp.eye(n))  # [PN, MN]
        full_cov = H_kron_I @ latent_K_block @ H_kron_I.T + sigma2 * jnp.eye(n * p)

        # Evaluate log N(vec(Y) | 0, full_cov)
        y_vec = y.T.ravel()  # [P*N]: all obs for output 0, then output 1, etc.

        # log N(y | 0, C) = -0.5 * (y^T C^{-1} y + log|C| + NP log(2pi))
        L = jnp.linalg.cholesky(full_cov)
        alpha = jax.scipy.linalg.solve_triangular(L, y_vec, lower=True)
        brute_mll = (
            -0.5 * jnp.dot(alpha, alpha)
            - jnp.sum(jnp.log(jnp.diag(L)))
            - 0.5 * n * p * jnp.log(2.0 * jnp.pi)
        )

        assert jnp.allclose(oilmm_val, brute_mll, atol=1e-4), (
            f"OILMM MLL {oilmm_val} != brute-force {brute_mll}"
        )

    def test_deterministic(self):
        """oilmm_mll produces identical results on repeated calls."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(101)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 10, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        val1 = oilmm_mll(model, data)
        val2 = oilmm_mll(model, data)

        assert jnp.allclose(val1, val2, atol=1e-10)


class TestKernelIndependence:
    """Tests for independent kernel instances per latent GP."""

    def test_single_kernel_creates_independent_copies(self):
        """Single kernel is deep-copied so latents have independent params.

        With equinox modules (frozen), we verify that different latent priors
        have distinct kernel parameter objects, not that mutation propagates.
        """
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        kernel = gpx.kernels.RBF()
        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=3,
            kernel=kernel,
            key=key,
        )

        # Verify the kernels are independent copies by checking they are
        # different objects (deep copy means separate parameter instances)
        ls0 = model.latent_priors[0].kernel.lengthscale
        ls1 = model.latent_priors[1].kernel.lengthscale
        # They should start with the same value
        assert jnp.allclose(ls0.unwrap(), ls1.unwrap())

        # Modify latent_priors[0].kernel.lengthscale via eqx.tree_at
        new_ls = gpx.parameters.PositiveReal(99.0)
        new_priors = list(model.latent_priors)
        new_priors[0] = eqx.tree_at(
            lambda p: p.kernel.lengthscale, model.latent_priors[0], new_ls
        )
        model = eqx.tree_at(lambda m: m.latent_priors, model, tuple(new_priors))

        # latent_priors[1] should be unchanged
        assert jnp.allclose(
            model.latent_priors[1].kernel.lengthscale.unwrap(), ls1.unwrap()
        )
        assert not jnp.allclose(
            model.latent_priors[0].kernel.lengthscale.unwrap(),
            model.latent_priors[1].kernel.lengthscale.unwrap(),
        )

    def test_list_of_kernels_used_directly(self):
        """List of kernels are used as-is."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(123)
        kernels = [gpx.kernels.RBF(), gpx.kernels.Matern52()]

        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=kernels,
            key=key,
        )

        assert isinstance(model.latent_priors[0].kernel, gpx.kernels.RBF)
        assert isinstance(model.latent_priors[1].kernel, gpx.kernels.Matern52)

    def test_wrong_kernel_list_length_raises(self):
        """Passing wrong number of kernels raises ValueError."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel
        import pytest

        key = jax.random.PRNGKey(456)
        kernels = [gpx.kernels.RBF(), gpx.kernels.Matern52()]

        with pytest.raises(ValueError, match="Expected 3 kernels, got 2"):
            OILMMModel(
                num_outputs=4,
                num_latent_gps=3,  # Mismatch: 3 latents but 2 kernels
                kernel=kernels,
                key=key,
            )


class TestSInitialization:
    """Tests for eigenvalue initialization of S."""

    def test_create_from_data_initializes_model(self):
        """create_oilmm_from_data creates a model with correct dimensions."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_from_data

        key = jax.random.PRNGKey(42)

        N = 50
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        base = jnp.sin(2 * jnp.pi * X.squeeze())
        y = jnp.column_stack(
            [
                base * 3.0,
                base * 2.0 + jr.normal(key, (N,)) * 0.1,
                -base,
                jr.normal(key, (N,)) * 0.5,
            ]
        )
        dataset = gpx.Dataset(X=X, y=y)

        model = create_oilmm_from_data(dataset=dataset, num_latent_gps=2, key=key)

        # Verify model structure
        assert model.num_outputs == 4
        assert model.num_latent_gps == 2
        # S should be positive (initialized to ones by default)
        assert jnp.all(model.mixing_matrix.S.unwrap() > 0)

    def test_create_from_data_s_is_positive(self):
        """S values are always positive."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_from_data

        key = jax.random.PRNGKey(123)

        N = 50
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        # One output is near-constant (eigenvalue ~= 0)
        y = jnp.column_stack(
            [
                jnp.sin(X.squeeze()),
                jnp.ones(N) * 5.0,  # constant -> zero variance
                jnp.cos(X.squeeze()),
            ]
        )
        dataset = gpx.Dataset(X=X, y=y)

        model = create_oilmm_from_data(dataset=dataset, num_latent_gps=3, key=key)

        assert jnp.all(model.mixing_matrix.S.unwrap() > 0)


class TestCovarianceEquivalence:
    """Test that einsum covariance matches old Kronecker computation."""

    def test_full_and_diagonal_covariance_paths_are_consistent(self):
        """Full-covariance and diagonal-covariance paths should share indexing."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(7)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        X_train = jnp.linspace(0, 1, 10).reshape(-1, 1)
        y_train = jr.normal(key, (10, 3))
        dataset = gpx.Dataset(X=X_train, y=y_train)
        posterior = model.condition(dataset)

        X_test = jnp.linspace(0.1, 0.9, 6).reshape(-1, 1)
        pred_full = posterior.predict(X_test, covariance="dense")
        pred_diag = posterior.predict(X_test, covariance="diagonal")

        full_diag = jnp.diag(pred_full.covariance())
        diag_only = jnp.diag(pred_diag.covariance())

        assert jnp.allclose(pred_full.mean, pred_diag.mean, atol=1e-8)
        assert jnp.allclose(full_diag, diag_only, atol=1e-6)

    def test_einsum_cov_matches_kronecker_reference(self):
        """New einsum covariance matches old Kronecker computation."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 8, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        posterior = model.condition(dataset)

        # Get einsum result from current implementation
        X_test = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)
        pred = posterior.predict(X_test, covariance="dense")
        einsum_cov = pred.covariance()

        # Compute reference via old Kronecker method
        N_test = X_test.shape[0]
        H = model.mixing_matrix.H

        latent_preds = [post(X_test) for post in posterior.latent_posteriors]
        latent_covs = [pred.covariance() for pred in latent_preds]

        latent_cov_block = jax.scipy.linalg.block_diag(*latent_covs)
        H_kron_I = jnp.kron(H, jnp.eye(N_test))
        kron_cov = H_kron_I @ latent_cov_block @ H_kron_I.T

        # (H ⊗ I) gives blocked order ([p0 all t], [p1 all t], ...), the
        # order of predict().
        assert jnp.allclose(einsum_cov, kron_cov, atol=1e-6)


def test_create_oilmm_from_data_recovers_planted_subspace():
    """PCA init must recover the column space of a known low-rank mixing.
    Random init does not (this is why the no-op shipped undetected)."""
    from gpjax.dataset import Dataset
    from gpjax.models.oilmm import create_oilmm_from_data

    key = jr.key(0)
    N, P, M = 60, 4, 2
    X = jnp.linspace(0.0, 1.0, N).reshape(-1, 1)
    # Two latent signals mixed into P=4 outputs via a planted orthonormal basis.
    latent = jnp.column_stack(
        [jnp.sin(6.0 * X.squeeze()), jnp.cos(4.0 * X.squeeze())]
    )  # [N, M]
    planted, _ = jnp.linalg.qr(jr.normal(key, (P, M)))  # [P, M] orthonormal
    Y = latent @ planted.T  # [N, P], rank M
    data = Dataset(X=X, y=Y)

    model = create_oilmm_from_data(dataset=data, num_latent_gps=M, key=jr.key(1))
    U = model.mixing_matrix.U  # [P, M] orthonormal

    # Principal angles between the recovered and planted column spaces.
    cos_angles = jnp.linalg.svd(planted.T @ U, compute_uv=False)
    assert jnp.all(cos_angles > 0.99)  # cos-angles ~= 1.0


def test_create_oilmm_from_data_requires_two_points():
    """N==1 -> jnp.cov divides by 0 -> NaN params. Guard, don't ship NaN."""
    from gpjax.dataset import Dataset
    from gpjax.models.oilmm import create_oilmm_from_data

    data = Dataset(X=jnp.zeros((1, 1)), y=jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="2"):
        create_oilmm_from_data(dataset=data, num_latent_gps=2, key=jr.key(0))


def test_create_oilmm_from_data_two_points_finite():
    """N==2, m==2 is valid (clamp path); parameters must be finite."""
    from gpjax.dataset import Dataset
    from gpjax.models.oilmm import create_oilmm_from_data

    data = Dataset(X=jnp.array([[0.0], [1.0]]), y=jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    model = create_oilmm_from_data(dataset=data, num_latent_gps=2, key=jr.key(0))
    assert jnp.all(jnp.isfinite(model.mixing_matrix.U))
    from gpjax.models.oilmm import val

    assert jnp.all(jnp.isfinite(val(model.mixing_matrix.S)))


def test_oilmm_predict_diagonal_returns_diagonal_operator():
    """The diagonal predict branch must return a DiagonalLinearOperator, not
    a densified MatrixLinearOperator."""
    import gpjax as gpx
    from gpjax.models.oilmm import OILMMModel
    import lineax as lx

    key = jax.random.PRNGKey(99)
    model = OILMMModel(
        num_outputs=2,
        num_latent_gps=2,
        kernel=gpx.kernels.RBF(),
        key=key,
    )

    N, P = 8, 2
    X = jnp.linspace(0.0, 1.0, N).reshape(-1, 1)
    y = jr.normal(key, (N, P))
    dataset = gpx.Dataset(X=X, y=y)
    posterior = model.condition(dataset)

    X_test = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)
    pred_diag = posterior.predict(X_test, covariance="diagonal")

    assert isinstance(pred_diag.scale, lx.DiagonalLinearOperator), (
        f"Expected DiagonalLinearOperator, got {type(pred_diag.scale).__name__}"
    )
    # Entries must still match the full-cov diagonal
    pred_full = posterior.predict(X_test, covariance="dense")
    assert jnp.allclose(
        jnp.diag(pred_full.covariance()), jnp.diag(pred_diag.covariance()), atol=1e-6
    )


def test_oilmm_predict_default_covariance_is_diagonal():
    """The default covariance must be "diagonal", not "dense" (issue #682): the
    dense joint covariance is O(m n^2 p^2) and forfeits OILMM's O(mn^3 + nmp)
    scaling, so the cheap marginal-variance path must be the default rather
    than an opt-in."""
    import gpjax as gpx
    from gpjax.models.oilmm import OILMMModel
    import lineax as lx

    key = jax.random.PRNGKey(17)
    model = OILMMModel(
        num_outputs=3,
        num_latent_gps=2,
        kernel=gpx.kernels.RBF(),
        key=key,
    )

    N, P = 8, 3
    X = jnp.linspace(0.0, 1.0, N).reshape(-1, 1)
    y = jr.normal(key, (N, P))
    dataset = gpx.Dataset(X=X, y=y)
    posterior = model.condition(dataset)

    X_test = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)

    # Neither __call__ nor predict() takes a covariance kwarg here.
    pred_call = posterior(X_test)
    pred_predict = posterior.predict(X_test)

    for pred in (pred_call, pred_predict):
        assert isinstance(pred.scale, lx.DiagonalLinearOperator), (
            f"Expected default covariance to be diagonal, got "
            f"{type(pred.scale).__name__}"
        )

    # And it must agree with the explicit diagonal call.
    pred_explicit_diag = posterior.predict(X_test, covariance="diagonal")
    assert jnp.allclose(pred_predict.mean, pred_explicit_diag.mean, atol=1e-10)
    assert jnp.allclose(
        pred_predict.covariance(), pred_explicit_diag.covariance(), atol=1e-10
    )


# --- Conformance with the v1.0 conditioning contract ---------------------------
#
# OILMM joined the contract at v1.0: `model.condition(D)` (sugar `model | D`)
# returns a `gpjax.conditioning.Posterior`, which owns the evidence and caches
# each latent factorisation. These pin that it behaves like every other
# conditionable object in the library.


def _oilmm_fixture(num_data: int = 20):
    import gpjax as gpx
    from gpjax.models import create_oilmm

    inputs = jnp.linspace(0.0, 5.0, num_data).reshape(-1, 1)
    outputs = jnp.hstack([jnp.sin(inputs), jnp.cos(inputs), jnp.sin(2.0 * inputs)])
    data = gpx.Dataset(X=inputs, y=outputs)
    model = create_oilmm(
        num_outputs=3, num_latent_gps=2, kernel=gpx.kernels.RBF(), key=jr.key(0)
    )
    return model, data


def test_condition_returns_a_conditioning_posterior():
    from gpjax.conditioning import Posterior

    model, data = _oilmm_fixture()
    assert isinstance(model.condition(data), Posterior)


def test_or_operator_is_sugar_for_condition():
    model, data = _oilmm_fixture()
    test_inputs = data.X[:4]
    assert jnp.allclose(
        (model | data)(test_inputs).mean, model.condition(data)(test_inputs).mean
    )


def test_predict_is_sugar_over_condition():
    """CONTEXT.md: sugar is never a second implementation."""
    model, data = _oilmm_fixture()
    test_inputs = data.X[:4]
    posterior = model.condition(data)
    assert jnp.allclose(
        posterior.predict(test_inputs).mean, posterior(test_inputs).mean
    )


def test_evidence_lives_on_the_posterior():
    from gpjax.models import oilmm_mll

    model, data = _oilmm_fixture()
    assert jnp.allclose(
        model.condition(data).log_marginal_likelihood, oilmm_mll(model, data)
    )


@pytest.mark.parametrize("num_data", [12, 20])
def test_diagonal_covariance_matches_the_dense_diagonal(num_data):
    model, data = _oilmm_fixture(num_data)
    posterior = model.condition(data)
    test_inputs = data.X[:5]
    dense = posterior(test_inputs, covariance="dense")
    diagonal = posterior(test_inputs, covariance="diagonal")
    assert jnp.allclose(dense.mean, diagonal.mean, atol=1e-10)
    assert jnp.allclose(jnp.diag(dense.covariance()), diagonal.variance, atol=1e-8)


def test_posterior_survives_a_pytree_round_trip():
    model, data = _oilmm_fixture()
    posterior = model.condition(data)
    leaves, treedef = jax.tree_util.tree_flatten(posterior)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    test_inputs = data.X[:4]
    assert jnp.allclose(restored(test_inputs).mean, posterior(test_inputs).mean)
    assert jnp.allclose(
        restored.log_marginal_likelihood, posterior.log_marginal_likelihood
    )


def test_conditioning_is_jittable():
    model, data = _oilmm_fixture()
    evidence = jax.jit(lambda m, d: m.condition(d).log_marginal_likelihood)(model, data)
    assert jnp.isfinite(evidence)


def test_condition_on_observations_is_deprecated():
    model, data = _oilmm_fixture()
    with pytest.warns(DeprecationWarning, match="condition"):
        deprecated = model.condition_on_observations(data)
    assert jnp.allclose(
        deprecated.log_marginal_likelihood,
        model.condition(data).log_marginal_likelihood,
    )


def test_return_full_cov_is_deprecated():
    model, data = _oilmm_fixture()
    posterior = model.condition(data)
    test_inputs = data.X[:4]
    with pytest.warns(DeprecationWarning, match="covariance"):
        legacy = posterior.predict(test_inputs, return_full_cov=False)
    assert jnp.allclose(legacy.mean, posterior(test_inputs, covariance="diagonal").mean)

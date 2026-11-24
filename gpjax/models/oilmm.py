"""Orthogonal Instantaneous Linear Mixing Model (OILMM) for multi-output GPs.

OILMM achieves O(n^3 m) complexity instead of O(n^3 m^3) by constraining the mixing
matrix to have orthogonal columns, which causes the projected noise to be
diagonal and enables inference to decompose into m independent single-output
GP problems.

Reference:
    Bruinsma et al. (2020). "Scalable Exact Inference in Multi-Output Gaussian
    Processes." ICML.
"""

from __future__ import annotations

import copy
import typing as tp
import warnings

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
import lineax as lx

from gpjax.conditioning import Posterior
from gpjax.distributions import GaussianDistribution
from gpjax.parameters import NonNegativeReal, PositiveReal, Real, val
from gpjax.typing import ScalarFloat

if tp.TYPE_CHECKING:
    from gpjax.dataset import Dataset
    from gpjax.kernels.base import AbstractKernel


class OrthogonalMixingMatrix(eqx.Module):
    """Mixing matrix H = U S^(1/2) with orthogonal columns.

    Parameterizes an orthogonal mixing matrix for OILMM where:
    - U in R^(p x m) has orthonormal columns (U^T U = I_m)
    - S > 0 is a diagonal scaling matrix (m x m)
    - H = U S^(1/2) is the mixing matrix
    - T = S^(-1/2) U^T is the projection matrix

    The orthogonality of U ensures that the projected noise is diagonal::

        Sigma_T = T Sigma T^T = sigma^2 S^(-1) + D

    where sigma^2 is observation noise and D is latent noise.

    Attributes:
        num_outputs: Number of output dimensions (p)
        num_latent_gps: Number of latent GP functions (m)
        U_latent: Unconstrained matrix for SVD orthogonalization
        S: Positive diagonal scaling
        obs_noise_variance: Homogeneous observation noise (sigma^2)
        latent_noise_variance: Per-latent heterogeneous noise (D), non-negative
    """

    num_outputs: int = eqx.field(static=True)
    num_latent_gps: int = eqx.field(static=True)
    U_latent: Real
    S: PositiveReal
    obs_noise_variance: PositiveReal
    latent_noise_variance: NonNegativeReal

    def __init__(
        self,
        num_outputs: int,
        num_latent_gps: int,
        key: Array,
    ):
        """Initialize orthogonal mixing matrix.

        Args:
            num_outputs: Number of output dimensions (p)
            num_latent_gps: Number of latent GPs (m), must satisfy m <= p
            key: JAX PRNG key for initialization
        """
        if num_latent_gps > num_outputs:
            raise ValueError(
                f"num_latent_gps ({num_latent_gps}) must be <= "
                f"num_outputs ({num_outputs})"
            )

        self.num_outputs = num_outputs
        self.num_latent_gps = num_latent_gps

        # Unconstrained latent representation (small init for stability)
        self.U_latent = Real(jr.normal(key, (num_outputs, num_latent_gps)) * 0.1)

        # Scaling diagonal (init to 1)
        self.S = PositiveReal(jnp.ones(num_latent_gps))

        # Noise parameters
        # obs_noise_variance is strictly positive (sigma^2 > 0)
        self.obs_noise_variance = PositiveReal(jnp.array(1.0))
        # latent_noise_variance (D) can be zero -- use NonNegativeReal
        self.latent_noise_variance = NonNegativeReal(jnp.zeros(num_latent_gps))

    @property
    def U(self) -> Float[Array, "P M"]:
        """Orthonormal columns via SVD.

        Uses SVD to project U_latent onto the Stiefel manifold (orthonormal columns).
        This ensures U^T U = I_m exactly.
        """
        U_svd, _, Vt_svd = jnp.linalg.svd(val(self.U_latent), full_matrices=False)
        return U_svd @ Vt_svd

    @property
    def sqrt_S(self) -> Float[Array, " M"]:
        """Square root of S diagonal: S^(1/2)."""
        return jnp.sqrt(val(self.S))

    @property
    def inv_sqrt_S(self) -> Float[Array, " M"]:
        """Inverse square root of S diagonal: S^(-1/2)."""
        return 1.0 / jnp.sqrt(val(self.S))

    @property
    def H(self) -> Float[Array, "P M"]:
        """Mixing matrix H = U S^(1/2).

        Maps from latent space (m dimensions) to output space (p dimensions).
        Each column is an orthogonal basis vector scaled by sqrt(S_i).
        """
        return self.U * self.sqrt_S[None, :]

    @property
    def T(self) -> Float[Array, "M P"]:
        """Projection matrix T = S^(-1/2) U^T.

        Projects from output space (p dimensions) to latent space (m dimensions).
        This is the left pseudo-inverse of H: T @ H = I_m.
        """
        return self.inv_sqrt_S[:, None] * self.U.T

    @property
    def H_squared(self) -> Float[Array, "P M"]:
        """Element-wise H^2 for fast diagonal variance reconstruction.

        When computing marginal variances, we need H^2 @ latent_vars::

            var_p = sum_m H^2_pm * var_m

        This property caches H^2 to avoid recomputation.
        """
        return self.H**2

    @property
    def projected_noise_variance(self) -> Float[Array, " M"]:
        """Diagonal projected noise: Sigma_T = sigma^2 S^(-1) + D.

        This is the noise variance for each independent latent GP after projection.
        The orthogonality of U ensures this is diagonal, which is what makes
        OILMM tractable.

        Returns:
            Array of shape [M] with noise variance for each latent GP.
        """
        return val(self.obs_noise_variance) * self.inv_sqrt_S**2 + val(
            self.latent_noise_variance
        )


class OILMMModel(eqx.Module):
    """Orthogonal Instantaneous Linear Mixing Model.

    OILMM decomposes multi-output GP inference into M independent single-output
    GP problems by using an orthogonal mixing matrix. This achieves O(n^3 m)
    complexity instead of O(n^3 m^3).

    The generative model is::

        x_i ~ GP(0, K(t,t'))          for i=1..M (latent GPs)
        f(t) = H x(t)                  (mixing)
        y | f ~ N(f(t), Sigma)         (noise: Sigma = sigma^2 I + H D H^T)

    The orthogonality constraint (U^T U = I) ensures the projected noise is diagonal::

        Sigma_T = T Sigma T^T = sigma^2 S^(-1) + D

    enabling independent inference for each latent GP.

    Attributes:
        num_outputs: Number of output dimensions (p)
        num_latent_gps: Number of latent GPs (m)
        mixing_matrix: OrthogonalMixingMatrix containing H, T, noise params
        latent_priors: Tuple of M independent Prior objects
    """

    num_outputs: int = eqx.field(static=True)
    num_latent_gps: int = eqx.field(static=True)
    mixing_matrix: OrthogonalMixingMatrix
    latent_priors: tuple

    def __init__(
        self,
        num_outputs: int,
        num_latent_gps: int,
        kernel: AbstractKernel | list[AbstractKernel],
        key: Array,
        mean_function: tp.Any = None,
    ):
        """Initialize OILMM model.

        Args:
            num_outputs: Number of output dimensions (p)
            num_latent_gps: Number of latent GPs (m), must satisfy m <= p
            kernel: Kernel for latent GPs. If a single kernel, it is deep-copied
                M times so each latent GP has independent hyperparameters. If a
                list of M kernels, each is used directly.
            key: JAX PRNG key
            mean_function: Mean function for latent GPs (default: Zero)
        """
        from gpjax.gps import Prior
        from gpjax.mean_functions import Zero

        self.num_outputs = num_outputs
        self.num_latent_gps = num_latent_gps

        # Orthogonal mixing matrix
        key, subkey = jr.split(key)
        self.mixing_matrix = OrthogonalMixingMatrix(
            num_outputs=num_outputs,
            num_latent_gps=num_latent_gps,
            key=subkey,
        )

        # Mean function (shared across latents)
        if mean_function is None:
            mean_function = Zero()

        # Build per-latent kernel list
        if isinstance(kernel, list):
            if len(kernel) != num_latent_gps:
                raise ValueError(
                    f"Expected {num_latent_gps} kernels, got {len(kernel)}"
                )
            kernels = kernel
        else:
            kernels = [copy.deepcopy(kernel) for _ in range(num_latent_gps)]

        self.latent_priors = tuple(
            Prior(kernel=k, mean_function=mean_function) for k in kernels
        )

    def _project_observations(
        self, dataset: Dataset
    ) -> tuple[Float[Array, "N D"], Float[Array, "M N"]]:
        """Project observations to latent space: y_latent = T @ y.

        This is the first phase of OILMM inference. The projection is cheap (O(nmp))
        and transforms the multi-output problem into M single-output problems.

        Args:
            dataset: Training data with X [N, D] and y [N, P]

        Returns:
            Tuple of (X, y_projected) where:
                - X: Input locations [N, D] (unchanged)
                - y_projected: Projected observations [M, N]
        """
        T = self.mixing_matrix.T  # [M, P]
        y_projected = T @ dataset.y.T  # [M, P] @ [P, N] = [M, N]
        return dataset.X, y_projected

    def condition(self, train_data: Dataset) -> OILMMPosterior:
        r"""Condition the model on data, returning the conditioned process.

        Projects the observations into latent space and conditions the ``M``
        independent latent GPs, caching each factorisation on the returned
        :class:`OILMMPosterior`. Operator sugar: ``model | train_data``.

        Args:
            train_data: Training data with ``X`` of shape ``(N, D)`` and ``y``
                of shape ``(N, P)``.

        Returns:
            OILMMPosterior: The conditioned OILMM process.
        """
        return OILMMPosterior(self, train_data)

    def __or__(self, train_data: Dataset) -> OILMMPosterior:
        r"""Operator sugar for :meth:`condition`: ``model | train_data``."""
        return self.condition(train_data)

    def condition_on_observations(self, dataset: Dataset) -> OILMMPosterior:
        """Deprecated alias for :meth:`condition`.

        Args:
            dataset: Training data with ``X`` of shape ``(N, D)`` and ``y`` of
                shape ``(N, P)``.

        Returns:
            OILMMPosterior: The conditioned OILMM process.
        """
        warnings.warn(
            "OILMMModel.condition_on_observations is deprecated; use "
            "model.condition(train_data) (or model | train_data), which is "
            "the conditioning interface shared by every GPJax model.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.condition(dataset)


def _projection_correction(model: OILMMModel, data: Dataset) -> ScalarFloat:
    r"""The OILMM evidence correction, Prop. 9 of Bruinsma et al. (2020).

    The projection :math:`T` discards the :math:`p - m` output dimensions
    orthogonal to the mixing matrix. These terms restore them, so that the
    latent log-likelihoods sum to the evidence of the full multi-output model
    rather than of the projected one.

    Args:
        model: The OILMM whose mixing matrix defines the projection.
        data: Training data with ``X`` of shape ``(N, D)`` and ``y`` of shape
            ``(N, P)``.

    Returns:
        ScalarFloat: The additive correction to the latent log-likelihoods.
    """
    num_data = data.n
    num_outputs = model.num_outputs
    num_latents = model.num_latent_gps
    mix = model.mixing_matrix

    U = mix.U  # [P, M]
    S = val(mix.S)  # [M]
    sigma2 = val(mix.obs_noise_variance)  # scalar

    # -(n/2) log|S|, with |S| = prod(S_i).
    term_log_S = -0.5 * num_data * jnp.sum(jnp.log(S))

    # -n(p - m)/2 log(2 pi sigma^2).
    term_noise = (
        -0.5 * num_data * (num_outputs - num_latents) * jnp.log(2.0 * jnp.pi * sigma2)
    )

    # -(1/(2 sigma^2)) ||(I_p - U U^T) Y||_F^2, computed without forming the
    # P x P projector.
    Y = data.y  # [N, P]
    UtY = U.T @ Y.T  # [M, N]
    residual = Y.T - U @ UtY  # [P, N]
    term_residual = -0.5 * jnp.sum(residual**2) / sigma2

    return term_log_S + term_noise + term_residual


class OILMMPosterior(Posterior):
    r"""The conditioned OILMM process.

    OILMM's orthogonal mixing matrix decouples a :math:`P`-output problem into
    :math:`M` independent single-output problems, so conditioning it is
    conditioning each latent GP on its projected observations. This object
    holds the resulting :math:`M` :class:`~gpjax.conditioning.ExactPosterior`
    factorisations and reconstructs predictions in output space on demand::

        posterior = model.condition(train_data)   # or: model | train_data
        predictive = posterior(test_inputs)

    Like every conditioned process in GPJax, the factorisations are computed
    once, at ``condition`` time, and cached here; each query is a view of them.

    Attributes:
        latent_posteriors: The ``M`` conditioned latent processes.
        mixing_matrix: The orthogonal mixing matrix used for reconstruction.
        evidence_correction: The Prop. 9 projection correction, cached so that
            ``log_marginal_likelihood`` is a view rather than a recomputation.
        num_outputs: The number of outputs, :math:`P`.
        num_latent_gps: The number of latent GPs, :math:`M`.
    """

    latent_posteriors: tuple
    mixing_matrix: OrthogonalMixingMatrix
    evidence_correction: ScalarFloat
    num_outputs: int = eqx.field(static=True)
    num_latent_gps: int = eqx.field(static=True)

    def __init__(self, model: OILMMModel, train_data: Dataset):
        r"""Condition an OILMM on data.

        Args:
            model: The model to condition.
            train_data: Training data with ``X`` of shape ``(N, D)`` and ``y``
                of shape ``(N, P)``.
        """
        from gpjax.conditioning import ExactPosterior
        from gpjax.dataset import Dataset as _Dataset
        from gpjax.likelihoods import Gaussian

        # Project the observations into latent space (O(nmp)), then condition
        # each latent GP on its own column. A Python loop rather than vmap:
        # each latent prior is an eqx.Module with independent state.
        inputs, projected_outputs = model._project_observations(train_data)
        projected_noise_vars = model.mixing_matrix.projected_noise_variance

        self.latent_posteriors = tuple(
            ExactPosterior(
                model.latent_priors[index],
                Gaussian(obs_stddev=jnp.sqrt(projected_noise_vars[index])),
                _Dataset(X=inputs, y=projected_outputs[index][:, None]),
            )
            for index in range(model.num_latent_gps)
        )
        self.mixing_matrix = model.mixing_matrix
        self.evidence_correction = _projection_correction(model, train_data)
        self.num_outputs = model.num_outputs
        self.num_latent_gps = model.num_latent_gps

    def __call__(
        self,
        test_inputs: Float[Array, "N D"],
        *,
        covariance: tp.Literal["dense", "diagonal"] = "diagonal",
    ) -> GaussianDistribution:
        r"""Evaluate the conditioned OILMM at the given test inputs.

        Each latent process is queried independently and mixed back into output
        space. Let :math:`\mu_m` and :math:`\Sigma_m` be the mean vector and the
        :math:`N \times N` covariance of latent process :math:`m` at the test
        inputs, and let :math:`h_m` be column :math:`m` of the mixing matrix
        :math:`H`.

        The flattened vector is interleaved (input-major): the value for test
        input :math:`n` and output :math:`p` is at index :math:`nP + p`. All
        :math:`P` outputs for test input 0 come first, then all :math:`P`
        outputs for test input 1, and so on. Thus ``mean.reshape(N, P)`` gives
        the output-space mean with one row for each test input. This is
        different from ``ExactPosterior`` with a multi-output likelihood and
        from the ICM and LCM kernels, which put all :math:`N` values for one
        output first (index :math:`pN + n`).

        In this ordering, the mean is :math:`\operatorname{vec}(H \Lambda)`,
        where :math:`\Lambda` is the :math:`M \times N` matrix whose rows are
        the :math:`\mu_m` and :math:`\operatorname{vec}` stacks the columns of
        the :math:`P \times N` matrix :math:`H \Lambda`. The covariance
        is :math:`\sum_{m=1}^{M} \Sigma_m \otimes h_m h_m^{\top}`, which is
        :math:`(I_N \otimes H) \tilde{\Sigma} (I_N \otimes H)^{\top}`, where
        :math:`\tilde{\Sigma}` is the joint latent covariance in the same
        interleaved ordering. The marginal variance at index :math:`nP + p` is
        :math:`\sum_{m} H_{pm}^2 \, [\Sigma_m]_{nn}`.

        Note that for this multi-output process ``covariance="dense"`` returns
        the joint :math:`(NP, NP)` covariance across both test inputs and
        outputs, in the same interleaved ordering as the mean; single-output
        processes return :math:`(N, N)`. ``covariance="diagonal"`` returns the
        :math:`NP` marginal variances, and asks the same of each latent
        process, so the dense latent covariances are never formed.

        The default is ``"diagonal"``: forming the joint covariance costs
        :math:`O(m n^2 p^2)` (an :math:`np \times np` matrix from :math:`m`
        :math:`n \times n` latent covariances), which forfeits the
        :math:`O(mn^3 + nmp)` scaling OILMM exists for. Marginal variances are
        the common query and stay on the cheap path; pass
        ``covariance="dense"`` to opt into the joint covariance explicitly.

        Args:
            test_inputs: Input locations of shape ``(N, D)``.
            covariance: Whether to return the dense joint covariance over test
                inputs and outputs, or only the marginal variances.

        Returns:
            GaussianDistribution: The predictive distribution, with ``loc`` of
            shape ``(NP,)`` in interleaved ordering (index :math:`nP + p`).
        """
        num_test = test_inputs.shape[0]
        mixing = self.mixing_matrix.H  # [P, M]

        latent_predictives = [
            latent(test_inputs, covariance=covariance)
            for latent in self.latent_posteriors
        ]
        latent_means = jnp.stack(
            [predictive.mean for predictive in latent_predictives]
        )  # [M, N]

        mixed_mean = jnp.einsum("pm,mn->pn", mixing, latent_means)  # [P, N]
        mixed_mean_flat = mixed_mean.T.ravel()  # [NP], index n * P + p

        if covariance == "dense":
            # Cov[p1, p2] = sum_m H[p1, m] H[p2, m] Sigma_latent_m.
            latent_covariances = jnp.stack(
                [predictive.covariance() for predictive in latent_predictives]
            )  # [M, N, N]
            blocks = jnp.einsum(
                "pm,qm,mij->pqij", mixing, mixing, latent_covariances
            )  # [P, P, N, N]
            # Reorder to [N, P, N, P] so that flattening gives the interleaved
            # ordering of the mean (index n * P + p).
            size = num_test * self.num_outputs
            scale = lx.MatrixLinearOperator(
                blocks.transpose(2, 0, 3, 1).reshape(size, size)
            )
        else:
            latent_variances = jnp.stack(
                [predictive.variance for predictive in latent_predictives]
            )  # [M, N]
            mixed_variances = jnp.einsum(
                "pm,mn->pn", self.mixing_matrix.H_squared, latent_variances
            )  # [P, N]
            # [N, P] then flatten: index n * P + p, the same as the mean.
            scale = lx.DiagonalLinearOperator(mixed_variances.T.ravel())

        return GaussianDistribution(
            loc=jnp.atleast_1d(mixed_mean_flat.squeeze()),
            scale=scale,
        )

    @property
    def log_marginal_likelihood(self) -> ScalarFloat:
        r"""The evidence :math:`\log p(Y)`, Prop. 9 of Bruinsma et al. (2020).

        The sum of the latent processes' log marginal likelihoods plus the
        projection correction, both cached at ``condition`` time.
        """
        latent_evidence = jnp.stack(
            [latent.log_marginal_likelihood for latent in self.latent_posteriors]
        )
        return self.evidence_correction + jnp.sum(latent_evidence)

    def predict(
        self,
        test_inputs: Float[Array, "N D"],
        train_data: Dataset | None = None,
        *,
        covariance: tp.Literal["dense", "diagonal"] = "diagonal",
        return_full_cov: bool | None = None,
    ) -> GaussianDistribution:
        r"""Sugar for calling the posterior: ``predict(t) == self(t)``.

        Defaults to ``covariance="diagonal"``: see :meth:`__call__` for why.

        Args:
            test_inputs: Input locations of shape ``(N, D)``.
            train_data: Accepted and ignored — this process is already
                conditioned on its training set.
            covariance: Whether to return the dense joint covariance or only
                the marginal variances.
            return_full_cov: Deprecated. ``True`` maps to ``covariance="dense"``
                and ``False`` to ``covariance="diagonal"``.

        Returns:
            GaussianDistribution: The predictive distribution at the inputs.
        """
        del train_data
        if return_full_cov is not None:
            warnings.warn(
                "OILMMPosterior.predict(return_full_cov=...) is deprecated; "
                'pass covariance="dense" or covariance="diagonal" instead, as '
                "every other GPJax process does.",
                DeprecationWarning,
                stacklevel=2,
            )
            covariance = "dense" if return_full_cov else "diagonal"
        return self(test_inputs, covariance=covariance)


def oilmm_mll(model: OILMMModel, data: Dataset) -> ScalarFloat:
    """Log marginal likelihood for the OILMM.

    Implements Prop. 9 from Bruinsma et al. (2020):

        log p(Y) = correction_terms + sum_i log N((TY)_i | 0, K_i + noise_i I_n)

    The correction terms prevent the projection from collapsing and account
    for data in the (p - m) dimensions orthogonal to the mixing matrix.

    Like the other objectives, this is a one-line view of the conditioned
    process: the evidence is owned by :class:`OILMMPosterior` and computed once
    when the model is conditioned.

    Args:
        model: OILMMModel with parameters to evaluate.
        data: Training data with X [N, D] and y [N, P].

    Returns:
        Scalar log marginal likelihood.
    """
    return model.condition(data).log_marginal_likelihood


# Convenience constructors


def create_oilmm(
    num_outputs: int,
    num_latent_gps: int,
    key: Array,
    kernel: AbstractKernel | list[AbstractKernel] | None = None,
    mean_function: tp.Any = None,
) -> OILMMModel:
    """Create OILMM model with shared kernel across latents.

    Args:
        num_outputs: Number of output dimensions (p)
        num_latent_gps: Number of latent GPs (m)
        key: JAX PRNG key
        kernel: Kernel for latent GPs (default: RBF)
        mean_function: Mean function for latent GPs (default: Zero)

    Returns:
        Initialized OILMMModel

    Example:
        >>> import gpjax as gpx
        >>> import jax.random as jr
        >>> model = gpx.models.create_oilmm(
        ...     num_outputs=5,
        ...     num_latent_gps=2,
        ...     key=jr.key(42),
        ...     kernel=gpx.kernels.Matern52()
        ... )
    """
    from gpjax.kernels.stationary import RBF

    if kernel is None:
        kernel = RBF()

    return OILMMModel(
        num_outputs=num_outputs,
        num_latent_gps=num_latent_gps,
        kernel=kernel,
        key=key,
        mean_function=mean_function,
    )


def create_oilmm_with_kernels(
    latent_kernels: list[AbstractKernel],
    num_outputs: int,
    key: Array,
    mean_function: tp.Any = None,
) -> OILMMModel:
    """Create OILMM with custom kernel per latent GP.

    Args:
        latent_kernels: List of M kernels, one per latent GP
        num_outputs: Number of output dimensions (p)
        key: JAX PRNG key
        mean_function: Mean function (shared, default: Zero)

    Returns:
        OILMMModel with heterogeneous latent kernels

    Example:
        >>> import gpjax as gpx
        >>> import jax.random as jr
        >>> model = gpx.models.create_oilmm_with_kernels(
        ...     latent_kernels=[gpx.kernels.RBF(), gpx.kernels.Matern52()],
        ...     num_outputs=6,
        ...     key=jr.key(42)
        ... )
    """
    import warnings

    warnings.warn(
        "create_oilmm_with_kernels is deprecated. Pass a list of kernels "
        "directly to OILMMModel or create_oilmm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return OILMMModel(
        num_outputs=num_outputs,
        num_latent_gps=len(latent_kernels),
        kernel=latent_kernels,
        key=key,
        mean_function=mean_function,
    )


def create_oilmm_from_data(
    dataset: Dataset,
    num_latent_gps: int,
    key: Array,
    kernel: AbstractKernel = None,
    mean_function: tp.Any = None,
) -> OILMMModel:
    """Create OILMM with data-informed initialization of mixing matrix.

    Initializes U to the top M eigenvectors and S to the top M eigenvalues of
    the empirical covariance matrix of the outputs. Near-zero eigenvalues are
    clamped to 1e-6 for numerical stability. This can provide better
    initialization than random, especially when outputs have clear correlation
    structure.

    Args:
        dataset: Training data with y [N, P]
        num_latent_gps: Number of latent GPs (m)
        key: JAX PRNG key
        kernel: Kernel for latent GPs (default: RBF)
        mean_function: Mean function (default: Zero)

    Returns:
        OILMMModel with U initialized to top M eigenvectors and S to
        top M eigenvalues

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> X = jnp.linspace(0, 1, 50).reshape(-1, 1)
        >>> y = jnp.column_stack([jnp.sin(X), jnp.cos(X)])
        >>> data = gpx.Dataset(X=X, y=y)
        >>> model = gpx.models.create_oilmm_from_data(
        ...     dataset=data,
        ...     num_latent_gps=2,
        ...     key=jr.key(42)
        ... )
    """
    from gpjax.kernels.stationary import RBF

    num_outputs = dataset.y.shape[1]

    if kernel is None:
        kernel = RBF()

    # Create base model
    model = OILMMModel(
        num_outputs=num_outputs,
        num_latent_gps=num_latent_gps,
        kernel=kernel,
        key=key,
        mean_function=mean_function,
    )

    if dataset.n < 2:  # jnp.cov divides by N-1; N==1 -> all-NaN covariance
        raise ValueError(
            "create_oilmm_from_data needs >=2 data points to estimate the "
            "output covariance for PCA initialisation; got "
            f"N={dataset.n}. Use OILMMModel/create_oilmm for smaller data."
        )

    Y = dataset.y  # [N, P]
    output_cov = jnp.cov(Y, rowvar=False)  # column-centred empirical cov [P, P]
    eigvals, eigvecs = jnp.linalg.eigh(output_cov)  # ascending
    top_eigvecs = eigvecs[:, ::-1][:, :num_latent_gps]  # [P, M]
    top_eigvals = jnp.clip(eigvals[::-1][:num_latent_gps], min=1e-6)  # [M]

    model = eqx.tree_at(
        lambda m: (m.mixing_matrix.U_latent, m.mixing_matrix.S),
        model,
        (Real(top_eigvecs), PositiveReal(top_eigvals)),
    )
    return model

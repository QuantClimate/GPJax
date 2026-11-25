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

import abc
from dataclasses import dataclass
import typing as tp

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import (
    Float,
    Int,
)
import lineax as lx

from gpjax.conditioning import (
    CollapsedPosterior,
    Posterior,
    SparsePosterior,
)
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    HeteroscedasticModel,
    JointModel,
    Prior,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractHeteroscedasticLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.linalg.utils import (
    add_jitter,
    stabilised_cholesky,
)
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import (
    LowerTriangular,
    Real,
    val,
)
from gpjax.summary import _SummaryMixin
from gpjax.typing import (
    Array,
    ScalarFloat,
)

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", Gaussian, NonGaussian)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)
HL = tp.TypeVar("HL", bound=AbstractHeteroscedasticLikelihood)
P = tp.TypeVar("P", bound=Prior)
PP = tp.TypeVar("PP", bound=JointModel)
HP = tp.TypeVar("HP", bound=HeteroscedasticModel)


def _tri_solve(
    factor: Float[Array, "M M"], rhs: Float[Array, "M N"]
) -> Float[Array, "M N"]:
    """Solve ``factor @ x = rhs`` for a lower-triangular ``factor``.

    The right-hand side may carry several columns; they are solved in one
    BLAS-3 call.

    Args:
        factor (Float[Array, "M M"]): A lower-triangular matrix.
        rhs (Float[Array, "M N"]): The right-hand side, one system per column.

    Returns:
        Float[Array, "M N"]: The solution $x$ of ``factor @ x = rhs``.
    """
    return jsp.linalg.solve_triangular(factor, rhs, lower=True)


def _symmetrise(matrix: Float[Array, "M M"]) -> Float[Array, "M M"]:
    """Return ``(matrix + matrix.T) / 2``.

    Args:
        matrix (Float[Array, "M M"]): A square matrix.

    Returns:
        Float[Array, "M M"]: The symmetric part of ``matrix``.
    """
    return 0.5 * (matrix + matrix.T)


class AbstractVariationalFamily(_SummaryMixin, eqx.Module, tp.Generic[L]):
    r"""
    Abstract base class used to represent families of distributions that can be
    used within variational inference.

    A variational family is a trainable approximate posterior over inducing
    values: it is to sparse GPs what :class:`~gpjax.gps.JointModel` is to
    exact ones. Conditioning an already-fit family yields a
    :class:`~gpjax.conditioning.Posterior` like any other.
    """

    model: JointModel

    def __init__(self, model: JointModel):
        self.model = model

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Evaluate the variational family's density.

        For a given set of parameters, compute the latent function's prediction
        under the variational approximation.

        Args:
            *args (Any): Arguments of the variational family's `predict` method.
            **kwargs (Any): Keyword arguments of the variational family's `predict`
                method.

        Returns:
            GaussianDistribution: The output of the variational family's `predict` method.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Predict the GP's output given the input.

        Args:
            *args (Any): Arguments of the variational family's ``predict``
                method.
            **kwargs (Any): Keyword arguments of the variational family's
                ``predict`` method.

        Returns:
            GaussianDistribution: The output of the variational family's ``predict`` method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prior_kl(self, *args: tp.Any, **kwargs: tp.Any) -> ScalarFloat:
        r"""The KL divergence from the variational distribution to the prior.

        Every ELBO-style objective subtracts this term, so each concrete
        family must provide it.

        Returns:
            ScalarFloat: The KL divergence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def condition(self, train_data: Dataset) -> Posterior:
        r"""Condition the family, yielding its posterior process.

        The signature is the one every conditionable object in GPJax shares:
        ``condition(train_data)``, with ``model | D`` as sugar. Families
        whose approximate posterior is already carried internally accept
        ``train_data`` for that uniformity and ignore it; the collapsed
        family, whose optimal $q(u)$ is solved from the data, consumes it.

        Args:
            train_data (Dataset): The training data to condition on.

        Returns:
            Posterior: The conditioned process, queried as
                ``q.condition(train_data)(test_inputs)``.
        """
        raise NotImplementedError

    def __or__(self, train_data: Dataset) -> Posterior:
        r"""Sugar for conditioning: ``q | D`` reads as $q(f \mid \mathcal{D})$."""
        return self.condition(train_data)


class AbstractVariationalGaussian(AbstractVariationalFamily[L]):
    r"""The variational Gaussian family of probability distributions."""

    inducing_inputs: tp.Any

    def __init__(
        self,
        model: JointModel,
        inducing_inputs: Int[Array, "N D"] | Float[Array, "N D"] | Real,
    ):
        if not isinstance(inducing_inputs, Real):
            inducing_inputs = Real(inducing_inputs)

        self.inducing_inputs = inducing_inputs

        super().__init__(model)

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return val(self.inducing_inputs).shape[0]

    def _fmt_Kzt_Ktt(self, Kzt, Ktt):
        """Adapt the cross- and test-covariances before they enter ``predict``.

        An identity pass-through for Euclidean inducing inputs. Subclasses over
        non-Euclidean index sets -- :class:`GraphVariationalGaussian` is the only one
        today -- override it to densify and to restore the second axis that a
        single-node query drops.
        """
        return Kzt, Ktt

    def _fmt_inducing_inputs(self):
        """Return the inducing inputs in the form the kernel expects.

        An identity pass-through except for families whose inducing inputs are node
        indices rather than continuous coordinates.
        """
        return val(self.inducing_inputs)


class VariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$, where
    $u = f(z)$ are the function values at the inducing inputs $z$
    and the distribution over the inducing inputs is
    $q(u) = \mathcal{N}(\mu, S)$.  We parameterise this over
    $\mu$ and $sqrt$ with $S = sqrt sqrt^{\top}$.
    """

    variational_mean: tp.Any
    variational_root_covariance: tp.Any

    def __init__(
        self,
        model: JointModel,
        inducing_inputs: Int[Array, "N D"] | Float[Array, "N D"],
        variational_mean: Float[Array, "N 1"] | None = None,
        variational_root_covariance: Float[Array, "N N"] | None = None,
    ):
        super().__init__(model, inducing_inputs)

        if variational_mean is None:
            variational_mean = jnp.zeros((self.num_inducing, 1))

        if variational_root_covariance is None:
            variational_root_covariance = jnp.eye(self.num_inducing)

        self.variational_mean = Real(variational_mean)
        self.variational_root_covariance = LowerTriangular(variational_root_covariance)

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the prior KL divergence.

        Compute the KL-divergence between our variational approximation and the
        Gaussian process prior.

        For this variational family, we have

        .. math::

            \begin{aligned}
            \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
            & = \operatorname{KL}[ \mathcal{N}(\mu, S) \mid\mid N(\mu z, \mathbf{K}_{zz}) ],
            \end{aligned}

        where $u = f(z)$ and $z$ are the inducing inputs.

        With $S = LL^{\top}$ for the stored triangular root $L$ and
        $\mathbf{K}_{zz} = L_z L_z^{\top}$, this evaluates in closed form as

        .. math::

            \tfrac{1}{2}\left(
                \lVert L_z^{-1}(\mu_z - \mu)\rVert^2
                + \lVert L_z^{-1} L\rVert_F^2
                - m
                + 2\sum_i \log [L_z]_{ii}
                - 2\sum_i \log \lvert L_{ii}\rvert
            \right),

        so the Cholesky factor of $\mathbf{K}_{zz}$ is the only factorisation
        required; $S$ is never formed and never re-factorised.

        Returns:
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        variational_mean = val(self.variational_mean)
        variational_sqrt = val(self.variational_root_covariance)
        inducing_inputs = self._fmt_inducing_inputs()
        num_inducing = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.model.prior.mean_function
        kernel = self.model.prior.kernel

        inducing_mean = mean_function(inducing_inputs)
        Kzz = kernel.gram(inducing_inputs)

        # Lz Lz^T = Kzz + jitter I. The single unavoidable factorisation, taken
        # through the one stabilise-and-factor seam.
        Lz = stabilised_cholesky(Kzz.as_matrix(), self.model.prior.jitter)

        # (muz - mu)^T Kzz^{-1} (muz - mu) = ||Lz^{-1} (muz - mu)||^2
        mahalanobis = jnp.sum(
            jnp.square(_tri_solve(Lz, inducing_mean - variational_mean))
        )

        # tr[Kzz^{-1} S] = ||Lz^{-1} sqrt||_F^2  [recall S = sqrt sqrt^T]
        trace = jnp.sum(jnp.square(_tri_solve(Lz, variational_sqrt)))

        # log|Kzz| and log|S|. The absolute value keeps log|S| = 2 sum log|sqrt_ii|
        # valid for any square root, though `LowerTriangular` already guarantees a
        # positive diagonal.
        log_det_prior = 2.0 * jnp.sum(jnp.log(jnp.diag(Lz)))
        log_det_variational = 2.0 * jnp.sum(
            jnp.log(jnp.abs(jnp.diag(variational_sqrt)))
        )

        return 0.5 * (
            mahalanobis - num_inducing + log_det_prior - log_det_variational + trace
        )

    def condition(self, train_data: Dataset | None) -> SparsePosterior:
        r"""Condition the family, yielding its posterior process.

        The family already carries everything conditioning needs — the joint
        model and the variational moments — so ``train_data`` is accepted for
        interface uniformity with :meth:`gpjax.gps.JointModel.condition` and
        is **not used**. The returned
        :class:`~gpjax.conditioning.SparsePosterior` caches the factorisation
        of $\mathbf{K}_{zz}$ and is queried directly:
        ``q.condition(train_data)(test_inputs)``.

        Args:
            train_data (Dataset | None): Accepted for interface uniformity and
                ignored; the fitted $q(u)$ already summarises the data.

        Returns:
            SparsePosterior: The conditioned sparse posterior process.
        """
        del train_data
        return SparsePosterior(
            self,
            val(self.variational_mean),
            val(self.variational_root_covariance),
            whitened=False,
        )

    def predict(
        self,
        test_inputs: Int[Array, "N D"] | Float[Array, "N D"],
        train_data: Dataset | None = None,
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral $q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u$, which
        can be computed in closed form as:

        .. math::

            \mathcal{N}\left(f(t); \mu t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu - \mu z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt}\right).

        Sugar for ``self.condition(train_data)(test_inputs)``.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.
            train_data (Dataset | None): Accepted for interface uniformity and
                ignored, exactly as by :meth:`condition`.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        return self.condition(train_data)(test_inputs)


class GraphVariationalGaussian(VariationalGaussian[L]):
    r"""A variational Gaussian defined over graph-structured inducing inputs.

    This subclass adapts the :class:`VariationalGaussian` family to the
    case where the inducing inputs are discrete graph node indices rather
    than continuous spatial coordinates.

    The main differences are:
      * Inducing inputs are integer node IDs.
      * Kernel matrices are ensured to be dense and 2D.
    """

    def __init__(
        self,
        model: JointModel,
        inducing_inputs: Int[Array, "N D"],
        variational_mean: Float[Array, "N 1"] | None = None,
        variational_root_covariance: Float[Array, "N N"] | None = None,
    ):
        super().__init__(
            model,
            inducing_inputs,
            variational_mean,
            variational_root_covariance,
        )
        self.inducing_inputs = val(self.inducing_inputs).astype(jnp.int64)

    def _fmt_Kzt_Ktt(self, Kzt, Ktt):
        Ktt = Ktt.as_matrix() if hasattr(Ktt, "as_matrix") else Ktt
        Kzt = Kzt.as_matrix() if hasattr(Kzt, "as_matrix") else Kzt
        Ktt = jnp.atleast_2d(Ktt)
        Kzt = (
            jnp.transpose(jnp.atleast_2d(Kzt)) if Kzt.ndim < 2 else jnp.atleast_2d(Kzt)
        )
        return Kzt, Ktt

    def _fmt_inducing_inputs(self):
        return self.inducing_inputs

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return val(self.inducing_inputs).shape[0]


class WhitenedVariationalGaussian(VariationalGaussian[L]):
    r"""The whitened variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$,
    where $u = f(z)$
    are the function values at the inducing inputs $z$ and the distribution over
    the inducing inputs is $q(u) = \mathcal{N}(Lz \mu + mz, Lz S Lz^{\top})$. We parameterise this
    over $\mu$ and $sqrt$ with $S = sqrt sqrt^{\top}$.
    """

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the KL-divergence between our variational approximation and
        the Gaussian process prior.

        For this variational family, we have

        .. math::

            \begin{aligned}
            \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
                & = \operatorname{KL}[N(\mu  , S)\mid\mid N(0, I)].
            \end{aligned}

        Against a standard normal prior the divergence has a closed form that
        needs no matrix factorisation at all. Writing $S = LL^{\top}$ for the
        stored triangular root $L$, and using
        $\operatorname{tr}[S] = \lVert L\rVert_F^2$ and
        $\log\lvert S\rvert = 2\sum_i \log\lvert L_{ii}\rvert$,

        .. math::

            \operatorname{KL}[\mathcal{N}(\mu, S)\mid\mid\mathcal{N}(0, I)] =
            \tfrac{1}{2}\left(
                \lVert\mu\rVert^2 + \lVert L\rVert_F^2 - m
                - 2\sum_i \log\lvert L_{ii}\rvert
            \right),

        where $m$ is the number of inducing points.

        Returns:
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = val(self.variational_mean)
        sqrt = val(self.variational_root_covariance)

        # mu^T I^{-1} mu, tr[S] = ||sqrt||_F^2 and log|S| = 2 sum log|sqrt_ii|.
        # The absolute value keeps the log-determinant valid for any square root,
        # though `LowerTriangular` already guarantees a positive diagonal.
        mahalanobis = jnp.sum(jnp.square(mu))
        trace = jnp.sum(jnp.square(sqrt))
        log_det_variational = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(sqrt))))

        return 0.5 * (mahalanobis + trace - self.num_inducing - log_det_variational)

    def condition(self, train_data: Dataset | None) -> SparsePosterior:
        r"""Condition the family, yielding its posterior process.

        Identical to :meth:`VariationalGaussian.condition` — ``train_data`` is
        likewise accepted for interface uniformity and **not used** — except
        that the stored moments parameterise the whitened distribution
        $q(u) = \mathcal{N}(\mathbf{L}_z\mu + \mu_z, \mathbf{L}_z S
        \mathbf{L}_z^{\top})$, which the returned posterior de-whitens at
        query time.

        Args:
            train_data (Dataset | None): Accepted for interface uniformity and
                ignored; the fitted $q(u)$ already summarises the data.

        Returns:
            SparsePosterior: The conditioned sparse posterior process.
        """
        del train_data
        return SparsePosterior(
            self,
            val(self.variational_mean),
            val(self.variational_root_covariance),
            whitened=True,
        )


class DualVariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The dual (site) parameterisation of a sparse variational Gaussian process.

    Following the t-SVGP parameterisation of Adam, Chang, Khan and Solin (2021),
    `arXiv:2111.03412 <https://arxiv.org/abs/2111.03412>`_, the variational
    distribution is stored as an unnormalised Gaussian *site* on the inducing outputs
    rather than as moments:

    .. math::

        t(u) = \exp\left(\lambda_1^{\top}\tilde u
            - \tfrac{1}{2}\tilde u^{\top}\Lambda_2\tilde u\right),
        \qquad q(u) \propto p_{\theta}(u)\,t(u),

    with $\tilde u = u - \mu_z$ the *centred* inducing outputs, giving

    .. math::

        S = \left(\mathbf{K}_{zz}^{-1} + \Lambda_2\right)^{-1},
        \qquad \tilde m = S\lambda_1, \qquad m = \mu_z + \tilde m .

    ``dual_vector`` is $\lambda_1\in\mathbb{R}^{M\times1}$ and ``dual_matrix`` is
    $\Lambda_2\in\mathbb{R}^{M\times M}$, stored in the **precision** convention (PSD,
    no $-\tfrac{1}{2}$ factor). Both default to zero, so that $q(u) = p(u)$ and the
    prior KL vanishes at initialisation.

    Neither field is wrapped in a constraining bijection: positive semi-definiteness
    of $\Lambda_2$ comes from the convex-combination structure of the natural-gradient
    update, and a bijection here would destroy that affine step. That guarantee holds
    only under :func:`~gpjax.fit.fit_natgrads`; plain :func:`~gpjax.fit.fit` can step
    $\Lambda_2$ out of the PSD cone and produce NaNs, so use ``fit_natgrads`` to
    train this family.

    Everything routes through the working matrix
    $\mathbf{R} = \mathbf{K}_{zz} + \mathbf{K}_{zz}\Lambda_2\mathbf{K}_{zz}
    = \mathbf{K}_{zz}\mathbf{S}^{-1}\mathbf{K}_{zz}$, so no matrix is ever explicitly
    inverted and $\Lambda_2$ is never factorised.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>>
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(), kernel=gpx.kernels.RBF()
        ... )
        >>> model = prior * gpx.likelihoods.Gaussian()
        >>> q = gpx.variational_families.DualVariationalGaussian(
        ...     model=model, inducing_inputs=jnp.linspace(0, 1, 4).reshape(-1, 1)
        ... )
        >>> bool(abs(q.prior_kl()) < 1e-10)
        True
    """

    dual_vector: tp.Any
    dual_matrix: tp.Any

    def __init__(
        self,
        model: JointModel,
        inducing_inputs: Int[Array, "N D"] | Float[Array, "N D"],
        dual_vector: Float[Array, "N 1"] | None = None,
        dual_matrix: Float[Array, "N N"] | None = None,
    ):
        super().__init__(model, inducing_inputs)

        if dual_vector is None:
            dual_vector = jnp.zeros((self.num_inducing, 1))

        if dual_matrix is None:
            dual_matrix = jnp.zeros((self.num_inducing, self.num_inducing))

        self.dual_vector = Real(dual_vector)
        self.dual_matrix = Real(dual_matrix)

    def _gram_and_root(
        self,
    ) -> tuple[Float[Array, "M M"], Float[Array, "M M"]]:
        r"""Return the jittered $\mathbf{K}_{zz}$ and its lower Cholesky factor.

        Split out of :meth:`_working_matrices` so that callers needing only
        $\mathbf{A} = \mathbf{K}_{zz}^{-1}\mathbf{K}_{zb}$ -- the natural-gradient step
        is the one in-tree example -- do not compute and discard the factor of
        $\mathbf{R}$.

        Returns:
            tuple[Float[Array, "M M"], Float[Array, "M M"]]: The jittered gram matrix
                $\mathbf{K}_{zz}$ and its lower-triangular Cholesky factor
                $\mathbf{L}_K$.
        """
        inducing_inputs = self._fmt_inducing_inputs()
        kernel = self.model.prior.kernel
        jitter = self.model.prior.jitter

        gram = kernel.gram(inducing_inputs).as_matrix()
        return add_jitter(gram, jitter), stabilised_cholesky(gram, jitter)

    def _working_matrices(
        self,
    ) -> tuple[Float[Array, "M M"], Float[Array, "M M"], Float[Array, "M M"]]:
        r"""Return $(\mathbf{K}_{zz},\ \mathbf{L}_K,\ \mathbf{L}_R)$.

        The working matrix is

        .. math::

            \mathbf{R} = \mathbf{K}_{zz} + \mathbf{K}_{zz}\Lambda_2\mathbf{K}_{zz}
                = \mathbf{K}_{zz}\mathbf{S}^{-1}\mathbf{K}_{zz}
                = \mathbf{L}_K\left(\mathbf{I}
                  + \mathbf{L}_K^{\top}\Lambda_2\mathbf{L}_K\right)\mathbf{L}_K^{\top},

        and it is the right-hand form that is factorised: with
        $\mathbf{G} = \operatorname{sym}(\mathbf{L}_K^{\top}\Lambda_2\mathbf{L}_K)$,
        $\mathbf{L}_R = \mathbf{L}_K\operatorname{chol}(\mathbf{I} + \mathbf{G})$,
        which is again lower triangular and satisfies
        $\mathbf{L}_R\mathbf{L}_R^{\top} = \mathbf{R}$. Exactly **two** Cholesky
        factorisations per call, as for the explicit triple product, plus one extra
        $M\times M$ product.

        Forming $\mathbf{R}$ explicitly instead is not safe. In exact arithmetic
        $\mathbf{R}\succeq\mathbf{K}_{zz}\succ0$ whenever $\Lambda_2\succeq0$, so
        $\operatorname{chol}(\mathbf{R})$ exists -- but the rounding error of the triple
        product is $\mathcal{O}(\lVert\mathbf{K}_{zz}\rVert^2\lVert\Lambda_2\rVert
        \varepsilon)$, which overwhelms
        $\lambda_{\min}(\mathbf{R})\approx\texttt{jitter}$ for a large-variance kernel
        or in single precision, and ``jnp.linalg.cholesky`` then returns ``NaN``
        silently. In the Cholesky basis the factorised matrix is
        $\mathbf{I} + \mathbf{G}$ with $\lambda_{\min}\ge1-\mathcal{O}(\lVert
        \mathbf{G}\rVert\varepsilon)$, so it is unconditionally factorisable however
        badly $\mathbf{K}_{zz}$ is scaled.

        Returns:
            tuple[Float[Array, "M M"], Float[Array, "M M"], Float[Array, "M M"]]: The
                jittered gram matrix and the lower-triangular Cholesky factors of
                $\mathbf{K}_{zz}$ and $\mathbf{R}$.
        """
        Kzz, Lk = self._gram_and_root()

        dual_matrix = val(self.dual_matrix)
        inner = _symmetrise(Lk.T @ dual_matrix @ Lk) + jnp.eye(
            self.num_inducing, dtype=Kzz.dtype
        )
        Lr = Lk @ jnp.linalg.cholesky(inner)
        return Kzz, Lk, Lr

    def moments(self) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
        r"""Return the implied moments $(\mathbf{m},\mathbf{S})$ of $q(u)$.

        .. math::

            \mathbf{S} = \mathbf{K}_{zz}\mathbf{R}^{-1}\mathbf{K}_{zz},
            \qquad
            \mathbf{m} = \mu_z
                + \mathbf{K}_{zz}\mathbf{R}^{-1}\left(\mathbf{K}_{zz}\lambda_1\right).

        For reporting, for interoperating with :class:`VariationalGaussian`, and for
        :meth:`condition`; the :func:`~gpjax.objectives.dual_elbo` training path
        never needs it. Nothing here is cached on the module: caching would silently
        turn :func:`~gpjax.objectives.dual_elbo` back into
        :func:`~gpjax.objectives.elbo` under differentiation.

        Returns:
            tuple[Float[Array, "M 1"], Float[Array, "M M"]]: The mean and covariance
                of $q(u)$.
        """
        Kzz, _, Lr = self._working_matrices()
        dual_vector = val(self.dual_vector)
        inducing_mean = self.model.prior.mean_function(self._fmt_inducing_inputs())

        covariance = _symmetrise(Kzz @ jsp.linalg.cho_solve((Lr, True), Kzz))
        centred_mean = Kzz @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)
        return inducing_mean + centred_mean, covariance

    def marginals(
        self, inputs: Float[Array, "P D"]
    ) -> tuple[Float[Array, " P"], Float[Array, " P"]]:
        r"""Batched marginal mean and variance of $q(f(\cdot))$ at ``inputs``.

        .. math::

            \mu_{\star} = \mu(X_{\star})
                + \mathbf{K}_{\star z}\mathbf{R}^{-1}\mathbf{K}_{zz}\lambda_1,
            \qquad
            \sigma^2_{\star} = \operatorname{diag}(\mathbf{K}_{\star\star})
                - \lVert\mathbf{L}_K^{-1}\mathbf{K}_{z\star}\rVert^2_{\mathrm{col}}
                + \lVert\mathbf{L}_R^{-1}\mathbf{K}_{z\star}\rVert^2_{\mathrm{col}}
                + \varepsilon .

        Costs $\mathcal{O}(M^3 + PM^2)$, the same order as ``vmap``-ing :meth:`predict`
        over single inputs the way :func:`~gpjax.objectives.elbo` does -- ``vmap``
        leaves the two factorisations unbatched, so they are not repeated per datum.
        The gain is in the constant: two BLAS-3 triangular solves against a
        $M\times P$ right-hand side, instead of $P$ rank-one solves and $P$
        ``GaussianDistribution`` constructions whose covariance is $1\times1$.

        The trailing ``+ jitter`` on the variance is load-bearing, not a numerical
        nicety. The conditioned :class:`~gpjax.conditioning.SparsePosterior` adds
        the model's ``Prior.jitter`` to its output covariance, so the per-point
        marginals that :func:`~gpjax.objectives.elbo` sees are inflated by exactly
        that amount. Dropping it here makes :func:`~gpjax.objectives.dual_elbo`
        disagree with :func:`~gpjax.objectives.elbo` at matched moments by
        $N\varepsilon/(2\sigma^2)$ -- a discrepancy that reads like a KL bug.

        Unlike :meth:`predict`, this routine does not route its kernel matrices through
        :meth:`_fmt_Kzt_Ktt`: it is for Euclidean inducing inputs only. A future
        non-Euclidean subclass must override it alongside the hook.

        Args:
            inputs (Float[Array, "P D"]): The inputs, of shape ``(P, D)``, at which the
                marginals of $q(f(\cdot))$ are required.

        Returns:
            tuple[Float[Array, " P"], Float[Array, " P"]]: The marginal mean and
                variance at each input.
        """
        Kzz, Lk, Lr = self._working_matrices()
        dual_vector = val(self.dual_vector)
        kernel = self.model.prior.kernel
        mean_function = self.model.prior.mean_function
        inducing_inputs = self._fmt_inducing_inputs()

        Kzs = kernel.cross_covariance(inducing_inputs, inputs)
        Kss_diagonal = lx.diagonal(kernel.diagonal(inputs))

        mean = mean_function(inputs).squeeze(-1) + (
            Kzs.T @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)
        ).squeeze(-1)

        prior_projection = _tri_solve(Lk, Kzs)
        site_projection = _tri_solve(Lr, Kzs)
        variance = (
            Kss_diagonal
            - jnp.sum(jnp.square(prior_projection), axis=0)
            + jnp.sum(jnp.square(site_projection), axis=0)
            + self.model.prior.jitter
        )
        return mean, variance

    def prior_kl(self) -> ScalarFloat:
        r"""Compute $\operatorname{KL}[q(u)\mid\mid p(u)]$ from the stored sites.

        .. math::

            \operatorname{KL} = \tfrac{1}{2}\left(
                \operatorname{tr}\left(\mathbf{R}^{-1}\mathbf{K}_{zz}\right) - M
                + \tilde m^{\top}\mathbf{K}_{zz}^{-1}\tilde m
                + \log\lvert\mathbf{R}\rvert - \log\lvert\mathbf{K}_{zz}\rvert
            \right),

        obtained from the standard Gaussian KL by substituting
        $\mathbf{S} = \mathbf{K}_{zz}\mathbf{R}^{-1}\mathbf{K}_{zz}$, which gives
        $\mathbf{K}_{zz}^{-1}\mathbf{S} = \mathbf{R}^{-1}\mathbf{K}_{zz}$ and
        $\log\lvert\mathbf{S}\rvert
        = 2\log\lvert\mathbf{K}_{zz}\rvert - \log\lvert\mathbf{R}\rvert$. The two
        log-determinants are read off the Cholesky diagonals; no matrix is inverted and
        $\mathbf{S}$ is never formed. The trace comes from
        $\operatorname{tr}(\mathbf{R}^{-1}\mathbf{K}_{zz})
        = \lVert\mathbf{L}_R^{-1}\mathbf{L}_K\rVert_F^2$, one triangular solve against a
        factor already in hand rather than a full ``cho_solve`` whose $M\times M$ result
        would only be traced.

        Returns:
            ScalarFloat: The KL divergence between the variational approximation and
                the GP prior.
        """
        Kzz, Lk, Lr = self._working_matrices()
        dual_vector = val(self.dual_vector)

        # tr[R^{-1} Kzz] = tr[Lk^T R^{-1} Lk] = ||Lr^{-1} Lk||_F^2.
        trace = jnp.sum(jnp.square(_tri_solve(Lr, Lk)))

        # The sites act on the centred process, so the Mahalanobis term is built from
        # the centred mean and the zero-mean prior N(0, Kzz).
        centred_mean = Kzz @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)
        mahalanobis = jnp.sum(jnp.square(_tri_solve(Lk, centred_mean)))

        log_det_ratio = 2.0 * (
            jnp.sum(jnp.log(jnp.diag(Lr))) - jnp.sum(jnp.log(jnp.diag(Lk)))
        )
        return 0.5 * (trace - self.num_inducing + mahalanobis + log_det_ratio)

    def condition(self, train_data: Dataset | None) -> SparsePosterior:
        r"""Condition the family, yielding its posterior process.

        The stored sites already summarise the data, so ``train_data`` is
        accepted for interface uniformity and **not used**. The sites are
        first converted to the implied moments $(\mathbf{m}, \mathbf{S})$ via
        :meth:`moments`, and the Cholesky root of $\mathbf{S}$ is handed to
        the shared sparse conditioning derivation. The conversion happens
        afresh on every call -- nothing is cached on the family -- so the
        implicit dependence of $q$ on the kernel hyperparameters through
        $\mathbf{K}_{zz}$ is preserved under differentiation.

        Args:
            train_data (Dataset | None): Accepted for interface uniformity and
                ignored; the fitted sites already summarise the data.

        Returns:
            SparsePosterior: The conditioned sparse posterior process.
        """
        del train_data
        variational_mean, variational_covariance = self.moments()
        return SparsePosterior(
            self,
            variational_mean,
            jnp.linalg.cholesky(variational_covariance),
            whitened=False,
        )

    def predict(
        self,
        test_inputs: Int[Array, "N D"] | Float[Array, "N D"],
        train_data: Dataset | None = None,
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        .. math::

            \mathcal{N}\left(f(t);\ \mu_t
            + \mathbf{K}_{tz}\mathbf{R}^{-1}\mathbf{K}_{zz}\lambda_1,\
            \mathbf{K}_{tt} - \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}\mathbf{K}_{zt}
            + \mathbf{K}_{tz}\mathbf{R}^{-1}\mathbf{K}_{zt}\right).

        Sugar for ``self.condition(train_data)(test_inputs)``.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.
            train_data (Dataset | None): Accepted for interface uniformity and
                ignored, exactly as by :meth:`condition`.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        return self.condition(train_data)(test_inputs)


class CollapsedVariationalGaussian(AbstractVariationalGaussian[GL]):
    r"""Collapsed variational Gaussian.

    Collapsed variational Gaussian family of probability distributions.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables
    in Sparse Gaussian Processes.

    The bound is *collapsed*: the variational parameters are solved for
    analytically, which requires a Gaussian likelihood and a pass over the full
    dataset. Lift either restriction and you need the uncollapsed bound of
    :class:`VariationalGaussian` instead.

    .. seealso::

        :doc:`/examples/collapsed_vi` works through the sparse regression setting
        this family is designed for.
    """

    def __init__(
        self,
        model: JointModel,
        inducing_inputs: Float[Array, "N D"],
    ):
        super().__init__(model, inducing_inputs)

        if not isinstance(model.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

    def condition(self, train_data: Dataset) -> CollapsedPosterior:
        r"""Condition the family on data, yielding its posterior process.

        Unlike the uncollapsed families, the optimal variational distribution
        here is a function of the data, so conditioning takes the training
        set — exactly as ``model.condition(train_data)`` does for a joint
        model. The returned :class:`~gpjax.conditioning.CollapsedPosterior`
        caches its factorisations; the predictive and the Titsias bound
        (``elbo_bound``) are views of them.

        Args:
            train_data (Dataset): The training data the optimal variational
                distribution is solved against.

        Returns:
            CollapsedPosterior: The conditioned collapsed posterior process.
        """
        return CollapsedPosterior(self, train_data)

    def predict(
        self, test_inputs: Float[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs.

        Sugar for ``self.condition(train_data)(test_inputs)``.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs $t$ at which to make
                predictions.
            train_data (Dataset): The training data that was used to fit the GP.

        Returns:
            GaussianDistribution: The predictive distribution of the collapsed
                variational Gaussian process at the test inputs $t$.
        """
        return self.condition(train_data)(test_inputs)

    def prior_kl(self, train_data: Dataset) -> ScalarFloat:
        r"""KL divergence from the optimal collapsed $q^{\star}(u)$ to the prior.

        The collapsed family's variational distribution is solved
        analytically from the data, so — unlike the uncollapsed families —
        its KL is a function of the training set. Sugar for
        ``self.condition(train_data).prior_kl``.

        Args:
            train_data (Dataset): The training data the optimal variational
                distribution is solved against.

        Returns:
            ScalarFloat: The KL divergence of the optimal collapsed
                variational distribution from the prior.
        """
        return self.condition(train_data).prior_kl


@dataclass(slots=True)
class VariationalGaussianInit:
    """Initialization parameters for a variational Gaussian distribution."""

    inducing_inputs: Int[Array, "N D"] | Float[Array, "N D"]
    variational_mean: Float[Array, "N 1"] | None = None
    variational_root_covariance: Float[Array, "N N"] | None = None


class HeteroscedasticPrediction(tp.NamedTuple):
    """Mean and variance of the signal and noise latent processes."""

    mean_f: Float[Array, "N 1"]
    variance_f: Float[Array, "N 1"]
    mean_g: Float[Array, "N 1"]
    variance_g: Float[Array, "N 1"]


class HeteroscedasticVariationalFamily(AbstractVariationalFamily[HL]):
    r"""Variational family for two independent latent processes f and g."""

    signal_variational: tp.Any
    noise_variational: tp.Any

    def __init__(
        self,
        model: HP,
        inducing_inputs: Int[Array, "N D"] | Float[Array, "N D"] = None,
        inducing_inputs_g: Int[Array, "M D"] | Float[Array, "M D"] | None = None,
        variational_mean_f: Float[Array, "N 1"] | None = None,
        variational_root_covariance_f: Float[Array, "N N"] | None = None,
        variational_mean_g: Float[Array, "M 1"] | None = None,
        variational_root_covariance_g: Float[Array, "M M"] | None = None,
        signal_init: VariationalGaussianInit | None = None,
        noise_init: VariationalGaussianInit | None = None,
    ):
        if signal_init is not None:
            self.signal_variational = VariationalGaussian(
                model=model,
                inducing_inputs=signal_init.inducing_inputs,
                variational_mean=signal_init.variational_mean,
                variational_root_covariance=signal_init.variational_root_covariance,
            )
        elif inducing_inputs is not None:
            self.signal_variational = VariationalGaussian(
                model=model,
                inducing_inputs=inducing_inputs,
                variational_mean=variational_mean_f,
                variational_root_covariance=variational_root_covariance_f,
            )
        else:
            raise ValueError("Either signal_init or inducing_inputs must be provided.")

        if noise_init is not None:
            self.noise_variational = VariationalGaussian(
                model=model.noise_model,
                inducing_inputs=noise_init.inducing_inputs,
                variational_mean=noise_init.variational_mean,
                variational_root_covariance=noise_init.variational_root_covariance,
            )
        else:
            noise_inducing = (
                inducing_inputs if inducing_inputs_g is None else inducing_inputs_g
            )
            if noise_inducing is None and signal_init is not None:
                noise_inducing = signal_init.inducing_inputs

            if noise_inducing is None:
                raise ValueError(
                    "Could not determine inducing inputs for noise process."
                )

            self.noise_variational = VariationalGaussian(
                model=model.noise_model,
                inducing_inputs=noise_inducing,
                variational_mean=variational_mean_g,
                variational_root_covariance=variational_root_covariance_g,
            )
        super().__init__(model)

    def prior_kl(self) -> ScalarFloat:
        return self.signal_variational.prior_kl() + self.noise_variational.prior_kl()

    def condition(self, train_data: Dataset | None) -> Posterior:
        r"""Not available: the heteroscedastic family has no single posterior.

        This family approximates *two* latent processes -- signal and noise --
        so there is no one conditioned process to return, matching the
        exclusion recorded for :class:`gpjax.gps.HeteroscedasticModel`.
        Condition the components instead, via
        :attr:`signal_variational` and :attr:`noise_variational`, or call
        :meth:`predict_latents` for the two predictive distributions and
        :meth:`predict` for their moments.

        Args:
            train_data (Dataset | None): Unused; present for interface
                uniformity.

        Raises:
            NotImplementedError: Always.
        """
        del train_data
        raise NotImplementedError(
            "HeteroscedasticVariationalFamily has no single conditioned process: "
            "it approximates both a signal and a noise latent. Use "
            "`predict_latents(test_inputs)`, or condition the components "
            "`signal_variational` / `noise_variational` individually."
        )

    def predict(
        self, test_inputs: Int[Array, "N D"] | Float[Array, "N D"]
    ) -> HeteroscedasticPrediction:
        dist_f = self.signal_variational.predict(test_inputs)
        dist_g = self.noise_variational.predict(test_inputs)

        mean_f = dist_f.mean[:, None] if dist_f.mean.ndim == 1 else dist_f.mean
        var_f = (
            dist_f.variance[:, None] if dist_f.variance.ndim == 1 else dist_f.variance
        )
        mean_g = dist_g.mean[:, None] if dist_g.mean.ndim == 1 else dist_g.mean
        var_g = (
            dist_g.variance[:, None] if dist_g.variance.ndim == 1 else dist_g.variance
        )

        return HeteroscedasticPrediction(
            mean_f=mean_f,
            variance_f=var_f,
            mean_g=mean_g,
            variance_g=var_g,
        )

    def predict_latents(
        self, test_inputs: Int[Array, "N D"] | Float[Array, "N D"]
    ) -> tuple[GaussianDistribution, GaussianDistribution]:
        return (
            self.signal_variational.predict(test_inputs),
            self.noise_variational.predict(test_inputs),
        )


__all__ = [
    "AbstractVariationalFamily",
    "AbstractVariationalGaussian",
    "CollapsedVariationalGaussian",
    "DualVariationalGaussian",
    "GraphVariationalGaussian",
    "HeteroscedasticPrediction",
    "HeteroscedasticVariationalFamily",
    "VariationalGaussian",
    "VariationalGaussianInit",
    "WhitenedVariationalGaussian",
]

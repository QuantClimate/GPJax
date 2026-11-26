# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
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

import functools
import typing as tp
import warnings

import equinox as eqx
import jax
import jax.core
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
from scipy.optimize import minimize

from gpjax.dataset import Dataset
from gpjax.natural_gradients import (
    _reject_frozen_coordinates,
    natural_gradient_step,
    partition_variational,
)
from gpjax.objectives import Objective
from gpjax.scan import vscan
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
from gpjax.variational_families import DualVariationalGaussian

Model = tp.TypeVar("Model", bound=eqx.Module)


def fit(
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    key: KeyArray = jr.key(42),
    num_iters: int = 100,
    batch_size: int = -1,
    log_rate: int = 10,
    verbose: bool = True,
    unroll: int = 1,
    safe: bool = True,
) -> tuple[Model, jax.Array]:
    r"""Train a Module model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import optax as ox
        >>> import gpjax as gpx
        >>>
        >>> xtrain = jnp.linspace(0, 1, 50).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)
        >>>
        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood
        >>>
        >>> nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
        >>> trained_model, history = gpx.fit(
        ...     model=posterior, objective=nmll, train_data=D,
        ...     optim=ox.adam(0.01), num_iters=100, verbose=False,
        ... )

    Args:
        model (Model): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (int): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (int): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (KeyArray): The random key to use for the optimisation batch
            selection. Defaults to jr.key(42).
        log_rate (int): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (bool): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.
        safe (bool): Whether to validate inputs before optimisation. Defaults to True.

    Returns:
        A tuple comprising the optimised model and training history. The value
        at index ``k`` of the history is the objective before step ``k``. With
        mini-batches, it is the objective on the mini-batch of step ``k``.

        ``fit`` also evaluates the objective of the model after the last step.
        If that objective is not finite (for example NaN), ``fit`` returns the
        last iterate whose objective is finite. All values in the history after
        that iterate are not finite. If no iterate has a finite objective,
        ``fit`` returns the model after the last step. With mini-batches, the
        objective of the model after the last step is evaluated on one more
        mini-batch.

    Warns:
        RuntimeWarning: If ``fit`` returns an earlier iterate because the
            objective after the last step is not finite. Inside ``jax.jit`` or
            ``jax.vmap``, ``fit`` gives no warning.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    model = _prepare_model(model, train_data)

    def loss(model: eqx.Module, batch: Dataset) -> ScalarFloat:
        return objective(model, batch)

    # Initialise optimiser state.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Mini-batch random keys to scan over. The scan has one more iteration,
    # which only evaluates the objective of the model after the last step.
    iter_keys = jnp.concatenate(
        [jr.split(key, num_iters), jr.fold_in(key, num_iters)[None]]
    )

    def mini_batch_or_all_data(key: KeyArray) -> Dataset:
        if batch_size != -1:
            return get_batch(train_data, batch_size, key)
        return train_data

    # Optimisation step. The carry also holds the parameters of the last
    # iterate whose objective is finite, and the index of that iterate (-1 if
    # there is none).
    def step(carry, key_and_index):
        model, opt_state, last_finite_params, last_finite_index = carry
        key, index = key_and_index

        loss_val, grads = eqx.filter_value_and_grad(loss)(
            model, mini_batch_or_all_data(key)
        )
        params = eqx.filter(model, eqx.is_array)
        finite = jnp.isfinite(loss_val)
        last_finite_params = _select(finite, params, last_finite_params)
        last_finite_index = jnp.where(finite, index, last_finite_index)

        # The last iteration does not change the model.
        is_step = index < num_iters
        updates, new_opt_state = optim.update(grads, opt_state, params)
        opt_state = _select(is_step, new_opt_state, opt_state)
        model = eqx.combine(
            _select(
                is_step,
                eqx.filter(eqx.apply_updates(model, updates), eqx.is_array),
                params,
            ),
            model,
        )

        carry = model, opt_state, last_finite_params, last_finite_index
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    params, static = eqx.partition(model, eqx.is_array)
    indices = jnp.arange(num_iters + 1)
    (model, _, last_finite_params, last_finite_index), losses = scan(
        step,
        (model, opt_state, params, jnp.full((), -1, dtype=indices.dtype)),
        (iter_keys, indices),
        unroll=unroll,
    )
    history = losses[:num_iters]

    # If the objective after the last step is not finite, return the last
    # iterate whose objective is finite.
    use_last_finite = (last_finite_index >= 0) & (last_finite_index < num_iters)
    model = eqx.combine(
        _select(use_last_finite, last_finite_params, eqx.filter(model, eqx.is_array)),
        static,
    )
    _warn_if_last_finite_iterate_returned(use_last_finite, "fit")

    return model, history


def fit_scipy(
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    max_iters: int = 500,
    verbose: bool = True,
    safe: bool = True,
) -> tuple[Model, Array]:
    r"""Train a Module model with respect to a supplied Objective function
    using SciPy's L-BFGS-B optimiser.

    Parameters are transformed to unconstrained space, flattened into a
    single vector, and passed to ``scipy.optimize.minimize``. Gradients
    are computed via JAX's ``value_and_grad``.

    Args:
        model (Module): The model to be optimised.
        objective (Objective): The objective function to minimise with respect
            to the model parameters.
        train_data (Dataset): The training data used to evaluate the objective.
        max_iters (int): Maximum number of L-BFGS-B iterations. Defaults to 500.
        verbose (bool): Whether to print optimisation progress. Defaults to True.
        safe (bool): Whether to validate inputs before optimisation. Defaults to
            True.

    Returns:
        tuple[Module, Array]: A tuple of the optimised model and an array of
            objective values recorded at each iteration. The last value of the
            history is the objective value of the returned model.

        If the objective after the last step is not finite (for example NaN),
        ``fit_scipy`` returns the last iterate whose objective is finite. The
        history then stops at that iterate. If no iterate has a finite
        objective, ``fit_scipy`` returns the model after the last step.

    Warns:
        RuntimeWarning: If ``fit_scipy`` returns an earlier iterate because the
            objective after the last step is not finite.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
        >>> trained_model, history = gpx.fit_scipy(
        ...     model=posterior, objective=nmll, train_data=D
        ... )
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_num_iters(max_iters)
        _check_verbose(verbose)

    model = _prepare_model(model, train_data)

    # Split model into trainable arrays and static parts
    params, static = eqx.partition(model, eqx.is_array)

    # Loss definition
    def loss(params) -> ScalarFloat:
        model = eqx.combine(params, static)
        return objective(model, train_data)

    # convert to numpy for interface with scipy
    x0, scipy_to_jnp = ravel_pytree(params)

    @jax.jit
    def scipy_wrapper(x0):
        value, grads = jax.value_and_grad(loss)(scipy_to_jnp(jnp.array(x0)))
        scipy_grads = ravel_pytree(grads)[0]
        return value, scipy_grads

    history = [scipy_wrapper(x0)[0]]
    last_finite = (0, x0) if np.isfinite(history[0]) else None

    def record_objective(x):
        nonlocal last_finite
        history.append(scipy_wrapper(x)[0])
        if np.isfinite(history[-1]):
            last_finite = (len(history) - 1, np.array(x))

    result = minimize(
        fun=scipy_wrapper,
        x0=x0,
        jac=True,
        callback=record_objective,
        options={"maxiter": max_iters, "disp": verbose},
    )
    x = result.x

    # SciPy can accept a last step to a NaN objective. Then return the last
    # iterate whose objective is finite, and stop the history at that iterate.
    use_last_finite = not np.isfinite(result.fun) and last_finite is not None
    if use_last_finite:
        index, x = last_finite
        history = history[: index + 1]
    _warn_if_last_finite_iterate_returned(use_last_finite, "fit_scipy")
    history = jnp.array(history)

    # convert back to pytree with JAX arrays
    params = scipy_to_jnp(x)

    # Reconstruct model
    model = eqx.combine(params, static)

    return model, history


def fit_lbfgs(
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    max_iters: int = 100,
    safe: bool = True,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
) -> tuple[Model, jax.Array]:
    r"""Train a Module model with respect to a supplied Objective function.

    Uses Optax's L-BFGS implementation with a ``jax.lax.while_loop``.

    Args:
        model (Module): The model to be optimised.
        objective (Objective): The objective function to minimise.
        train_data (Dataset): The training data used to evaluate the objective.
        max_iters (int): Maximum number of L-BFGS iterations. Defaults to 100.
        safe (bool): Whether to validate inputs before optimisation. Defaults to
            True.
        max_linesearch_steps (int): Maximum number of line-search steps per
            iteration. Defaults to 32.
        gtol (float): Terminate if the L2 norm of the gradient falls below this
            threshold. Defaults to 1e-5.

    Returns:
        tuple[Module, Array]: A tuple of the optimised model and the final loss
            value.

        If the objective or the parameters after the last step are not finite
        (for example NaN), ``fit_lbfgs`` returns the last iterate whose
        objective and parameters are finite, and the objective of that
        iterate. If no iterate is finite, ``fit_lbfgs`` returns the model after
        the last step.

    Warns:
        RuntimeWarning: If ``fit_lbfgs`` returns an earlier iterate because the
            objective or the parameters after the last step are not finite.
            Inside ``jax.jit`` or ``jax.vmap``, ``fit_lbfgs`` gives no warning.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1, 20).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
        >>> trained_model, final_loss = gpx.fit_lbfgs(
        ...     model=posterior, objective=nmll, train_data=D
        ... )
    """
    if safe:
        # Check inputs
        _check_model(model)
        _check_train_data(train_data)
        _check_num_iters(max_iters)

    model = _prepare_model(model, train_data)

    # Split model into trainable arrays and static parts
    params, static = eqx.partition(model, eqx.is_array)

    # Loss definition
    def loss(params) -> ScalarFloat:
        model = eqx.combine(params, static)
        return objective(model, train_data)

    # Initialise optimiser
    optim = ox.lbfgs(
        linesearch=ox.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            initial_guess_strategy="one",
        )
    )
    opt_state = optim.init(params)
    loss_value_and_grad = ox.value_and_grad_from_state(loss)

    def all_finite(params) -> jax.Array:
        return jnp.all(
            jnp.array([jnp.all(jnp.isfinite(p)) for p in jax.tree.leaves(params)])
        )

    # Optimisation step. The carry also holds the last iterate whose objective
    # and parameters are finite, and the objective of that iterate.
    def step(carry):
        params, opt_state, last_finite_params, last_finite_loss = carry

        # Using optax's value_and_grad_from_state is more efficient given LBFGS uses a linesearch
        loss_val, loss_gradient = loss_value_and_grad(params, state=opt_state)
        finite = jnp.isfinite(loss_val) & all_finite(params)
        last_finite_params = _select(finite, params, last_finite_params)
        last_finite_loss = jnp.where(finite, loss_val, last_finite_loss)

        updates, opt_state = optim.update(
            loss_gradient,
            opt_state,
            params,
            value=loss_val,
            grad=loss_gradient,
            value_fn=loss,
        )
        params = ox.apply_updates(params, updates)

        return params, opt_state, last_finite_params, last_finite_loss

    def continue_fn(carry):
        _, opt_state, _, _ = carry
        n = ox.tree_utils.tree_get(opt_state, "count")
        g = ox.tree_utils.tree_get(opt_state, "grad")
        g_l2_norm = ox.tree_utils.tree_l2_norm(g)
        return (n == 0) | ((n < max_iters) & (g_l2_norm >= gtol))

    # Optimisation loop. No iterate is finite before the first step.
    no_loss = jnp.full((), jnp.nan, dtype=jax.eval_shape(loss, params).dtype)
    params, opt_state, last_finite_params, last_finite_loss = jax.lax.while_loop(
        continue_fn,
        step,
        (params, opt_state, params, no_loss),
    )
    final_loss = ox.tree_utils.tree_get(opt_state, "value")

    # The line search rejects a step to a non-finite objective, but a NaN
    # gradient can still make the parameters NaN. Then return the last iterate
    # whose objective and parameters are finite, and its objective.
    use_last_finite = ~(jnp.isfinite(final_loss) & all_finite(params)) & jnp.isfinite(
        last_finite_loss
    )
    params = _select(use_last_finite, last_finite_params, params)
    final_loss = jnp.where(use_last_finite, last_finite_loss, final_loss)
    _warn_if_last_finite_iterate_returned(use_last_finite, "fit_lbfgs")

    # Reconstruct model
    model = eqx.combine(params, static)

    return model, final_loss


def fit_natgrads(
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    natgrad_lr: ScalarFloat | int | ox.Schedule = 1e-1,
    key: KeyArray = jr.key(42),
    num_iters: int = 100,
    batch_size: int = -1,
    map_jitter: ScalarFloat | int = 0.0,
    backoff: ScalarFloat | int = 0.5,
    max_backoff: int = 5,
    beta_floor: ScalarFloat | int = 1e-8,
    log_rate: int = 10,
    verbose: bool = True,
    unroll: int = 1,
    safe: bool = True,
) -> tuple[Model, jax.Array]:
    r"""Train a variational family by alternating natural-gradient and Optax steps.

    Implements the NGD+Adam scheme of Salimbeni, Eleftheriadis and Hensman (2018),
    arXiv:1803.09151. Each iteration takes one natural-gradient step on the
    exponential-family coordinates of the variational distribution, then one step of
    the supplied Optax optimiser on everything else -- kernel and likelihood
    hyperparameters, the mean function, and the inducing inputs, which the paper counts
    as hyperparameters.

    The natural gradient with respect to the natural parameters
    $\boldsymbol\theta$ is the ordinary gradient with respect to the expectation
    parameters $\boldsymbol\eta$, so the update
    $\boldsymbol\theta\leftarrow\boldsymbol\theta
    -\gamma\,\partial\ell/\partial\boldsymbol\eta$ needs no Fisher matrix. For a
    conjugate (Gaussian-likelihood) model on the full batch, ``natgrad_lr=1.0``
    reaches the exact optimal $q$ in a single iteration.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import optax as ox
        >>> import gpjax as gpx
        >>>
        >>> xtrain = jnp.linspace(0, 1, 20).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)
        >>>
        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood
        >>>
        >>> z = jnp.linspace(0, 1, 5).reshape(-1, 1)
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     model=posterior, inducing_inputs=z
        ... )
        >>>
        >>> negative_elbo = lambda p, d: -gpx.objectives.elbo(p, d)
        >>> trained_model, history = gpx.fit_natgrads(
        ...     model=q, objective=negative_elbo, train_data=D,
        ...     optim=ox.adam(0.01), natgrad_lr=1.0, num_iters=10, verbose=False,
        ... )

    Args:
        model (Model): The variational family to be optimised.
        objective (Objective): The loss to minimise, e.g.
            ``lambda q, d: -gpjax.objectives.elbo(q, d)``.
        train_data (Dataset): The training data used to evaluate the objective.
        optim (GradientTransformation): The Optax optimiser applied to the
            hyperparameter partition.
        natgrad_lr (float | int | jax.Array | optax.Schedule): The natural-gradient
            step size $\gamma\in(0,1]$, or an Optax schedule mapping the iteration
            number to a step size. Defaults to ``1e-1``, the value Salimbeni et al.
            recommend in the stochastic, non-conjugate regime; ``natgrad_lr=1.0`` is
            optimal only when the model is conjugate *and* the batch is full. Adam,
            Chang, Khan and Solin (2021) write this step size $\rho$ for the dual
            parameterisation; it is the same quantity, and started from the same $q$
            the two branches produce identical iterates -- provided the dual branch's
            computed $\boldsymbol\beta$ stays non-negative, so that its ``beta_floor``
            never engages. GPJax's clipped probit link breaks that in the far tails.
            On a ``DualVariationalGaussian`` a value above $1$ is rejected, because
            the site update is a convex combination towards its target -- for a
            schedule this is checked over the whole ``num_iters``-long trajectory, not
            just at construction.
        key (KeyArray): The random key used for mini-batch selection. Defaults to
            ``jr.key(42)``.
        num_iters (int): The number of alternating iterations to run. Defaults to 100.
        batch_size (int): The size of the mini-batch to use. Defaults to -1 (i.e. full
            batch). The same batch feeds both sub-steps of an iteration.
        map_jitter (float): Jitter added inside the
            $\boldsymbol\theta\leftrightarrow\boldsymbol\xi$ maps. Defaults to ``0.0``
            and is deliberately **not** inherited from the model's ``Prior.jitter``: a
            non-zero value biases the recovered covariance by
            $\approx\varepsilon\lVert\mathbf S\rVert^2$ regardless of conditioning,
            which destroys the exactness of the conjugate one-step solution. Raise it
            to $10^{-12}$--$10^{-10}$ only when fighting an ill-conditioned
            $\mathbf S$, and note that a non-zero value also shifts every entry of
            ``history`` by $\mathcal O(\varepsilon)$, because the logged loss is read
            off the differentiated $\boldsymbol\eta$ closure.
        backoff (float): Multiplicative shrink factor applied to $\gamma$ when a step
            would leave the negative-definite cone. Defaults to 0.5.
        max_backoff (int): The number of shrink attempts after the first, so $\gamma$
            can fall by $\beta^{K}$. Defaults to 5.
        beta_floor (float): Lower clip on the expected negative curvature $\beta$ in
            the dual step, which keeps $\boldsymbol\Lambda_2$ inside the positive
            semi-definite cone for likelihoods that are not log-concave. The
            Salimbeni-family step ignores it. Defaults to ``1e-8``.
        log_rate (int): How frequently the objective value should be printed. Defaults
            to 10.
        verbose (bool): Whether to display the training progress bar. Defaults to
            True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.
        safe (bool): Whether to validate inputs before optimisation. Defaults to True.

    Returns:
        tuple[Model, jax.Array]: A tuple of the optimised model and a 1-D history of
            length ``num_iters``.

    Notes:
        **Step ordering.** Within one iteration the natural-gradient step runs *first*
        and the Optax step second, on the already-updated $q$. Salimbeni et al.
        describe the reverse order and explicitly allow either; natgrad-first is
        chosen here because the forward pass that produces
        $\partial\ell/\partial\boldsymbol\eta$ also yields
        $\ell(\boldsymbol\xi_t,\boldsymbol\phi_t)$ for free, which is exactly
        ``fit()``'s ``history[t]`` convention, and because it decouples a bad
        hyperparameter step from the Cholesky factorisations of the natural-gradient
        step by one iteration. The ordering changes traces bit-for-bit, so do not
        reverse it casually.

        **Choice of family.** The step differentiates the loss through
        $\boldsymbol\xi(\boldsymbol\eta)$, which subtracts
        $\boldsymbol\eta_1\boldsymbol\eta_1^\top$ from $\mathbf H_2$. When
        $\lVert\mathbf m\rVert^2\gg\lVert\mathbf S\rVert$ that cancellation loses
        digits quietly -- finite, unguarded and increasingly wrong -- so prefer
        ``WhitenedVariationalGaussian``, whose $q(\mathbf v)$ stays close to
        $\mathcal N(\mathbf 0,\mathbf I)$, in that regime.
        ``DualVariationalGaussian`` is immune to this particular cancellation for a
        different reason: its step is affine in the stored sites and takes no
        $\boldsymbol\xi(\boldsymbol\eta)$ round trip at all, so
        $\boldsymbol\eta_1\boldsymbol\eta_1^\top$ is never formed. It buys that with a
        second $M\times M$ factorisation per objective evaluation and a step size
        capped at $1$.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_log_rate(log_rate)
        _check_verbose(verbose)
        _check_natgrad_lr(natgrad_lr, model, num_iters)
        # Surface a frozen coordinate as an ordinary argument error, before `vscan`
        # opens a progress bar and buries the traceback in the scan trace. The step
        # itself repeats the check unconditionally as a backstop.
        _reject_frozen_coordinates(model)

    # The natural-gradient steps assume one latent function: one mean column
    # and one root matrix.
    num_latent_functions = model.model.likelihood.num_latent_functions
    if num_latent_functions > 1:
        raise ValueError(
            "fit_natgrads supports one latent function only, but the likelihood "
            f"has {num_latent_functions}. Use fit with the elbo objective instead."
        )

    model = _prepare_model(model, train_data)

    # Split once, before the scan: the exponential-family coordinates are driven by
    # the natural-gradient rule and everything else by `optim`.
    variational, hyper = partition_variational(model)

    # Initialise optimiser state on the hyperparameter partition only.
    opt_state = optim.init(eqx.filter(hyper, eqx.is_array))

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # A plain float is resolved to a constant schedule; the `callable` branch is a
    # Python-level check on a static object, so only the resolved value is traced.
    schedule = natgrad_lr if callable(natgrad_lr) else (lambda _: natgrad_lr)

    def hyper_loss(hyper, variational, batch):
        model = eqx.combine(variational, hyper)
        return objective(model, batch)

    # Optimisation step.
    def step(carry, iteration_and_key):
        variational, hyper, opt_state = carry
        iteration, iter_key = iteration_and_key

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, iter_key)
        else:
            batch = train_data

        # (a) natural-gradient step on the exponential-family coordinates.
        variational, loss_val = natural_gradient_step(
            variational,
            hyper,
            batch,
            objective,
            # Coerced to the default float type so that an integer step size, or a
            # schedule returning one, still reaches the step as a `ScalarFloat`.
            jnp.asarray(schedule(iteration), dtype=jnp.result_type(float)),
            map_jitter=map_jitter,
            backoff=backoff,
            max_backoff=max_backoff,
            beta_floor=beta_floor,
        )

        # (b) Optax step on hyperparameters and inducing inputs, at the updated q.
        _, grads = eqx.filter_value_and_grad(hyper_loss)(hyper, variational, batch)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(hyper, eqx.is_array)
        )
        hyper = eqx.apply_updates(hyper, updates)

        carry = variational, hyper, opt_state
        return carry, loss_val

    # Optimisation scan. `jax.lax.scan` has no `log_rate`, so it is bound only on the
    # verbose branch, where it actually drives the progress bar.
    scan = functools.partial(vscan, log_rate=log_rate) if verbose else jax.lax.scan

    # Optimisation loop.
    (variational, hyper, _), history = scan(
        step,
        (variational, hyper, opt_state),
        (jnp.arange(num_iters), iter_keys),
        unroll=unroll,
    )

    return eqx.combine(variational, hyper), history


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr

        >>> X = jnp.linspace(0, 1, 100).reshape(-1, 1)
        >>> y = jnp.sin(X)
        >>> D = gpx.Dataset(X=X, y=y)

        >>> from gpjax.fit import get_batch
        >>> batch = get_batch(D, batch_size=16, key=jr.key(0))

    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indices with replacement.
    indices = jr.choice(key, n, (batch_size,), replace=True)

    # Stamp the parent's size onto the batch so minibatch objectives can rescale the
    # expected log-likelihood. `full_size` already falls back to `n` when the parent
    # is itself a full dataset, and re-batching a batch keeps the original size.
    return Dataset(X=x[indices], y=y[indices], n_total=train_data.full_size)


def _select(condition: jax.Array, if_true: tp.Any, if_false: tp.Any) -> tp.Any:
    """Select ``if_true`` where ``condition`` is true, else ``if_false``, leaf by leaf."""
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), if_true, if_false)


def _warn_if_last_finite_iterate_returned(
    returned: jax.Array | bool, optimiser: str
) -> None:
    """Warn that ``optimiser`` returns the last iterate whose objective is finite.

    Inside ``jax.jit`` or ``jax.vmap``, ``returned`` is not known, and there is no
    warning.
    """
    try:
        returned = bool(returned)
    except jax.errors.ConcretizationTypeError:
        return
    if returned:
        warnings.warn(
            f"The objective after the last step of {optimiser} is not finite. "
            f"{optimiser} returns the last iterate whose objective is finite.",
            RuntimeWarning,
            stacklevel=3,
        )


def _prepare_model(model: Model, train_data: Dataset) -> Model:
    """Run any data-dependent initialisation the model defines.

    JointModels use this to size lazily-initialised state (e.g. the
    non-conjugate latent vector) from the training data.
    """
    from gpjax.gps import JointModel

    if isinstance(model, JointModel):
        return model._prepare(train_data)
    return model


def _check_model(model: tp.Any) -> None:
    """Check that the model is a subclass of eqx.Module."""
    if not isinstance(model, eqx.Module):
        raise TypeError(
            "Expected model to be a subclass of eqx.Module. "
            f"Got {model} of type {type(model)}."
        )


def _check_train_data(train_data: tp.Any) -> None:
    """Check that the train_data is of type gpjax.Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError(
            "Expected train_data to be of type gpjax.Dataset. "
            f"Got {train_data} of type {type(train_data)}."
        )


def _check_optim(optim: tp.Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError(
            "Expected optim to be of type optax.GradientTransformation. "
            f"Got {optim} of type {type(optim)}."
        )


def _check_num_iters(num_iters: tp.Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError(
            "Expected num_iters to be of type int. "
            f"Got {num_iters} of type {type(num_iters)}."
        )

    if num_iters <= 0:
        raise ValueError(f"Expected num_iters to be positive. Got {num_iters}.")


def _check_log_rate(log_rate: tp.Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError(
            "Expected log_rate to be of type int. "
            f"Got {log_rate} of type {type(log_rate)}."
        )

    if not log_rate > 0:
        raise ValueError(f"Expected log_rate to be positive. Got {log_rate}.")


def _check_verbose(verbose: tp.Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError(
            "Expected verbose to be of type bool. "
            f"Got {verbose} of type {type(verbose)}."
        )


def _check_natgrad_lr(
    natgrad_lr: tp.Any, model: tp.Any = None, num_iters: tp.Any = None
) -> None:
    r"""Check the natural-gradient step size is a positive float or an optax schedule.

    Args:
        natgrad_lr (Any): The candidate step size.
        model (Any): The model being fitted. Unconstrained above for the Salimbeni
            families, which tolerate $\gamma>1$ in principle; capped at $1$ for
            ``DualVariationalGaussian``.
        num_iters (Any): The number of iterations the schedule will be evaluated at.
            Supply it to have a schedule bound-checked over its whole trajectory;
            without it a callable ``natgrad_lr`` is accepted unexamined.

    Raises:
        TypeError: If ``natgrad_lr`` is neither a float, an int, a 0-d JAX array, nor
            a callable schedule.
        ValueError: If ``natgrad_lr`` is non-positive, or exceeds $1$ for a
            ``DualVariationalGaussian``.

    Notes:
        A 0-d JAX array is accepted, because the driver immediately does
        ``jnp.asarray(schedule(iteration))`` and the dispatched step is annotated
        ``ScalarFloat``. ``bool`` is rejected despite being an ``int`` subclass:
        silently reading ``True`` as $\gamma=1$ is never what the caller meant. The
        positivity check is skipped for traced values, which have no concrete sign at
        trace time.

        The dual branch requires $\rho\in(0,1]$: the site update is a convex
        combination towards the target, so $\rho>1$ overshoots it and can push
        $\boldsymbol\Lambda_2$ out of the positive semi-definite cone, from which the
        run never recovers -- the ``NaN`` is silent and poisons every later iterate. A
        schedule is fully determined at construction time, so when ``num_iters`` is
        known the whole trajectory ``natgrad_lr(jnp.arange(num_iters))`` is checked up
        front, exactly as a scalar is. Schedules that cannot be evaluated on an
        integer array are left alone.

        Positivity is checked for *every* family, scalar or schedule. A rate of zero
        is a wasted iteration and a negative rate extrapolates away from the target,
        which can leave the cone in either parameterisation; a decaying schedule that
        reaches or crosses zero inside the horizon is the realistic way to hit this by
        accident.
    """
    if callable(natgrad_lr):
        _check_natgrad_schedule(natgrad_lr, model, num_iters)
        return

    is_scalar_array = isinstance(natgrad_lr, jax.Array) and jnp.ndim(natgrad_lr) == 0
    if isinstance(natgrad_lr, bool) or not (
        is_scalar_array or isinstance(natgrad_lr, (float, int))
    ):
        raise TypeError(
            "Expected natgrad_lr to be of type float, a 0-d JAX array, or an optax "
            f"schedule. Got {natgrad_lr} of type {type(natgrad_lr)}."
        )

    if isinstance(natgrad_lr, jax.core.Tracer):
        return

    if natgrad_lr <= 0:
        raise ValueError(f"Expected natgrad_lr to be positive. Got {natgrad_lr}.")

    if isinstance(model, DualVariationalGaussian) and natgrad_lr > 1.0:
        raise ValueError(
            "Expected natgrad_lr to lie in (0, 1] for a DualVariationalGaussian, "
            f"whose site update is a convex combination. Got {natgrad_lr}."
        )


def _check_natgrad_schedule(
    natgrad_lr: tp.Callable, model: tp.Any, num_iters: tp.Any
) -> None:
    """Bound-check an optax schedule over the trajectory ``fit_natgrads`` will take.

    The schedule is evaluated at the same iteration indices ``fit_natgrads`` scans
    over, so the check sees exactly the step sizes the run will take.

    The lower bound applies to *every* family: a non-positive rate is an extrapolation
    away from the target rather than towards it, which can leave the positive
    semi-definite cone in either parameterisation, and the scalar path already rejects
    it. Only the upper bound is dual-specific, because only the site update is a convex
    combination.

    Args:
        natgrad_lr (Callable): The candidate schedule.
        model (Any): The model being fitted; only ``DualVariationalGaussian`` carries
            the upper bound.
        num_iters (Any): The scan length. A non-integer or non-positive value means
            the trajectory is unknown, and the schedule is left unexamined.

    Raises:
        ValueError: If the schedule is non-positive anywhere in the first
            ``num_iters`` iterations, or exceeds $1$ there for a
            ``DualVariationalGaussian``.
    """
    if not isinstance(num_iters, int) or isinstance(num_iters, bool) or num_iters <= 0:
        return

    # A schedule need only be defined on scalars: Optax's are all vectorised, but a
    # hand-written one may branch in Python, index a list, or call `float()` on the
    # step. Evaluating such a schedule on an integer array raises `TypeError` (a
    # scalar conversion or index was demanded of a 1-d array, which also covers JAX's
    # own `ConcretizationTypeError`), `ValueError` (an array was used as a truth
    # value, or the result is ragged), or `IndexError` (a lookup table shorter than
    # the horizon). Those three mean "not evaluable in bulk", so the trajectory check
    # is skipped, as documented. Anything else is a genuine bug inside the caller's
    # schedule and must not be swallowed here -- it would resurface as a `NaN`
    # thousands of iterations later, or not at all.
    try:
        rates = jnp.asarray(natgrad_lr(jnp.arange(num_iters)))
    except (TypeError, ValueError, IndexError):
        return

    smallest = float(jnp.min(rates))
    if smallest <= 0.0:
        raise ValueError(
            "Expected natgrad_lr to be positive. The supplied schedule reaches "
            f"{smallest} within the first {num_iters} iterations."
        )

    if not isinstance(model, DualVariationalGaussian):
        return

    largest = float(jnp.max(rates))
    if largest > 1.0:
        raise ValueError(
            "Expected natgrad_lr to lie in (0, 1] for a DualVariationalGaussian, "
            "whose site update is a convex combination. The supplied schedule reaches "
            f"{largest} within the first {num_iters} iterations."
        )


def _check_batch_size(batch_size: tp.Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError(
            "Expected batch_size to be of type int. "
            f"Got {batch_size} of type {type(batch_size)}."
        )

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError(f"Expected batch_size to be positive or -1. Got {batch_size}.")


__all__ = [
    "fit",
    "fit_lbfgs",
    "fit_natgrads",
    "fit_scipy",
    "get_batch",
]

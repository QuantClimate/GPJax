# The classifier uses variational inference with an inducing point at every row

`GPClassifier` fits a Gaussian process prior with GPJax's `Bernoulli` likelihood, whose link is the probit, by variational inference with an inducing point at every training row, held there. The fitted distribution is therefore over the latent values at the training rows themselves, and nothing is summarised. The optimiser the exact regressor uses, L-BFGS-B, maximises the evidence lower bound over that distribution and the kernel settings together, and stops once the fit has finished or after the `iterations` given to `fit`. We chose this because every prediction keeps the uncertainty in the latent values, the kernel settings come from a bound on the marginal likelihood, and GPJax 1.1.0 supports it without any change. A sparse classifier is then the same model with fewer inducing points.

## Considered Options

- **The most probable latent values**, which is what GPJax's classification example does. Least code. Rejected because predictions leave out the uncertainty in the latent values, so probabilities near the data come out too confident, and because learning the kernel settings from that same objective puts nothing in the way of the variation growing.
- **The Laplace approximation**, which scikit-learn's `GaussianProcessClassifier` uses. It keeps the uncertainty in the latent values and learns the kernel settings from the approximate marginal likelihood. Not chosen because GPJax has no Laplace function; its example builds one by hand. It could replace this choice if GPJax gains one, and skgp would call it rather than write its own.
- **Natural gradients**, GPJax's `fit_natgrads` on all rows, in place of L-BFGS-B. Its memory does not grow with the optimiser's history. We built it first and replaced it: it runs a set number of iterations, and with two inputs on a CPU its 2,000 iterations took 200 seconds at 500 rows, where L-BFGS-B finished in 19 seconds at the same or a slightly better objective.

## Consequences

- Two classes only: GPJax 1.1.0 has no multi-class likelihood. More are tracked in issue 37.
- The optimiser fits about N²/2 numbers for N training rows, and L-BFGS-B keeps ten pairs of vectors that long: about 320 MB at 2,000 rows and 8 GB at 10,000. Time grows with the cube of the rows. With two inputs on a CPU, a fit took about 1 second at 100 rows, 19 seconds at 500 rows and 150 seconds at 1,000 rows. `fit` issues a `LargeDataWarning` above `skgp.config.size_warning_rows`.
- Whether a fit finished properly is judged by the same rule as an exact fit's, and [ADR 0003](0003-the-fit-is-not-constrained-or-restarted.md) holds: one start, no bounds.
- GPJax's probit link keeps every probability between 0.001 and 0.999.
- `SparseGPClassifier`, for large data, uses the same variational family with fewer inducing points: L-BFGS-B on all rows, or natural gradients alternating with Adam on minibatches, which L-BFGS-B cannot work on.

# Location functions are a separate type from mean functions

Nonstationary kernels (#812) need a kernel parameter that changes with input
location, such as a standard deviation or a lengthscale. We decided to add a
separate abstract type for these location functions in
`gpjax/kernels/location_functions.py`, and not to reuse
`gpjax.mean_functions`. A mean function describes the process, while a
location function describes the covariance. They also have different
contracts: a location function evaluates one point, selects its own input
columns with `active_dims`, and returns a value on the log scale, to which the
kernel applies `exp`.

## Considered options

- **Reuse `AbstractMeanFunction`.** Rejected. It works on `(N, D)` batches,
  has no column selection, and would mix two different concepts in one type.
- **Accept any `eqx.Module` callable.** Rejected. It gives no contract that
  beartype can check, and a plain Python callable silently puts its parameters
  in a static field, so `fit` does not train them.

## Consequences

- The user documentation must explain why a location function is not a mean
  function, because a reader will expect the two to be the same.
- A location function that has an intercept overlaps with the scale parameters
  of the base kernel. For this reason, `location_functions.Linear` has no
  intercept by default.
- Kernels evaluate their location functions inside `__call__`, and no special
  compute engine is used. This is not wasteful: `vmap` batches only the
  operations that depend on the mapped input, so the dense engine evaluates
  a location function once for each row ($N + M$ times for an $N \times M$
  matrix), also inside sum and product kernels. A dedicated engine would add
  code and give no saving.

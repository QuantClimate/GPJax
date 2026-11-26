# Only a fit by L-BFGS-B is judged to have converged; a fit on minibatches is not

skgp says whether a fit finished properly only when L-BFGS-B made it: a fit on all rows at once, of a `GPRegressor`, a `GPClassifier`, or a sparse model without a `minibatch_size`. A fit on minibatches, by Adam alone or alternating with natural gradients, runs exactly the `iterations` given to `fit`, and skgp says nothing about whether it converged: no `ConvergenceWarning`, and no `finished_properly` or `stop_reason` in `to_json()`, in the printed model or in the `fit finished` log record. We chose this because skgp cannot determine it. L-BFGS-B minimises one fixed objective and has its own test of when to stop. A fit on minibatches sees a different random minibatch at each iteration, so its objective is a noisy series with no such test, and any verdict on it is a guess that a user would read as a fact.

## Considered Options

- **Judge from the objective's last ten passes through the data**, which skgp did until this decision: the mean objective of the last five passes was compared with that of the five before, against how much the pass means varied. Rejected. It rested on three thresholds with no basis in the model. It needed ten passes to say anything, so at the default 2,000 iterations any data with more than 200 minibatches in a pass (over 50,000 rows at a `minibatch_size` of 250) always got a warning that the fit was too short to judge.
- **Stop a fit on minibatches early once its objective stops improving.** Not done: it needs the same judgement of a noisy objective.

## Consequences

- The user decides how long a fit on minibatches runs, through `iterations`, and judges the result themselves. `to_json()["fitting"]` gives the iterations it ran and its final objective, the mean over its last pass.
- A fit on minibatches whose numbers become infinite or undefined issues no warning either. Its `final_objective` is `null` in `to_json()`.
- `finished_properly` and the `ConvergenceWarning` have one meaning: L-BFGS-B stopped, and skgp judged from where it ended whether the fit had settled ([ADR 0003](0003-the-fit-is-not-constrained-or-restarted.md)).
- Anyone adding a convergence verdict, or early stopping, for Adam or natural gradients is reopening this decision; read this record first.

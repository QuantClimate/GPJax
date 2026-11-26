# The fit is not constrained or restarted; an unfinished fit raises a warning

skgp learns the values of a model's kernel and its noise with a single run of the optimiser from one starting point. It adds no limits on those values (no minimum noise, no bounds on length scales) and does not retry from other starting points. When the optimiser stops without finishing, `fit` still returns the model and issues a `ConvergenceWarning`. (Only a fit by L-BFGS-B is judged; [ADR 0008](0008-only-a-fit-by-l-bfgs-b-is-judged-to-have-converged.md) records why a fit on minibatches is not.) We chose this because limits would be assumptions about the data that the user never made, much like a prior, and because the model should be whatever the data and the likelihood support. Restarts would also multiply fitting time. Noise-free data is a common and legitimate case for Gaussian processes, and a zero learned noise has to be allowed.

## Considered Options

- **A minimum noise and bounds on length scales:** stops fits collapsing to zero noise. Rejected because it changes the model rather than fitting it.
- **Several starting points, keeping the best fit:** the same model, better optimised. Not done by default because it inflates fitting time; it could return later as an opt-in setting.

## Consequences

- With four or more input columns an independent review found fits that settle on near-zero noise while reporting success, so intervals came out too narrow (about 83 to 86% coverage for a nominal 95% interval). A different starting point reached a better fit with the right noise. This is a known cost of the decision.
- Anyone changing the optimiser's limits or starting points is changing what skgp models mean; read this record first.

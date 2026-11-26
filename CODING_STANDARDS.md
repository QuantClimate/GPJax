# Coding standards

Read during review, for GPJax and skgp. Anything a tool checks lives in the tools, not here: ruff (lint and format), ty (types), the beartype import hook (array shapes) and pytest, run by the prek hooks and CI. This file holds the judgement calls. skgp's own rules are in `skgp/CODING_STANDARDS.md`.

## Tests

- **Well-posed.** A test that fits a model fits a problem with one clear answer: enough distinct inputs, real noise, and parameters the data can identify. Repeated inputs with free kernel hyperparameters, a noise-free exact formula, or two parameters of which only the sum is identifiable give a fit with no settled answer. A test whose pass depends on how far the optimiser got before it stopped is deleted, not repaired.
- **Next to its counterpart.** A test sits in the file that matches its module (for example, kernel tests in `tests/test_kernels/`), so a stack trace leads to the right file.
- **Every precision.** A test of numerical code is parametrised over the dtypes and precisions it supports, to catch silent promotion.
- **A regression test for each fix.** A bug fix adds the test that fails without it.
- **Functions.** Tests are functions, not classes. Mocks use `pytest-mock`; property-based tests use `hypothesis`.

## Examples

- An example in `docs/examples/` changes in the same pull request as the behaviour it shows.

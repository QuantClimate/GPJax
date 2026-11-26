# Coding standards

Read during review. Anything a tool can check lives in the tools, not here: `ruff` (lint and format), `ty` (types) and `pytest`, run by the commit hook and CI. This file holds the judgement calls.

The rules in `CODING_STANDARDS.md` at the repository root also apply to skgp. Among them: a test fits a well-posed problem, and a test whose pass depends on how far the optimiser got is deleted, not repaired.

## Code

- **User choices are settings or arguments; constants live in one place.** First ask whether a value is a choice a user would reasonably make for their own model or fit: π is a constant, the number of optimisation iterations is a user's choice. A user's choice is a setting of the model or an argument of the method, with its default in the signature. In `src/`, every other number, and every name of an optimiser, that sets how skgp fits, checks or reports is a documented field of `Constant` in `src/skgp/_constants.py`, read as `C.<NAME>` after `from skgp._constants import Constant as C`. Lookup tables, objects, the text of messages, number formats, values written into a type (such as `Literal[...]`) and the defaults written in signatures stay where they are used. (`ruff` flags a number in a comparison anywhere else in `src/`; a number used any other way is a review call.)

## Tests

- **Concrete.** The test shows a small literal input and an answer a reader can check by eye: "fit on these five rows, predict at 0.5, expect about 0.5". A reader should not have to rebuild the expected value in their head.
- **One owner.** Test what skgp itself does. What GPJax, scipy, pandera, polars, pandas, optax or packaging guarantee is their tests' job.
- **Expected values come from outside the code.** Work the number out by hand. Do not compute it with the logic the library uses, and do not compare one library call with another. A pickle round trip that checks the loaded model predicts like the saved one is the exception, with a tolerance.
- **Say why it was refused.** A test of an error or a refusal matches the wording that is specific to that case, so a refusal for the wrong reason fails it. (`ruff` already requires a `match`; whether the match is specific is a review call.)
- **Values, not types.** Checks of types, shapes and dtypes belong to the type checker.
- **Through the public API.** Test what a user sees. Log wording is tested only where a user reads it. The module that judges how a fit ended (`src/skgp/_how_a_fit_ended.py`) is the exception, tested with literal numbers: no public fit can be made to end in each of its ways, and how a fit ends differs between operating systems.
- **The same on every platform.** Use tolerances. Do not compare floats exactly across refits or reloads, tune a threshold on one operating system, or assert memory or timing.
- **One behaviour, one test.** Before adding a test, check that no other test already pins the behaviour. Delete overlap.
- **Leave nothing behind.** A test that changes global state (a logger, `skgp.config`) restores it.
- **Mark the slow ones.** A test that takes over a second carries `@pytest.mark.slow`. CI runs them; a plain `pytest` skips them.

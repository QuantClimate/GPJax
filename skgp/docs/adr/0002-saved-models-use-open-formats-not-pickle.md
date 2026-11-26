---
status: superseded by ADR-0005
---

# Saved models use open formats, never pickle

A saved model is one zip file holding JSON (the settings, how the fit ended and the skgp version), the model's arrays in numpy's `.npy` format, written and read by equinox, and Parquet (the training data the model needs to predict). We do not use Python's pickle, which would be less code but runs arbitrary code when a file is loaded, so loading a model from someone else would be unsafe; numpy reads the arrays with pickle turned off. Loading accepts any file from the same `0.x` release line and rejects a different one with an error naming both versions.

## Considered Options

- **Every learned number as JSON text, with limits on what a file may claim:** the format while 0.0.2 was developed. skgp wrote each learned value, and the fitted distribution of a sparse model, as JSON and rebuilt the GPJax model from them, and `load` refused a file that claimed more rows, inputs or inducing points than fixed ceilings and a user setting allowed. Replaced because it was a third of skgp's code, wrote millions of numbers as text, wrote into a private field of GPJax, and refused large files that skgp itself had saved. The safety that matters, never running code from a file, is kept. The limit on how much memory a crafted file can make `load` use was dropped.

## Consequences

- The saved file contains the training data, so it is as sensitive as that data. Users are responsible for what they save. The one exception is a sparse model fitted on minibatches, whose file holds no training data (see [ADR 0004](0004-sparse-models-are-a-separate-class.md)).
- The user's own data schema is not saved, because it is code. It is passed again with `load(path, schema=...)`. Saving it automatically is tracked in issue #1.
- A file from someone else cannot run code when loaded, but it can make `load` use a lot of memory.
- The arrays are stored as GPJax lays out its models, so a change in that layout changes the saved format ([docs/version-rule.md](../version-rule.md)).

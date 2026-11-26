# skgp does not save models; users pickle a fitted model

skgp has no save or load function. A user saves a fitted model with Python's `pickle`, as scikit-learn users do, and `docs/saving.md` says how and what to watch for. This replaces [ADR 0002](0002-saved-models-use-open-formats-not-pickle.md), which saved a model as a zip of open formats that could never run code. We dropped that before the first release that would have shipped it. It took about a tenth of skgp's code. It tied a file format and a version rule to GPJax's internal layout of a model. And every new feature had to answer how it would be saved: a user's own kernel or schema is code, which the open format could not hold. A fitted model could already be pickled and predict the same after loading, so the format bought safety and portability at a cost no user had yet asked us to pay.

## Considered Options

- **Keep the open format, and pickle only a user's own kernel or schema inside it:** files from built-in kernels stay safe. Not chosen because it keeps the format and the version rule, and makes safety depend on which settings a model used.

## Consequences

- Loading a pickle file can run code, so users must only load files they trust. The docs say so.
- A pickle file loads reliably only with the versions of skgp, GPJax and JAX that saved it. skgp does not check: `to_json()` names the skgp version that fitted the model, and that version is kept when the model is pickled.
- A pickled model holds its schema when pickle can save it, so a schema no longer has to be passed again.
- Pickling leaves out the conditioned model that the first prediction keeps, which grows with the square of the training rows; the next prediction after loading makes it again.
- Pickling saves the GPJax model's arrays as numpy arrays and loading makes them 64-bit JAX arrays again. JAX makes an array 32-bit when it is unpickled outside its 64-bit mode, which skgp only switches on while it runs, so a plainly pickled model would come back with its learned numbers rounded.
- Adding a save function later breaks no user; removing one after a release would have. Anyone adding one is reopening this decision; read this record first.

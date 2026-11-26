# Saving a model

skgp has no save or load function of its own. A fitted model, a regressor's or a classifier's, is an ordinary Python object, so you save it with Python's `pickle` module, as you would a scikit-learn model:

```python
import pickle

import numpy as np
import polars as pl
from skgp import GPRegressor

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 30)
train = pl.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.1, 30)})
fitted = GPRegressor(features=["x"], target="y").fit(train)

# Save the fitted model to a file.
with open("model.pkl", "wb") as file:
    pickle.dump(fitted, file)

# Load it back, in this session or a later one.
with open("model.pkl", "rb") as file:
    loaded = pickle.load(file)

new = pl.DataFrame({"x": [2.5, 7.5]})
print(loaded.predict(new).to_frame())  # the same predictions as fitted.predict(new)
```

The file holds everything the model needs to predict: its settings, what it learned, how skgp rescaled your data, and the training data. Your schema is saved too, if pickle can save it (a pandera schema can). The first prediction after loading takes longer, because skgp prepares the training data again; it leaves that out of the file to keep the file small.

## Before you rely on it

- **Only load files you trust.** Loading a pickle file can run any code its author chose, so never load a model from someone you do not trust. See the warning in [Python's pickle documentation](https://docs.python.org/3/library/pickle.html).
- **Load with the versions that saved it.** The file holds skgp's and GPJax's objects as those versions lay them out. Load it with the same versions of skgp, GPJax and JAX: another version may fail to load it or, rarely, give a model that behaves differently. `fitted.to_json()` names the skgp version that fitted the model. To keep the versions, record your environment, for example in a lock file.
- **The file holds your training data.** Treat it as you treat the data. A {py:class}`~skgp.SparseGPRegressor` fitted with a `minibatch_size` or with heteroscedastic noise, and a {py:class}`~skgp.SparseGPClassifier`, keep no training data. A {py:class}`~skgp.GPClassifier`, and a {py:class}`~skgp.GPRegressor` with heteroscedastic noise, keep your input columns, rescaled, but not the target column.
- **When in doubt, fit again.** If you keep the data and the model's settings, you can fit the model again with any version of skgp.

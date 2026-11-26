# skgp

**Fit Gaussian process models from dataframes, without learning JAX.**

A Gaussian process is a flexible way to learn a smooth curve from data and to say how uncertain you are about it. The libraries that fit them well ask you to build several pieces by hand. skgp does that setup for you: you describe the model with your column names, call `.fit()` and `.predict()`, and get answers back as dataframes. skgp is built on [GPJax](https://github.com/QuantClimate/GPJax), which does the fitting.

## tl;dr

Install it:

```bash
pip install skgp
```

Describe a model with your column names, fit it to a dataframe, and predict for new rows:

```python
import numpy as np
import polars as pl
from skgp import GPRegressor

# Made-up data: temperature falls with latitude and altitude, plus some noise.
rng = np.random.default_rng(0)
latitude = rng.uniform(35, 60, 40)
altitude = rng.uniform(0, 2000, 40)
noise = rng.normal(0, 0.5, 40)
temperature = 30 - 0.4 * (latitude - 35) - 0.006 * altitude + noise
train = pl.DataFrame(
    {"latitude": latitude, "altitude": altitude, "temperature": temperature}
)
new = pl.DataFrame({"latitude": [45.0, 52.0], "altitude": [100.0, 800.0]})

model = GPRegressor(features=["latitude", "altitude"], target="temperature")
fitted = model.fit(train)

prediction = fitted.predict(new)
print(prediction.to_frame())  # your rows, then the predicted mean and std
print(prediction.interval(0.95))  # lower and upper bounds for each row
```

That is the whole workflow. The rest of what you need to know:

- **Three steps, three objects.** A {py:class}`~skgp.GPRegressor` describes a model: the input columns (`features`), the `target` column and the `kernel`. Its `.fit(df)` returns a separate {py:class}`~skgp.FittedGPRegressor`, whose `.predict(rows)` returns a {py:class}`~skgp.Prediction`. No object changes after it is made; `model.replace(kernel="rbf")` returns a new model.
- **Dataframes in, dataframes out.** Polars and pandas dataframes both work, and results come back in the type you passed in. You never import JAX.
- **Results are in your units.** skgp rescales the inputs and the target before fitting and converts predictions back. `prediction.mean()` and `prediction.std()` are numpy arrays in the target's units.
- **A prediction is for a new observation unless you ask for the curve.** By default the standard deviation includes measurement noise. `fitted.predict(new, include_noise=False)` gives the underlying curve: the same mean, with a smaller standard deviation. `prediction.sample(n_samples)` gives random samples, one value per row in each, chosen jointly so that nearby rows hang together, in the shape scikit-learn's `sample_y` gives.
- **Three kernels by name, or any GPJax kernel.** The kernel sets the shape of the curve: `"matern52"` (the default), `"matern32"` for a rougher curve, or `"rbf"` for a very smooth one. For any other shape, such as one that repeats, pass a GPJax kernel, for example `gpjax.kernels.RBF() + gpjax.kernels.Periodic(period=52.0)`. Every kernel sees your inputs in their own units, so a value that measures distance along an input, such as a length scale or a period, is in those units, both as you give it and as the fit reports it. A date input, an input column of dates or of dates and times, is counted in days since 1970-01-01, so a period of 365.25 is one year.
- **A mean function for far from the data.** Near the data, the curve follows the data; far from it, the curve goes to its `mean_function`: `"constant"` (the default), learned from the data, `"zero"`, in the target's units, `"linear"`, which keeps rising or falling as the data does, or a GPJax mean function.
- **Noise that is constant, that varies, or that you know.** A regressor learns one noise level for every row unless you pass `learned_noise="heteroscedastic"`, which learns noise that varies with the inputs. A {py:class}`~skgp.GPRegressor` then fits by variational inference with an inducing point at every row, as a classifier does. When each row comes with its own known noise, such as an instrument's stated error, name that column with `noise_column` on a {py:class}`~skgp.GPRegressor`: the model learns a constant noise on top of it, or none with `learned_noise=None`.
- **Mistakes stop early, with a plain message.** A wrong setting raises `ValueError` when you define the model. Missing or non-numeric columns, missing or infinite values, and fewer than 5 training rows raise an error that names the column or the number of rows. `model.check(df)` runs the data checks ahead of a long fit and returns a {py:class}`~skgp.CheckedData` that `fit` accepts without checking again. Pass `schema=` any object with a `.validate(frame)` method, such as a pandera schema, to add your own checks.
- **Large tables need the sparse model.** The time an exact fit takes grows with the cube of the number of rows, and its memory with the square, so {py:class}`~skgp.GPRegressor` becomes impractical somewhere past 10,000 rows and warns you with a {py:class}`~skgp.LargeDataWarning`. {py:class}`~skgp.SparseGPRegressor` takes the same settings and is fitted and used in the same way. It summarises the data with a smaller set of points (inducing points), so it trades some accuracy for speed and memory. {py:class}`~skgp.SparseGPClassifier` does the same for a classifier. skgp never switches to either for you.
- **Two classes, too.** {py:class}`~skgp.GPClassifier` takes the same settings, and its target may hold any two values: numbers, text, or true and false. Its `.fit(df)` returns a {py:class}`~skgp.FittedGPClassifier`, whose `.predict(rows)` returns a {py:class}`~skgp.ClassPrediction`: the probability of each class at each row, and the likelier class.
- **A fit that does not finish warns you.** `fit` still returns a model, but issues a {py:class}`~skgp.ConvergenceWarning` that says why fitting stopped. Every `fit` takes `iterations`, 2,000 unless you give another number: exactly the number of iterations a fit on minibatches runs, and the most any other fit may take. A fit on minibatches is not judged, so it never warns.
- **Score a fit on held-out data.** `fitted.score(held_out)` returns a {py:class}`~skgp.Score`: by default the mean CRPS of a regressor's predictions, in the target's units, or the Brier loss of a classifier's, from 0 (perfect) to 1, each with its sign flipped so that higher is better. `scoring=` takes scikit-learn's scorer names too, such as `"r2"`.
- **Works with scikit-learn.** `skgp.sklearn` has an estimator for each model, such as `skgp.sklearn.GaussianProcessRegressor`, which takes `X` and `y` and works in a `Pipeline`, `cross_val_score` and `GridSearchCV`. See [Using skgp with scikit-learn](sklearn.md).
- **Save a model with pickle.** skgp has no save function of its own: save a fitted model with Python's `pickle`, as you would a scikit-learn model. Only load files you trust, with the same versions of skgp, GPJax and JAX that saved them. See [Saving a model](saving.md).
- **You can see what skgp does.** Add `verbose=True` to `check`, `fit` or `predict` to log each check and each choice. `print(fitted)` summarises what the model learned, with length scales in your units, and `fitted.to_json()` gives the same as JSON.

Not supported yet: classification with more than two classes, models for data on graphs, extreme-value models, fully Bayesian fitting, and non-Gaussian noise in regression.

## Where next

- [Installation](installation.md): pip, uv, the development version, and running on a GPU.
- [Saving a model](saving.md): saving a fitted model with pickle, and what to watch for.
- [Using skgp with scikit-learn](sklearn.md): estimators for pipelines, cross-validation and grid searches.
- [API reference](reference/index.md): every public class and function, with all their settings.

```{toctree}
:hidden:

installation
saving
sklearn
reference/index
```

# skgp

skgp lets anyone fit Gaussian process models from dataframes, without needing to know JAX or GPJax. It is for QuantClimate's own work first and for anyone who finds it useful.

## Language

**GPRegressor**:
A user-defined description of a regression model: which dataframe columns are the inputs, which is the target, and optionally which kernel and mean function to use, what noise it learns, which column holds known noise, and a schema for checking data. It never changes; fitting it produces a separate fitted model. It is the first model skgp offers.

**SparseGPRegressor**:
A user-defined description of a regression model for datasets too large for a GPRegressor. It names the same things (inputs, target, kernel, mean function, learned noise, optionally a schema), but no known noise, and fits an approximation that summarises the data with a smaller set of points, trading some accuracy for speed and memory. The user chooses whether it works through the data in small batches. It never changes; fitting it produces a separate fitted model. skgp never picks it for the user: a GPRegressor with constant or known noise always means an exact model.

**Known noise**:
Noise on each observation that the user supplies: a standard deviation for every row, in the target's units, held in a column of the data, such as an instrument's stated error.
_Avoid_: output noise, alpha

**Learned noise**:
Noise on each observation that a regressor learns from the data: constant or heteroscedastic. A GPRegressor with known noise may learn constant noise on top, and then the two variances add, or learn none.

**Heteroscedastic noise**:
Learned noise whose size varies with the inputs, as opposed to constant noise, one size for every row.

**Noise process**:
In a regressor with heteroscedastic noise, the second Gaussian process, beside the curve: its value at each input is the logarithm of the noise variance there. Its kernel is the noise kernel.

**Mean function**:
What a model's curve is far from the training data, where the data says nothing: zero, a constant learned from the data, a number plus a slope along each input learned from the data, or a function the user gives. Near the data, the curve follows the data whatever the mean function is.
_Avoid_: prior mean, trend

**Date input**:
An input column that holds dates, or dates and times, rather than numbers. A model counts it in days since 1970-01-01 in UTC.
_Avoid_: date column

**FittedGPRegressor**:
A GPRegressor or SparseGPRegressor after it has been fitted to data; both produce this same kind of object. It can make predictions and describe itself as JSON or as a short printed summary. It never changes.

**GPClassifier**:
A user-defined description of a classification model: which dataframe columns are the inputs, which column is the target, and optionally which kernel and mean function to use and a schema for checking data. The target holds two classes. It never changes; fitting it produces a separate fitted model.

**Class**:
One of the two values a classifier's target holds, such as "rain" and "dry", or true and false. It is the user's own value, never a code.

**SparseGPClassifier**:
A user-defined description of a classification model for datasets too large for a GPClassifier. It names the same things and summarises the data with a smaller set of points, as a SparseGPRegressor does, and the user chooses whether it works through the data in small batches. It never changes; fitting it produces a separate fitted model. skgp never picks it for the user.

**FittedGPClassifier**:
A GPClassifier or SparseGPClassifier after it has been fitted to data; both produce this same kind of object. It can make predictions and describe itself as JSON or as a short printed summary. It never changes.

**Score**:
How well a fitted model's predictions match held-out data with known targets, by one or more scorers. Following scikit-learn, every value is one where higher is better.

**Estimator**:
A model that follows scikit-learn's rules, so that it works in a pipeline, cross-validation and a grid search: it takes its inputs as `X` and its target as `y`, and fitting changes it rather than returning a separate fitted model. skgp has one for each of its four models, each describing that model and, once fitted, holding its fitted model.
_Avoid_: adapter, wrapper

**CheckedData**:
A dataframe that has passed every data check of a model. It is valid for any model of the same task, regression or classification, with the same inputs, target and schema, whether exact or sparse and whatever its kernel, because only those decide what the checks are.

**Prediction**:
What a fitted regression model returns when asked about new rows: an object describing what the model expects at each row, in the user's original units. By default that is a new observation, noise included; asked for without noise, it is the underlying curve. It is more than a table of mean and spread, so that richer results can be added later.

**ClassPrediction**:
What a fitted classifier returns when asked about new rows: the probability of each of its two classes at each row, and the likelier class.

**Sample**:
One set of values of a prediction, a value at every row of it, chosen at random from what the model expects of all the rows together, so that the rows hang together as it expects. A classifier's sample holds a class at every row. Never a subset of rows.
_Avoid_: draw, realisation

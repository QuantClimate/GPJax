class ConvergenceWarning(UserWarning):
    """Fitting did not finish properly, so the fitted model may be less reliable.

    `fit` issues it once per fit when the search for the model's settings had not
    settled when it stopped, judged from where it ended rather than from what the
    optimiser reported. The message says why it stopped, in plain words. The fit
    still returns normally. Filter it with the `warnings` module, or turn it into
    an error with `warnings.simplefilter("error", skgp.ConvergenceWarning)`.
    """


class LargeDataWarning(UserWarning):
    """`GPRegressor.fit` or `GPClassifier.fit` was given more rows than
    `skgp.config.size_warning_rows`.

    Both models get slow and need a lot of memory as rows grow, so the warning suggests
    `SparseGPRegressor` or `SparseGPClassifier`. The fit still goes ahead and returns normally,
    and the limit never changes which model is fitted. Filter it with the `warnings`
    module, or raise the limit.
    """

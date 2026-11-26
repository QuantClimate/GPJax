"""scikit-learn's own checks of an estimator, run against skgp's four estimators.

They fit many times, so they are marked slow. The checks each estimator is expected to fail
are listed with the reason.
"""

import pytest
from sklearn.base import is_regressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from skgp.sklearn import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    SparseGaussianProcessClassifier,
    SparseGaussianProcessRegressor,
)

# A short fit is enough to check that each estimator follows scikit-learn's rules.
ITERATIONS = 50


def expected_failures(estimator):
    failures = {
        "check_fit2d_1sample": (
            "skgp refuses to fit fewer than 5 rows, with its own message, which does not "
            "use the words scikit-learn looks for."
        )
    }
    if is_regressor(estimator):
        failures["check_regressors_train"] = (
            "score is the mean CRPS with its sign flipped, which is never above 0, while "
            "this check expects an R² above 0.5."
        )
    return failures


@pytest.mark.slow
@parametrize_with_checks(
    [
        GaussianProcessRegressor(iterations=ITERATIONS),
        SparseGaussianProcessRegressor(iterations=ITERATIONS),
        GaussianProcessClassifier(iterations=ITERATIONS),
        SparseGaussianProcessClassifier(iterations=ITERATIONS),
    ],
    expected_failed_checks=expected_failures,
)
# scikit-learn's checks fit with a small iteration limit on purpose.
@pytest.mark.filterwarnings("ignore::skgp.ConvergenceWarning")
def test_each_estimator_passes_scikit_learns_checks(estimator, check):
    check(estimator)

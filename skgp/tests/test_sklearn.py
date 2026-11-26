"""The scikit-learn estimators in `skgp.sklearn`: skgp's models with scikit-learn's rules."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from helpers import WEATHER_GROUPS
from skgp.sklearn import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    SparseGaussianProcessRegressor,
    neg_crps,
)

# Twenty readings of y = 2x from x = 0 to 9.5, wobbling by 0.05 either way: at x = 4.25 the
# line is at 8.5.
X_LINE = np.arange(0.0, 10.0, 0.5).reshape(-1, 1)
Y_LINE = 2 * X_LINE[:, 0] + 0.05 * (-1) ** np.arange(20)


@pytest.fixture(scope="module")
def line_estimator():
    estimator = GaussianProcessRegressor().fit(X_LINE, Y_LINE)
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    estimator.predict(X_LINE[:1], return_std=True)
    return estimator


def test_fit_returns_the_estimator_fitted_and_it_predicts(line_estimator):
    assert line_estimator.predict([[4.25]])[0] == pytest.approx(8.5, abs=0.1)


def test_an_array_s_columns_are_x0_x1_and_a_dataframe_s_keep_their_names(
    line_estimator,
):
    assert line_estimator.fitted_.model.features == ("x0",)
    assert not hasattr(line_estimator, "feature_names_in_")

    named = GaussianProcessRegressor().fit(pd.DataFrame({"week": X_LINE[:, 0]}), Y_LINE)

    assert named.fitted_.model.features == ("week",)
    assert list(named.feature_names_in_) == ["week"]


def test_predict_refuses_to_give_both_the_standard_deviation_and_the_covariance(
    line_estimator,
):
    with pytest.raises(
        RuntimeError, match=r"^At most one of return_std and return_cov"
    ):
        line_estimator.predict([[2.0]], return_std=True, return_cov=True)


def test_predict_includes_measurement_noise_unless_asked_for_the_curve(line_estimator):
    _, observation = line_estimator.predict([[2.0]], return_std=True)
    _, curve = line_estimator.predict([[2.0]], return_std=True, include_noise=False)

    assert curve[0] < observation[0]


def test_sample_y_gives_samples_around_the_line(line_estimator):
    samples = line_estimator.sample_y([[2.0], [3.0]], n_samples=200)

    np.testing.assert_allclose(samples.mean(axis=1), [4.0, 6.0], atol=0.1)


def test_neg_crps_is_the_mean_crps_with_its_sign_flipped():
    # A prediction of mean 0 and standard deviation 1 for an observation of 0 has a CRPS of
    # 2 times the normal density at 0 minus 1 over the square root of pi:
    # 2 * 0.398942 - 0.564190 = 0.233695.
    standard_normal = SimpleNamespace(
        predict=lambda X, return_std: (np.zeros(len(X)), np.ones(len(X)))
    )

    assert neg_crps(standard_normal, [[0.0]], [0.0]) == pytest.approx(
        -0.233695, abs=1e-6
    )


def test_a_regressor_s_score_is_the_mean_crps_of_its_predictions_sign_flipped(
    line_estimator,
):
    # The CRPS of a normal prediction with mean m and standard deviation s of an observed y
    # is s (z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)), where z = (y - m) / s.
    mean, std = line_estimator.predict(X_LINE, return_std=True)
    z = (Y_LINE - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    assert line_estimator.score(X_LINE, Y_LINE) == pytest.approx(-crps.mean(), rel=1e-9)


def test_x_without_column_names_is_matched_to_the_named_columns_by_position():
    named = GaussianProcessRegressor().fit(pd.DataFrame({"week": X_LINE[:, 0]}), Y_LINE)

    with pytest.warns(UserWarning, match="X does not have valid feature names"):
        mean = named.predict(np.array([[4.25]]))

    assert mean[0] == pytest.approx(8.5, abs=0.1)


@pytest.fixture(scope="module")
def weather_estimator():
    X = pd.DataFrame({"x": WEATHER_GROUPS["x"].to_numpy()})
    estimator = GaussianProcessClassifier().fit(X, WEATHER_GROUPS["weather"].to_numpy())
    # JAX compiles on the first prediction; pay for it here, once, not in a test.
    estimator.predict_proba(X.head(1))
    return estimator


def test_a_classifier_predicts_your_classes_with_their_probabilities_in_order(
    weather_estimator,
):
    rows = pd.DataFrame({"x": [-3.0, 3.0]})

    assert list(weather_estimator.classes_) == ["dry", "rain"]
    assert list(weather_estimator.predict(rows)) == ["dry", "rain"]
    probability_of_rain = weather_estimator.predict_proba(rows)[:, 1]
    assert probability_of_rain[0] < 0.05
    assert probability_of_rain[1] > 0.95


def test_a_classifier_s_score_is_the_brier_loss_with_its_sign_flipped(
    weather_estimator,
):
    # Rows far from the boundary, each predicted with a probability within 0.05 of its own
    # class, so the squared differences, and their mean, are below 0.05 squared.
    rows = pd.DataFrame({"x": [-3.0, 3.0]})

    assert -0.0025 < weather_estimator.score(rows, ["dry", "rain"]) <= 0.0


def test_a_classifier_refuses_more_than_two_classes():
    with pytest.raises(
        ValueError, match=r"^Only binary classification is supported\. The type of the"
    ):
        GaussianProcessClassifier().fit(X_LINE, ["a", "b", "c", "d"] * 5)


def test_settings_are_kept_as_given_and_checked_at_fit():
    estimator = GaussianProcessRegressor(learned_noise="varying")  # ty: ignore[invalid-argument-type]  # the wrong value is the point

    assert clone(estimator).get_params()["learned_noise"] == "varying"
    with pytest.raises(ValueError, match=r"^learned_noise must be"):
        estimator.fit(X_LINE, Y_LINE)


def test_a_column_of_x_named_y_is_refused():
    with pytest.raises(ValueError, match=r"^X has a column named 'y'"):
        GaussianProcessRegressor().fit(pd.DataFrame({"y": X_LINE[:, 0]}), Y_LINE)


def test_known_noise_comes_from_a_column_of_x_and_is_not_an_input():
    # Column x1 is the known noise: 0.05 on every row.
    X = np.column_stack([X_LINE[:, 0], np.full(20, 0.05)])

    estimator = GaussianProcessRegressor(noise_column="x1", learned_noise=None).fit(
        X, Y_LINE
    )

    assert estimator.fitted_.model.features == ("x0",)
    # The prediction of the curve alone still takes the column, as scikit-learn wants the
    # columns of fit, but does not use it.
    _, observation = estimator.predict([[2.0, 0.5]], return_std=True)
    _, curve = estimator.predict([[2.0, 99.0]], return_std=True, include_noise=False)
    assert observation[0] ** 2 - curve[0] ** 2 == pytest.approx(0.25, abs=1e-9)


@pytest.mark.slow
def test_an_estimator_works_in_a_pipeline_in_cross_validation_and_in_a_grid_search():
    pipeline = make_pipeline(
        ColumnTransformer([("scale", StandardScaler(), [0])]),
        SparseGaussianProcessRegressor(num_inducing=10),
    )

    scores = cross_val_score(pipeline, X_LINE, Y_LINE, cv=2)
    search = GridSearchCV(
        GaussianProcessRegressor(),
        {"kernel": ["rbf", "matern32"]},
        cv=2,
        scoring=neg_crps,
    ).fit(X_LINE, Y_LINE)

    assert len(scores) == 2
    assert all(score < 0 for score in scores)  # a mean CRPS above 0, its sign flipped
    assert search.best_params_["kernel"] in {"rbf", "matern32"}

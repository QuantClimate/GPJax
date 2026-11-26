import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
DOCS_LANDING_PAGE = ROOT / "docs" / "index.md"
DOCS_SAVING_PAGE = ROOT / "docs" / "saving.md"
DOCS_SKLEARN_PAGE = ROOT / "docs" / "sklearn.md"


def run_python_blocks(page: Path) -> dict[str, Any]:
    """Run every Python block of `page` in order, in one namespace, and return it.

    Blocks share one namespace, in page order, so a later block may use names from an
    earlier one. Call it from an empty folder, so any file a block saves lands there.
    """
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.DOTALL)
    assert blocks, f"{page.name} has no Python block"
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), namespace)
    return namespace


@pytest.mark.slow
def test_readme_python_blocks_run_in_order_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    namespace = run_python_blocks(README)

    prediction = namespace["prediction"]
    assert {"mean", "std"} <= set(prediction.to_frame().columns)
    assert list(prediction.interval(0.95).columns) == ["lower", "upper"]
    description = json.loads(namespace["sparse_fitted"].to_json())
    assert description["model"]["class"] == "SparseGPRegressor"
    assert description["model"]["minibatch_size"] is not None
    rain_prediction = namespace["rain_prediction"]
    assert rain_prediction.classes == ("dry", "rain")
    # More rain further north: 55 degrees is more likely to be wet than 40 degrees.
    assert rain_prediction.probabilities()[1, 1] > rain_prediction.probabilities()[0, 1]
    # The season repeats every 52 weeks, and the period is learned in weeks.
    learned = json.loads(namespace["seasonal_fitted"].to_json())["learned"]
    assert learned["kernel"]["kernels[1].period"] == pytest.approx(52.0, rel=0.02)


@pytest.mark.slow
def test_docs_landing_page_python_blocks_run_in_order_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    namespace = run_python_blocks(DOCS_LANDING_PAGE)

    prediction = namespace["prediction"]
    assert {"mean", "std"} <= set(prediction.to_frame().columns)
    assert list(prediction.interval(0.95).columns) == ["lower", "upper"]


@pytest.mark.slow
def test_docs_saving_page_python_blocks_run_and_the_loaded_model_predicts_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    namespace = run_python_blocks(DOCS_SAVING_PAGE)

    new, fitted, loaded = namespace["new"], namespace["fitted"], namespace["loaded"]
    np.testing.assert_allclose(
        loaded.predict(new).mean(), fitted.predict(new).mean(), rtol=1e-9
    )


@pytest.mark.slow
def test_docs_sklearn_page_python_blocks_run_and_the_known_noise_widens_the_spread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    namespace = run_python_blocks(DOCS_SKLEARN_PAGE)

    assert namespace["search"].best_params_["kernel"] in {"rbf", "matern32", "matern52"}
    # Each spread of a new reading is at least that reading's known noise.
    errors = namespace["readings"]["error"].to_numpy()[:3]
    assert np.all(namespace["known_std"] >= errors)

"""Tests for GPJax imports and exports."""

import gpjax


def test_all_exports_importable():
    """Test that all items in __all__ are actually importable."""
    for item in gpjax.__all__:
        assert hasattr(gpjax, item), f"{item} not available in gpjax module"


def test_common_imports():
    """Test that common import patterns work correctly."""
    # Test basic imports
    assert hasattr(gpjax, "gps")
    assert hasattr(gpjax, "kernels")
    assert hasattr(gpjax, "likelihoods")
    assert hasattr(gpjax, "Dataset")
    assert hasattr(gpjax, "fit")
    assert hasattr(gpjax, "natural_gradients")

    # Test callable functions
    assert callable(gpjax.fit)
    assert callable(gpjax.fit_lbfgs)
    assert callable(gpjax.fit_natgrads)
    assert callable(gpjax.fit_scipy)
    assert callable(gpjax.cite)


def test_gev_names_are_exported():
    """The GEV distribution, its functions and the GEV likelihoods are public."""
    import gpjax.distributions
    import gpjax.likelihoods

    for name in (
        "GeneralisedExtremeValue",
        "gev_return_level",
        "gev_return_period",
    ):
        assert name in gpjax.distributions.__all__
        assert hasattr(gpjax.distributions, name)
    for name in ("GEV", "HeteroscedasticGEV"):
        assert name in gpjax.likelihoods.__all__
        assert hasattr(gpjax.likelihoods, name)

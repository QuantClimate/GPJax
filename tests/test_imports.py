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

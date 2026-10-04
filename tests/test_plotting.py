import sys

import pytest

mpl = pytest.importorskip("matplotlib")

from gpjax import plotting


def test_use_style_applies_ledger_style_when_qc_core_is_installed() -> None:
    pytest.importorskip("qc_core")
    with mpl.rc_context():
        assert plotting.use_style() == "ledger"
        assert mpl.rcParams["figure.facecolor"] == "#faf9f7"


def test_use_style_falls_back_to_packaged_style_without_qc_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A `None` entry in sys.modules makes `import qc_core` raise ImportError.
    monkeypatch.setitem(sys.modules, "qc_core", None)
    expected = mpl.rc_params_from_file(plotting.STYLE_PATH, use_default_template=False)
    with mpl.rc_context():
        assert plotting.use_style() == "gpjax"
        assert mpl.rcParams["axes.prop_cycle"] == expected["axes.prop_cycle"]


def test_use_style_raises_clear_error_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="needs Matplotlib"):
        plotting.use_style()


def test_fallback_style_ships_with_the_package() -> None:
    assert plotting.STYLE_PATH.is_file()

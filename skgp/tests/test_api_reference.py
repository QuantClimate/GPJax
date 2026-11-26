"""The API reference page lists every public class, so none is left without a page."""

from pathlib import Path
import re

import skgp

API_REFERENCE_PAGE = Path(__file__).parent.parent / "docs" / "reference" / "index.md"


def test_every_public_class_has_a_page_in_the_api_reference():
    # An autosummary entry is a class name alone on a line, indented three spaces.
    listed = set(
        re.findall(r"^   (\w+)$", API_REFERENCE_PAGE.read_text(), flags=re.MULTILINE)
    )
    public_classes = {
        name for name in skgp.__all__ if isinstance(getattr(skgp, name), type)
    }

    assert public_classes - listed == set()

"""How skgp writes counts and lists of names in its messages, errors and printed results."""

from skgp._constants import Constant as C


def count(n: int, noun: str) -> str:
    """`n` followed by `noun`, plural unless `n` is 1: "1 row", "2 rows", "1,000 rows"."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def joined(items: list[str], conjunction: str = "and") -> str:
    """'a', 'a and b', or 'a, b and c', with another `conjunction` in place of 'and' if given."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def quoted(names: list[str]) -> str:
    """The names in single quotes, joined: "'a', 'b' and 'c'"."""
    return joined([f"'{name}'" for name in names])


def listed(items: list[str]) -> str:
    """`items` joined by commas, the first `C.MAX_COLUMNS_LISTED` of them and how many more
    there are."""
    shown = ", ".join(items[: C.MAX_COLUMNS_LISTED])
    hidden = len(items) - C.MAX_COLUMNS_LISTED
    return shown if hidden <= 0 else f"{shown} and {hidden:,} more"

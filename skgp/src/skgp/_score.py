"""`Score`: how well a fitted model's predictions match held-out data with known targets."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from skgp._text import count


@dataclass(frozen=True, eq=False, repr=False)
class Score(Mapping[str, float]):
    """How well a fitted model's predictions match held-out data with known targets, by one
    or more scorers: what `fitted.score(data)` returns.

    It maps the name of each scorer to its value, so `score["neg_crps"]` reads one, and
    `dict(score)` gives them all. Following scikit-learn, every value is one where higher is
    better: a loss, such as the CRPS or the Brier loss, has its sign flipped, and its name
    starts with `neg_`. Printing it shows a short table. It never changes.

    Like a dict, it is equal to any mapping with the same names and values.
    """

    _values: Mapping[str, float]
    number_of_rows: int
    """How many rows of data the predictions were scored on."""

    def __post_init__(self) -> None:
        # A copy that cannot be changed, so neither the caller's dict nor anyone holding
        # the score can change it afterwards.
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def __getitem__(self, name: str) -> float:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        width = max((len(name) for name in self._values), default=0)
        lines = [
            f"  {name:<{width}}  {value:.6g}" for name, value in self._values.items()
        ]
        title = f"Score on {count(self.number_of_rows, 'row')} (higher is better):"
        return "\n".join([title, *lines])

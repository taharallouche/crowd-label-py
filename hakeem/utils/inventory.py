from dataclasses import dataclass


@dataclass(frozen=True)
class _DEFAULT_RELIABILITY_BOUNDS:
	lower: float = 10**-3
	upper: float = 1 - 10**-3


DEFAULT_RELIABILITY_BOUNDS = _DEFAULT_RELIABILITY_BOUNDS()


@dataclass(frozen=True)
class _COLUMNS:
	question: str = "Question"
	voter: str = "Voter"


COLUMNS = _COLUMNS()

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
	path: str
	alternatives: list
	nbr_questions: int
	nbr_voters: int


@dataclass(frozen=True)
class _COLUMNS:
	interface: str = "Interface"
	mechanism: str = "Mechanism"
	question: str = "Question"
	true_answer: str = "TrueAnswer"
	answer: str = "Answer"
	comments: str = "Comments"
	voter: str = "Voter"
	weight: str = "Weight"


COLUMNS = _COLUMNS()


ANIMALS_DATASET = Dataset(
	"data/animals/raw.csv",
	["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"],
	16,
	110,
)

TEXTURES_DATASET = Dataset(
	"data/textures/raw.csv",
	["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"],
	16,
	96,
)

LANGUAGE_DATASET = Dataset(
	"data/languages/raw.csv",
	[
		"Hebrew",
		"Russian",
		"Japanese",
		"Thai",
		"Chinese",
		"Tamil",
		"Latin",
		"Hindi",
	],
	25,
	109,
)

DATASETS = {
	"animals": ANIMALS_DATASET,
	"textures": TEXTURES_DATASET,
	"languages": LANGUAGE_DATASET,
}

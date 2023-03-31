from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    alternatives: list
    nbr_questions: int


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


@dataclass(frozen=True)
class _RULES:
    standard_approval_voting: str = "Standard Approval Voting"
    euclid: str = "Euclid"
    jaccard: str = "Jaccard"
    dice: str = "Dice"
    condorcet: str = "Condorcet"


RULES = _RULES()


COLUMNS = _COLUMNS()


ANIMALS_DATASET = Dataset(
    "animals",
    "data/data_animals.csv",
    ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"],
    16,
)

TEXTURES_DATASET = Dataset(
    "textures",
    "data/data_textures.csv",
    ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"],
    16,
)

LANGUAGE_DATASET = Dataset(
    "languages",
    "data/data_languages.csv",
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
)

DATASETS = {
    "animals": ANIMALS_DATASET,
    "textures": TEXTURES_DATASET,
    "languages": LANGUAGE_DATASET,
}


PLOT_OPTIONS = {
    RULES.standard_approval_voting: {"linestyle": "solid", "index": 0},
    RULES.euclid: {"linestyle": "dashdot", "index": 1},
    RULES.jaccard: {"linestyle": "dashed", "index": 2},
    RULES.dice: {"linestyle": (0, (3, 5, 1, 5)), "index": 3},
    RULES.condorcet: {"linestyle": "dotted", "index": 4},
}

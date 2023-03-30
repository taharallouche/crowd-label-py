from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    path: str
    alternatives: list
    nbr_questions: int


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
    "SAV": {"linestyle": "solid", "index": 0},
    "Euclid": {"linestyle": "dashdot", "index": 1},
    "Jaccard": {"linestyle": "dashed", "index": 2},
    "Dice": {"linestyle": (0, (3, 5, 1, 5)), "index": 3},
    "Condorcet": {"linestyle": "dotted", "index": 4},
}

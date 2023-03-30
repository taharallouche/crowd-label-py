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
